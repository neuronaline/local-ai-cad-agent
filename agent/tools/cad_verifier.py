"""Deterministic CAD verification tool for OpenSCAD code and STL mesh topology.

Performs rule-based, non-speculative static analysis of OpenSCAD code and
exact mathematical mesh topology validation without hallucination or fuzzy guessing.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.llm_base import RequestCancelled
from agent.print_advisor import parse_nozzle_spec, parse_printer_spec
from agent.revisions import MODEL_FILENAME, model_is_built
from agent.tools.cad_scripts.renderer import load_stl
from agent.tools.cad_scripts.runner import mesh_metrics
from agent.tools.file_tool import OpenScadPreflight
from agent.tools.tool_events import publish_tool_phase


@dataclass(frozen=True)
class Finding:
    category: str  # "CRITICAL", "WARNING", "INFO"
    code: str
    message: str
    line: int | None = None
    impact: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "category": self.category,
            "code": self.code,
            "message": self.message,
        }
        if self.line is not None:
            result["line"] = self.line
        if self.impact is not None:
            result["impact"] = self.impact
        return result


_clean_code_preserve_lines = OpenScadPreflight.clean_code_preserve_lines


def _strip_comments_preserve_lines(code: str) -> str:
    """Strip line and block comments while preserving string literals and line numbers."""
    def _repl(match: re.Match) -> str:
        s = match.group(0)
        if s.startswith('"'):
            return s
        return "\n" * s.count("\n")

    return re.sub(r'"(?:\\.|[^"\\])*"|/\*.*?\*/|//[^\n]*', _repl, code, flags=re.DOTALL)


def _detect_expected_part_count(code: str) -> int | None:
    """Extract the expected number of parts when PART_TYPE defines explicit single-part variants."""
    clean = _strip_comments_preserve_lines(code)
    multi_tokens = {
        "ALL",
        "PLATE",
        "BOTH",
        "ASSEMBLY",
        "SET",
        "COMBINED",
        "PRINT_BED",
    }
    variants: set[str] = set()
    for m in re.finditer(
        r"\bPART_TYPE\s*==\s*[\"\']([A-Za-z0-9_]+)[\"\']",
        clean,
    ):
        token = m.group(1).upper()
        if token not in multi_tokens:
            variants.add(token)
    if len(variants) >= 2:
        return len(variants)

    # Also check Customizer option comment: PART_TYPE = "ALL"; // [ALL, BOTTLE, CAP]
    comment_match = re.search(
        r"^[ \t]*PART_TYPE[ \t]*=[ \t]*[\"\'][^\"\']+[\"\'][ \t]*;[ \t]*//[ \t]*\[([^\]]+)\]",
        code,
        re.MULTILINE,
    )
    if comment_match:
        opts = [
            opt.strip().split(":")[0].strip().strip("\"'").upper()
            for opt in comment_match.group(1).split(",")
        ]
        single_opts = {o for o in opts if o and o not in multi_tokens}
        if len(single_opts) >= 2:
            return len(single_opts)
    return None


def _detect_multi_part_intent(code: str) -> bool:
    """Detect whether model.scad intentionally defines a multi-part assembly or plate layout."""
    clean_code = _strip_comments_preserve_lines(code)
    param_matches = re.finditer(
        r"^[ \t]*([A-Z][A-Z0-9_]*)[ \t]*=[ \t]*[\"\']?([A-Za-z0-9_]+)[\"\']?[ \t]*;",
        clean_code,
        re.MULTILINE,
    )
    selector_found = False
    for m in param_matches:
        param_name, param_val = m.group(1).upper(), m.group(2).upper()
        if param_name in ("PART_TYPE", "LAYOUT", "MODE", "VIEW", "ARRANGEMENT"):
            selector_found = True
            if param_val in (
                "ALL",
                "PLATE",
                "BOTH",
                "ASSEMBLY",
                "SET",
                "COMBINED",
                "PRINT_BED",
            ):
                return True
        if param_name in ("ASSEMBLY", "MULTI_PART", "PRINT_BED", "PLATE_LAYOUT") and param_val in (
            "TRUE",
            "1",
        ):
            return True

    # If an explicit selector was found (e.g. PART_TYPE = "50"), the user specifically
    # selected a single variant. Do not fall back to generic comment matching.
    if selector_found:
        return False

    return bool(
        re.search(
            r"//.*?\b(?:multi[- ]part|assembly|plate layout|print bed|side[- ]by[- ]side)\b",
            code,
            re.IGNORECASE,
        )
    )


class CadVerifier:
    """Deterministic verifier for CAD models, producing categorized findings and a risk score."""

    def __init__(
        self,
        project_dir: Path,
        publish: Any = None,
        *,
        user_printer: str = "",
        user_nozzle: str = "",
        user_filament: str = "PLA",
        stop_event: threading.Event | None = None,
    ) -> None:
        self.project_dir = project_dir.resolve()
        self._publish = publish
        self.user_printer = user_printer
        self.user_nozzle = user_nozzle
        self.user_filament = user_filament
        self._call_id = ""
        self._stop_event = stop_event

    def with_call_id(self, call_id: str) -> CadVerifier:
        verifier = CadVerifier(
            self.project_dir,
            self._publish,
            user_printer=self.user_printer,
            user_nozzle=self.user_nozzle,
            user_filament=self.user_filament,
            stop_event=self._stop_event,
        )
        verifier._call_id = call_id or ""
        return verifier

    def _check_cancelled(self) -> None:
        if self._stop_event is not None and self._stop_event.is_set():
            raise RequestCancelled("CAD verification cancelled.")

    def _publish_status(self, status: str, message: str) -> None:
        if self._publish:
            publish_tool_phase(
                self._publish,
                project=self.project_dir.name,
                tool="cad_build",
                call_id=self._call_id,
                status=status,
                message=message,
            )

    def verify(
        self,
        *,
        precomputed_metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run complete deterministic verification on model.scad and preview.stl."""
        self._check_cancelled()
        self._publish_status("verifying", "Running deterministic CAD verification…")

        model_path = self.project_dir / MODEL_FILENAME
        if not model_path.is_file():
            finding = Finding(
                category="CRITICAL",
                code="MODEL_MISSING",
                message=f"{MODEL_FILENAME} does not exist.",
                impact="Cannot verify a missing model file.",
            )
            return self._build_envelope(
                findings=[finding],
                code_integrity={"exists": False},
                mesh_integrity={"exists": False},
            )

        model_code = model_path.read_text(encoding="utf-8", errors="replace")
        code_findings, code_meta = self._verify_code(model_code)
        self._check_cancelled()

        preview_path = self.project_dir / "preview.stl"
        build_findings: list[Finding] = []
        is_stale = False

        if (
            precomputed_metrics is None
            and preview_path.is_file()
            and preview_path.stat().st_size > 0
            and not model_is_built(self.project_dir)
        ):
            is_stale = True
            build_findings.append(
                Finding(
                    category="CRITICAL",
                    code="STALE_PREVIEW_BUILD",
                    message="preview.stl is outdated compared to latest model.scad. Rebuild before verifying.",
                    impact="Mesh validation would inspect obsolete geometry from an earlier revision.",
                )
            )

        if is_stale:
            mesh_findings: list[Finding] = []
            mesh_meta: dict[str, Any] = {"exists": True, "stale": True}
        else:
            mesh_findings, mesh_meta = self._verify_mesh(
                preview_path,
                multi_part_intent=bool(code_meta.get("multi_part_intent")),
                expected_part_count=code_meta.get("expected_part_count"),
                precomputed_metrics=precomputed_metrics,
            )

        all_findings = code_findings + build_findings + mesh_findings
        return self._build_envelope(all_findings, code_meta, mesh_meta)

    def _verify_code(self, code: str) -> tuple[list[Finding], dict[str, Any]]:
        findings: list[Finding] = []

        # 1. Delimiter & Syntax checking
        delimiter_err = OpenScadPreflight.check_delimiters(code)
        syntax_error = bool(delimiter_err)
        if delimiter_err:
            err_code, err_msg, lineno = delimiter_err
            findings.append(
                Finding(
                    category="CRITICAL",
                    code=err_code,
                    message=err_msg,
                    line=lineno,
                    impact="OpenSCAD compiler parse error.",
                )
            )

        # 2. Python syntax checks via centralized OpenScadPreflight
        clean_code = _clean_code_preserve_lines(code)
        for line_num, kw, impact in OpenScadPreflight.check_python_syntax(clean_code):
            findings.append(
                Finding(
                    category="CRITICAL",
                    code="PYTHON_SYNTAX_DETECTED",
                    message=f"Python syntax '{kw}' found at line {line_num}.",
                    line=line_num,
                    impact=impact,
                )
            )
            syntax_error = True

        for line_num, line in enumerate(clean_code.splitlines(), start=1):
            stripped = line.strip()

            # 3. Minkowski ban
            if re.search(r"\bminkowski\s*\(", line):
                findings.append(
                    Finding(
                        category="CRITICAL",
                        code="MINKOWSKI_DETECTED",
                        message=f"Use of 3D minkowski() detected at line {line_num}.",
                        line=line_num,
                        impact="Causes severe OpenSCAD compiler freezing and timeouts. Use hull() or 2D offset() instead.",
                    )
                )

        # Active scene modifiers
        # In OpenSCAD, '!', '*', and '%' modifiers prefix a module or primitive instantiation at statement
        # boundaries (start of file or after ';', '{', or '}'). Operators in multi-line expressions
        # ('total_height = a\n    * b;') or comparisons ('!=', '*=') are not scene modifiers.
        _BOOLEAN_BUILTINS = {
            "is_undef",
            "is_num",
            "is_string",
            "is_bool",
            "is_list",
            "is_function",
        }
        for mod_m in re.finditer(
            r"(?:^|(?<=[;{}]))\s*([!*%])(?!=)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*[\(\{]",
            clean_code,
        ):
            mod_char = mod_m.group(1)
            target = mod_m.group(2)
            if mod_char == "!" and target in _BOOLEAN_BUILTINS:
                continue
            line_num = clean_code[: mod_m.start(1)].count("\n") + 1
            if mod_char == "!":
                findings.append(
                    Finding(
                        category="CRITICAL",
                        code="ACTIVE_ROOT_SCENE_MODIFIER",
                        message=f"Root scene modifier '!' found at line {line_num} ('!{target}').",
                        line=line_num,
                        impact="OpenSCAD will isolate this element and discard the rest of the model from the STL export.",
                    )
                )
            elif mod_char == "*":
                findings.append(
                    Finding(
                        category="CRITICAL",
                        code="ACTIVE_DISABLE_SCENE_MODIFIER",
                        message=f"Disable scene modifier '*' found at line {line_num} ('*{target}').",
                        line=line_num,
                        impact="This feature is disabled and will be omitted from the exported STL mesh.",
                    )
                )
            elif mod_char == "%":
                findings.append(
                    Finding(
                        category="WARNING",
                        code="BACKGROUND_MODIFIER_OMITTED",
                        message=f"Background modifier '%' found at line {line_num} ('%{target}').",
                        line=line_num,
                        impact="Background modifier geometry is visual-only in OpenSCAD GUI and omitted from STL export.",
                    )
                )

        # 4. Degree-to-radian trig conversions
        for line_num, trig_fn in OpenScadPreflight.check_radian_trig(clean_code):
            findings.append(
                Finding(
                    category="WARNING",
                    code="RADIAN_TRIG_USAGE",
                    message=f"Degree-to-radian conversion in '{trig_fn}()' at line {line_num}.",
                    line=line_num,
                    impact="OpenSCAD trig functions expect degrees directly, not radians. Scaling by PI/180 crushes angles by ~57.3x.",
                )
            )

        # 5. 3D primitives directly inside 2D extrusions
        for line_num, ext_name, prim_name in OpenScadPreflight.check_extrusion_children(clean_code):
            findings.append(
                Finding(
                    category="CRITICAL",
                    code="3D_PRIMITIVE_IN_2D_EXTRUSION",
                    message=f"3D primitive '{prim_name}()' used inside {ext_name}() at line {line_num}.",
                    line=line_num,
                    impact=f"{ext_name}() only accepts 2D child shapes (circle, square, polygon). 3D shapes cause empty geometry or CGAL evaluation failure.",
                )
            )

        # 6. Lexical Scope Block Reassignment Detection
        top_level_vars: set[str] = set()
        brace_depth = 0
        top_paren_depth = 0
        for line in clean_code.splitlines():
            open_c = line.count("{")
            close_c = line.count("}")
            open_p = line.count("(")
            close_p = line.count(")")
            if brace_depth == 0 and top_paren_depth == 0:
                assign_m = re.match(r"^[ \t]*([A-Za-z0-9_]+)[ \t]*=(?!=)", line)
                if assign_m:
                    var = assign_m.group(1)
                    if not var.startswith("$") and var != "EPS":
                        top_level_vars.add(var)
            brace_depth = max(0, brace_depth + open_c - close_c)
            top_paren_depth = max(0, top_paren_depth + open_p - close_p)

        if top_level_vars:
            for block_m in re.finditer(r"\b(if|for)\s*\(", clean_code):
                block_kind = block_m.group(1)
                paren_depth = 1
                p_idx = block_m.end()
                while p_idx < len(clean_code) and paren_depth > 0:
                    if clean_code[p_idx] == "(":
                        paren_depth += 1
                    elif clean_code[p_idx] == ")":
                        paren_depth -= 1
                    p_idx += 1
                if paren_depth != 0:
                    continue
                while p_idx < len(clean_code) and clean_code[p_idx].isspace():
                    p_idx += 1
                if p_idx >= len(clean_code) or clean_code[p_idx] != "{":
                    continue
                start_idx = p_idx
                b_depth = 1
                cur_idx = start_idx + 1
                while cur_idx < len(clean_code) and b_depth > 0:
                    if clean_code[cur_idx] == "{":
                        b_depth += 1
                    elif clean_code[cur_idx] == "}":
                        b_depth -= 1
                    cur_idx += 1
                inner_content = clean_code[start_idx + 1 : cur_idx - 1]
                after_block = clean_code[cur_idx:]
                for var in top_level_vars:
                    var_pat = rf"\b{re.escape(var)}\b"
                    for reassign_m in re.finditer(
                        rf"(?:^|[;{{}}])\s*({re.escape(var)})\s*=(?!=)",
                        inner_content,
                        re.MULTILINE,
                    ):
                        stmt_tail = inner_content[reassign_m.end() :]
                        semi_idx = stmt_tail.find(";")
                        rhs_text = stmt_tail[:semi_idx] if semi_idx != -1 else stmt_tail
                        rem_in_block = stmt_tail[semi_idx + 1 :] if semi_idx != -1 else ""
                        used_in_rhs = bool(re.search(var_pat, rhs_text))
                        used_later_in_block = bool(re.search(var_pat, rem_in_block))
                        used_after_block = bool(re.search(var_pat, after_block))
                        if used_later_in_block and not used_in_rhs and not used_after_block:
                            continue
                        abs_pos = start_idx + 1 + reassign_m.start(1)
                        line_num = clean_code[:abs_pos].count("\n") + 1
                        findings.append(
                            Finding(
                                category="CRITICAL",
                                code="SHADOWED_BLOCK_ASSIGNMENT",
                                message=f"Variable '{var}' is reassigned inside {block_kind}() block at line {line_num}.",
                                line=line_num,
                                impact=f"OpenSCAD variables are immutable and lexical. Reassignment inside {block_kind}() only shadows the inner scope; outer '{var}' remains unchanged. Use a ternary expression '(cond ? a : b)' or 'let()'.",
                            )
                        )

        # 7. Golden Rules: $fn check
        fn_matches = list(re.finditer(r"\$fn\s*=\s*([0-9]+)", clean_code))
        fn_val = None
        global_fn_m = re.search(r"^[ \t]*\$fn\s*=\s*([0-9]+)\s*;", clean_code, re.MULTILINE)
        if global_fn_m:
            fn_val = int(global_fn_m.group(1))
        elif fn_matches:
            fn_val = int(fn_matches[0].group(1))

        fa_match = re.search(r"\$fa\s*=\s*([0-9\.]+)\s*;", clean_code)
        fs_match = re.search(r"\$fs\s*=\s*([0-9\.]+)\s*;", clean_code)
        has_adaptive_facets = bool(fa_match and fs_match)

        if fn_val is None and not has_adaptive_facets:
            findings.append(
                Finding(
                    category="WARNING",
                    code="MISSING_FN",
                    message="No global $fn declared at top of model.scad (e.g. $fn = 32; or $fa = 6; $fs = 1;).",
                    impact="Circles and cylinders will render with coarse, unpredictable default facets.",
                )
            )
        elif fn_val is not None and fn_val < 16:
            findings.append(
                Finding(
                    category="WARNING",
                    code="LOW_FN",
                    message=f"Global $fn is set to {fn_val} (< 16).",
                    impact="Cylindrical surfaces will have noticeable polygonal faceting.",
                )
            )

        for fm in fn_matches:
            val = int(fm.group(1))
            if val > 100:
                line_num = clean_code[: fm.start()].count("\n") + 1
                findings.append(
                    Finding(
                        category="WARNING",
                        code="EXCESSIVE_FN",
                        message=f"$fn is set to {val} (> 100) at line {line_num}.",
                        line=line_num,
                        impact="Excessive triangle count causes slow rendering and sandbox timeouts.",
                    )
                )

        # 8. Golden Rules: EPS constant & difference() cutter overshoot rule
        eps_match = re.search(r"\bEPS\s*=\s*([0-9\.]+(?:[eE][-+]?[0-9]+)?)\s*;", clean_code)
        has_eps = eps_match is not None
        diff_matches = list(re.finditer(r"\bdifference\s*\(\s*\)", clean_code))
        if diff_matches:
            if not has_eps:
                findings.append(
                    Finding(
                        category="WARNING",
                        code="MISSING_EPS",
                        message="difference() used without EPS constant declared (e.g. EPS = 0.01;).",
                        impact="Subtractions and cutter geometry risk non-manifold coincident faces and Z-fighting.",
                    )
                )
            else:
                eps_count = len(re.findall(r"\bEPS\b", clean_code))
                if eps_count <= 1:
                    findings.append(
                        Finding(
                            category="WARNING",
                            code="UNUSED_EPS_IN_CUTTERS",
                            message="EPS is declared but never referenced in difference() cutters (expected -EPS offset or +2*EPS length).",
                            impact="Cutters sharing exact boundaries with target create non-manifold coincident faces and Z-fighting.",
                        )
                    )

        # 9. Parameters check (accepts numeric literals and parametric expressions)
        declared_params = re.findall(
            r"^[ \t]*([A-Z][A-Z0-9_]*)[ \t]*=[^;]+;",
            clean_code,
            re.MULTILINE,
        )
        param_names = [p for p in declared_params if p != "EPS" and not p.startswith("$")]
        if not param_names:
            findings.append(
                Finding(
                    category="WARNING",
                    code="NO_UPPERCASE_PARAMETERS",
                    message="No UPPER_CASE parameters defined at top of model.scad.",
                    impact="Hardcoded magic numbers inside geometry violate design principles and reduce maintainability.",
                )
            )

        # 10. Hardware-aware wall thickness parameter inspection
        nozzle_spec = parse_nozzle_spec(self.user_nozzle)
        nozzle_dia = float(nozzle_spec.get("diameter_mm") or 0.40)
        wall_param_match = re.search(
            r"^[ \t]*(WALL_THICKNESS|WALL|THICKNESS|SHELL_THICKNESS|WALL_T)\s*=\s*([0-9\.]+)\s*;",
            clean_code,
            re.MULTILINE,
        )
        if wall_param_match:
            wall_name = wall_param_match.group(1)
            wall_val = float(wall_param_match.group(2))
            if wall_val < nozzle_dia:
                findings.append(
                    Finding(
                        category="WARNING",
                        code="THIN_WALL_PARAMETER",
                        message=f"Declared wall thickness parameter {wall_name} = {wall_val:.2f} mm is thinner than nozzle diameter ({nozzle_dia:.2f} mm).",
                        impact="Slicers will drop walls thinner than a single extrusion perimeter.",
                    )
                )
            elif wall_val < 2 * nozzle_dia * 0.95:
                findings.append(
                    Finding(
                        category="WARNING",
                        code="SINGLE_PERIMETER_WALL",
                        message=f"Declared wall thickness {wall_name} = {wall_val:.2f} mm produces only a single perimeter for a {nozzle_dia:.2f} mm nozzle.",
                        impact="Single-perimeter walls are prone to leaking, stringing, and delamination. Minimum 2 perimeters recommended.",
                    )
                )

        multi_part = _detect_multi_part_intent(code)
        expected_parts = _detect_expected_part_count(code) if multi_part else None
        code_meta = {
            "exists": True,
            "syntax_valid": not syntax_error,
            "parameters_defined": param_names,
            "fn_value": fn_val,
            "eps_defined": has_eps,
            "multi_part_intent": multi_part,
            "expected_part_count": expected_parts,
        }
        return findings, code_meta

    def _verify_mesh(
        self,
        stl_path: Path,
        multi_part_intent: bool = False,
        expected_part_count: int | None = None,
        precomputed_metrics: dict[str, Any] | None = None,
    ) -> tuple[list[Finding], dict[str, Any]]:
        findings: list[Finding] = []
        self._check_cancelled()

        if not stl_path.is_file() or stl_path.stat().st_size == 0:
            findings.append(
                Finding(
                    category="CRITICAL",
                    code="PREVIEW_STL_MISSING",
                    message="No compiled preview.stl found.",
                    impact="Mesh topology and 3D geometry cannot be verified.",
                )
            )
            return findings, {"exists": False}

        cached_metrics: dict[str, Any] | None = None
        if (
            isinstance(precomputed_metrics, dict)
            and "dimensions_mm" in precomputed_metrics
            and "volume_mm3" in precomputed_metrics
            and "boundary_edges" in precomputed_metrics
        ):
            cached_metrics = precomputed_metrics

        metrics_file = self.project_dir / ".cad_metrics.json"
        if cached_metrics is None and metrics_file.is_file():
            try:
                cached_data = json.loads(metrics_file.read_text(encoding="utf-8"))
                if isinstance(cached_data, dict) and isinstance(cached_data.get("metrics"), dict):
                    expected_sha = cached_data.get("preview_sha256")
                    is_valid_cache = True
                    if expected_sha and isinstance(expected_sha, str):
                        self._check_cancelled()
                        actual_sha = hashlib.sha256(stl_path.read_bytes()).hexdigest()
                        if actual_sha != expected_sha:
                            is_valid_cache = False
                    if is_valid_cache:
                        m = cached_data["metrics"]
                        if "dimensions_mm" in m and "volume_mm3" in m and "boundary_edges" in m:
                            cached_metrics = m
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                pass

        if cached_metrics is None:
            self._check_cancelled()
            try:
                vertices, triangles = load_stl(stl_path)
            except Exception as err:  # noqa: BLE001
                findings.append(
                    Finding(
                        category="CRITICAL",
                        code="CORRUPT_STL",
                        message=f"Failed to load STL mesh: {err}",
                        impact="STL file is malformed or unreadable.",
                    )
                )
                return findings, {"exists": True, "valid": False}

            if len(vertices) == 0 or len(triangles) == 0:
                findings.append(
                    Finding(
                        category="CRITICAL",
                        code="EMPTY_MESH",
                        message="The compiled mesh contains zero vertices or triangles.",
                        impact="OpenSCAD produced an empty shape.",
                    )
                )
                return findings, {"exists": True, "valid": False}

            self._check_cancelled()
            try:
                cached_metrics = mesh_metrics(vertices, triangles)
            except Exception as err:  # noqa: BLE001
                findings.append(
                    Finding(
                        category="CRITICAL",
                        code="CORRUPT_STL",
                        message=f"Failed to calculate mesh metrics: {err}",
                        impact="STL mesh topology calculation failed.",
                    )
                )
                return findings, {"exists": True, "valid": False}

        dims = cached_metrics.get("dimensions_mm", {})
        dim_x = float(dims.get("x", 0))
        dim_y = float(dims.get("y", 0))
        dim_z = float(dims.get("z", 0))
        volume = float(cached_metrics.get("volume_mm3", 0.0))
        raw_volume = float(cached_metrics.get("raw_volume", volume))
        boundary_edges = int(cached_metrics.get("boundary_edges", 0))
        multi_face_edges = int(cached_metrics.get("multi_face_edges", 0))
        solid_count = int(cached_metrics.get("solid_count", 1))
        vertex_count = int(cached_metrics.get("vertex_count", 0))
        triangle_count = int(cached_metrics.get("triangle_count", 0))

        # 0. OpenSCAD compiler diagnostics captured during execution
        for cw in cached_metrics.get("compiler_warnings", []):
            msg = cw.get("message", "")
            lineno = cw.get("line")
            msg_lower = msg.lower()
            if any(sym in msg_lower for sym in ("unknown variable", "unknown module", "unknown function", "can't open library", "cannot open library")):
                findings.append(
                    Finding(
                        category="CRITICAL",
                        code="COMPILER_UNDEFINED_SYMBOL",
                        message=f"OpenSCAD reported undefined symbol: {msg}",
                        line=lineno,
                        impact="OpenSCAD ignored the undefined symbol/module, resulting in missing or corrupt geometry.",
                    )
                )
            else:
                findings.append(
                    Finding(
                        category="WARNING",
                        code="COMPILER_GEOMETRY_WARNING",
                        message=f"OpenSCAD compiler warning: {msg}",
                        line=lineno,
                        impact="OpenSCAD reported geometry/manifold warning during CSG evaluation.",
                    )
                )

        if vertex_count == 0 or triangle_count == 0:
            findings.append(
                Finding(
                    category="CRITICAL",
                    code="EMPTY_MESH",
                    message="The compiled mesh contains zero vertices or triangles.",
                    impact="OpenSCAD produced an empty shape.",
                )
            )
            return findings, {"exists": True, "valid": False}

        # 1. Bounding box & non-zero dimensions
        is_finite_geom = all(math.isfinite(d) for d in (dim_x, dim_y, dim_z)) and math.isfinite(volume)
        nozzle_spec = parse_nozzle_spec(self.user_nozzle)
        nozzle_dia = float(nozzle_spec.get("diameter_mm") or 0.40)

        if not is_finite_geom:
            findings.append(
                Finding(
                    category="CRITICAL",
                    code="NON_FINITE_GEOMETRY",
                    message="Mesh dimensions or volume contain non-finite values (NaN or Inf).",
                    impact="Invalid floating-point geometry.",
                )
            )
            dim_x = dim_x if math.isfinite(dim_x) else 0.0
            dim_y = dim_y if math.isfinite(dim_y) else 0.0
            dim_z = dim_z if math.isfinite(dim_z) else 0.0
            volume = volume if math.isfinite(volume) else 0.0
        elif any(d <= 0.0001 for d in (dim_x, dim_y, dim_z)):
            findings.append(
                Finding(
                    category="CRITICAL",
                    code="ZERO_THICKNESS",
                    message=f"Model has zero or near-zero thickness in one axis ({dim_x:.3f}x{dim_y:.3f}x{dim_z:.3f} mm).",
                    impact="2D flat surfaces cannot be 3D printed or rendered as solid.",
                )
            )
        else:
            # Build plate bounds check
            if self.user_printer:
                printer_spec = parse_printer_spec(self.user_printer)
                bed_x = float(printer_spec.get("bed_x_mm") or 256.0)
                bed_y = float(printer_spec.get("bed_y_mm") or 256.0)
                fits_normal = (dim_x <= bed_x and dim_y <= bed_y)
                fits_rotated = (dim_x <= bed_y and dim_y <= bed_x)
                if not (fits_normal or fits_rotated):
                    findings.append(
                        Finding(
                            category="WARNING",
                            code="BUILD_VOLUME_EXCEEDED",
                            message=f"Model dimensions ({dim_x:.1f} × {dim_y:.1f} mm) exceed build plate bounds ({bed_x:.1f} × {bed_y:.1f} mm) for {self.user_printer}.",
                            impact="The model will not fit on the configured 3D printer bed without scaling or splitting.",
                        )
                    )

            # Nozzle-aware thin wall check
            min_dim = min(dim_x, dim_y, dim_z)
            if min_dim < nozzle_dia:
                findings.append(
                    Finding(
                        category="WARNING",
                        code="THIN_BOUNDING_DIMENSION",
                        message=f"Model has a thin dimension under nozzle diameter {nozzle_dia:.2f} mm ({min_dim:.3f} mm).",
                        impact=f"Features below nozzle diameter ({nozzle_dia:.2f} mm) cannot be extruded by standard slicers.",
                    )
                )

        # 2. Signed Polyhedron Volume
        if is_finite_geom:
            if volume <= 0.0001:
                findings.append(
                    Finding(
                        category="CRITICAL",
                        code="INVALID_VOLUME",
                        message="Calculated mesh volume is zero mm³.",
                        impact="Zero-volume or self-intersecting collapsed geometry.",
                    )
                )
            elif raw_volume < -0.0001:
                findings.append(
                    Finding(
                        category="WARNING",
                        code="INVERTED_NORMALS",
                        message="Mesh has inverted surface normals (clockwise face winding).",
                        impact="Face normals point inward; slicers may require auto-repair.",
                    )
                )

        # 3. Watertight Manifold Edge-sharing Check & Winding Orientation
        is_watertight = boundary_edges == 0 and multi_face_edges == 0
        if boundary_edges > 0:
            findings.append(
                Finding(
                    category="CRITICAL",
                    code="NON_WATERTIGHT_MESH",
                    message=f"Mesh has {boundary_edges} boundary edge(s) (holes in surface).",
                    impact="The 3D model is not watertight (manifold), causing slicer and export failures.",
                )
            )
        if multi_face_edges > 0:
            findings.append(
                Finding(
                    category="CRITICAL",
                    code="NON_MANIFOLD_EDGES",
                    message=f"Mesh has {multi_face_edges} non-manifold edge(s) shared by more than 2 faces.",
                    impact="Self-intersecting internal walls or intersecting sheets.",
                )
            )

        inconsistent_winding_edges = int(cached_metrics.get("inconsistent_winding_edges", 0) or 0)
        if inconsistent_winding_edges > 0:
            findings.append(
                Finding(
                    category="CRITICAL",
                    code="INCONSISTENT_WINDING",
                    message=f"Mesh has {inconsistent_winding_edges} edge(s) shared by adjacent faces with inconsistent normal orientation (flipped face winding).",
                    impact="Opposing surface normals create non-orientable topology that slicers cannot determine inside/outside for.",
                )
            )

        trapped_cavities = cached_metrics.get("trapped_cavities") or []
        if trapped_cavities:
            cav_descs = [
                f"Vol: {c.get('volume_mm3', 0)} mm³, Z: [{c.get('min_z', 0)}..{c.get('max_z', 0)}] mm"
                for c in trapped_cavities[:3]
            ]
            findings.append(
                Finding(
                    category="CRITICAL",
                    code="TRAPPED_INTERNAL_CAVITY",
                    message=f"Model contains {len(trapped_cavities)} trapped internal sealed cavity/void(s) with negative signed volume ({'; '.join(cav_descs)}).",
                    impact="Subtractive cutter did not break through outer shell or neck was sealed off. Extend cutters through openings using EPS.",
                )
            )

        micro_debris_count = int(cached_metrics.get("micro_debris_count", 0) or 0)
        if micro_debris_count > 0:
            findings.append(
                Finding(
                    category="WARNING",
                    code="DEGENERATE_CSG_DEBRIS",
                    message=f"Model has {micro_debris_count} microscopic disconnected CSG sliver(s)/debris (<0.01 mm³).",
                    impact="Coincident cutter surfaces or boolean tolerance slivers detected. Ensure boolean cutters extend past boundaries with EPS.",
                )
            )

        # 4. Disconnected Solid Components & Floating Part Check
        floating_solid_count = int(cached_metrics.get("floating_solid_count", 0) or 0)
        solids_min_z = cached_metrics.get("solids_min_z") or []
        printability = cached_metrics.get("printability") or {}
        floating_solids = cached_metrics.get("floating_solids") or []

        if solid_count > 1:
            if floating_solid_count > 0:
                if floating_solids:
                    solids_desc = [
                        f"Z: [{s.get('min_z')}..{s.get('max_z')}] mm, center: ({(s.get('center_xy') or [0, 0])[0]}, {(s.get('center_xy') or [0, 0])[1]}) mm, size: {s.get('bbox_mm') or []} mm"
                        for s in floating_solids[:3]
                    ]
                    part_detail = f" ({'; '.join(solids_desc)})"
                else:
                    floating_zs = [z for z in solids_min_z[1:] if z > (solids_min_z[0] if solids_min_z else 0.0) + 0.08]
                    part_detail = f" (at Z = {floating_zs} mm)" if floating_zs else ""

                findings.append(
                    Finding(
                        category="CRITICAL",
                        code="FLOATING_DISCONNECTED_SOLID",
                        message=(
                            f"Model has {floating_solid_count} disconnected solid component(s) floating in mid-air{part_detail} "
                            f"out of {solid_count} total solids."
                        ),
                        impact=(
                            "A feature (such as a bottle neck, boss, or wall) was severed from its base by an "
                            "excessive difference() cavity or Z-offset gap, or a separate part is not resting on Z = 0."
                        ),
                    )
                )
            if multi_part_intent and expected_part_count is not None and solid_count != expected_part_count:
                findings.append(
                    Finding(
                        category="CRITICAL" if solid_count > expected_part_count else "WARNING",
                        code="UNEXPECTED_SOLID_COUNT",
                        message=(
                            f"PART_TYPE defines {expected_part_count} part variant(s), but the compiled mesh "
                            f"contains {solid_count} disconnected solids."
                        ),
                        impact=(
                            "One of the components is split into disconnected pieces (if solid_count > expected) "
                            "or fused/missing (if solid_count < expected)."
                        ),
                    )
                )
            if multi_part_intent and floating_solid_count == 0:
                overlapping_part_count = int(cached_metrics.get("overlapping_part_count", 0) or 0)
                min_part_gap_mm = cached_metrics.get("min_part_gap_mm")
                if overlapping_part_count > 0:
                    findings.append(
                        Finding(
                            category="WARNING",
                            code="OVERLAPPING_PARTS",
                            message=f"Model has {overlapping_part_count} pair(s) of grounded parts with overlapping XY footprints.",
                            impact="Adjacent parts in plate layout will fuse together during 3D printing. Increase layout spacing between parts.",
                        )
                    )
                elif min_part_gap_mm is not None and min_part_gap_mm < 2.0:
                    findings.append(
                        Finding(
                            category="WARNING",
                            code="CROWDED_PART_LAYOUT",
                            message=f"Minimum clearance between grounded parts is only {min_part_gap_mm:.2f} mm (< 2.0 mm).",
                            impact="Parts are arranged too close together on the build plate and may fuse or cause nozzle collisions. Increase PART_GAP to >= 10 mm (minimum >= 3 mm).",
                        )
                    )
                if expected_part_count is None or solid_count == expected_part_count:
                    findings.append(
                        Finding(
                            category="INFO",
                            code="MULTI_PART_ASSEMBLY",
                            message=f"Model contains {solid_count} intentional components resting on the build plate.",
                            impact="Multiple separate components arranged in one file.",
                        )
                    )
            elif not multi_part_intent and floating_solid_count == 0:
                findings.append(
                    Finding(
                        category="WARNING",
                        code="DISCONNECTED_SOLIDS",
                        message=(
                            f"Model contains {solid_count} disconnected solid bodies. "
                            "If intended as a multi-part assembly or plate layout, set PART_TYPE = \"ALL\" "
                            "or add a '// multi-part' comment."
                        ),
                        impact="Parts may be detached unless explicitly intended as a multi-part assembly.",
                    )
                )

        # 5. Printability & Severe Overhang Check
        if isinstance(printability, dict) and printability:
            bed_area = float(printability.get("bed_contact_area_mm2", 0.0) or 0.0)
            overhang_60 = float(printability.get("overhang_area_60deg_mm2", 0.0) or 0.0)
            overhang_ratio = float(printability.get("overhang_ratio", 0.0) or 0.0)
            surf_area = float(printability.get("surface_area_mm2", 0.0) or 0.0)

            if volume > 100.0 and bed_area < 1.0:
                findings.append(
                    Finding(
                        category="WARNING",
                        code="LOW_BED_CONTACT",
                        message=f"Model has near-zero build plate contact area ({bed_area:.2f} mm²).",
                        impact="Part will not adhere to the print bed without supports; orient flat base at Z = 0.",
                    )
                )

            if surf_area > 0 and volume > 0 and not any(d <= 0.0001 for d in (dim_x, dim_y, dim_z)):
                effective_t = 2.0 * volume / surf_area
                if effective_t < 0.35 * nozzle_dia:
                    findings.append(
                        Finding(
                            category="WARNING",
                            code="THIN_WALL_SHELL",
                            message=f"Calculated average mesh wall thickness ({effective_t:.2f} mm) is critically thin for a {nozzle_dia:.2f} mm nozzle.",
                            impact="Thin shells risk slicing holes and brittle prints.",
                        )
                    )
            footprint_xy = max(dim_x * dim_y, 1.0)
            if overhang_60 >= 150.0 and (
                overhang_ratio >= 0.06 or overhang_60 >= 0.15 * footprint_xy
            ):
                findings.append(
                    Finding(
                        category="WARNING",
                        code="SEVERE_OVERHANG",
                        message=(
                            f"Model has {overhang_60:.1f} mm² of steep (>60 deg) unsupported overhang area."
                        ),
                        impact=(
                            "Horizontal internal roofs or unchamfered transitions will sag during FDM printing; "
                            "use >= 45 deg conical shoulders or orient caps upside-down."
                        ),
                    )
                )

            if (
                solid_count == 1
                and floating_solid_count == 0
                and dim_z >= 5.0
                and bed_area >= 1.0
            ):
                is_stable = printability.get("is_statically_stable", True)
                stability_margin = float(
                    printability.get("stability_margin_mm", 0.0) or 0.0
                )
                if is_stable is False and stability_margin < -0.5:
                    com_mm = printability.get("center_of_mass_mm")
                    support_bounds = printability.get("support_footprint_bounds_mm")
                    com_str = (
                        f"({com_mm[0]:.1f}, {com_mm[1]:.1f}, {com_mm[2]:.1f}) mm"
                        if isinstance(com_mm, list) and len(com_mm) == 3
                        else "outside base"
                    )
                    bounds_str = (
                        f"X=[{support_bounds['min_x']:.1f}, {support_bounds['max_x']:.1f}], "
                        f"Y=[{support_bounds['min_y']:.1f}, {support_bounds['max_y']:.1f}] mm"
                        if isinstance(support_bounds, dict)
                        else "Z=0 contact polygon"
                    )
                    findings.append(
                        Finding(
                            category="WARNING",
                            code="UNSTABLE_CENTER_OF_MASS",
                            message=(
                                f"Center of Mass {com_str} projects {abs(stability_margin):.2f} mm outside "
                                f"the convex Z=0 base support footprint ({bounds_str})."
                            ),
                            impact=(
                                "Freestanding part is statically unstable and will tip over under gravity. "
                                "Widen the Z=0 base footprint, add a support foot under the overhang, or re-orient the part."
                            ),
                        )
                    )

        mesh_meta = {
            "exists": True,
            "watertight": is_watertight,
            "solid_count": solid_count,
            "floating_solid_count": floating_solid_count,
            "solids_min_z": solids_min_z,
            "volume_mm3": round(volume, 2),
            "dimensions_mm": {
                "x": round(dim_x, 2),
                "y": round(dim_y, 2),
                "z": round(dim_z, 2),
            },
            "triangle_count": triangle_count,
            "vertex_count": vertex_count,
            "boundary_edge_count": boundary_edges,
            "non_manifold_edge_count": multi_face_edges,
            "printability": printability,
        }
        return findings, mesh_meta

    def _build_envelope(
        self,
        findings: list[Finding],
        code_integrity: dict[str, Any],
        mesh_integrity: dict[str, Any],
    ) -> dict[str, Any]:
        criticals = [f for f in findings if f.category == "CRITICAL"]
        warnings = [f for f in findings if f.category == "WARNING"]

        # Risk score calculation: 0 (perfect) to 100 (critical fail)
        warning_score = min(40, len(warnings) * 10)
        critical_score = min(100, len(criticals) * 50)
        risk_score = min(100, critical_score + warning_score)

        if criticals:
            status = "FAILED"
            risk_level = "HIGH_RISK"
        elif warnings:
            status = "WARNING"
            risk_level = "MODERATE_RISK" if risk_score >= 30 else "LOW_RISK"
        else:
            status = "PASSED"
            risk_level = "EXCELLENT"

        summary_parts = []
        if criticals:
            summary_parts.append(f"{len(criticals)} critical issue(s)")
        if warnings:
            summary_parts.append(f"{len(warnings)} warning(s)")
        if not criticals and not warnings:
            summary_parts.append("All checks passed cleanly")

        watertight_str = "watertight" if mesh_integrity.get("watertight") else "non-manifold/unverified"
        summary = (
            f"Verification {status}: {', '.join(summary_parts)}. "
            f"Risk score: {risk_score}/100 ({risk_level}). Mesh is {watertight_str}."
        )

        return {
            "status": status,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "summary": summary,
            "findings": [f.to_dict() for f in findings],
            "code_integrity": code_integrity,
            "mesh_integrity": mesh_integrity,
        }
