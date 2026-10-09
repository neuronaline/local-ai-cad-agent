"""Deterministic CAD verification tool for OpenSCAD code and STL mesh topology.

Performs rule-based, non-speculative static analysis of OpenSCAD code and
exact mathematical mesh topology validation without hallucination or fuzzy guessing.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


def _clean_code_preserve_lines(code: str) -> str:
    """Strip comments and string contents while keeping exact line numbering."""
    def repl_block(match: re.Match) -> str:
        return "\n" * match.group(0).count("\n")

    cleaned = re.sub(r"/\*.*?\*/", repl_block, code, flags=re.DOTALL)
    cleaned = re.sub(r"//.*", "", cleaned)
    cleaned = re.sub(r'"(?:\\.|[^"\\])*"', '""', cleaned)
    return cleaned


def _detect_multi_part_intent(code: str) -> bool:
    """Detect whether model.scad intentionally defines a multi-part assembly or plate layout."""
    param_matches = re.finditer(
        r"^[ \t]*([A-Z][A-Z0-9_]*)[ \t]*=[ \t]*[\"\']?([A-Za-z0-9_]+)[\"\']?[ \t]*;",
        code,
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

    def __init__(self, project_dir: Path, publish: Any = None) -> None:
        self.project_dir = project_dir.resolve()
        self._publish = publish
        self._call_id = ""

    def with_call_id(self, call_id: str) -> CadVerifier:
        self._call_id = call_id
        return self

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

    def verify(self) -> dict[str, Any]:
        """Run complete deterministic verification on model.scad and preview.stl."""
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

        preview_path = self.project_dir / "preview.stl"
        build_findings: list[Finding] = []
        is_stale = False

        if (
            preview_path.is_file()
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

        # 2. Python syntax & banned ops checks with line-number preservation
        clean_code = _clean_code_preserve_lines(code)

        for line_num, line in enumerate(clean_code.splitlines(), start=1):
            if re.search(r"^[ \t]*def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(", line):
                findings.append(
                    Finding(
                        category="CRITICAL",
                        code="PYTHON_SYNTAX_DETECTED",
                        message=f"Python function definition 'def' found at line {line_num}.",
                        line=line_num,
                        impact="OpenSCAD is not Python; use 'module name() { ... }'.",
                    )
                )
                syntax_error = True
            elif re.search(r"^[ \t]*class\s+[a-zA-Z_]", line):
                findings.append(
                    Finding(
                        category="CRITICAL",
                        code="PYTHON_SYNTAX_DETECTED",
                        message=f"Python 'class' keyword found at line {line_num}.",
                        line=line_num,
                        impact="OpenSCAD does not support classes.",
                    )
                )
                syntax_error = True
            elif re.search(r"^[ \t]*elif\b", line):
                findings.append(
                    Finding(
                        category="CRITICAL",
                        code="PYTHON_SYNTAX_DETECTED",
                        message=f"Python 'elif' keyword found at line {line_num}.",
                        line=line_num,
                        impact="OpenSCAD uses 'else if'.",
                    )
                )
                syntax_error = True
            elif re.search(r"^[ \t]*from\s+[a-zA-Z_][a-zA-Z0-9_]*\s+import\b", line):
                findings.append(
                    Finding(
                        category="CRITICAL",
                        code="PYTHON_SYNTAX_DETECTED",
                        message=f"Python 'from ... import' found at line {line_num}.",
                        line=line_num,
                        impact="OpenSCAD uses 'use <...>' or 'include <...>'.",
                    )
                )
                syntax_error = True
            elif re.search(r"^[ \t]*import\s+[a-zA-Z_][a-zA-Z0-9_]*(?!\s*\()", line):
                findings.append(
                    Finding(
                        category="CRITICAL",
                        code="PYTHON_SYNTAX_DETECTED",
                        message=f"Python-style import found at line {line_num}.",
                        line=line_num,
                        impact="OpenSCAD uses 'use <...>' or 'include <...>'. Built-in import() requires parentheses.",
                    )
                )
                syntax_error = True

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

        # 4. Golden Rules: $fn check
        fn_match = re.search(r"\$fn\s*=\s*([0-9]+)\s*;", clean_code)
        fn_val = int(fn_match.group(1)) if fn_match else None
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
        elif fn_val is not None:
            if fn_val > 100:
                findings.append(
                    Finding(
                        category="WARNING",
                        code="EXCESSIVE_FN",
                        message=f"$fn is set to {fn_val} (> 100).",
                        impact="Excessive triangle count causes slow rendering and sandbox timeouts.",
                    )
                )
            elif fn_val < 16:
                findings.append(
                    Finding(
                        category="WARNING",
                        code="LOW_FN",
                        message=f"$fn is set to {fn_val} (< 16).",
                        impact="Cylindrical surfaces will have noticeable polygonal faceting.",
                    )
                )

        # 5. Golden Rules: EPS constant & difference() cutter overshoot rule
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

        # 6. Parameters check (accepts numeric literals and parametric expressions)
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

        code_meta = {
            "exists": True,
            "syntax_valid": not syntax_error,
            "parameters_defined": param_names,
            "fn_value": fn_val,
            "eps_defined": has_eps,
            "multi_part_intent": _detect_multi_part_intent(code),
        }
        return findings, code_meta

    def _verify_mesh(
        self, stl_path: Path, multi_part_intent: bool = False
    ) -> tuple[list[Finding], dict[str, Any]]:
        findings: list[Finding] = []

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

        metrics_file = self.project_dir / ".cad_metrics.json"
        cached_metrics: dict[str, Any] | None = None
        if metrics_file.is_file():
            try:
                cached_data = json.loads(metrics_file.read_text(encoding="utf-8"))
                if isinstance(cached_data, dict) and isinstance(cached_data.get("metrics"), dict):
                    m = cached_data["metrics"]
                    if "dimensions_mm" in m and "volume_mm3" in m and "boundary_edges" in m:
                        cached_metrics = m
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                pass

        if cached_metrics is None:
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
        if not all(math.isfinite(d) for d in (dim_x, dim_y, dim_z)):
            findings.append(
                Finding(
                    category="CRITICAL",
                    code="NON_FINITE_GEOMETRY",
                    message="Mesh dimensions contain non-finite values (NaN or Inf).",
                    impact="Invalid floating-point geometry.",
                )
            )
        elif any(d <= 0.0001 for d in (dim_x, dim_y, dim_z)):
            findings.append(
                Finding(
                    category="CRITICAL",
                    code="ZERO_THICKNESS",
                    message=f"Model has zero or near-zero thickness in one axis ({dim_x:.3f}x{dim_y:.3f}x{dim_z:.3f} mm).",
                    impact="2D flat surfaces cannot be 3D printed or rendered as solid.",
                )
            )
        elif any(d < 0.4 for d in (dim_x, dim_y, dim_z)):
            findings.append(
                Finding(
                    category="WARNING",
                    code="THIN_WALL",
                    message=f"Model has a thin dimension under 0.4 mm ({min(dim_x, dim_y, dim_z):.3f} mm).",
                    impact="Features below 0.4 mm may fail to print on standard FDM nozzles.",
                )
            )

        # 2. Signed Polyhedron Volume
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

        # 3. Watertight Manifold Edge-sharing Check
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

        # 4. Disconnected Solid Components
        if solid_count > 1:
            if multi_part_intent:
                findings.append(
                    Finding(
                        category="INFO",
                        code="MULTI_PART_ASSEMBLY",
                        message=f"Model contains {solid_count} intentional components (multi-part assembly or plate arrangement detected).",
                        impact="Multiple separate components arranged in one file.",
                    )
                )
            else:
                findings.append(
                    Finding(
                        category="WARNING",
                        code="DISCONNECTED_SOLIDS",
                        message=(
                            f"Model contains {solid_count} disconnected solid bodies. "
                            "If intended as a multi-part assembly or plate layout, set PART_TYPE = \"ALL\" "
                            "or add a '// multi-part' comment."
                        ),
                        impact="Parts may be floating in air unattached unless explicitly intended as a multi-part assembly.",
                    )
                )

        mesh_meta = {
            "exists": True,
            "watertight": is_watertight,
            "solid_count": solid_count,
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
