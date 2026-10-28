from __future__ import annotations

import re
import threading
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from agent.revisions import MODEL_FILENAME, RevisionOrigin, RevisionStore

EDITABLE_FILES = {MODEL_FILENAME}
MAX_FILE_BYTES = 1 * 1024 * 1024
# Maximum number of lines a single ``read_file`` call may return. Mirrors
# the JSON schema's ``maximum`` so the runtime guard and the model-facing
# limit cannot drift.
MAX_READ_LINES = 2000
class _ProjectLock:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.active_count = 0


_MAX_PROJECT_FILE_LOCKS = 256
_MODEL_FILE_LOCKS: OrderedDict[str, _ProjectLock] = OrderedDict()
_MODEL_FILE_LOCKS_GUARD: threading.Lock = threading.Lock()


@contextmanager
def _file_lock(path: Path) -> Iterator[threading.RLock]:
    """Serialise writes to ``model.scad`` per project with active reference tracking."""
    key = str(path.parent)
    with _MODEL_FILE_LOCKS_GUARD:
        entry = _MODEL_FILE_LOCKS.get(key)
        if entry is None:
            entry = _ProjectLock()
            _MODEL_FILE_LOCKS[key] = entry
        entry.active_count += 1
        _MODEL_FILE_LOCKS.move_to_end(key)
        # Only evict idle locks (active_count == 0) to prevent race conditions
        if len(_MODEL_FILE_LOCKS) > _MAX_PROJECT_FILE_LOCKS:
            for k in list(_MODEL_FILE_LOCKS):
                if len(_MODEL_FILE_LOCKS) <= _MAX_PROJECT_FILE_LOCKS:
                    break
                if _MODEL_FILE_LOCKS[k].active_count == 0:
                    _MODEL_FILE_LOCKS.pop(k, None)

    entry.lock.acquire()
    try:
        yield entry.lock
    finally:
        entry.lock.release()
        with _MODEL_FILE_LOCKS_GUARD:
            entry.active_count = max(0, entry.active_count - 1)


def invalidate_model_file_lock(project_dir: Path) -> None:
    """Drop the per-project file lock when its directory goes away."""
    key = str(Path(project_dir))
    with _MODEL_FILE_LOCKS_GUARD:
        _MODEL_FILE_LOCKS.pop(key, None)


class OpenScadPreflight:
    """Catch deterministic OpenSCAD mistakes and unsafe patterns before running CLI."""

    def __init__(self) -> None:
        self.warnings: list[str] = []
        self.blocked_errors: list[str] = []

    PYTHON_KEYWORD_PATTERNS = (
        (re.compile(r"^[ \t]*def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\("), "def", "OpenSCAD uses 'module name() { ... }'."),
        (re.compile(r"^[ \t]*class\s+[a-zA-Z_]"), "class", "OpenSCAD does not support classes."),
        (re.compile(r"^[ \t]*elif\b"), "elif", "OpenSCAD uses 'else if'."),
        (re.compile(r"^[ \t]*from\s+[a-zA-Z_][a-zA-Z0-9_]*\s+import\b"), "from ... import", "OpenSCAD uses 'use <...>' or 'include <...>'."),
        (re.compile(r"^[ \t]*import\s+[a-zA-Z_][a-zA-Z0-9_]*(?!\s*\()"), "import", "OpenSCAD uses 'use <...>' or 'include <...>'. Built-in import() requires parentheses."),
        (re.compile(r"\bTrue\b"), "True", "OpenSCAD uses lowercase 'true'. Capital 'True' evaluates to undef."),
        (re.compile(r"\bFalse\b"), "False", "OpenSCAD uses lowercase 'false'. Capital 'False' evaluates to undef."),
        (re.compile(r"\bNone\b"), "None", "OpenSCAD uses 'undef'. 'None' is a Python literal that evaluates to undef."),
        (re.compile(r"(?<!\*)\*\*(?!\*)"), "**", "OpenSCAD uses '^' or pow(base, exponent) for exponentiation, not '**'."),
    )

    @classmethod
    def check_python_syntax(cls, clean_code: str) -> list[tuple[int, str, str]]:
        """Check for Python keywords in OpenSCAD code. Returns list of (line_num, keyword, impact)."""
        findings: list[tuple[int, str, str]] = []
        for line_num, line in enumerate(clean_code.splitlines(), start=1):
            for pattern, kw, impact in cls.PYTHON_KEYWORD_PATTERNS:
                if pattern.search(line):
                    findings.append((line_num, kw, impact))
        return findings

    @staticmethod
    def clean_code_preserve_lines(code: str) -> str:
        """Strip comments and string contents while keeping exact line numbering."""
        def repl_block(match: re.Match) -> str:
            return "\n" * match.group(0).count("\n")

        cleaned = re.sub(r"/\*.*?\*/", repl_block, code, flags=re.DOTALL)
        cleaned = re.sub(r"//.*", "", cleaned)
        return re.sub(r'"(?:\\.|[^"\\])*"', '""', cleaned)

    def validate(self, code: str) -> None:
        clean_code = self.clean_code_preserve_lines(code)

        # 1. Check for accidental Python syntax
        python_matches = self.check_python_syntax(clean_code)
        if python_matches:
            line_num, kw, impact = python_matches[0]
            self.blocked_errors.append(
                f"Unsafe or invalid OpenSCAD syntax: Python keyword '{kw}' found at line {line_num}. {impact}"
            )
            return

        # 2. Check for unsafe file inclusions / path traversal
        inc_matches = re.findall(r"(?:include|use)\s*<([^>]+)>", clean_code)
        for inc_path in inc_matches:
            inc_path = inc_path.strip()
            if inc_path.startswith("/") or ".." in inc_path or "\\" in inc_path:
                self.blocked_errors.append(
                    f"Unsafe include/use path blocked: '{inc_path}'. "
                    "Path must be relative and cannot escape the project directory."
                )
                return

        # 3. Check balanced delimiters
        delimiter_err = self.check_delimiters(code)
        if delimiter_err:
            _err_code, err_msg, _lineno = delimiter_err
            self.blocked_errors.append(err_msg)
            return

        # 4. Block minkowski() and extreme facet/slice counts that lock up OpenSCAD
        minkowski_match = re.search(r"\bminkowski\s*\(", clean_code)
        if minkowski_match:
            line_num = clean_code[: minkowski_match.start()].count("\n") + 1
            self.blocked_errors.append(
                f"Unsafe CSG operation: minkowski() at line {line_num} causes severe OpenSCAD "
                "CSG lockups and timeouts. Use 2D offset() before extrusion or hull() instead."
            )
            return

        for fn_m in re.finditer(r"\$fn\s*=\s*([0-9]+)", clean_code):
            fn_val = int(fn_m.group(1))
            line_num = clean_code[: fn_m.start()].count("\n") + 1
            if fn_val > 256:
                self.blocked_errors.append(
                    f"Excessive facet resolution: $fn = {fn_val} at line {line_num} exceeds the "
                    "maximum safe limit (256) and will stall OpenSCAD compilation. Use $fn <= 64 (or <= 36 for threads)."
                )
                return
            if fn_val > 100:
                self.warnings.append(
                    f"PRE-FLIGHT WARNING: $fn = {fn_val} (> 100) at line {line_num} causes slow rendering. Prefer $fn <= 64."
                )

        for sl_m in re.finditer(r"\bslices\s*=\s*([0-9]+)", clean_code):
            sl_val = int(sl_m.group(1))
            if sl_val > 180:
                line_num = clean_code[: sl_m.start()].count("\n") + 1
                self.blocked_errors.append(
                    f"Excessive extrusion slices: slices = {sl_val} at line {line_num} exceeds the "
                    "maximum safe limit (180) and will stall OpenSCAD compilation. Use slices <= turns * 18."
                )
                return

        # 5. Block 3D primitives directly inside 2D extrusions
        extrusion_errors = self.check_extrusion_children(clean_code)
        if extrusion_errors:
            line_num, ext_name, prim_name = extrusion_errors[0]
            self.blocked_errors.append(
                f"Invalid OpenSCAD CSG: 3D primitive '{prim_name}()' used inside {ext_name}() at line {line_num}. "
                f"{ext_name}() only accepts 2D shapes (circle, square, polygon) as children."
            )
            return

        # 6. Check for UPPER_CASE parameter definition
        has_params = bool(re.search(r"^[ \t]*[A-Z][A-Z0-9_]*[ \t]*=", clean_code, re.MULTILINE))
        if not has_params:
            self.warnings.append(
                "PRE-FLIGHT WARNING: No UPPER_CASE parameters declared at top of model.scad. "
                "Define numeric parameters at top (e.g. WIDTH = 50;)."
            )

        # 7. Check radian conversion in OpenSCAD degree-based trig functions
        for line_num, trig_fn in self.check_radian_trig(clean_code):
            self.warnings.append(
                f"PRE-FLIGHT WARNING: Degree-to-radian conversion in trig function at line {line_num} ('{trig_fn}()'). "
                "OpenSCAD sin(), cos(), and tan() expect degrees directly, not radians."
            )

    @staticmethod
    def check_radian_trig(clean_code: str) -> list[tuple[int, str]]:
        """Return list of (line_num, trig_fn) where degree-to-radian conversion is detected."""
        results: list[tuple[int, str]] = []
        for m in re.finditer(
            r"\b(sin|cos|tan)\s*\([^;\n)]*(?:PI\s*\*\s*[^;\n)]*/\s*180|[^;\n)]*\*\s*PI\s*/\s*180|[^;\n)]*/\s*180\s*\*\s*PI|\*\s*3\.14159\s*/\s*180)",
            clean_code,
            re.IGNORECASE,
        ):
            line_num = clean_code[: m.start()].count("\n") + 1
            results.append((line_num, m.group(1)))
        return results

    @staticmethod
    def _consume_balanced_parens(text: str, start_after_open_paren: int) -> int | None:
        """Return index immediately after the matching ')' or None if unbalanced."""
        depth = 1
        idx = start_after_open_paren
        while idx < len(text) and depth > 0:
            ch = text[idx]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            idx += 1
        return idx if depth == 0 else None

    @classmethod
    def check_extrusion_children(cls, clean_code: str) -> list[tuple[int, str, str]]:
        """Return list of (line_num, ext_name, prim_name) where a 3D primitive is placed in a 2D extrusion."""
        results: list[tuple[int, str, str]] = []
        for ext_m in re.finditer(r"\b(linear_extrude|rotate_extrude)\s*\(", clean_code):
            ext_name = ext_m.group(1)
            rest_idx = cls._consume_balanced_parens(clean_code, ext_m.end())
            if rest_idx is None:
                continue
            while rest_idx < len(clean_code) and clean_code[rest_idx].isspace():
                rest_idx += 1
            if rest_idx >= len(clean_code):
                continue
            if clean_code[rest_idx] == "{":
                brace_depth = 1
                end_idx = rest_idx + 1
                while end_idx < len(clean_code) and brace_depth > 0:
                    if clean_code[end_idx] == "{":
                        brace_depth += 1
                    elif clean_code[end_idx] == "}":
                        brace_depth -= 1
                    end_idx += 1
                block = clean_code[rest_idx:end_idx]

                # Identify projection ranges inside the block
                proj_ranges: list[tuple[int, int]] = []
                for p_m in re.finditer(r"\bprojection\s*\(", block):
                    p_rest = cls._consume_balanced_parens(block, p_m.end())
                    if p_rest is None:
                        continue
                    while p_rest < len(block) and block[p_rest].isspace():
                        p_rest += 1
                    if p_rest < len(block) and block[p_rest] == "{":
                        p_depth = 1
                        p_end = p_rest + 1
                        while p_end < len(block) and p_depth > 0:
                            if block[p_end] == "{":
                                p_depth += 1
                            elif block[p_end] == "}":
                                p_depth -= 1
                            p_end += 1
                        proj_ranges.append((p_m.start(), p_end))
                    else:
                        stmt_end = block.find(";", p_rest)
                        proj_ranges.append((p_m.start(), stmt_end + 1 if stmt_end != -1 else len(block)))

                for prim_m in re.finditer(r"\b(cube|cylinder|sphere|polyhedron)\s*\(", block):
                    p_start = prim_m.start()
                    if any(pr_start <= p_start < pr_end for pr_start, pr_end in proj_ranges):
                        continue
                    prim_name = prim_m.group(1)
                    line_num = clean_code[: rest_idx + p_start].count("\n") + 1
                    results.append((line_num, ext_name, prim_name))
            else:
                child_m = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", clean_code[rest_idx:])
                if child_m:
                    child_name = child_m.group(1)
                    if child_name in ("cube", "cylinder", "sphere", "polyhedron"):
                        line_num = clean_code[:rest_idx].count("\n") + 1
                        results.append((line_num, ext_name, child_name))
        return results

    @staticmethod
    def check_delimiters(code: str) -> tuple[str, str, int | None] | None:
        """Validate delimiter balance and unclosed literals in OpenSCAD code.

        Returns (error_code, error_message, lineno) if an error is found, or None.
        """
        pairs = {"{": "}", "[": "]", "(": ")"}
        stack: list[tuple[str, int]] = []
        in_line_comment = False
        in_block_comment = False
        in_string = False

        i = 0
        n = len(code)
        lineno = 1
        while i < n:
            char = code[i]
            if char == "\n":
                lineno += 1
                in_line_comment = False
                i += 1
                continue

            if in_line_comment:
                i += 1
                continue

            if in_block_comment:
                if code[i : i + 2] == "*/":
                    in_block_comment = False
                    i += 2
                else:
                    i += 1
                continue

            if in_string:
                if char == "\\" and i + 1 < n:
                    i += 2
                elif char == '"':
                    in_string = False
                    i += 1
                else:
                    i += 1
                continue

            if code[i : i + 2] == "//":
                in_line_comment = True
                i += 2
                continue

            if code[i : i + 2] == "/*":
                in_block_comment = True
                i += 2
                continue

            if char == '"':
                in_string = True
                i += 1
                continue

            if char in pairs:
                stack.append((char, lineno))
            elif char in pairs.values():
                if not stack:
                    return (
                        "UNMATCHED_CLOSING_DELIMITER",
                        f"Unmatched closing '{char}' at line {lineno}.",
                        lineno,
                    )
                top, top_line = stack.pop()
                if pairs[top] != char:
                    return (
                        "MISMATCHED_DELIMITER",
                        f"Mismatched delimiter: opened '{top}' at line {top_line} but closed with '{char}' at line {lineno}.",
                        lineno,
                    )
            i += 1

        if in_string:
            return ("UNCLOSED_STRING", "Unclosed string literal in model.scad.", None)

        if in_block_comment:
            return ("UNCLOSED_BLOCK_COMMENT", "Unclosed block comment (/* ... */) in model.scad.", None)

        if stack:
            unclosed, start_line = stack[-1]
            return ("UNCLOSED_DELIMITER", f"Unclosed delimiter '{unclosed}' opened at line {start_line}.", start_line)

        return None


class MatchError(ValueError):
    """Raised by :meth:`FileTool._resolve_match` when ``old_string`` is ambiguous.

    Subclassing :class:`ValueError` keeps the public tool contract intact —
    callers that catch ``ValueError`` still see the failure — while giving
    :meth:`FileTool.edit_file` a typed handle so it can normalise the
    message (e.g. capitalise the first letter for end-user display)
    without resorting to lowercase-first-character string sniffing.
    """



class FileTool:
    def __init__(
        self,
        project_dir: Path,
        revisions: RevisionStore | None = None,
        tool_call_id: str | None = None,
    ) -> None:
        self.project_dir = project_dir.resolve()
        self._revisions = revisions or RevisionStore(project_dir)
        self._tool_call_id = tool_call_id

    def with_call_id(self, tool_call_id: str) -> FileTool:
        """Return a copy of this tool bound to a specific tool-call ID."""
        return FileTool(
            self.project_dir,
            self._revisions,
            tool_call_id,
        )

    def _path(self, filename: str = MODEL_FILENAME) -> Path:
        if filename not in EDITABLE_FILES:
            raise ValueError(f"Only {MODEL_FILENAME} can be edited.")
        path = (self.project_dir / filename).resolve()
        if path.parent != self.project_dir:
            raise ValueError("Path escapes the project directory.")
        return path

    @staticmethod
    def validate_model(code: str) -> list[str]:
        preflight = OpenScadPreflight()
        preflight.validate(code)
        if preflight.blocked_errors:
            raise ValueError(preflight.blocked_errors[0])
        return preflight.warnings

    def read_file(
        self,
        filename: str = MODEL_FILENAME,
        offset: int = 1,
        limit: int | None = None,
    ) -> dict:
        """Return ``model.scad`` content as a plain dict.

        The dispatcher wraps this in the standard :func:`tool_success`
        envelope, which performs the JSON encoding exactly once. Returning
        a dict here — instead of a pre-serialised JSON string — avoids a
        double-encoded payload whose ``data`` field is a stringified JSON
        blob. A double-encoded payload wastes context tokens and trips up
        models when they copy exact ``old_string`` snippets into
        ``edit_file``.
        """
        path = self._path(filename)
        if offset < 1:
            raise ValueError("offset must be at least 1.")
        if not path.exists():
            return {
                "exists": False,
                "content": "",
                "total_lines": 0,
                "offset": offset,
                "returned_lines": 0,
                "next_offset": None,
            }
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(
                f"{filename} is too large to read safely (max {MAX_FILE_BYTES // 1024} KiB)."
            )
        content = path.read_text(encoding="utf-8")
        lines = content.splitlines(keepends=True)
        if limit is not None and (limit < 1 or limit > MAX_READ_LINES):
            raise ValueError(f"limit must be between 1 and {MAX_READ_LINES} lines.")
        if offset > len(lines) + 1:
            raise ValueError(
                f"offset {offset} exceeds {filename}'s {len(lines)} lines."
            )
        chunk = "".join(
            lines[offset - 1 :] if limit is None else lines[offset - 1 : offset - 1 + limit]
        )
        next_offset = offset + len(chunk.splitlines())
        return {
            "exists": True,
            "content": chunk,
            "total_lines": len(lines),
            "offset": offset,
            "returned_lines": len(chunk.splitlines()),
            "next_offset": next_offset if next_offset <= len(lines) else None,
        }

    def write_file(
        self,
        filename: str = MODEL_FILENAME,
        content: str = "",
    ) -> str:
        path = self._path(filename)
        with _file_lock(path):
            return self._write_model(content, "write_file")

    def _write_model(self, content: str, operation: str) -> str:
        """Validate, commit revision, and atomically write model.scad."""
        warnings = self.validate_model(content)
        revision = self._revisions.commit(
            content,
            RevisionOrigin(
                kind="agent_edit",
                operation=operation,
                tool_call_id=self._tool_call_id,
            ),
        )
        msg = (
            f"Wrote model.scad ({len(content.splitlines())} lines, "
            f"{len(content)} chars, revision {revision.id[:8]})."
        )
        if warnings:
            msg += "\n" + "\n".join(warnings)
        return msg

    @classmethod
    def _resolve_match(
        cls, current: str, old_string: str, new_string: str
    ) -> tuple[tuple[int, int], str, str]:
        # 1. Exact match fast-path
        matches = current.count(old_string)
        if matches == 1:
            start = current.find(old_string)
            end = start + len(old_string)
            return ((start, end), old_string, new_string)
        if matches > 1:
            raise MatchError(
                f"expected one exact match, found {matches}; file was not changed. "
                "Call read_file to inspect the current file and find a larger unique context block."
            )

        # 2. Whitespace-tolerant match (only when exact match found 0 matches)
        file_lines = current.splitlines(keepends=True)
        old_lines = old_string.splitlines(keepends=True)
        k = len(old_lines)
        if k == 0 or len(file_lines) < k:
            raise MatchError(
                "expected one exact match, found 0; file was not changed. "
                "Call read_file to inspect the current file content."
            )

        # Attempt A: Trailing-whitespace/line-ending normalization (indent matches)
        candidates = [
            i
            for i in range(len(file_lines) - k + 1)
            if [f.rstrip("\r\n \t") for f in file_lines[i : i + k]]
            == [o.rstrip("\r\n \t") for o in old_lines]
        ]

        # Attempt B: Full whitespace normalization (both leading indent and trailing whitespace)
        if not candidates and any(line.strip() for line in old_lines):
            candidates = [
                i
                for i in range(len(file_lines) - k + 1)
                if [f.strip() for f in file_lines[i : i + k]]
                == [o.strip() for o in old_lines]
            ]

        if len(candidates) == 1:
            idx = candidates[0]
            matched_slice = file_lines[idx : idx + k]
            start_char = sum(len(file_lines[j]) for j in range(idx))
            matched_text = "".join(matched_slice)

            if old_string.endswith(("\n", "\r")):
                end_char = start_char + len(matched_text)
                replacement = new_string
                if replacement and not replacement.endswith(("\n", "\r")):
                    replacement += "\n"
            else:
                line_ending = (
                    "\r\n"
                    if matched_text.endswith("\r\n")
                    else ("\n" if matched_text.endswith("\n") else "")
                )
                end_char = start_char + len(matched_text) - len(line_ending)
                replacement = new_string

            matched_sub = current[start_char:end_char]
            return ((start_char, end_char), matched_sub, replacement)

        if len(candidates) > 1:
            raise MatchError(
                f"expected one match, found {len(candidates)} after whitespace normalization; file was not changed. "
                "Call read_file to inspect the current file and find a larger unique context block."
            )

        raise MatchError(
            "expected one exact match, found 0; file was not changed. "
            "Call read_file to inspect the current file content."
        )

    def edit_file(
        self,
        filename: str = MODEL_FILENAME,
        old_string: str = "",
        new_string: str = "",
    ) -> str:
        """Replace a unique occurrence of ``old_string`` with ``new_string``."""
        if filename not in EDITABLE_FILES:
            # Caller passed 2 positional arguments: edit_file(old_string, new_string)
            new_string = old_string
            old_string = filename
            filename = MODEL_FILENAME
        if not isinstance(old_string, str) or not old_string:
            raise ValueError("old_string must not be empty.")
        if new_string is not None and not isinstance(new_string, str):
            raise ValueError("new_string must be a string.")
        return self.edit_file_atomic(
            filename,
            [{"old_string": old_string, "new_string": new_string if new_string is not None else ""}],
        )

    def edit_file_atomic(
        self,
        filename: str | list[dict[str, str]] = MODEL_FILENAME,
        edits: list[dict[str, str]] | None = None,
    ) -> str:
        """Apply several ``{old_string, new_string}`` replacements atomically.

        Every ``old_string`` is verified against the **original** file contents
        first; only if all matches resolve uniquely and without overlap do the
        replacements get applied. Overlap detection rejects two edits whose
        resolved byte ranges intersect so the caller can never silently mutate
        text the LLM copied from the pre-edit file. A failed validation aborts
        the whole batch with no write.
        """
        if edits is None and isinstance(filename, list):
            edits = filename
            actual_filename = MODEL_FILENAME
        else:
            actual_filename = str(filename) if isinstance(filename, str) else MODEL_FILENAME
        if not edits:
            raise ValueError("edit_file_atomic requires at least one edit.")
        normalised: list[tuple[str, str]] = []
        for index, entry in enumerate(edits):
            if not isinstance(entry, dict):
                raise ValueError(f"edits[{index}] must be an object.")
            old_string = entry.get("old_string")
            new_string = entry.get("new_string")
            if new_string is None:
                new_string = ""
            if not isinstance(old_string, str) or not old_string:
                raise ValueError(f"edits[{index}].old_string must be non-empty.")
            if not isinstance(new_string, str):
                raise ValueError(f"edits[{index}].new_string must be a string.")
            normalised.append((old_string, new_string))
        path = self._path(actual_filename)
        with _file_lock(path):
            if not path.exists():
                raise FileNotFoundError(
                    f"{actual_filename} does not exist; use write_file to create it."
                )
            current = path.read_text(encoding="utf-8")
            # Validate every match against the original buffer (not the running
            # ``updated`` string) so an edit cannot invalidate another edit's
            # ``old_string`` after a previous replace has rewritten the region.
            resolved: list[tuple[tuple[int, int], str, str]] = []
            for index, (old_string, new_string) in enumerate(normalised):
                try:
                    (start, end), old_match, replacement = self._resolve_match(
                        current, old_string, new_string
                    )
                except ValueError as err:
                    prefix = f"edits[{index}] " if len(normalised) > 1 else ""
                    msg = str(err)
                    if not prefix and msg and msg[0].islower():
                        msg = msg[0].upper() + msg[1:]
                    raise ValueError(f"{prefix}{msg}") from None
                resolved.append(((start, end), old_match, replacement))
            # Reject any pair of edits whose resolved byte ranges overlap.
            ordered = sorted(resolved, key=lambda entry: entry[0][0])
            previous_end = -1
            for (start, end), _old, _new in ordered:
                if start < previous_end:
                    raise ValueError(
                        "edit_file_atomic rejected overlapping edits; verify "
                        "each old_string is copied from the same file state."
                    )
                previous_end = end
            parts: list[str] = []
            cursor = 0
            ranges: list[tuple[int, int, int]] = []
            for (start, end), old_match, replacement in ordered:
                parts.append(current[cursor:start])
                parts.append(replacement)
                start_line = current[:start].count("\n") + 1
                old_lines = len(old_match.splitlines()) or 1
                end_line = start_line + old_lines - 1
                new_lines = len(replacement.splitlines()) or 1
                new_end_line = start_line + new_lines - 1
                ranges.append((start_line, end_line, new_end_line))
                cursor = end
            parts.append(current[cursor:])
            updated = "".join(parts)
            base = self._write_model(updated, "edit_file")
        ranges_str = ", ".join(
            f"lines {start}-{end}→{new_end}"
            for start, end, new_end in ranges
        )
        return f"{base} (replaced {ranges_str})."
