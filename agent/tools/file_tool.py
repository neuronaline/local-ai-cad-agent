from __future__ import annotations

import re
import threading
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
# Per-project locks for ``model.scad`` writes.
_MODEL_FILE_LOCKS: dict[str, threading.RLock] = {}
_MODEL_FILE_LOCKS_GUARD: threading.Lock = threading.Lock()


@contextmanager
def _file_lock(path: Path) -> Iterator[threading.RLock]:
    """Serialise writes to ``model.scad`` per project.

    Only ``model.scad`` is editable (see ``EDITABLE_FILES``), so keying by
    ``path.parent`` (= the project directory) is sufficient.
    """
    key = str(path.parent)
    with _MODEL_FILE_LOCKS_GUARD:
        lock = _MODEL_FILE_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _MODEL_FILE_LOCKS[key] = lock
    lock.acquire()
    try:
        yield lock
    finally:
        lock.release()


class OpenScadPreflight:
    """Catch deterministic OpenSCAD mistakes and unsafe patterns before running CLI."""

    def __init__(self) -> None:
        self.warnings: list[str] = []
        self.blocked_errors: list[str] = []

    def validate(self, code: str) -> None:
        # Strip comments to prevent false positives in comments/documentation
        clean_code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
        clean_code = re.sub(r"//.*", "", clean_code)

        # 1. Check for accidental Python syntax
        python_keywords = ("def ", "import ", "from ", "class ", "elif ")
        for kw in python_keywords:
            if re.search(r"^[ \t]*" + re.escape(kw), clean_code, re.MULTILINE):
                self.blocked_errors.append(
                    f"Unsafe or invalid OpenSCAD syntax: Python keyword '{kw.strip()}' found. "
                    "Write native OpenSCAD code (use 'module', 'use <...>', etc.)."
                )
                return

        # 2. Check for unsafe file inclusions / path traversal
        inc_matches = re.findall(r"(?:include|use)\s*<([^>]+)>", code)
        for inc_path in inc_matches:
            inc_path = inc_path.strip()
            if inc_path.startswith("/") or ".." in inc_path or "\\" in inc_path:
                self.blocked_errors.append(
                    f"Unsafe include/use path blocked: '{inc_path}'. "
                    "Path must be relative and cannot escape the project directory."
                )
                return

        # 3. Check balanced delimiters
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
                    self.blocked_errors.append(f"Unmatched closing '{char}' at line {lineno}.")
                    return
                top, top_line = stack.pop()
                if pairs[top] != char:
                    self.blocked_errors.append(
                        f"Mismatched delimiter: opened '{top}' at line {top_line} but closed with '{char}' at line {lineno}."
                    )
                    return
            i += 1

        if in_string:
            self.blocked_errors.append("Unclosed string literal.")
            return

        if in_block_comment:
            self.blocked_errors.append("Unclosed block comment.")
            return

        if stack:
            unclosed, start_line = stack[-1]
            self.blocked_errors.append(f"Unclosed delimiter '{unclosed}' opened at line {start_line}.")
            return

        # 4. Check for UPPER_CASE parameter definition
        has_params = bool(re.search(r"^[ \t]*[A-Z][A-Z0-9_]*[ \t]*=", code, re.MULTILINE))
        if not has_params:
            self.warnings.append(
                "PRE-FLIGHT WARNING: No UPPER_CASE parameters declared at top of model.scad. "
                "Define numeric parameters at top (e.g. WIDTH = 50;)."
            )


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

    def _path(self, filename: str) -> Path:
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
        filename: str,
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
        filename: str,
        content: str,
    ) -> str:
        path = self._path(filename)
        with _file_lock(path):
            return self._write_model(content, "write_file")

    def _write_model(self, content: str, operation: str) -> str:
        """Validate, commit revision, and atomically write model.scad."""
        self.validate_model(content)
        revision = self._revisions.commit(
            content,
            RevisionOrigin(
                kind="agent_edit",
                operation=operation,
                tool_call_id=self._tool_call_id,
            ),
        )
        return (
            f"Wrote model.scad ({len(content.splitlines())} lines, "
            f"{len(content)} chars, revision {revision.id[:8]})."
        )

    @staticmethod
    def _reindent(
        file_lines: list[str], old_lines: list[str], new_string: str
    ) -> str:
        def get_indent(lines: list[str]) -> int:
            for line in lines:
                if line.strip():
                    return len(line) - len(line.lstrip(" "))
            return 0

        file_indent = get_indent(file_lines)
        old_indent = get_indent(old_lines)
        delta = file_indent - old_indent
        if delta == 0 or not new_string:
            return new_string

        new_lines = new_string.splitlines(keepends=True)
        new_indent = get_indent(new_lines)
        if new_indent == file_indent:
            return new_string

        if new_indent == old_indent:
            adjusted: list[str] = []
            for line in new_lines:
                if not line.strip():
                    adjusted.append(line)
                else:
                    curr_indent = len(line) - len(line.lstrip(" "))
                    target_indent = max(0, curr_indent + delta)
                    adjusted.append(" " * target_indent + line.lstrip(" "))
            return "".join(adjusted)
        return new_string

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
                f"expected one exact match, found {matches}; file was not changed."
            )

        # 2. Whitespace-tolerant match (only when exact match found 0 matches)
        file_lines = current.splitlines(keepends=True)
        old_lines = old_string.splitlines(keepends=True)
        k = len(old_lines)
        if k == 0 or len(file_lines) < k:
            raise MatchError(
                "expected one exact match, found 0; file was not changed."
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
                replacement = cls._reindent(matched_slice, old_lines, new_string)
                if replacement and not replacement.endswith(("\n", "\r")):
                    replacement += "\n"
            else:
                line_ending = (
                    "\r\n"
                    if matched_text.endswith("\r\n")
                    else ("\n" if matched_text.endswith("\n") else "")
                )
                end_char = start_char + len(matched_text) - len(line_ending)
                replacement = cls._reindent(matched_slice, old_lines, new_string)

            matched_sub = current[start_char:end_char]
            return ((start_char, end_char), matched_sub, replacement)

        if len(candidates) > 1:
            raise MatchError(
                f"expected one match, found {len(candidates)} after whitespace normalization; file was not changed."
            )

        raise MatchError(
            "expected one exact match, found 0; file was not changed."
        )

    def edit_file(
        self,
        filename: str,
        old_string: str,
        new_string: str = "",
    ) -> str:
        if not isinstance(old_string, str) or not old_string:
            raise ValueError("old_string must not be empty.")
        if new_string is None:
            new_string = ""
        elif not isinstance(new_string, str):
            raise ValueError("new_string must be a string.")
        path = self._path(filename)
        with _file_lock(path):
            if not path.exists():
                raise ValueError(
                    f"{filename} does not exist; use write_file to create it."
                )
            current = path.read_text(encoding="utf-8")
            try:
                (start, end), old_match, replacement = self._resolve_match(
                    current, old_string, new_string
                )
            except MatchError as err:
                # MatchError messages are formatted in lowercase (intended
                # to slot into compound sentences). Capitalise the first
                # letter for end-user display — this branch is reached
                # only when we *know* the exception came from our own
                # resolver, so the casing assumption is safe.
                msg = str(err)
                if msg and msg[0].islower():
                    msg = msg[0].upper() + msg[1:]
                raise ValueError(msg) from None
            updated = current[:start] + replacement + current[end:]
            start_line = current[:start].count("\n") + 1
            old_lines = len(old_match.splitlines()) or 1
            end_line = start_line + old_lines - 1
            new_lines = len(replacement.splitlines()) or 1
            new_end_line = start_line + new_lines - 1
            base = self._write_model(updated, "edit_file")
        return (
            f"{base} "
            f"(replaced lines {start_line}-{end_line} with lines "
            f"{start_line}-{new_end_line})."
        )

    def edit_file_atomic(
        self,
        filename: str,
        edits: list[dict[str, str]],
    ) -> str:
        """Apply several ``{old_string, new_string}`` replacements atomically.

        Every ``old_string`` is verified against the **original** file contents
        first; only if all matches resolve uniquely and without overlap do the
        replacements get applied. Overlap detection rejects two edits whose
        resolved byte ranges intersect so the caller can never silently mutate
        text the LLM copied from the pre-edit file. A failed validation aborts
        the whole batch with no write.
        """
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
        path = self._path(filename)
        with _file_lock(path):
            if not path.exists():
                raise ValueError(
                    f"{filename} does not exist; use write_file to create it."
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
                    raise ValueError(f"edits[{index}] {err}") from None
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
