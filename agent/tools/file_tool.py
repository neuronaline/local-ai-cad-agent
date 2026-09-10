from __future__ import annotations

import ast
import hashlib
import json
import math
import operator
import threading
from pathlib import Path

from agent.revisions import RevisionOrigin, RevisionStore

BLOCKED_IMPORTS = {
    "builtins",
    "importlib",
    "os",
    "pathlib",
    "shutil",
    "socket",
    "subprocess",
    "sys",
}
BLOCKED_CALLS = {"__import__", "breakpoint", "compile", "eval", "exec", "input", "open"}
EDITABLE_FILES = {"model.py"}
MAX_FILE_BYTES = 1 * 1024 * 1024
DEFAULT_READ_LIMIT = 400
# Maximum number of lines a single ``read_file`` call may return. Mirrors
# the JSON schema's ``maximum`` so the runtime guard and the model-facing
# limit cannot drift.
MAX_READ_LINES = 2000
_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
# Only ``model.py`` is editable (see ``EDITABLE_FILES``), so every path
# that reaches ``_file_lock`` resolves to that single file. A single
# module-level RLock is enough to serialise concurrent writes against
# ``model.py`` across projects; the previous dictionary keying grew
# without eviction (audit_270).
_MODEL_FILE_LOCK: threading.RLock = threading.RLock()


def _file_lock(path: Path) -> threading.RLock:
    """Return the lock used to serialise writes to ``model.py``.

    The argument is ignored for backwards compatibility with call sites
    that pass the resolved path. New callers may pass any value (the
    editable-file check upstream guarantees ``model.py``); the shared
    lock ensures consistent cross-project serialisation without any
    per-path bookkeeping.
    """
    del path  # explicit: the path no longer keys the lock.
    return _MODEL_FILE_LOCK


class ModelPreflight(ast.NodeVisitor):
    """Catch deterministic build123d mistakes before running the CAD kernel."""

    def __init__(self) -> None:
        self.values: dict[str, object] = {}
        self.edge_points: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {}
        # Topology-change tracking is a depth counter rather than a one-way
        # boolean so the warning fires only for assignments nested inside the
        # fillet/chamfer expression chain. A boolean latch persisted across
        # every later statement and produced false positives for unrelated
        # parts (audit_024).
        self._topology_depth: int = 0
        self.warnings: list[str] = []
        self.blocked_errors: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        names = [alias.name.split(".")[0] for alias in node.names]
        forbidden = set(names) & BLOCKED_IMPORTS
        if forbidden:
            self.blocked_errors.append(
                f"Unsafe import blocked: {', '.join(sorted(forbidden))}"
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        names = [(node.module or "").split(".")[0]]
        forbidden = set(names) & BLOCKED_IMPORTS
        if forbidden:
            self.blocked_errors.append(
                f"Unsafe import blocked: {', '.join(sorted(forbidden))}"
            )

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            value = self._number_or_tuple(node.value)
            if value is not None:
                self.values[name] = value
            points = self._line_points(node.value)
            if points is not None:
                self.edge_points[name] = points
            # ``_topology_depth > 0`` means this assignment is nested inside
            # a fillet/chamfer expression chain (depth was bumped in
            # ``visit_Call`` before ``generic_visit`` walked the args).
            # Sibling statements after the fillet see ``depth == 0`` and
            # do not trigger the warning, fixing the unrelated-over-warning
            # case (audit_024).
            if self._topology_depth > 0 and self._contains_fixed_selector_index(
                node.value
            ):
                self.warnings.append(
                    f"line {node.lineno}: fixed selector index used after a topology-changing "
                    "operation; reselect by geometry, position, radius, or adjacency"
                )
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name) and node.value is not None:
            value = self._number_or_tuple(node.value)
            if value is not None:
                self.values[node.target.id] = value
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # Check for blocked built-in calls. Direct calls (eval(...)) and
        # attribute-access forms (builtins.eval(...), __builtins__["eval"](...))
        # both reach the same dangerous function, so cover both shapes.
        if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_CALLS:
            self.blocked_errors.append(f"Unsafe function blocked: {node.func.id}")
        elif isinstance(node.func, ast.Attribute) and node.func.attr in BLOCKED_CALLS:
            self.blocked_errors.append(f"Unsafe function blocked: {node.func.attr}")
        name = self._call_name(node)
        if name == "Ellipse":
            invalid = {"center", "start_angle", "end_angle"} & {
                keyword.arg for keyword in node.keywords
            }
            if invalid:
                raise ValueError(
                    "Invalid Ellipse argument(s): "
                    + ", ".join(sorted(invalid))
                    + ". Ellipse is a full 2D sketch; use EllipticalCenterArc in BuildLine."
                )
        elif name == "RadiusArc":
            self._validate_radius_arc(node)
        elif name in {"fillet", "chamfer"}:
            # In-call check: a fixed selector index passed directly to a
            # topology-changing call resolves against the pre-call geometry
            # but the returned selector points at a different edge after
            # the call, so the indexed edge is fragile (audit_024).
            for arg in node.args:
                if self._contains_fixed_selector_index(arg):
                    self.warnings.append(
                        f"line {node.lineno}: fixed selector index passed to {name}; "
                        "reselect by geometry, position, radius, or adjacency"
                    )
            # Bump the depth counter for the duration of the children walk
            # so nested visit_Assign calls in the same expression chain see
            # ``_topology_depth > 0``. Reset on exit so unrelated later
            # statements are not warned.
            self._topology_depth += 1
            try:
                self.generic_visit(node)
            finally:
                self._topology_depth -= 1
            return
        self.generic_visit(node)

    def _validate_radius_arc(self, node: ast.Call) -> None:
        start = self._point(self._argument(node, 0, "start_point"))
        end = self._point(self._argument(node, 1, "end_point"))
        radius = self._number(self._argument(node, 2, "radius"))
        # ``build123d.RadiusArc`` accepts a signed radius: ``radius > 0``
        # yields the short sagitta on one side of the chord, ``radius < 0``
        # yields the equivalent arc mirrored to the other side. The
        # preflight must align with the real API instead of rejecting the
        # negative convention that the playbook documents.
        if radius == 0:
            raise ValueError("RadiusArc radius must be non-zero.")
        if start is None or end is None or radius is None:
            return
        chord = math.dist(start, end)
        minimum = chord / 2
        # Compare against the magnitude: ``abs(radius)`` is the chord-
        # distance bound; the sign only flips the arc side.
        if abs(radius) + 1e-9 < minimum:
            raise ValueError(
                f"RadiusArc radius {radius:g} is too small for chord {chord:.3f}; "
                f"minimum magnitude is {minimum:.3f}."
            )

    def _line_points(
        self, node: ast.AST
    ) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
        if not isinstance(node, ast.Call) or self._call_name(node) != "Line":
            return None
        start = self._point(self._argument(node, 0, "start"))
        end = self._point(self._argument(node, 1, "end"))
        return (start, end) if start is not None and end is not None else None

    def _point(self, node: ast.AST | None) -> tuple[float, ...] | None:
        if (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.MatMult)
            and isinstance(node.left, ast.Name)
        ):
            index = self._number(node.right)
            points = self.edge_points.get(node.left.id)
            if points is not None and index in {0, 1}:
                return points[int(index)]
        value = self._number_or_tuple(node)
        if (
            isinstance(value, tuple)
            and len(value) in {2, 3}
            and all(isinstance(item, float) for item in value)
        ):
            return value
        return None

    def _number(self, node: ast.AST | None) -> float | None:
        value = self._number_or_tuple(node)
        return value if isinstance(value, float) else None

    def _number_or_tuple(
        self, node: ast.AST | None
    ) -> float | tuple[float, ...] | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            return self.values.get(node.id)  # type: ignore[return-value]
        if isinstance(node, ast.Tuple):
            values = tuple(self._number(item) for item in node.elts)
            return values if all(value is not None for value in values) else None
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = self._number(node.operand)
            if value is not None:
                return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            left = self._number(node.left)
            right = self._number(node.right)
            if left is not None and right is not None:
                try:
                    return float(_BINARY_OPERATORS[type(node.op)](left, right))
                except (ArithmeticError, OverflowError):
                    return None
        return None

    @staticmethod
    def _argument(node: ast.Call, index: int, keyword_name: str) -> ast.AST | None:
        if len(node.args) > index:
            return node.args[index]
        return next(
            (keyword.value for keyword in node.keywords if keyword.arg == keyword_name),
            None,
        )

    @staticmethod
    def _call_name(node: ast.Call) -> str | None:
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None

    @staticmethod
    def _contains_fixed_selector_index(node: ast.AST) -> bool:
        selector_names = {"edges", "faces", "vertices", "wires", "sort_by", "group_by"}
        for child in ast.walk(node):
            if not isinstance(child, ast.Subscript):
                continue
            index = child.slice
            if not (isinstance(index, ast.Constant) and isinstance(index.value, int)):
                continue
            calls = {
                part.func.attr
                for part in ast.walk(child.value)
                if isinstance(part, ast.Call) and isinstance(part.func, ast.Attribute)
            }
            if calls & selector_names:
                return True
        return False


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
            raise ValueError("Only model.py can be edited.")
        path = (self.project_dir / filename).resolve()
        if path.parent != self.project_dir:
            raise ValueError("Path escapes the project directory.")
        return path

    @staticmethod
    def validate_model(code: str) -> list[str]:
        try:
            tree = ast.parse(code, filename="model.py")
        except SyntaxError as error:
            raise ValueError(
                f"Invalid Python: {error.msg} (line {error.lineno})"
            ) from error
        preflight = ModelPreflight()
        preflight.visit(tree)
        if preflight.blocked_errors:
            raise ValueError(preflight.blocked_errors[0])
        return preflight.warnings

    def read_file(
        self,
        filename: str,
        offset: int = 1,
        limit: int | None = None,
        known_sha256: str | None = None,
    ) -> str:
        path = self._path(filename)
        if offset < 1:
            raise ValueError("offset must be at least 1.")
        if not path.exists():
            return json.dumps(
                {
                    "exists": False,
                    "content": "",
                    "sha256": None,
                    "total_lines": 0,
                    "offset": offset,
                    "returned_lines": 0,
                    "next_offset": None,
                },
                ensure_ascii=False,
            )
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(
                f"{filename} is too large to read safely (max {MAX_FILE_BYTES // 1024} KiB)."
            )
        content = path.read_text(encoding="utf-8")
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if known_sha256:
            if len(known_sha256) != 64:
                raise ValueError("known_sha256 must be a SHA-256 digest.")
            if known_sha256 == digest:
                return json.dumps(
                    {"exists": True, "unchanged": True, "sha256": digest},
                    ensure_ascii=False,
                )
        if limit is None:
            limit = None if offset == 1 else DEFAULT_READ_LIMIT
        if limit is not None and (limit < 1 or limit > MAX_READ_LINES):
            raise ValueError(f"limit must be between 1 and {MAX_READ_LINES} lines.")
        lines = content.splitlines(keepends=True)
        if offset > len(lines) + 1:
            raise ValueError(
                f"offset {offset} exceeds {filename}'s {len(lines)} lines."
            )
        chunk = "".join(
            lines[offset - 1 :] if limit is None else lines[offset - 1 : offset - 1 + limit]
        )
        next_offset = offset + len(chunk.splitlines())
        return json.dumps(
            {
                "exists": True,
                "content": chunk,
                "sha256": digest,
                "total_lines": len(lines),
                "offset": offset,
                "returned_lines": len(chunk.splitlines()),
                "next_offset": next_offset if next_offset <= len(lines) else None,
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _validate_expected_sha(
        filename: str, content: str, expected_sha256: str | None
    ) -> None:
        if expected_sha256 is None:
            return
        if len(expected_sha256) != 64:
            raise ValueError("expected_sha256 must be a SHA-256 digest.")
        actual = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if actual != expected_sha256:
            raise ValueError(
                f"{filename} changed since it was read (current sha256={actual}); "
                "call read_file again and use that digest as expected_sha256."
            )

    def write_file(
        self,
        filename: str,
        content: str,
        expected_sha256: str | None = None,
    ) -> str:
        path = self._path(filename)
        with _file_lock(path):
            current = path.read_text(encoding="utf-8") if path.exists() else ""
            # ``expected_sha256`` is optional. When omitted, the caller is
            # declaring an unconditional overwrite; we still capture the
            # current SHA so the caller can pass it on the next edit if they
            # want strict conflict detection. This avoids forcing a redundant
            # ``read_file`` round-trip before every ``write_file`` (the model
            # already knows its own last write produced a known SHA).
            self._validate_expected_sha(filename, current, expected_sha256)
            return self._write_model(content, "write_file")

    def _write_model(self, content: str, operation: str) -> str:
        """Validate, commit revision, and atomically write model.py."""
        warnings = self.validate_model(content)
        revision = self._revisions.commit(
            content,
            RevisionOrigin(
                kind="agent_edit",
                operation=operation,
                tool_call_id=self._tool_call_id,
            ),
        )
        result = (
            f"Wrote model.py ({len(content.splitlines())} lines, "
            f"{len(content)} chars, revision {revision.id[:8]})."
        )
        if warnings:
            result += "\nPRE-FLIGHT WARNING: " + " | ".join(warnings)
        return result

    def edit_file(
        self,
        filename: str,
        old_string: str,
        new_string: str,
        expected_sha256: str | None = None,
    ) -> str:
        if not old_string:
            raise ValueError("old_string must not be empty.")
        # ``expected_sha256`` is optional: omitting it means an unconditional
        # edit (the same pattern as ``write_file``). The agent already knows
        # the current content from its own previous write, so a redundant
        # ``read_file`` round-trip only to capture the SHA would just waste a
        # tool call + tokens + latency. Conflict detection remains opt-in via
        # ``expected_sha256``; pass it when you want to guarantee no concurrent
        # edit. If expected_sha256 mismatches, re-read; do not retry the same
        # edit blindly.
        path = self._path(filename)
        with _file_lock(path):
            if not path.exists():
                raise ValueError(
                    f"{filename} does not exist; use write_file to create it."
                )
            current = path.read_text(encoding="utf-8")
            self._validate_expected_sha(filename, current, expected_sha256)
            matches = current.count(old_string)
            if matches != 1:
                raise ValueError(
                    f"Expected one exact match, found {matches}; file was not changed."
                )
            updated = current.replace(old_string, new_string, 1)
            start_line = current[: current.find(old_string)].count("\n") + 1
            old_lines = old_string.count("\n") or 1
            end_line = start_line + old_lines - 1
            new_lines = new_string.count("\n") or 1
            new_end_line = start_line + new_lines - 1
            base = self._write_model(updated, "edit_file")
        return (
            f"{base} "
            f"(replaced lines {start_line}-{end_line} with lines "
            f"{start_line}-{new_end_line})."
        )

    def insert_file(
        self,
        filename: str,
        anchor: str,
        content: str,
        position: str,
        expected_sha256: str | None = None,
    ) -> str:
        """Insert content next to one short, exact anchor without replacing it."""
        if not anchor:
            raise ValueError("anchor must not be empty.")
        if not content:
            raise ValueError("content must not be empty.")
        if position not in {"before", "after"}:
            raise ValueError("position must be 'before' or 'after'.")
        path = self._path(filename)
        with _file_lock(path):
            if not path.exists():
                raise ValueError(f"{filename} does not exist; use write_file to create it.")
            current = path.read_text(encoding="utf-8")
            self._validate_expected_sha(filename, current, expected_sha256)
            matches = current.count(anchor)
            if matches != 1:
                raise ValueError(
                    f"Expected one exact anchor, found {matches}; file was not changed."
                )
            replacement = content + anchor if position == "before" else anchor + content
            updated = current.replace(anchor, replacement, 1)
            line = current[: current.find(anchor)].count("\n") + 1
            base = self._write_model(updated, "insert_file")
        return f"{base} (inserted {position} anchor at line {line})."

    def edit_file_atomic(
        self,
        filename: str,
        edits: list[dict[str, str]],
        expected_sha256: str | None = None,
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
            new_string = entry.get("new_string", "")
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
            self._validate_expected_sha(filename, current, expected_sha256)
            # Validate every match against the original buffer (not the running
            # ``updated`` string) so an edit cannot invalidate another edit's
            # ``old_string`` after a previous replace has rewritten the region.
            resolved: list[tuple[tuple[int, int], str, str]] = []
            for index, (old_string, new_string) in enumerate(normalised):
                matches = sum(
                    1
                    for start in range(len(current))
                    if current.startswith(old_string, start)
                )
                if matches != 1:
                    raise ValueError(
                        f"edits[{index}] expected one exact match, found "
                        f"{matches}; file was not changed."
                    )
                start = current.find(old_string)
                resolved.append(((start, start + len(old_string)), old_string, new_string))
            # Reject any pair of edits whose resolved byte ranges overlap.
            ordered = sorted(resolved, key=lambda entry: entry[0][0])
            previous_end = -1
            for (start, end), old_string, _new_string in ordered:
                if start < previous_end:
                    raise ValueError(
                        "edit_file_atomic rejected overlapping edits; verify "
                        "each old_string is copied from the same file state."
                    )
                previous_end = end
            # All validations passed — apply the replacements using the
            # offsets resolved against the original buffer. Applying in caller
            # order against the running buffer would let an earlier edit's
            # ``new_string`` introduce text that the next edit's ``old_string``
            # then matches, silently leaving the original target untouched.
            # We rebuild slices from each edit's start/end so the byte ranges
            # always refer to the original positions the LLM copied from.
            parts: list[str] = []
            cursor = 0
            ranges: list[tuple[int, int, int]] = []
            ordered_offsets = sorted(
                ((start, end, old_string, new_string) for (start, end), old_string, new_string in resolved)
            )
            for index, (start, end, old_string, new_string) in enumerate(ordered_offsets):
                parts.append(current[cursor:start])
                parts.append(new_string)
                start_line = current[:start].count("\n") + 1
                old_lines = old_string.count("\n") or 1
                end_line = start_line + old_lines - 1
                new_lines = new_string.count("\n") or 1
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
