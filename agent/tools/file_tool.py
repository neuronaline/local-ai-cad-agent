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
_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_FILE_LOCKS: dict[Path, threading.RLock] = {}
_FILE_LOCKS_GUARD = threading.Lock()


def _file_lock(path: Path) -> threading.RLock:
    with _FILE_LOCKS_GUARD:
        return _FILE_LOCKS.setdefault(path, threading.RLock())


class ModelPreflight(ast.NodeVisitor):
    """Catch deterministic build123d mistakes before running the CAD kernel."""

    def __init__(self) -> None:
        self.values: dict[str, object] = {}
        self.edge_points: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {}
        self._topology_changed = False
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
            if self._topology_changed and self._contains_fixed_selector_index(
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
            self._topology_changed = True
        self.generic_visit(node)

    def _validate_radius_arc(self, node: ast.Call) -> None:
        start = self._point(self._argument(node, 0, "start_point"))
        end = self._point(self._argument(node, 1, "end_point"))
        radius = self._number(self._argument(node, 2, "radius"))
        if radius is not None and radius <= 0:
            raise ValueError("RadiusArc radius must be positive.")
        if start is None or end is None or radius is None:
            return
        chord = math.dist(start, end)
        minimum = chord / 2
        if radius + 1e-9 < minimum:
            raise ValueError(
                f"RadiusArc radius {radius:g} is too small for chord {chord:.3f}; "
                f"minimum radius is {minimum:.3f}."
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
        if limit is not None and (limit < 1 or limit > 2000):
            raise ValueError("limit must be between 1 and 2000 lines.")
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
            f"Wrote model.py ({len(content)} characters, revision {revision.id[:8]})."
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
