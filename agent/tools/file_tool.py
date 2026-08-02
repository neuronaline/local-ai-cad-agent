from __future__ import annotations

import ast
import json
import math
import operator
import re
from pathlib import Path
from typing import ClassVar

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
EDITABLE_FILES = {"model.py", "summary.md"}
_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}


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
            self.blocked_errors.append(f"Unsafe import blocked: {', '.join(sorted(forbidden))}")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        names = [(node.module or "").split(".")[0]]
        forbidden = set(names) & BLOCKED_IMPORTS
        if forbidden:
            self.blocked_errors.append(f"Unsafe import blocked: {', '.join(sorted(forbidden))}")

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            value = self._number_or_tuple(node.value)
            if value is not None:
                self.values[name] = value
            points = self._line_points(node.value)
            if points is not None:
                self.edge_points[name] = points
            if self._topology_changed and self._contains_fixed_selector_index(node.value):
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
        # Check for blocked built-in calls.
        if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_CALLS:
            self.blocked_errors.append(f"Unsafe function blocked: {node.func.id}")
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

    def _number_or_tuple(self, node: ast.AST | None) -> float | tuple[float, ...] | None:
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
            if not (
                isinstance(index, ast.Constant)
                and isinstance(index.value, int)
            ):
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
    __tool_schema__: ClassVar[dict[str, object]] = {
        "type": "function",
        "function": {
            "name": "file",
            "description": "Read or safely edit model.py or summary.md.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": ["read", "write", "replace", "regex_replace"]},
                    "filename": {"type": "string", "enum": ["model.py", "summary.md"]},
                    "content": {"type": "string"},
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                    "pattern": {"type": "string"},
                    "replacement": {"type": "string"},
                    "count": {"type": "integer"},
                },
                "required": ["operation", "filename"],
            },
        },
    }

    def __init__(self, project_dir: Path) -> None:
        self.project_dir = project_dir.resolve()

    def _path(self, filename: str) -> Path:
        if filename not in EDITABLE_FILES:
            raise ValueError("Only model.py and summary.md can be edited.")
        path = (self.project_dir / filename).resolve()
        if path.parent != self.project_dir:
            raise ValueError("Path escapes the project directory.")
        return path

    @staticmethod
    def validate_model(code: str) -> list[str]:
        try:
            tree = ast.parse(code, filename="model.py")
        except SyntaxError as error:
            raise ValueError(f"Invalid Python: {error.msg} (line {error.lineno})") from error
        preflight = ModelPreflight()
        preflight.visit(tree)
        if preflight.blocked_errors:
            raise ValueError(preflight.blocked_errors[0])
        return preflight.warnings

    def read(self, filename: str) -> str:
        path = self._path(filename)
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def write(self, filename: str, content: str) -> str:
        path = self._path(filename)
        warnings: list[str] = []
        if filename == "model.py":
            warnings = self.validate_model(content)
        path.write_text(content, encoding="utf-8")
        result = f"Wrote {filename} ({len(content)} characters)."
        if warnings:
            result += "\nPRE-FLIGHT WARNING: " + " | ".join(warnings)
        return result

    def replace(self, filename: str, old: str, new: str) -> str:
        current = self.read(filename)
        if old not in current:
            raise ValueError("The requested text was not found; file was not changed.")
        updated = current.replace(old, new, 1)
        return self.write(filename, updated)

    def execute(self, args: dict) -> tuple[str, bool]:
        """Dispatch file operations by name. Returns (result_json, waiting=False)."""
        operation = args["operation"]
        filename = args["filename"]
        if operation == "read":
            result = self.read(filename)
        elif operation == "write":
            result = self.write(filename, args.get("content", ""))
        elif operation == "replace":
            result = self.replace(filename, args.get("old", ""), args.get("new", ""))
        elif operation == "regex_replace":
            result = self.regex_replace(
                filename, args.get("pattern", ""), args.get("replacement", ""), args.get("count", 1)
            )
        else:
            raise ValueError("Unsupported file operation.")
        return json.dumps(result) if not isinstance(result, str) else result, False

    def regex_replace(self, filename: str, pattern: str, replacement: str, count: int = 1) -> str:
        if len(pattern) > 2000 or len(replacement) > 10000:
            raise ValueError("Patch pattern or replacement is too large.")
        if not 1 <= count <= 20:
            raise ValueError("Regex replacement count must be between 1 and 20.")
        current = self.read(filename)
        try:
            updated, replacements = re.subn(pattern, replacement, current, count=count)
        except re.error as error:
            raise ValueError(f"Invalid regex pattern: {error}") from error
        if replacements == 0:
            raise ValueError("The regex did not match; file was not changed.")
        self.write(filename, updated)
        return f"Updated {filename} with {replacements} regex replacement(s)."
