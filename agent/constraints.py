"""User-owned source constraints (pins) for model.py parameters and features.

Pins are stored outside model.py in .cad-agent/constraints.json. The AI receives
them as context but cannot mutate them through any tool. Only user-facing HTTP
endpoints may create or remove constraints.

Two enforceable protection kinds:
  1. parameter — protects a top-level typed assignment (e.g. HOLE_DIAMETER: float = 3.2)
  2. source_feature — protects an explicitly marked, named source region
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import tempfile
import textwrap
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
_FEATURE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_PARAMETER_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MAX_CONSTRAINTS = 100
_CONSTRAINT_LOCK = threading.RLock()


class ConstraintError(Exception):
    """Raised when constraint validation fails or constraint data is corrupt."""


@dataclass(frozen=True)
class Constraint:
    id: str
    kind: str  # parameter, source_feature
    name: str
    expected_ast: str | None = None  # For parameter pins
    normalized_sha256: str | None = None  # For source_feature pins
    created_at: str = ""

    def to_dict(self) -> dict:
        d: dict = {
            "id": self.id,
            "kind": self.kind,
            "name": self.name,
        }
        if self.expected_ast is not None:
            d["expected_ast"] = self.expected_ast
        if self.normalized_sha256 is not None:
            d["normalized_sha256"] = self.normalized_sha256
        d["created_at"] = self.created_at
        return d

    @classmethod
    def from_dict(cls, data: dict) -> Constraint:
        return cls(
            id=str(data["id"]),
            kind=str(data["kind"]),
            name=str(data["name"]),
            expected_ast=data.get("expected_ast"),
            normalized_sha256=data.get("normalized_sha256"),
            created_at=str(data.get("created_at", "")),
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- #
#  AST normalization
# --------------------------------------------------------------------------- #

def _normalize_ast_string(node: ast.AST) -> str:
    """Return a canonical string representation of an AST node.

    Whitespace and formatting changes must not affect the result.
    """
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def _find_parameter_assignments(tree: ast.Module, name: str) -> list[ast.AST]:
    """Find every top-level assignment value for a parameter name."""
    values = _find_typed_parameter_assignments(tree, name)
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == name:
                values.append(node.value)
    return values


def _find_typed_parameter_assignments(tree: ast.Module, name: str) -> list[ast.AST]:
    """Find every top-level typed parameter assignment value for a name."""
    values: list[ast.AST] = []
    for node in ast.iter_child_nodes(tree):
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
            and node.value is not None
        ):
            values.append(node.value)
    return values


def _find_parameter_assignment(tree: ast.Module, name: str) -> ast.AST | None:
    values = _find_parameter_assignments(tree, name)
    return values[0] if len(values) == 1 else None


# --------------------------------------------------------------------------- #
#  Named feature markers
# --------------------------------------------------------------------------- #

_FEATURE_START_RE = re.compile(
    r"^\s*#\s*cad-feature:\s*([a-z][a-z0-9_]*)\s+start\s*$"
)
_FEATURE_END_RE = re.compile(
    r"^\s*#\s*cad-feature:\s*([a-z][a-z0-9_]*)\s+end\s*$"
)
_FEATURE_MARKER_RE = re.compile(r"^\s*#\s*cad-feature:")


@dataclass(frozen=True)
class FeatureRegion:
    name: str
    start_line: int  # 1-based line number of the start marker
    end_line: int  # 1-based line number of the end marker
    content: str  # normalized source between markers (excluding markers and surrounding blanks)


def parse_feature_regions(source: str) -> dict[str, FeatureRegion]:
    """Parse all named feature regions from source code.

    Returns a dict mapping feature name to FeatureRegion.
    Raises ConstraintError for malformed markers, duplicates, nesting, or mismatches.
    """
    lines = source.splitlines()
    open_features: dict[str, int] = {}
    regions: dict[str, FeatureRegion] = {}

    for i, line in enumerate(lines, start=1):
        start_match = _FEATURE_START_RE.match(line)
        end_match = _FEATURE_END_RE.match(line)

        if _FEATURE_MARKER_RE.match(line) and not start_match and not end_match:
            raise ConstraintError(f"Malformed feature marker at line {i}.")

        if start_match:
            name = start_match.group(1)
            if not _FEATURE_NAME_RE.fullmatch(name):
                raise ConstraintError(f"Invalid feature name: {name}")
            if open_features:
                raise ConstraintError(
                    f"Feature '{name}' cannot start while '{next(iter(open_features))}' "
                    "is still open (nested features not allowed)."
                )
            if name in regions:
                raise ConstraintError(f"Duplicate feature marker: '{name}'.")
            open_features[name] = i
        elif end_match:
            name = end_match.group(1)
            if name not in open_features:
                raise ConstraintError(f"Feature end marker for '{name}' has no matching start.")
            start_line = open_features.pop(name)
            content_lines = lines[start_line : i - 1]  # lines between markers
            # Normalize: strip surrounding blank lines but preserve internal structure.
            content = "\n".join(content_lines).strip("\n")
            regions[name] = FeatureRegion(
                name=name,
                start_line=start_line,
                end_line=i,
                content=content,
            )

    if open_features:
        missing = ", ".join(sorted(open_features))
        raise ConstraintError(f"Unclosed feature marker(s): {missing}")

    return regions


def _normalize_feature_content(content: str) -> str:
    """Normalize feature content to executable AST, ignoring formatting/comments."""
    normalized = textwrap.dedent(content).strip("\n")
    try:
        tree = ast.parse(normalized or "pass", filename="model.py")
    except SyntaxError as error:
        raise ConstraintError(
            "Protected feature content must be a self-contained Python block: "
            f"{error.msg} (line {error.lineno})."
        ) from error
    return _normalize_ast_string(tree)


def _feature_content_hash(content: str) -> str:
    return _sha256(_normalize_feature_content(content).encode("utf-8"))


# --------------------------------------------------------------------------- #
#  ConstraintStore
# --------------------------------------------------------------------------- #

class ConstraintStore:
    """Manages constraint persistence in .cad-agent/constraints.json."""

    def __init__(self, project_dir: Path) -> None:
        self.project_dir = project_dir.resolve()
        self._path = self.project_dir / ".cad-agent" / "constraints.json"

    def list(self) -> list[Constraint]:
        """Return all constraints. Returns empty list if none or file missing."""
        with _CONSTRAINT_LOCK:
            data = self._read_safe()
            constraints_data = data.get("constraints", []) if data else []
            if not isinstance(constraints_data, list):
                raise ConstraintError("Invalid constraints list structure.")
            constraints: list[Constraint] = []
            for item in constraints_data:
                if not isinstance(item, dict):
                    raise ConstraintError("Invalid constraint entry structure.")
                try:
                    constraint = Constraint.from_dict(item)
                except (KeyError, TypeError) as error:
                    raise ConstraintError("Malformed constraint entry.") from error
                self._validate_constraint(constraint)
                constraints.append(constraint)
            return constraints

    def get(self, constraint_id: str) -> Constraint | None:
        for c in self.list():
            if c.id == constraint_id:
                return c
        return None

    def add(self, constraint: Constraint) -> None:
        with _CONSTRAINT_LOCK:
            self._validate_constraint(constraint)
            constraints = self.list()
            if len(constraints) >= _MAX_CONSTRAINTS:
                raise ConstraintError(f"Maximum constraint limit ({_MAX_CONSTRAINTS}) reached.")
            # Check for duplicate name+kind.
            for existing in constraints:
                if existing.kind == constraint.kind and existing.name == constraint.name:
                    raise ConstraintError(
                        f"A {constraint.kind} constraint for '{constraint.name}' already exists."
                    )
            constraints.append(constraint)
            self._write(constraints)

    def remove(self, constraint_id: str) -> Constraint | None:
        with _CONSTRAINT_LOCK:
            constraints = self.list()
            removed = None
            remaining = []
            for c in constraints:
                if c.id == constraint_id:
                    removed = c
                else:
                    remaining.append(c)
            if removed is not None:
                self._write(remaining)
            return removed

    def clear(self) -> None:
        """Remove all constraints (for testing)."""
        with _CONSTRAINT_LOCK:
            self._write([])

    # ------------------------------------------------------------------ #
    #  Factory methods — derive constraints from active model
    # ------------------------------------------------------------------ #

    def create_parameter_constraint(self, name: str, source: str) -> Constraint:
        """Create a parameter pin from the active model source."""
        if not _PARAMETER_NAME_RE.fullmatch(name):
            raise ConstraintError(f"Invalid parameter name: {name}")
        try:
            tree = ast.parse(source, filename="model.py")
        except SyntaxError as error:
            raise ConstraintError(f"Cannot parse model.py: {error.msg} (line {error.lineno})") from error

        values = _find_typed_parameter_assignments(tree, name)
        if not values:
            raise ConstraintError(
                f"Parameter '{name}' not found as a top-level assignment in model.py."
            )
        if len(values) > 1:
            raise ConstraintError(
                f"Parameter '{name}' has multiple top-level assignments in model.py."
            )

        expected_ast = _normalize_ast_string(values[0])
        return Constraint(
            id=str(uuid.uuid4()),
            kind="parameter",
            name=name,
            expected_ast=expected_ast,
            created_at=_utc_now(),
        )

    def create_source_feature_constraint(self, name: str, source: str) -> Constraint:
        """Create a source_feature pin from the active model source."""
        if not _FEATURE_NAME_RE.fullmatch(name):
            raise ConstraintError(f"Invalid feature name: {name}")
        regions = parse_feature_regions(source)
        if name not in regions:
            raise ConstraintError(
                f"Feature '{name}' not found in model.py. "
                "Use '# cad-feature: {name} start' and '# cad-feature: {name} end' markers."
            )
        region = regions[name]
        normalized_hash = _feature_content_hash(region.content)
        return Constraint(
            id=str(uuid.uuid4()),
            kind="source_feature",
            name=name,
            normalized_sha256=normalized_hash,
            created_at=_utc_now(),
        )

    def discover_targets(self, source: str) -> dict[str, list[dict[str, object]]]:
        """Return protectable parameters and named features from active source."""
        try:
            tree = ast.parse(source, filename="model.py")
        except SyntaxError as error:
            raise ConstraintError(
                f"Cannot parse model.py: {error.msg} (line {error.lineno})"
            ) from error

        pinned = {(constraint.kind, constraint.name) for constraint in self.list()}
        parameters: list[dict[str, object]] = []
        for node in tree.body:
            if not (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.value is not None
            ):
                continue
            parameters.append({
                "name": node.target.id,
                "value": ast.unparse(node.value),
                "line": node.lineno,
                "pinned": ("parameter", node.target.id) in pinned,
            })

        features = [
            {
                "name": region.name,
                "start_line": region.start_line,
                "end_line": region.end_line,
                "pinned": ("source_feature", region.name) in pinned,
            }
            for region in parse_feature_regions(source).values()
        ]
        return {"parameters": parameters, "features": features}

    # ------------------------------------------------------------------ #
    #  Private helpers
    # ------------------------------------------------------------------ #

    def _read_safe(self) -> dict | None:
        if not self._path.is_file():
            return None
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            raise ConstraintError(f"Malformed constraints file: {error}") from error
        if not isinstance(data, dict):
            raise ConstraintError("Invalid constraints file structure.")
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ConstraintError("Unsupported constraints schema version.")
        return data

    @staticmethod
    def _validate_constraint(constraint: Constraint) -> None:
        try:
            uuid.UUID(constraint.id)
        except (ValueError, AttributeError) as error:
            raise ConstraintError("Constraint has an invalid ID.") from error
        if constraint.kind == "parameter":
            if not _PARAMETER_NAME_RE.fullmatch(constraint.name):
                raise ConstraintError("Constraint has an invalid parameter name.")
            if not isinstance(constraint.expected_ast, str) or not constraint.expected_ast:
                raise ConstraintError("Parameter constraint is missing its expected value.")
        elif constraint.kind == "source_feature":
            if not _FEATURE_NAME_RE.fullmatch(constraint.name):
                raise ConstraintError("Constraint has an invalid feature name.")
            digest = constraint.normalized_sha256
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ConstraintError("Feature constraint has an invalid digest.")
        else:
            raise ConstraintError("Constraint has an unsupported kind.")

    def _write(self, constraints: list[Constraint]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "schema_version": SCHEMA_VERSION,
            "constraints": [c.to_dict() for c in constraints],
        }
        with tempfile.NamedTemporaryFile(
            mode="w", dir=self._path.parent, encoding="utf-8", delete=False, suffix=".tmp"
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(data, temporary, ensure_ascii=False, indent=2)
        try:
            temporary_path.replace(self._path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise


# --------------------------------------------------------------------------- #
#  ModelConstraintValidator
# --------------------------------------------------------------------------- #

class ModelConstraintValidator:
    """Validates candidate model source against active constraints."""

    def __init__(self, store: ConstraintStore) -> None:
        self._store = store

    def active_constraints(self) -> list[Constraint]:
        return self._store.list()

    def validate(self, candidate_source: str) -> None:
        """Raise ConstraintError if the candidate violates any active constraint.

        Reports all violated pins in one bounded error.
        """
        constraints = self._store.list()
        if not constraints:
            return

        try:
            tree = ast.parse(candidate_source, filename="model.py")
        except SyntaxError as error:
            raise ConstraintError(
                f"Cannot parse candidate model.py: {error.msg} (line {error.lineno})"
            ) from error

        violations: list[str] = []

        # Parse feature regions once for all source_feature constraints.
        feature_regions: dict[str, FeatureRegion] = {}
        source_feature_constraints = [c for c in constraints if c.kind == "source_feature"]
        if source_feature_constraints:
            try:
                feature_regions = parse_feature_regions(candidate_source)
            except ConstraintError as error:
                violations.append(str(error))

        for constraint in constraints:
            if constraint.kind == "parameter":
                self._validate_parameter(tree, constraint, violations)
            elif constraint.kind == "source_feature":
                self._validate_source_feature(feature_regions, constraint, violations)

        if violations:
            raise ConstraintError(
                "Protected constraint(s) violated: " + "; ".join(violations)
            )

    def _validate_parameter(
        self, tree: ast.Module, constraint: Constraint, violations: list[str]
    ) -> None:
        values = _find_parameter_assignments(tree, constraint.name)
        if not values:
            violations.append(
                f"parameter '{constraint.name}' was deleted or renamed"
            )
            return
        if len(values) > 1:
            violations.append(
                f"parameter '{constraint.name}' has multiple top-level assignments"
            )
            return
        actual_ast = _normalize_ast_string(values[0])
        if actual_ast != constraint.expected_ast:
            violations.append(
                f"parameter '{constraint.name}' value was changed"
            )

    def _validate_source_feature(
        self,
        regions: dict[str, FeatureRegion],
        constraint: Constraint,
        violations: list[str],
    ) -> None:
        if constraint.name not in regions:
            violations.append(
                f"feature '{constraint.name}' was deleted or markers removed"
            )
            return
        region = regions[constraint.name]
        try:
            actual_hash = _feature_content_hash(region.content)
        except ConstraintError as error:
            violations.append(f"feature '{constraint.name}' is invalid: {error}")
            return
        if actual_hash != constraint.normalized_sha256:
            violations.append(
                f"feature '{constraint.name}' content was changed"
            )

    def constraint_summary(self) -> str:
        """Return a concise machine-readable list of active pins for the agent context."""
        constraints = self._store.list()
        if not constraints:
            return ""
        lines = []
        for c in constraints:
            if c.kind == "parameter":
                lines.append(f"- parameter: {c.name}")
            elif c.kind == "source_feature":
                lines.append(f"- source_feature: {c.name}")
        return "Protected (do not change):\n" + "\n".join(lines)
