"""Tests for constraint storage and model constraint validation."""
from __future__ import annotations

from pathlib import Path

import pytest

from agent.constraints import (
    Constraint,
    ConstraintError,
    ConstraintStore,
    ModelConstraintValidator,
    parse_feature_regions,
)


def _add_param_store(store, source, name="WIDTH"):
    """Helper: create and add a parameter constraint."""
    c = store.create_parameter_constraint(name, source)
    store.add(c)
    return c


def _add_feature_store(store, source, name="base_plate"):
    """Helper: create and add a source_feature constraint."""
    c = store.create_source_feature_constraint(name, source)
    store.add(c)
    return c


# --------------------------------------------------------------------------- #
#  Feature region parsing
# --------------------------------------------------------------------------- #

def test_parse_single_feature_region():
    source = (
        "# cad-feature: base_plate start\n"
        "base = Box(10, 20, 5)\n"
        "result = base\n"
        "# cad-feature: base_plate end\n"
    )
    regions = parse_feature_regions(source)
    assert "base_plate" in regions
    assert "Box(10, 20, 5)" in regions["base_plate"].content


def test_parse_multiple_feature_regions():
    source = (
        "# cad-feature: holes start\n"
        "holes = Cylinder(3, 5)\n"
        "result = result - holes\n"
        "# cad-feature: holes end\n"
        "\n"
        "# cad-feature: base start\n"
        "base = Box(20, 20, 5)\n"
        "# cad-feature: base end\n"
    )
    regions = parse_feature_regions(source)
    assert set(regions) == {"holes", "base"}


def test_parse_unclosed_feature_raises():
    source = "# cad-feature: base start\nbase = Box(10, 20, 5)\n"
    with pytest.raises(ConstraintError, match="Unclosed"):
        parse_feature_regions(source)


def test_parse_unopened_feature_end_raises():
    source = "base = Box(10, 20, 5)\n# cad-feature: base end\n"
    with pytest.raises(ConstraintError, match="no matching start"):
        parse_feature_regions(source)


def test_parse_duplicate_feature_raises():
    source = (
        "# cad-feature: base start\n"
        "base = Box(10, 20, 5)\n"
        "# cad-feature: base end\n"
        "# cad-feature: base start\n"
        "base2 = Box(30, 20, 5)\n"
        "# cad-feature: base end\n"
    )
    with pytest.raises(ConstraintError, match="Duplicate"):
        parse_feature_regions(source)


def test_parse_nested_feature_raises():
    source = (
        "# cad-feature: outer start\n"
        "# cad-feature: inner start\n"
        "Box(10, 10, 10)\n"
        "# cad-feature: inner end\n"
        "# cad-feature: outer end\n"
    )
    with pytest.raises(ConstraintError, match="nested"):
        parse_feature_regions(source)


def test_parse_invalid_feature_name_rejected():
    source = (
        "# cad-feature: 123bad start\n"
        "Box(10, 10, 10)\n"
        "# cad-feature: 123bad end\n"
    )
    with pytest.raises(ConstraintError, match="Malformed feature marker"):
        parse_feature_regions(source)


# --------------------------------------------------------------------------- #
#  Constraint creation
# --------------------------------------------------------------------------- #

def test_create_parameter_constraint(tmp_path: Path):
    store = ConstraintStore(tmp_path)
    source = "WIDTH: float = 10.0\nresult = Box(WIDTH, 20, 5)\n"
    c = store.create_parameter_constraint("WIDTH", source)
    assert c.kind == "parameter"
    assert c.name == "WIDTH"
    assert c.expected_ast is not None


def test_create_parameter_constraint_missing(tmp_path: Path):
    store = ConstraintStore(tmp_path)
    with pytest.raises(ConstraintError, match="not found"):
        store.create_parameter_constraint("NONEXISTENT", "result = Box(10, 20, 5)\n")


def test_create_source_feature_constraint(tmp_path: Path):
    store = ConstraintStore(tmp_path)
    source = (
        "# cad-feature: holes start\n"
        "holes = Cylinder(3, 5)\n"
        "# cad-feature: holes end\n"
    )
    c = store.create_source_feature_constraint("holes", source)
    assert c.kind == "source_feature"
    assert c.name == "holes"
    assert c.normalized_sha256 is not None


def test_create_source_feature_constraint_missing(tmp_path: Path):
    store = ConstraintStore(tmp_path)
    with pytest.raises(ConstraintError, match="not found"):
        store.create_source_feature_constraint("nonexistent", "result = 1\n")


# --------------------------------------------------------------------------- #
#  ConstraintStore persistence
# --------------------------------------------------------------------------- #

def test_constraint_store_list_empty(tmp_path: Path):
    assert ConstraintStore(tmp_path).list() == []


def test_constraint_store_add_and_list(tmp_path: Path):
    store = ConstraintStore(tmp_path)
    source = "WIDTH: float = 10.0\n"
    c = store.create_parameter_constraint("WIDTH", source)
    store.add(c)
    assert len(store.list()) == 1
    assert store.list()[0].name == "WIDTH"


def test_constraint_store_remove(tmp_path: Path):
    store = ConstraintStore(tmp_path)
    c = store.create_parameter_constraint("WIDTH", "WIDTH: float = 10.0\n")
    store.add(c)
    removed = store.remove(c.id)
    assert removed is not None
    assert removed.name == "WIDTH"
    assert store.list() == []


def test_constraint_store_remove_nonexistent(tmp_path: Path):
    store = ConstraintStore(tmp_path)
    assert store.remove("00000000-0000-0000-0000-000000000000") is None


def test_constraint_store_duplicate_rejected(tmp_path: Path):
    store = ConstraintStore(tmp_path)
    source = "WIDTH: float = 10.0\n"
    c1 = store.create_parameter_constraint("WIDTH", source)
    c2 = store.create_parameter_constraint("WIDTH", source)
    store.add(c1)
    with pytest.raises(ConstraintError, match="already exists"):
        store.add(c2)


def test_constraint_store_max_limit(tmp_path: Path):
    store = ConstraintStore(tmp_path)
    for i in range(100):
        c = Constraint(
            id=f"00000000-0000-0000-0000-{i:012d}",
            kind="parameter",
            name=f"WIDTH_{i}",
            expected_ast="Constant(value=10.0)",
        )
        store.add(c)
    with pytest.raises(ConstraintError, match="Maximum constraint limit"):
        c = Constraint(
            id="00000000-0000-0000-0000-000000000100",
            kind="parameter",
            name="OVERFLOW",
            expected_ast="Constant(value=1.0)",
        )
        store.add(c)


def test_constraint_store_malformed_file_is_preserved(tmp_path: Path):
    path = tmp_path / ".cad-agent" / "constraints.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("garbage", encoding="utf-8")

    with pytest.raises(ConstraintError, match="Malformed"):
        ConstraintStore(tmp_path).list()


# --------------------------------------------------------------------------- #
#  ModelConstraintValidator — parameter pins
# --------------------------------------------------------------------------- #

def test_validator_pinned_parameter_survives_whitespace_changes(tmp_path: Path):
    store = ConstraintStore(tmp_path)
    source = "WIDTH: float = 10.0\nresult = Box(WIDTH, 20, 5)\n"
    _add_param_store(store, source)

    candidate = "WIDTH:   float   =   10.0\nresult = Box(WIDTH, 20, 5)\n"
    ModelConstraintValidator(store).validate(candidate)  # Should not raise.


def test_validator_changing_pinned_parameter_rejected(tmp_path: Path):
    store = ConstraintStore(tmp_path)
    source = "WIDTH: float = 10.0\nresult = Box(WIDTH, 20, 5)\n"
    _add_param_store(store, source)

    candidate = "WIDTH: float = 20.0\nresult = Box(WIDTH, 20, 5)\n"
    with pytest.raises(ConstraintError, match="WIDTH"):
        ModelConstraintValidator(store).validate(candidate)


def test_validator_deleting_pinned_parameter_rejected(tmp_path: Path):
    store = ConstraintStore(tmp_path)
    source = "WIDTH: float = 10.0\nresult = Box(WIDTH, 20, 5)\n"
    _add_param_store(store, source)

    candidate = "result = Box(10.0, 20, 5)\n"
    with pytest.raises(ConstraintError, match="WIDTH.*deleted"):
        ModelConstraintValidator(store).validate(candidate)


def test_validator_renaming_pinned_parameter_rejected(tmp_path: Path):
    store = ConstraintStore(tmp_path)
    source = "WIDTH: float = 10.0\nresult = Box(WIDTH, 20, 5)\n"
    _add_param_store(store, source)

    candidate = "NEW_WIDTH: float = 10.0\nresult = Box(NEW_WIDTH, 20, 5)\n"
    with pytest.raises(ConstraintError, match="WIDTH.*deleted"):
        ModelConstraintValidator(store).validate(candidate)


def test_validator_rejects_duplicate_pinned_parameter_assignment(tmp_path: Path):
    store = ConstraintStore(tmp_path)
    source = "WIDTH: float = 10.0\nresult = Box(WIDTH, 20, 5)\n"
    _add_param_store(store, source)

    candidate = source + "WIDTH = 20.0\n"
    with pytest.raises(ConstraintError, match="WIDTH.*multiple"):
        ModelConstraintValidator(store).validate(candidate)


# --------------------------------------------------------------------------- #
#  ModelConstraintValidator — source_feature pins
# --------------------------------------------------------------------------- #

def test_validator_pinned_feature_unchanged_passes(tmp_path: Path):
    store = ConstraintStore(tmp_path)
    source = (
        "# cad-feature: holes start\n"
        "holes = Cylinder(3, 5)\n"
        "# cad-feature: holes end\n"
    )
    _add_feature_store(store, source, "holes")
    ModelConstraintValidator(store).validate(source)


def test_validator_pinned_feature_allows_formatting_and_comment_changes(tmp_path: Path):
    store = ConstraintStore(tmp_path)
    source = (
        "# cad-feature: holes start\n"
        "holes = Cylinder(3, 5)\n"
        "# cad-feature: holes end\n"
    )
    _add_feature_store(store, source, "holes")
    candidate = (
        "# cad-feature: holes start\n"
        "# Formatting does not change the feature.\n"
        "holes=Cylinder(3,5)\n"
        "# cad-feature: holes end\n"
    )

    ModelConstraintValidator(store).validate(candidate)


def test_validator_pinned_feature_deleted_rejected(tmp_path: Path):
    store = ConstraintStore(tmp_path)
    source = (
        "# cad-feature: holes start\n"
        "holes = Cylinder(3, 5)\n"
        "# cad-feature: holes end\n"
    )
    _add_feature_store(store, source, "holes")

    candidate = "result = Box(10, 20, 5)\n"
    with pytest.raises(ConstraintError, match="holes"):
        ModelConstraintValidator(store).validate(candidate)


def test_validator_pinned_feature_changed_rejected(tmp_path: Path):
    store = ConstraintStore(tmp_path)
    source = (
        "# cad-feature: holes start\n"
        "holes = Cylinder(3, 5)\n"
        "# cad-feature: holes end\n"
    )
    _add_feature_store(store, source, "holes")

    candidate = (
        "# cad-feature: holes start\n"
        "holes = Cylinder(6, 5)\n"
        "# cad-feature: holes end\n"
    )
    with pytest.raises(ConstraintError, match="holes.*content was changed"):
        ModelConstraintValidator(store).validate(candidate)


def test_validator_edit_outside_pinned_region_succeeds(tmp_path: Path):
    store = ConstraintStore(tmp_path)
    source = (
        "# cad-feature: holes start\n"
        "holes = Cylinder(3, 5)\n"
        "# cad-feature: holes end\n"
        "\n"
        "result = Box(10, 20, 5)\n"
    )
    _add_feature_store(store, source, "holes")

    candidate = (
        "# cad-feature: holes start\n"
        "holes = Cylinder(3, 5)\n"
        "# cad-feature: holes end\n"
        "\n"
        "result = Box(15, 25, 5)\n"
    )
    ModelConstraintValidator(store).validate(candidate)  # Should not raise.


def test_validator_multiple_violations_reported(tmp_path: Path):
    store = ConstraintStore(tmp_path)
    source = (
        "WIDTH: float = 10.0\n"
        "# cad-feature: holes start\n"
        "holes = Cylinder(3, 5)\n"
        "# cad-feature: holes end\n"
    )
    _add_param_store(store, source)
    _add_feature_store(store, source, "holes")

    candidate = (
        "WIDTH: float = 99.0\n"
        "result = Box(99, 20, 5)\n"
    )
    with pytest.raises(ConstraintError) as exc:
        ModelConstraintValidator(store).validate(candidate)
    msg = str(exc.value)
    assert "WIDTH" in msg
    assert "holes" in msg


# --------------------------------------------------------------------------- #
#  Full-file write cannot bypass pins
# --------------------------------------------------------------------------- #

def test_validator_full_rewrite_cannot_bypass_pins(tmp_path: Path):
    store = ConstraintStore(tmp_path)
    source = "WIDTH: float = 10.0\nresult = Box(WIDTH, 20, 5)\n"
    _add_param_store(store, source)

    # A full file write without the pinned parameter.
    candidate = "result = Box(99.0, 20, 5)\n"
    with pytest.raises(ConstraintError, match="WIDTH"):
        ModelConstraintValidator(store).validate(candidate)


# --------------------------------------------------------------------------- #
#  Malformed marker penalty in constraint validation
# --------------------------------------------------------------------------- #

def test_validator_malformed_markers_in_candidate_fails_constraint_check(tmp_path: Path):
    store = ConstraintStore(tmp_path)
    source = (
        "# cad-feature: base start\n"
        "base = Box(10, 20, 5)\n"
        "# cad-feature: base end\n"
    )
    _add_feature_store(store, source, "base")

    # Candidate has unclosed marker.
    candidate = (
        "# cad-feature: base start\n"
        "base = Box(30, 20, 5)\n"
    )
    with pytest.raises(ConstraintError):
        ModelConstraintValidator(store).validate(candidate)


# --------------------------------------------------------------------------- #
#  Constraint summary for agent context
# --------------------------------------------------------------------------- #

def test_validator_constraint_summary(tmp_path: Path):
    store = ConstraintStore(tmp_path)
    source = (
        "WIDTH: float = 10.0\n"
        "# cad-feature: holes start\n"
        "holes = Cylinder(3, 5)\n"
        "# cad-feature: holes end\n"
    )
    _add_param_store(store, source)
    _add_feature_store(store, source, "holes")

    summary = ModelConstraintValidator(store).constraint_summary()
    assert "WIDTH" in summary
    assert "holes" in summary
    assert "do not change" in summary.lower()


def test_validator_empty_constraints_summary(tmp_path: Path):
    store = ConstraintStore(tmp_path)
    assert ModelConstraintValidator(store).constraint_summary() == ""


def test_discover_targets_returns_typed_parameters_and_features(tmp_path: Path):
    store = ConstraintStore(tmp_path)
    source = (
        "WIDTH: float = 10.0\n"
        "UNTYPED = 5\n"
        "# cad-feature: holes start\n"
        "holes = Cylinder(3, 5)\n"
        "# cad-feature: holes end\n"
    )

    targets = store.discover_targets(source)

    assert [item["name"] for item in targets["parameters"]] == ["WIDTH"]
    assert targets["parameters"][0]["value"] == "10.0"
    assert [item["name"] for item in targets["features"]] == ["holes"]
