"""Critical coverage for user-owned parameter and feature pins."""

from pathlib import Path

import pytest

from agent.constraints import (
    ConstraintError,
    ConstraintStore,
    ModelConstraintValidator,
    parse_feature_regions,
)

PARAM_SOURCE = "WIDTH: float = 10.0\nresult = Box(WIDTH, 20, 5)\n"
FEATURE_SOURCE = (
    "# cad-feature: holes start\n"
    "holes = Cylinder(3, 5)\n"
    "# cad-feature: holes end\n"
    "result = Box(10, 20, 5)\n"
)


def _pin_parameter(store: ConstraintStore) -> None:
    store.add(store.create_parameter_constraint("WIDTH", PARAM_SOURCE))


def _pin_feature(store: ConstraintStore) -> None:
    store.add(store.create_source_feature_constraint("holes", FEATURE_SOURCE))


def test_feature_parser_finds_regions_and_rejects_ambiguous_markers():
    source = (
        "# cad-feature: base start\n"
        "base = Box(20, 20, 5)\n"
        "# cad-feature: base end\n"
        "# cad-feature: holes start\n"
        "holes = Cylinder(3, 5)\n"
        "# cad-feature: holes end\n"
    )

    regions = parse_feature_regions(source)

    assert set(regions) == {"base", "holes"}
    assert regions["holes"].content == "holes = Cylinder(3, 5)"

    invalid_sources = (
        "# cad-feature: base start\nbase = Box(10, 20, 5)\n",
        "base = Box(10, 20, 5)\n# cad-feature: base end\n",
        (
            "# cad-feature: outer start\n"
            "# cad-feature: inner start\n"
            "# cad-feature: inner end\n"
            "# cad-feature: outer end\n"
        ),
        "# cad-feature: 123bad start\nBox(1, 1, 1)\n# cad-feature: 123bad end\n",
    )
    for invalid_source in invalid_sources:
        with pytest.raises(ConstraintError):
            parse_feature_regions(invalid_source)


def test_constraint_store_persists_and_removes_pins(tmp_path: Path):
    store = ConstraintStore(tmp_path)
    parameter = store.create_parameter_constraint("WIDTH", PARAM_SOURCE)
    feature = store.create_source_feature_constraint("holes", FEATURE_SOURCE)

    store.add(parameter)
    store.add(feature)

    reloaded = ConstraintStore(tmp_path)
    assert [(pin.kind, pin.name) for pin in reloaded.list()] == [
        ("parameter", "WIDTH"),
        ("source_feature", "holes"),
    ]
    assert reloaded.remove(parameter.id) == parameter
    assert [(pin.kind, pin.name) for pin in reloaded.list()] == [
        ("source_feature", "holes")
    ]

    with pytest.raises(ConstraintError, match="already exists"):
        reloaded.add(feature)


def test_malformed_constraint_store_is_preserved(tmp_path: Path):
    path = tmp_path / ".cad-agent" / "constraints.json"
    path.parent.mkdir(parents=True)
    original = "{broken json"
    path.write_text(original, encoding="utf-8")

    with pytest.raises(ConstraintError, match="Malformed"):
        ConstraintStore(tmp_path).list()

    assert path.read_text(encoding="utf-8") == original


def test_parameter_pin_allows_formatting_but_blocks_semantic_changes(tmp_path: Path):
    store = ConstraintStore(tmp_path)
    _pin_parameter(store)
    validator = ModelConstraintValidator(store)

    validator.validate("WIDTH:   float = 10.0\nresult = Box(WIDTH, 30, 5)\n")

    invalid_candidates = (
        "WIDTH: float = 20.0\nresult = Box(WIDTH, 20, 5)\n",
        "result = Box(10.0, 20, 5)\n",
        PARAM_SOURCE + "WIDTH = 20.0\n",
    )
    for candidate in invalid_candidates:
        with pytest.raises(ConstraintError, match="WIDTH"):
            validator.validate(candidate)


def test_feature_pin_allows_external_edits_but_blocks_feature_changes(tmp_path: Path):
    store = ConstraintStore(tmp_path)
    _pin_feature(store)
    validator = ModelConstraintValidator(store)

    validator.validate(
        "# cad-feature: holes start\n"
        "# Cosmetic formatting is allowed.\n"
        "holes=Cylinder(3,5)\n"
        "# cad-feature: holes end\n"
        "result = Box(30, 40, 5)\n"
    )

    invalid_candidates = (
        "result = Box(10, 20, 5)\n",
        FEATURE_SOURCE.replace("Cylinder(3, 5)", "Cylinder(6, 5)"),
        "# cad-feature: holes start\nholes = Cylinder(3, 5)\n",
    )
    for candidate in invalid_candidates:
        with pytest.raises(ConstraintError, match="holes"):
            validator.validate(candidate)


def test_validator_reports_all_broken_pins(tmp_path: Path):
    store = ConstraintStore(tmp_path)
    combined_source = PARAM_SOURCE.split("result", 1)[0] + FEATURE_SOURCE
    store.add(store.create_parameter_constraint("WIDTH", combined_source))
    store.add(store.create_source_feature_constraint("holes", combined_source))

    with pytest.raises(ConstraintError) as error:
        ModelConstraintValidator(store).validate("WIDTH: float = 99.0\nresult = 1\n")

    assert "WIDTH" in str(error.value)
    assert "holes" in str(error.value)


def test_agent_context_discovers_and_summarizes_active_pins(tmp_path: Path):
    store = ConstraintStore(tmp_path)
    source = PARAM_SOURCE.split("result", 1)[0] + FEATURE_SOURCE
    store.add(store.create_parameter_constraint("WIDTH", source))
    store.add(store.create_source_feature_constraint("holes", source))

    targets = store.discover_targets(source)
    summary = ModelConstraintValidator(store).constraint_summary()

    assert targets == {
        "parameters": [
            {"name": "WIDTH", "value": "10.0", "line": 1, "pinned": True}
        ],
        "features": [
            {"name": "holes", "start_line": 2, "end_line": 4, "pinned": True}
        ],
    }
    assert summary == (
        "Protected (do not change):\n"
        "- parameter: WIDTH\n"
        "- source_feature: holes"
    )
