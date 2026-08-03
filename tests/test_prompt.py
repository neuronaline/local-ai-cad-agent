from agent.prompt import BUILD123D_PLAYBOOK, SYSTEM_PROMPT


def test_build123d_playbook_is_injected_once_into_the_static_system_prompt():
    assert BUILD123D_PLAYBOOK.startswith("# build123d CAD CLI Playbook")
    assert "```markdown" not in BUILD123D_PLAYBOOK
    assert SYSTEM_PROMPT.count("<build123d_cli_playbook>") == 1
    assert BUILD123D_PLAYBOOK in SYSTEM_PROMPT
    assert "cad_build_and_verify performs the build" in SYSTEM_PROMPT
    assert "cad.run" not in SYSTEM_PROMPT


def test_build123d_playbook_has_versioned_curve_and_topology_guidance():
    assert "`build123d` **0.11.1**" in BUILD123D_PLAYBOOK
    assert "`Ellipse` creates a **complete filled 2D sketch**" in BUILD123D_PLAYBOOK
    assert "EllipticalCenterArc" in BUILD123D_PLAYBOOK
    assert "radius >= endpoint_distance / 2" in BUILD123D_PLAYBOOK
    assert "There is no top-level `max_fillet` function" in BUILD123D_PLAYBOOK
    assert "discard any cached edge/face indices" in SYSTEM_PROMPT
