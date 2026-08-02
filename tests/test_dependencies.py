from pathlib import Path


def test_build123d_version_matches_the_versioned_playbook():
    requirements = Path("requirements.txt").read_text(encoding="utf-8").splitlines()

    assert "build123d==0.11.1" in requirements
