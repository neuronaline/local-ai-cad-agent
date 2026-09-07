from pathlib import Path

import pytest

from agent.revisions import RevisionStore
from agent.tools.file_tool import FileTool


def test_model_write_creates_a_revision_and_rejects_unsafe_code(tmp_path: Path) -> None:
    tool = FileTool(tmp_path)

    tool.write_file("model.py", "from build123d import Box\nresult = Box(10, 20, 30)\n")

    assert RevisionStore(tmp_path).head() is not None
    with pytest.raises(ValueError, match="Unsafe import blocked"):
        tool.write_file("model.py", "import subprocess\n")
    assert "result = Box" in (tmp_path / "model.py").read_text(encoding="utf-8")
