"""Build and render the project's CAD model inside the bubblewrap sandbox."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

from agent.revisions import RevisionIntegrityError, RevisionStore
from agent.sandbox import command as sandbox_command
from agent.tools.file_tool import FileTool
from agent.tools.terminal_tool import _stream_with_limit, _TimedOut

_SCRIPTS_DIR = Path(__file__).resolve().parent / "cad_scripts"


def _read_script(name: str) -> str:
    return (_SCRIPTS_DIR / name).read_text(encoding="utf-8")


RUNNER = _read_script("runner.py")
RENDERER = _read_script("renderer.py")


# Hard validation bounds shared with the runner; enforced here so callers see
# a useful error before the sandbox subprocess spins up.
_MIN_DIMENSION_MM = 0.001
_MAX_DIMENSION_MM = 1_000_000.0
_MIN_VOLUME_MM3 = 0.0


def _kill_process_group(process: subprocess.Popen[str], *, force: bool) -> None:
    """Send a signal to the sandbox process group so descendant pythons exit.

    The bubblewrap subprocess is launched with ``start_new_session=True`` and
    in turn spawns a python interpreter inside its namespace. ``Popen.terminate``
    only signals the direct child (bwrap); without killing the process group
    the inner python keeps running and holding memory/file handles.
    """
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
    except ProcessLookupError:
        pass


class CadTool:
    def __init__(
        self,
        project_dir: Path,
        publish: Any = None,
        revisions: RevisionStore | None = None,
    ) -> None:
        self.project_dir = project_dir.resolve()
        self._publish = publish
        self._revisions = revisions or RevisionStore(project_dir)
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()

    def _execute(self, render: bool = False) -> dict[str, Any]:
        model_path = self.project_dir / "model.py"
        if not model_path.exists():
            raise ValueError("model.py does not exist yet.")
        model_code = model_path.read_text(encoding="utf-8")
        FileTool.validate_model(model_code)
        code = RUNNER + (RENDERER if render else "")

        with tempfile.TemporaryDirectory(prefix="cad-agent-") as temporary:
            workspace = Path(temporary)
            (workspace / "model.py").write_text(model_code, encoding="utf-8")
            (workspace / "runner.py").write_text(code, encoding="utf-8")
            command, seccomp_fd = sandbox_command(
                workspace,
                ["runner.py"],
                writable=True,
                timeout_seconds=120,
            )
            try:
                with self._lock:
                    self._process = subprocess.Popen(
                        command,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        pass_fds=(seccomp_fd,),
                        start_new_session=True,
                    )
                    process = self._process
                try:
                    stdout, stderr = _stream_with_limit(process, timeout=120)
                except _TimedOut as error:
                    self._record_build_failure("CAD operation timed out after 120 seconds.")
                    raise RuntimeError(
                        "CAD operation timed out after 120 seconds."
                    ) from error
                except RuntimeError as error:
                    detail = f"CAD subprocess failed: {error}"
                    self._record_build_failure(detail)
                    raise RuntimeError(detail) from error
                if process.returncode:
                    detail = self._failure_detail(stderr or stdout)
                    self._record_build_failure(detail)
                    raise RuntimeError(f"CAD execution failed:\n{detail}")
            finally:
                os.close(seccomp_fd)
                with self._lock:
                    self._process = None

            metrics_path = workspace / ".cad_metrics.json"
            preview_path = workspace / "preview.stl"
            if not metrics_path.is_file() or not preview_path.is_file():
                error_msg = "CAD execution did not produce preview and geometry metrics."
                self._record_build_failure(error_msg)
                raise RuntimeError(error_msg)
            cached = json.loads(metrics_path.read_text(encoding="utf-8"))
            if not isinstance(cached, dict) or not isinstance(
                cached.get("metrics"), dict
            ):
                error_msg = "CAD geometry metrics are malformed."
                self._record_build_failure(error_msg)
                raise TypeError(error_msg)
            metrics = cached["metrics"]
            self._enforce_basic_geometry(metrics)
            self._atomic_copy(preview_path, self.project_dir / "preview.stl")
            if render:
                render_path = workspace / "render.png"
                if not render_path.is_file() or render_path.stat().st_size == 0:
                    error_msg = "CAD execution did not produce a render."
                    self._record_build_failure(error_msg)
                    raise RuntimeError(error_msg)
                self._atomic_copy(render_path, self.project_dir / "render.png")
            self._atomic_copy(metrics_path, self.project_dir / ".cad_metrics.json")
            self._record_build_success(metrics)
            return {"metrics": metrics}

    def _enforce_basic_geometry(self, metrics: dict[str, Any]) -> None:
        """Apply the basic geometry checks the runner pre-validates.

        Keeps the simple "valid geometry, at least one solid, positive volume,
        plausible bounding box" contract even when the quality verifiers are
        no longer in the loop.
        """
        try:
            solid_count = int(metrics.get("solid_count", 0) or 0)
            is_valid = bool(metrics.get("is_valid"))
            dimensions = metrics.get("dimensions_mm") or {}
            volume = float(metrics.get("volume_mm3", 0.0) or 0.0)
        except (TypeError, ValueError):
            raise ValueError("CAD geometry metrics are malformed.") from None
        if solid_count < 1:
            raise ValueError("Build did not produce a solid.")
        if not is_valid:
            raise ValueError("CAD geometry is not valid.")
        try:
            dim_values = [float(dimensions[axis]) for axis in ("x", "y", "z")]
        except (KeyError, TypeError, ValueError):
            raise ValueError("CAD dimensions are missing or malformed.") from None
        for axis, value in zip(("x", "y", "z"), dim_values):
            if not (value > 0):
                raise ValueError(f"CAD dimension {axis} must be positive.")
            if value < _MIN_DIMENSION_MM or value > _MAX_DIMENSION_MM:
                raise ValueError(
                    f"CAD dimension {axis}={value} mm is outside the plausible range."
                )
        if not (volume > _MIN_VOLUME_MM3):
            raise ValueError("CAD volume must be positive.")

    def _record_build_success(self, metrics: dict[str, Any]) -> None:
        """Record a successful build against the active revision."""
        try:
            head = self._revisions.head()
            if head is None:
                return
            self._revisions.record_build_success(
                head.id, metrics, self.project_dir / "preview.stl"
            )
        except (OSError, RevisionIntegrityError):
            pass  # Best-effort; don't block CAD results on revision issues.

    def _record_build_failure(self, error: str) -> None:
        """Record a failed build against the active revision."""
        try:
            head = self._revisions.head()
            if head is None:
                return
            self._revisions.record_build_failure(head.id, error)
        except (OSError, RevisionIntegrityError):
            pass  # Best-effort; don't block CAD errors on revision issues.

    @staticmethod
    def _atomic_copy(source: Path, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
        try:
            shutil.copyfile(source, temporary_path)
            temporary_path.replace(target)
        finally:
            temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _failure_detail(output: str) -> str:
        """Extract the most relevant traceback frames and the final error message."""
        lines = [line.rstrip() for line in output.strip().splitlines()]
        frames: list[str] = []
        in_traceback = False
        for i, line in enumerate(lines):
            if line.startswith("Traceback "):
                in_traceback = True
                continue
            if in_traceback:
                if line.startswith("  File "):
                    if "runner.py" not in line and ".cad_runner" not in line:
                        frames.append(line)
                        if i + 1 < len(lines) and lines[i + 1].startswith("    "):
                            frames.append(lines[i + 1])
                elif not line.startswith("  ") and line.strip():
                    frames.append(line)
                    break
        if not frames:
            for i, line in enumerate(lines):
                if 'File "model.py"' in line:
                    frames.append(line)
                    if i + 1 < len(lines):
                        frames.append(lines[i + 1])
            final = next(
                (l for l in reversed(lines) if l.strip()), "Unknown CAD error."
            )
            frames.append(final)
        return "\n".join(dict.fromkeys(frames))[-2000:]

    def run(self) -> dict[str, Any]:
        """Build once and return the geometry metrics."""
        payload = self._execute()
        return dict(payload.get("metrics") or {})

    def build_and_verify(self) -> dict[str, Any]:
        """Build once, validate metrics, and produce both preview and render."""
        payload = self._execute(render=True)
        metrics = payload.get("metrics") if isinstance(payload, dict) else None
        return {
            "metrics": metrics,
            "preview": "preview.stl",
            "render": "render.png",
        }

    def stop(self) -> None:
        with self._lock:
            process = self._process
        if process is None:
            return
        _kill_process_group(process, force=False)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            _kill_process_group(process, force=True)
