from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import uuid
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import Any, ClassVar

from PIL import Image, UnidentifiedImageError

from agent.revisions import RevisionIntegrityError, RevisionStore
from agent.sandbox import command as sandbox_command
from agent.tools.file_tool import FileTool

_SCRIPTS_DIR = Path(__file__).resolve().parent / "cad_scripts"


def _read_script(name: str) -> str:
    return (_SCRIPTS_DIR / name).read_text(encoding="utf-8")


RUNNER = _read_script("runner.py")
RENDERER = _read_script("renderer.py")


class CadTool:
    __tool_schema__: ClassVar[dict[str, Any]] = {
        "type": "function",
        "function": {
            "name": "cad",
            "description": "Run, inspect, or render the CAD model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["run", "inspect", "render"],
                    },
                },
                "required": ["operation"],
            },
        },
    }

    def __init__(
        self,
        project_dir: Path,
        publish: Callable[[str, dict], None] | None = None,
        revisions: RevisionStore | None = None,
    ) -> None:
        self.project_dir = project_dir.resolve()
        self._publish = publish
        self._revisions = revisions or RevisionStore(project_dir)
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self._screenshot_lock = threading.Lock()
        self._screenshot_event = threading.Event()
        self._screenshot_data: dict[str, str] = {}
        self._pending_screenshot_id: str | None = None

    def _execute(
        self, export_dir: str | None = None, render: bool = False
    ) -> dict[str, Any]:
        model_path = self.project_dir / "model.py"
        if not model_path.exists():
            raise ValueError("model.py does not exist yet.")
        model_code = model_path.read_text(encoding="utf-8")
        FileTool.validate_model(model_code)
        code = RUNNER
        if render:
            code += RENDERER
        target: Path | None = None
        if export_dir is not None:
            target = (self.project_dir / export_dir).resolve()
            if target == self.project_dir or not target.is_relative_to(
                self.project_dir
            ):
                raise ValueError("Export directory must be inside the active project.")
            target.mkdir(parents=True, exist_ok=True)
            code += (
                "\nPath('output').mkdir(exist_ok=True)\n"
                "export_step(shape, 'output/model.step')\n"
                "export_stl(shape, 'output/model.stl')\n"
            )

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
                finally:
                    os.close(seccomp_fd)
                try:
                    stdout, stderr = process.communicate(timeout=120)
                finally:
                    with self._lock:
                        self._process = None
            except subprocess.TimeoutExpired as error:
                self._terminate(process, force=True)
                process.communicate()
                self._record_build_failure("CAD operation timed out after 120 seconds.")
                raise RuntimeError(
                    "CAD operation timed out after 120 seconds."
                ) from error
            finally:
                with self._lock:
                    self._process = None
            if process.returncode:
                detail = self._failure_detail(stderr or stdout)
                self._record_build_failure(detail)
                raise RuntimeError(f"CAD execution failed:\n{detail}")

            metrics_path = workspace / ".cad_metrics.json"
            preview_path = workspace / "preview.stl"
            if not metrics_path.is_file() or not preview_path.is_file():
                error_msg = (
                    "CAD execution did not produce preview and geometry metrics."
                )
                self._record_build_failure(error_msg)
                raise RuntimeError(error_msg)
            cached = json.loads(metrics_path.read_text(encoding="utf-8"))
            if not isinstance(cached, dict) or not isinstance(
                cached.get("metrics"), dict
            ):
                error_msg = "CAD geometry metrics are malformed."
                self._record_build_failure(error_msg)
                raise TypeError(error_msg)
            self._atomic_copy(preview_path, self.project_dir / "preview.stl")
            if render:
                render_path = workspace / "render.png"
                if not render_path.is_file() or render_path.stat().st_size == 0:
                    error_msg = "CAD execution did not produce a render."
                    self._record_build_failure(error_msg)
                    raise RuntimeError(error_msg)
                self._atomic_copy(render_path, self.project_dir / "render.png")
            if target is not None:
                for name in ("model.step", "model.stl"):
                    artifact = workspace / "output" / name
                    if not artifact.is_file() or artifact.stat().st_size == 0:
                        error_msg = f"CAD execution did not produce {name}."
                        self._record_build_failure(error_msg)
                        raise RuntimeError(error_msg)
                    self._atomic_copy(artifact, target / name)
            self._atomic_copy(metrics_path, self.project_dir / ".cad_metrics.json")
            self._record_build_success(cached["metrics"])
            return cached["metrics"]

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
    def _terminate(process: subprocess.Popen[str], *, force: bool) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
        except ProcessLookupError:
            pass

    @staticmethod
    def _failure_detail(output: str) -> str:
        """Extract the most relevant traceback frames and the final error message."""
        lines = [line.rstrip() for line in output.strip().splitlines()]
        # Collect File/line frames, filtering out runner.py internal frames.
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
        return self._execute()

    def build_and_verify(self) -> dict[str, Any]:
        """Build once, validate metrics, and produce both preview and render."""
        metrics = self._execute(render=True)
        return {
            "metrics": metrics,
            "preview": "preview.stl",
            "render": "render.png",
        }

    def execute(self, args: dict) -> tuple[str, bool]:
        """Dispatch CAD operations. Returns (result_json, waiting=False)."""
        operation = args["operation"]
        if operation not in {"run", "inspect", "render"}:
            raise ValueError("Unsupported CAD operation; use run, inspect, or render.")
        result = getattr(self, operation)()
        return json.dumps(result) if not isinstance(result, str) else result, False

    def inspect(self) -> dict[str, Any]:
        model_path = self.project_dir / "model.py"
        metrics_path = self.project_dir / ".cad_metrics.json"
        info: dict[str, Any] = {}
        if model_path.is_file():
            info["model_lines"] = len(
                model_path.read_text(encoding="utf-8").splitlines()
            )
            info["render_available"] = (self.project_dir / "render.png").is_file()
            info["preview_available"] = (self.project_dir / "preview.stl").is_file()
        if model_path.is_file() and metrics_path.is_file():
            try:
                cached = json.loads(metrics_path.read_text(encoding="utf-8"))
                digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
                if cached.get("model_sha256") == digest and isinstance(
                    cached.get("metrics"), dict
                ):
                    info |= cached["metrics"]
                    return info
            except (OSError, json.JSONDecodeError, AttributeError):
                pass
        return info if info else {"error": "model.py not found"}

    def finalize(self, export_dir: str = "output") -> dict[str, Any]:
        """Run the model, render a preview image, and export STEP+STL in one subprocess call."""
        model_path = self.project_dir / "model.py"
        if not model_path.exists():
            raise ValueError("model.py does not exist yet.")
        target = (self.project_dir / export_dir).resolve()
        if target == self.project_dir or not target.is_relative_to(self.project_dir):
            raise ValueError("Export directory must be inside the active project.")
        target.mkdir(parents=True, exist_ok=True)
        try:
            metrics = self._execute(export_dir=export_dir, render=True)
        except RuntimeError as error:
            raise RuntimeError(
                str(error).replace("CAD execution", "CAD finalization")
            ) from error
        export_step_path = self.project_dir / export_dir / "model.step"
        export_stl_path = self.project_dir / export_dir / "model.stl"
        if not export_step_path.is_file() or not export_stl_path.is_file():
            raise RuntimeError("CAD finalization did not produce export artifacts.")
        return {
            "metrics": metrics,
            "step": f"{export_dir}/model.step",
            "stl": f"{export_dir}/model.stl",
        }

    def render(self) -> dict[str, str]:
        self._execute(render=True)
        return {"render": "render.png"}

    def export(self, output_dir: str = "output") -> dict[str, Any]:
        metrics = self._execute(export_dir=output_dir)
        return {
            "metrics": metrics,
            "step": f"{output_dir}/model.step",
            "stl": f"{output_dir}/model.stl",
        }

    def screenshot(self, view: str, proximity: float = 1.0) -> dict[str, str]:
        if not self._publish:
            raise RuntimeError("Screenshot capture is not available in this context.")
        if view not in {
            "front",
            "back",
            "top",
            "bottom",
            "left",
            "right",
            "isometric",
            "current",
        }:
            raise ValueError("Unsupported screenshot view.")
        proximity = float(proximity)
        if not 0.1 <= proximity <= 5:
            raise ValueError("Screenshot proximity must be between 0.1 and 5.0.")
        request_id = uuid.uuid4().hex
        with self._screenshot_lock:
            if self._pending_screenshot_id:
                raise RuntimeError("Another screenshot capture is already pending.")
            self._screenshot_event.clear()
            self._screenshot_data.clear()
            self._pending_screenshot_id = request_id
        self._publish(
            "screenshot_request",
            {
                "project": self.project_dir.name,
                "request_id": request_id,
                "view": view,
                "proximity": proximity,
            },
        )
        if not self._screenshot_event.wait(timeout=30):
            with self._screenshot_lock:
                self._pending_screenshot_id = None
            raise RuntimeError("Screenshot capture timed out after 30 seconds.")
        with self._screenshot_lock:
            image_b64 = self._screenshot_data.get("image")
            capture_error = self._screenshot_data.get("error")
            self._screenshot_data.clear()
        if capture_error:
            raise RuntimeError(f"Screenshot capture failed: {capture_error}")
        if not image_b64:
            raise RuntimeError("Screenshot capture returned no image data.")
        path = self.project_dir / "screenshot.png"
        try:
            raw = base64.b64decode(image_b64, validate=True)
            if len(raw) > 5 * 1024 * 1024:
                raise ValueError("Screenshot exceeds 5 MB.")
            with Image.open(BytesIO(raw)) as image:
                if image.format != "PNG" or image.width > 4096 or image.height > 4096:
                    raise ValueError(
                        "Screenshot must be a PNG no larger than 4096 × 4096."
                    )
                image.verify()
            path.write_bytes(raw)
        except (ValueError, UnidentifiedImageError, OSError) as error:
            raise RuntimeError(f"Failed to save screenshot: {error}") from error
        return {"screenshot": "screenshot.png", "view": view, "proximity": proximity}

    def receive_screenshot(
        self, request_id: str, image_base64: str = "", error: str = ""
    ) -> bool:
        with self._screenshot_lock:
            if request_id != self._pending_screenshot_id:
                return False
            self._pending_screenshot_id = None
            self._screenshot_data = {"image": image_base64, "error": error}
            self._screenshot_event.set()
            return True

    def stop(self) -> None:
        with self._screenshot_lock:
            if self._pending_screenshot_id:
                self._pending_screenshot_id = None
                self._screenshot_data = {"error": "Task stopped."}
                self._screenshot_event.set()
        with self._lock:
            if self._process and self._process.poll() is None:
                self._terminate(self._process, force=True)
