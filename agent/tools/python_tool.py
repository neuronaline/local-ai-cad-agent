"""Execute restricted mathematical and engineering Python calculations inside a sandbox."""
from __future__ import annotations

import json
import logging
import tempfile
import threading
from pathlib import Path
from typing import Any

from agent.sandbox import command as sandbox_command
from agent.tools.process_runner import ProcessSlot, run_sandbox_subprocess, terminate

_LOG = logging.getLogger(__name__)

_SCRIPTS_DIR = Path(__file__).resolve().parent / "cad_scripts"
_RUNNER_FILENAME = "python_runner.py"
_SCRIPT_FILENAME = "script.py"

PYTHON_TIMEOUT_SECONDS = 5
MAX_OUTPUT_CHARS = 8000

_RUNNER_SOURCE_CACHE: str | None = None


def _get_runner_source() -> str:
    global _RUNNER_SOURCE_CACHE
    if _RUNNER_SOURCE_CACHE is None:
        _RUNNER_SOURCE_CACHE = (_SCRIPTS_DIR / _RUNNER_FILENAME).read_text(encoding="utf-8")
    return _RUNNER_SOURCE_CACHE


def _truncate_output(val: Any, max_chars: int = MAX_OUTPUT_CHARS) -> Any:
    """Ensure returned text and structured representations fit within LLM context budget."""
    if val is None or isinstance(val, (bool, int, float)):
        return val
    if isinstance(val, str):
        if len(val) > max_chars:
            return val[:max_chars].rstrip() + "\n... [truncated]"
        return val
    try:
        rendered = json.dumps(val, ensure_ascii=False)
    except Exception:
        rendered = str(val)
    if len(rendered) > max_chars:
        return rendered[:max_chars].rstrip() + "\n... [truncated]"
    return val


class PythonTool:
    """Restricted Python execution engine for math and engineering computations."""

    def __init__(
        self,
        project_dir: Path,
        publish: Any = None,
        stop_event: threading.Event | None = None,
        *,
        _active_slots: set[ProcessSlot] | None = None,
        _lock: threading.Lock | None = None,
        _call_id: str = "",
    ) -> None:
        self.project_dir = project_dir.resolve()
        self._publish = publish
        self._stop_event = stop_event
        self._call_id = _call_id
        self._active_slots: set[ProcessSlot] = (
            _active_slots if _active_slots is not None else set()
        )
        self._lock: threading.Lock = _lock if _lock is not None else threading.Lock()

    def with_call_id(self, call_id: str) -> PythonTool:
        """Return a lightweight view bound to ``call_id`` sharing active process slots."""
        return PythonTool(
            self.project_dir,
            self._publish,
            self._stop_event,
            _active_slots=self._active_slots,
            _lock=self._lock,
            _call_id=call_id or "",
        )

    def stop(self) -> None:
        """Terminate all currently executing sandbox calculations."""
        with self._lock:
            processes = [
                slot.process
                for slot in self._active_slots
                if slot.process is not None
            ]
        for proc in processes:
            terminate(proc, force=True)

    def run(self, code: str) -> dict[str, Any]:
        """Execute Python code in an isolated, read-only bubblewrap container.

        Returns a dictionary containing captured ``stdout`` and evaluated ``result``.
        Raises ``RuntimeError`` on timeout, sandbox fault, or execution failure.
        """
        if self._stop_event is not None and self._stop_event.is_set():
            raise RuntimeError("Operation cancelled before execution.")

        cleaned_code = code.strip() if isinstance(code, str) else ""
        if not cleaned_code:
            return {"stdout": "", "result": None}

        runner_source = _get_runner_source()
        slot = ProcessSlot()
        with self._lock:
            self._active_slots.add(slot)

        try:
            with tempfile.TemporaryDirectory(prefix="cad-python-") as temporary:
                workspace = Path(temporary)
                (workspace / _SCRIPT_FILENAME).write_text(cleaned_code, encoding="utf-8")
                (workspace / _RUNNER_FILENAME).write_text(runner_source, encoding="utf-8")

                if self._stop_event is not None and self._stop_event.is_set():
                    raise RuntimeError("Operation cancelled before execution.")

                cmd, seccomp_fd = sandbox_command(
                    workspace,
                    [_RUNNER_FILENAME],
                    writable=False,
                    writable_tmp=True,
                    timeout_seconds=PYTHON_TIMEOUT_SECONDS,
                )

                try:
                    stdout, stderr, returncode = run_sandbox_subprocess(
                        argv=cmd,
                        seccomp_fd=seccomp_fd,
                        timeout_seconds=PYTHON_TIMEOUT_SECONDS,
                        lock=self._lock,
                        process_slot=slot,
                        timeout_message=(
                            f"Python calculation timed out after {PYTHON_TIMEOUT_SECONDS} seconds."
                        ),
                    )
                except RuntimeError as error:
                    detail = f"Python sandbox failed: {error}"
                    _LOG.warning(detail)
                    raise RuntimeError(detail) from error

            if returncode != 0 and not stdout.strip():
                error_msg = stderr.strip() or f"Subprocess exited with code {returncode}."
                raise RuntimeError(f"Python execution failed:\n{error_msg}")

            try:
                payload = json.loads(stdout.strip())
            except (json.JSONDecodeError, ValueError) as err:
                raw_out = (stdout + "\n" + stderr).strip()
                raise RuntimeError(
                    f"Python runner emitted invalid response:\n{raw_out[:1000]}"
                ) from err

            if not isinstance(payload, dict):
                raise TypeError("Python runner produced an unexpected payload.")

            if not payload.get("ok"):
                err_text = payload.get("error", "Unknown execution error.")
                stdout_text = payload.get("stdout", "")
                full_msg = f"{err_text}\n{stdout_text}".strip()
                raise RuntimeError(full_msg)

            raw_stdout = payload.get("stdout") or ""
            raw_result = payload.get("result")

            return {
                "stdout": _truncate_output(raw_stdout),
                "result": _truncate_output(raw_result),
            }
        finally:
            with self._lock:
                self._active_slots.discard(slot)
