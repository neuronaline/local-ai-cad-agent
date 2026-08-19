"""Internal subprocess streaming helper used by
``cad_tool`` and ``cad_screenshot_tool``.

The bubblewrap subprocesses that run CAD jobs stream their stdout/stderr
through a bounded ring buffer so a runaway python process cannot exhaust
the host's memory. ``terminate`` kills a process group so the bubblewrap
namespace and any spawned python interpreter exit cleanly.

Public surface (kept stable for callers):

- :func:`stream_with_limit`
- :class:`TimedOut`
- :func:`terminate`
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading

MAX_OUTPUT_BYTES = 1 * 1024 * 1024  # 1 MB per stream


class TimedOut(RuntimeError):
    """Subprocess timed out; carries partial output captured before the kill."""

    def __init__(self, stdout: str, stderr: str) -> None:
        super().__init__(f"Command timed out.\n{stdout}{stderr}")
        self.stdout = stdout
        self.stderr = stderr


class _RingBuffer:
    """Thread-safe circular buffer capped at ``max_bytes`` bytes."""

    def __init__(self, max_bytes: int) -> None:
        self._max = max_bytes
        self._buf: list[bytes] = []
        self._size = 0
        self._lock = threading.Lock()

    def append(self, chunk: str) -> bool:
        """Append a string chunk. Returns False if the buffer overflowed."""
        data = chunk.encode("utf-8", errors="replace")
        with self._lock:
            if self._size + len(data) > self._max:
                return False
            self._buf.append(data)
            self._size += len(data)
            return True

    def value(self) -> str:
        with self._lock:
            return b"".join(self._buf).decode("utf-8", errors="replace")


def terminate(process: subprocess.Popen[str], *, force: bool) -> None:
    """Kill a subprocess process group."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
    except ProcessLookupError:
        pass


def stream_with_limit(
    process: subprocess.Popen[str],
    timeout: float,
) -> tuple[str, str]:
    """Read stdout/stderr with a per-stream ring buffer; kill on overflow."""
    stdout_buf = _RingBuffer(MAX_OUTPUT_BYTES)
    stderr_buf = _RingBuffer(MAX_OUTPUT_BYTES)
    killed = threading.Event()

    def _reader(pipe, buf):
        try:
            for chunk in iter(lambda: pipe.read(65536), ""):
                if killed.is_set():
                    break
                if not buf.append(chunk):
                    killed.set()
                    terminate(process, force=True)
                    break
        except (OSError, ValueError):
            pass

    stdout_thread = threading.Thread(
        target=_reader, args=(process.stdout, stdout_buf), daemon=True
    )
    stderr_thread = threading.Thread(
        target=_reader, args=(process.stderr, stderr_buf), daemon=True
    )
    stdout_thread.start()
    stderr_thread.start()

    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        killed.set()
        terminate(process, force=True)
        stdout_thread.join(timeout=1)
        stderr_thread.join(timeout=1)
        raise TimedOut(stdout_buf.value(), stderr_buf.value())
    finally:
        stdout_thread.join(timeout=1)
        stderr_thread.join(timeout=1)

    if killed.is_set():
        raise RuntimeError("Subprocess output exceeded the memory limit; terminated.")

    return stdout_buf.value(), stderr_buf.value()


# --------------------------------------------------------------------------- #
#  Backwards-compatible aliases for tests + dynamic imports. These were the
#  private names exposed by the legacy ``terminal_tool`` module; keep them
#  around so existing imports keep working until they are migrated.
# --------------------------------------------------------------------------- #

_terminate = terminate
_stream_with_limit = stream_with_limit
_TimedOut = TimedOut
