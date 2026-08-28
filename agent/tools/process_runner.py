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

# Sandbox subprocess timeout ceiling shared by ``cad_screenshot``,
# ``cad_review``, and the corresponding JSON schema. Centralising the value
# keeps the runtime cap, the tool class constant, and the schema's
# ``maximum`` in lock-step so they cannot drift.
MAX_SANDBOX_TIMEOUT_SECONDS = 120


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
#  Backwards-compatible aliases for tests + dynamic imports.
#
#  These were the private names exposed by earlier refactors of
#  ``cad_tool`` / ``cad_screenshot_tool`` and are kept so any external
#  caller (or future migration step) keeps working without churn.
# --------------------------------------------------------------------------- #

_terminate = terminate
_stream_with_limit = stream_with_limit
_TimedOut = TimedOut


# --------------------------------------------------------------------------- #
#  Sandbox subprocess lifecycle.
#
#  ``cad_tool`` and ``cad_screenshot_tool`` both stage a small bubblewrap
#  workspace, spawn ``bwrap`` with the seccomp FD, stream its stdout/stderr,
#  and tear the process group down on timeout, error, or non-zero return. The
#  only differences are the argv, the timeout, the tool's ``_lock``/active
#  process slot, and the error message raised on timeout. Centralising the
#  common shape keeps both call sites focused on "what" and prevents the
#  boilerplate from drifting (e.g. the cleanup finally block must force-kill
#  even when ``stream_with_limit`` raised, otherwise the inner python escapes
#  ``stop()``).
# --------------------------------------------------------------------------- #


def run_sandbox_subprocess(
    *,
    argv: list[str],
    seccomp_fd: int,
    timeout_seconds: float,
    lock: threading.Lock,
    process_slot: list[subprocess.Popen[str]],
    timeout_message: str,
) -> tuple[str, str, int]:
    """Spawn ``argv`` inside bubblewrap and stream stdout/stderr.

    ``process_slot`` must be a single-element list; the helper assigns the
    live :class:`subprocess.Popen` into ``process_slot[0]`` under ``lock`` so
    :meth:`stop` can find it, and clears the slot inside ``finally`` once the
    helper returns. Returns ``(stdout, stderr, returncode)`` on success.

    Raises :class:`RuntimeError` on subprocess timeout (with
    ``timeout_message``) or when stdout/stderr exceeds the memory limit. The
    caller decides how to map a non-zero ``returncode`` to a build failure
    message (each tool surfaces different traceback extraction).
    """
    process: subprocess.Popen[str] | None = None
    try:
        with lock:
            process = subprocess.Popen(
                argv,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                pass_fds=(seccomp_fd,),
                start_new_session=True,
            )
            process_slot[0] = process
        try:
            stdout, stderr = stream_with_limit(process, timeout=timeout_seconds)
        except TimedOut as error:
            raise RuntimeError(timeout_message) from error
        return stdout, stderr, process.returncode or 0
    finally:
        os.close(seccomp_fd)
        with lock:
            process_slot[0] = None
        # If ``stream_with_limit`` raised (timeout, memory-limit overflow, or
        # any other exception), ``process`` is still alive and would otherwise
        # be leaked: the slot has already been cleared, so ``stop()`` becomes
        # a no-op for this child. Force-kill the whole process group so the
        # bubblewrap namespace and any spawned python interpreter exit.
        if process is not None:
            terminate(process, force=True)
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
