from __future__ import annotations

import ast
import json
import os
import signal
import subprocess
import threading
from pathlib import Path
from typing import ClassVar

from agent.sandbox import command as sandbox_command

# Maximum bytes to buffer from subprocess stdout/stderr to prevent OOM.
_MAX_OUTPUT_BYTES = 1 * 1024 * 1024  # 1 MB per stream


class _TimedOut(RuntimeError):
    """Subprocess timed out; carries partial output captured before the kill."""

    def __init__(self, stdout: str, stderr: str) -> None:
        super().__init__(f"Command timed out.\n{stdout}{stderr}")
        self.stdout = stdout
        self.stderr = stderr


def _terminate(process: subprocess.Popen[str], *, force: bool) -> None:
    """Kill a subprocess process group."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
    except ProcessLookupError:
        pass


def _stream_with_limit(
    process: subprocess.Popen[str],
    timeout: float,
) -> tuple[str, str]:
    """Read stdout/stderr with a per-stream ring buffer; kill on overflow."""
    stdout_buf = _RingBuffer(_MAX_OUTPUT_BYTES)
    stderr_buf = _RingBuffer(_MAX_OUTPUT_BYTES)
    killed = threading.Event()

    def _reader(pipe, buf):
        try:
            for chunk in iter(lambda: pipe.read(65536), ""):
                if killed.is_set():
                    break
                if not buf.append(chunk):
                    killed.set()
                    _terminate(process, force=True)
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
        _terminate(process, force=True)
        stdout_thread.join(timeout=1)
        stderr_thread.join(timeout=1)
        raise _TimedOut(stdout_buf.value(), stderr_buf.value())
    finally:
        stdout_thread.join(timeout=1)
        stderr_thread.join(timeout=1)

    if killed.is_set():
        raise RuntimeError("Subprocess output exceeded the memory limit; terminated.")

    return stdout_buf.value(), stderr_buf.value()


def _drain_remaining(process: subprocess.Popen[str]) -> tuple[str, str]:
    """Best-effort drain after process has been terminated."""
    try:
        out, err = process.communicate(timeout=2)
        return (out or ""), (err or "")
    except (subprocess.TimeoutExpired, OSError, ValueError):
        return "", ""


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

_ALLOWED_CHECK_IMPORTS = {"build123d", "cadquery", "math", "numpy", "ocp_vscode"}
_BLOCKED_CHECK_NAMES = frozenset(
    {
        "__import__",
        "breakpoint",
        "compile",
        "eval",
        "exec",
        "input",
        "open",
        "os",
        "pathlib",
        "shutil",
        "socket",
        "subprocess",
        "sys",
    }
)
_ALLOWED_CHECK_CALLS = frozenset(
    {
        "callable",
        "dir",
        "getattr",
        "hasattr",
        "help",
        "isinstance",
        "issubclass",
        "len",
        "print",
        "repr",
        "type",
    }
)


def _validate_check_code(code: str) -> None:
    """Reject check -c snippets that could execute arbitrary code.

    Allows only imports from a strict module allowlist followed by a single
    trailing call to a read-only inspection function.
    """
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as error:
        raise ValueError(f"check -c code is not valid Python: {error}") from error
    statements = list(tree.body)
    if not statements:
        raise ValueError("check -c requires inline code.")

    def _import_allowed(module: str) -> bool:
        return module.split(".")[0] in _ALLOWED_CHECK_IMPORTS or module.startswith("ocp_")

    for index, statement in enumerate(statements):
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            for alias in statement.names:
                module = alias.name if isinstance(statement, ast.Import) else statement.module
                if not _import_allowed(module or ""):
                    raise ValueError(f"check -c import of '{module}' is not permitted.")
                if alias.asname in _BLOCKED_CHECK_NAMES:
                    raise ValueError(f"check -c does not allow the name '{alias.asname}'.")
                if (
                    isinstance(statement, ast.ImportFrom)
                    and alias.name in _BLOCKED_CHECK_NAMES
                ):
                    raise ValueError(f"check -c does not allow the name '{alias.name}'.")
            continue
        if index != len(statements) - 1:
            raise ValueError(
                "check -c allows only imports followed by a single inspection call."
            )
        if not (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Name)
            and statement.value.func.id in _ALLOWED_CHECK_CALLS
        ):
            raise ValueError(
                "check -c final statement must call one of: "
                + ", ".join(sorted(_ALLOWED_CHECK_CALLS))
                + "."
            )
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in _BLOCKED_CHECK_NAMES:
            raise ValueError(f"check -c does not allow the name '{node.id}'.")


class TerminalTool:
    """Runs only the current Python interpreter within a single workspace."""

    __tool_schema__: ClassVar[dict[str, object]] = {
        "type": "function",
        "function": {
            "name": "terminal",
            "description": "Run a project-local Python script or a quick validation check.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": ["run", "check"]},
                    "arguments": {"type": "array", "items": {"type": "string"}},
                    "timeout_seconds": {"type": "integer"},
                },
                "required": ["arguments"],
            },
        },
    }

    def __init__(self, project_dir: Path) -> None:
        self.project_dir = project_dir.resolve()
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()

    def execute(self, args: dict) -> tuple[str, bool]:
        """Dispatch terminal run or check. Returns (result_json, waiting=False)."""
        operation = args.get("operation", "run")
        if operation == "check":
            result = self.check(args["arguments"], args.get("timeout_seconds", 15))
        else:
            result = self.run(args["arguments"], args.get("timeout_seconds", 30))
        return json.dumps(result), False

    def check(self, arguments: list[str], timeout_seconds: int = 15) -> dict[str, object]:
        """Run a quick Python validation: -c '<safe_expr>' or -m pytest / pip list.

        Restricted to read-only, project-scoped operations. No file modification,
        shell commands, or network access beyond pip list.
        """
        if not arguments or arguments[0] != "python":
            raise ValueError("Only Python commands are permitted.")
        if len(arguments) < 2:
            raise ValueError("check requires -c '<code>' or -m <module>.")
        flag = arguments[1]
        if flag == "-c":
            code = " ".join(arguments[2:])
            if not code:
                raise ValueError("check -c requires inline code.")
            # Allow only allowlisted imports plus a single inspection call.
            _validate_check_code(code)
            return self._run_command(["-c", code], timeout_seconds)
        if flag == "-m":
            module = arguments[2] if len(arguments) > 2 else ""
            # Allow test modules and pip list/inspect.
            allowed_modules = {"pytest", "pip"}
            if module not in allowed_modules:
                raise ValueError(
                    f"check -m supports: {', '.join(sorted(allowed_modules))}. "
                    "Use terminal.run for custom scripts."
                )
            # pip list / pip show are read-only; block install/uninstall.
            sub_args = arguments[3:] if len(arguments) > 3 else []
            if module == "pip":
                pip_verb = sub_args[0] if sub_args else "list"
                if pip_verb not in {"list", "show", "freeze"}:
                    raise ValueError("check -m pip supports: list, show, freeze.")
            return self._run_command(["-m", module, *sub_args], timeout_seconds)
        raise ValueError("check supports only -c '<code>' or -m <module>.")

    def _run_command(self, command: list[str], timeout_seconds: int) -> dict[str, object]:
        timeout = max(1, min(timeout_seconds, 120))
        command, seccomp_fd = sandbox_command(
            self.project_dir,
            command,
            writable=False,
            timeout_seconds=timeout,
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
            stdout, stderr = _stream_with_limit(process, timeout=timeout)
        except _TimedOut:
            raise  # The exception carries the partial output in its message.
        finally:
            with self._lock:
                self._process = None
        if process.returncode:
            raise RuntimeError(
                f"Python command failed with exit code {process.returncode}.\n"
                f"stdout:\n{stdout[-8000:]}\nstderr:\n{stderr[-8000:]}"
            )
        return {"returncode": process.returncode, "stdout": stdout[-8000:], "stderr": stderr[-8000:], "actual_timeout_seconds": timeout}

    def run(self, arguments: list[str], timeout_seconds: int = 30) -> dict[str, object]:
        if not arguments or arguments[0] != "python":
            raise ValueError("Only Python commands are permitted.")
        if any(token in {"-c", "-m", "-i"} for token in arguments[1:]):
            raise ValueError("Inline, module, and interactive Python commands are not permitted.")
        if len(arguments) != 2 or Path(arguments[1]).name != arguments[1]:
            raise ValueError("Run a single Python script (no subdirectories) from the active project directory.")
        script = (self.project_dir / arguments[1]).resolve()
        if script.parent != self.project_dir or not script.is_file():
            raise ValueError("Python script must exist in the active project directory.")
        return self._run_command([script.name], timeout_seconds)

    def stop(self) -> None:
        with self._lock:
            if self._process and self._process.poll() is None:
                _terminate(self._process, force=True)
