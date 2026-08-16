from __future__ import annotations

import ast
import json
import os
import re
import shlex
import signal
import subprocess
import threading
from pathlib import Path

from agent.sandbox import command as sandbox_command

# Maximum bytes to buffer from subprocess stdout/stderr to prevent OOM.
_MAX_OUTPUT_BYTES = 1 * 1024 * 1024  # 1 MB per stream
_READ_ONLY_COMMANDS = frozenset({"find", "git", "grep", "head", "ls", "pwd", "rg", "sed", "stat", "tail", "wc"})
_SHELL_METACHARS = re.compile(r"[;&|<>`\n]|\$\(")
_NATIVE_EXEC_CODE = (
    "import json, os, sys\n"
    "args = json.loads(sys.argv[1])\n"
    "os.execvp(args[0], args)\n"
)


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


_ALLOWED_CHECK_IMPORTS = {"build123d", "cadquery", "math", "numpy"}
# Blocked identifiers: any AST Name or Attribute attr matching this set is
# rejected. The terminal_check AST walker also bans chained accesses through
# ``__builtins__``, ``__class__``, ``__bases__``, ``__mro__``, ``__subclasses__``,
# ``__globals__``, ``globals``, ``locals``, and ``vars`` so the validator cannot
# be tricked into evaluating ``getattr``/``__builtins__['eval']`` style escapes.
_BLOCKED_CHECK_NAMES = frozenset(
    {
        "__import__",
        "__build_class__",
        "__builtins__",
        "__class__",
        "__bases__",
        "__mro__",
        "__subclasses__",
        "__globals__",
        "__dict__",
        "breakpoint",
        "builtins",
        "compile",
        "eval",
        "exec",
        "getattr",
        "globals",
        "input",
        "locals",
        "open",
        "os",
        "pathlib",
        "setattr",
        "shutil",
        "socket",
        "subprocess",
        "sys",
        "vars",
    }
)
# Function names that may be called directly. Any other call — including calls
# whose target is an ``Attribute`` (``builtins.eval``) or a ``Subscript``
# (``__builtins__['eval']``) — is rejected.
_ALLOWED_CHECK_CALLS = frozenset(
    {
        "callable",
        "dir",
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

# Root attributes whose lookup is always banned because they unlock any builtin.
_BANNED_ATTRIBUTE_ROOTS = frozenset(
    {
        "__builtins__",
        "__class__",
        "__bases__",
        "__mro__",
        "__subclasses__",
        "__globals__",
        "__dict__",
        "builtins",
        "globals",
        "locals",
        "vars",
    }
)


def _check_root_identifier(node: ast.AST) -> str | None:
    """Return the leftmost root identifier of an Attribute/Subscript chain.

    ``__builtins__['eval']`` -> ``__builtins__``. ``(1).__class__.__mro__[1]``
    -> ``__class__``. ``a.b`` -> ``a``.
    """
    while True:
        if isinstance(node, ast.Attribute):
            node = node.value
            continue
        if isinstance(node, ast.Subscript):
            node = node.value
            continue
        if isinstance(node, ast.Name):
            return node.id
        return None


def _subscript_constant(node: ast.Subscript) -> str | None:
    """If the Subscript's slice is a string/numeric Constant, return its string form."""
    slice_node = node.slice
    if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
        return slice_node.value
    return None


class _CheckASTValidator(ast.NodeVisitor):
    """Walk the snippet AST and reject any path that could escape the sandbox.

    The validator mirrors the seccomp filter's intent: even with the bubblewrap
    namespace in place, Python builtins like ``eval``, ``open``, ``__import__``,
    and ``getattr`` are pure-Python functions the kernel does not see, so the
    only real barrier is an AST-level allowlist. We track four conditions:

    1. The final statement must be a single direct call to an allowed function
       (``print``, ``type``, ``len``, ...). Indirect targets (``getattr``,
       ``__builtins__['eval']``, ``(1).__class__``) are rejected.
    2. Any ``ast.Name`` whose id matches the blocked set is rejected.
    3. Any ``ast.Attribute`` whose attr matches the blocked set, or whose root
       identifier is a banned introspection root (``__builtins__``, ``globals``,
       ...) is rejected.
    4. Any ``ast.Subscript`` whose root identifier is banned, or whose slice is
       a string constant naming a blocked function (``__builtins__['eval']``) is
       rejected.
    """

    def __init__(self) -> None:
        self._errors: list[str] = []

    def fail(self, message: str) -> None:
        if message not in self._errors:
            self._errors.append(message)

    @property
    def errors(self) -> list[str]:
        return self._errors

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in _BLOCKED_CHECK_NAMES:
            self.fail(f"check -c does not allow the name '{node.id}'.")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in _BLOCKED_CHECK_NAMES:
            self.fail(f"check -c does not allow the name '{node.attr}'.")
        root = _check_root_identifier(node)
        if root in _BANNED_ATTRIBUTE_ROOTS:
            self.fail(
                f"check -c does not allow accessing '{root}' attributes."
            )
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        slice_value = _subscript_constant(node)
        if slice_value and slice_value in _BLOCKED_CHECK_NAMES:
            self.fail(
                f"check -c does not allow the name '{slice_value}'."
            )
        root = _check_root_identifier(node)
        if root in _BANNED_ATTRIBUTE_ROOTS:
            self.fail(
                f"check -c does not allow subscripting '{root}'."
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # Only direct ``Name`` calls are allowed at the top level. Indirect
        # call targets like ``getattr(__builtins__, 'eval')`` or
        # ``__builtins__['eval']`` are rejected because we cannot prove which
        # function they will resolve to without executing the code. Nested
        # calls used as *arguments* (e.g. ``math.sqrt(4)`` inside
        # ``print(math.sqrt(4))``) are inspected by the visitor walk but the
        # restrictive top-level check is enforced separately against the
        # final statement only.
        self.generic_visit(node)


def _validate_check_code(code: str) -> None:
    """Reject check -c snippets that could execute arbitrary code.

    Allows only imports from a strict module allowlist followed by a single
    trailing call to a read-only inspection function. The validator walks the
    full AST and bans any ``Attribute``/``Subscript`` chain that could resolve
    to a blocked builtin (``eval``, ``exec``, ``open``, ``__import__``, ...),
    including computed-name chains (``__builtins__['eval']``,
    ``getattr(__builtins__, 'open')``) and MRO introspection
    (``().__class__.__bases__``).
    """
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as error:
        raise ValueError(f"check -c code is not valid Python: {error}") from error
    statements = list(tree.body)
    if not statements:
        raise ValueError("check -c requires inline code.")

    def _import_allowed(module: str) -> bool:
        return module.split(".")[0] in _ALLOWED_CHECK_IMPORTS or module.startswith(
            "ocp_"
        )

    final_statement: ast.stmt | None = None
    for index, statement in enumerate(statements):
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            for alias in statement.names:
                module = (
                    alias.name
                    if isinstance(statement, ast.Import)
                    else statement.module
                )
                if not _import_allowed(module or ""):
                    raise ValueError(f"check -c import of '{module}' is not permitted.")
                if alias.asname in _BLOCKED_CHECK_NAMES:
                    raise ValueError(
                        f"check -c does not allow the name '{alias.asname}'."
                    )
                if (
                    isinstance(statement, ast.ImportFrom)
                    and alias.name in _BLOCKED_CHECK_NAMES
                ):
                    raise ValueError(
                        f"check -c does not allow the name '{alias.name}'."
                    )
            continue
        if index != len(statements) - 1:
            raise ValueError(
                "check -c allows only imports followed by a single inspection call."
            )
        if not (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
        ):
            raise ValueError(
                "check -c final statement must call one of: "
                + ", ".join(sorted(_ALLOWED_CHECK_CALLS))
                + "."
            )
        final_statement = statement

    validator = _CheckASTValidator()
    validator.visit(tree)
    if validator.errors:
        raise ValueError(validator.errors[0])
    # ``getattr`` was removed from the allow-list because its only safe use in
    # a sandbox-only check was reaching blocked builtins, which the AST walker
    # now bans. The second final-statement check below preserves the original
    # "direct ``Name`` call to an allowed function" contract: the in-loop check
    # only verifies the statement is an ``Expr`` wrapping a ``Call``, but the
    # ``func.id in _ALLOWED_CHECK_CALLS`` membership is enforced here.
    if final_statement is not None:
        call = final_statement.value  # type: ignore[attr-defined]
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id in _ALLOWED_CHECK_CALLS
        ):
            raise ValueError(
                "check -c final statement must call one of: "
                + ", ".join(sorted(_ALLOWED_CHECK_CALLS))
                + "."
            )


class TerminalTool:
    """Runs only the current Python interpreter within a single workspace."""

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

    def bash(self, command: str, timeout_seconds: int = 15) -> dict[str, object]:
        """Run one read-only shell-style inspection without shell expansion."""
        if not command or len(command) > 2000:
            raise ValueError("bash command must be between 1 and 2000 characters.")
        if _SHELL_METACHARS.search(command):
            raise ValueError("bash does not allow shell operators, redirects, or substitutions.")
        try:
            arguments = shlex.split(command)
        except ValueError as error:
            raise ValueError(f"Invalid shell quoting: {error}") from error
        if not arguments or Path(arguments[0]).name != arguments[0]:
            raise ValueError("bash command must use a workspace-safe command name.")
        program = arguments[0]
        if program not in _READ_ONLY_COMMANDS:
            raise ValueError(
                "bash supports only read-only commands: "
                + ", ".join(sorted(_READ_ONLY_COMMANDS))
            )
        forbidden = {"-delete", "-exec", "-execdir", "-i", "--in-place", "--upload-pack"}
        if any(argument in forbidden for argument in arguments[1:]):
            raise ValueError("bash command includes a non-read-only option.")
        if program == "git" and (len(arguments) < 2 or arguments[1] not in {"diff", "log", "rev-parse", "show", "status"}):
            raise ValueError("bash git supports: diff, log, rev-parse, show, status.")
        return self._run_command(arguments, timeout_seconds, native=True)

    def check(
        self, arguments: list[str], timeout_seconds: int = 15
    ) -> dict[str, object]:
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

    def _run_command(
        self, command: list[str], timeout_seconds: int, *, native: bool = False
    ) -> dict[str, object]:
        timeout = max(1, min(timeout_seconds, 120))
        sandbox_arguments = (
            ["-c", _NATIVE_EXEC_CODE, json.dumps(command)] if native else command
        )
        command, seccomp_fd = sandbox_command(
            self.project_dir,
            sandbox_arguments,
            writable=False,
            timeout_seconds=timeout,
            writable_tmp=not native,
        )
        process: subprocess.Popen[str] | None = None
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
        finally:
            with self._lock:
                self._process = None
            # If ``_stream_with_limit`` raised (timeout, memory-limit overflow,
            # or any other exception), ``process`` is still alive and would
            # otherwise be leaked: the outer reference on ``self._process`` has
            # already been cleared, so ``stop()`` becomes a no-op for this
            # child. Force-kill the whole process group so the bubblewrap
            # namespace and any spawned python interpreter exit cleanly.
            if process is not None:
                _terminate(process, force=True)
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
        # ``_stream_with_limit`` only runs once ``Popen`` has captured a
        # process reference, so reaching this point means ``process`` is
        # populated. ``Popen`` failures propagate through the inner ``try``
        # above and surface as ``OSError``/``FileNotFoundError`` with their
        # original detail, which is more informative than a generic fallback.
        if process.returncode:
            raise RuntimeError(
                f"Python command failed with exit code {process.returncode}.\n"
                f"stdout:\n{stdout[-8000:]}\nstderr:\n{stderr[-8000:]}"
            )
        return {
            "returncode": process.returncode,
            "stdout": stdout[-8000:],
            "stderr": stderr[-8000:],
            "actual_timeout_seconds": timeout,
        }

    def run(self, arguments: list[str], timeout_seconds: int = 30) -> dict[str, object]:
        if not arguments or arguments[0] != "python":
            raise ValueError("Only Python commands are permitted.")
        if any(token in {"-c", "-m", "-i"} for token in arguments[1:]):
            raise ValueError(
                "Inline, module, and interactive Python commands are not permitted."
            )
        if len(arguments) != 2 or Path(arguments[1]).name != arguments[1]:
            raise ValueError(
                "Run a single Python script (no subdirectories) from the active project directory."
            )
        script = (self.project_dir / arguments[1]).resolve()
        if script.parent != self.project_dir or not script.is_file():
            raise ValueError(
                "Python script must exist in the active project directory."
            )
        return self._run_command([script.name], timeout_seconds)

    def stop(self) -> None:
        with self._lock:
            if self._process and self._process.poll() is None:
                _terminate(self._process, force=True)
