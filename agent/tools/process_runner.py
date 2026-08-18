"""Internal subprocess streaming + sandbox-escape AST validator used by
``cad_tool`` and ``cad_screenshot_tool``.

The previous ``terminal_tool`` module also exposed a user-facing ``TerminalTool``
that ran arbitrary Python scripts in the bubblewrap sandbox. That tool has been
removed: the agent no longer has a direct terminal surface, and the only
remaining legitimate use of this module is the bounded streaming helper that
sandbox-isolated CAD subprocesses rely on, plus the AST validator whose
regression coverage must keep running.

Public surface (kept stable for callers and tests):

- :func:`stream_with_limit`
- :class:`TimedOut`
- :func:`drain_remaining`
- :func:`terminate`
- :func:`validate_check_code`

Names that begin with ``_`` are intentionally module-private; the public
aliases above exist solely so ``cad_tool``/``cad_screenshot_tool`` and the
test suite can import them without the leading underscore.
"""

from __future__ import annotations

import ast
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


def drain_remaining(process: subprocess.Popen[str]) -> tuple[str, str]:
    """Best-effort drain after process has been terminated."""
    try:
        out, err = process.communicate(timeout=2)
        return (out or ""), (err or "")
    except (subprocess.TimeoutExpired, OSError, ValueError):
        return "", ""


# --------------------------------------------------------------------------- #
#  AST validator used by the (removed) ``terminal_check`` surface.
# --------------------------------------------------------------------------- #
#
# Kept here because the historical regression tests in ``tests/test_tools.py``
# guard against re-introducing sandbox-escape vectors (``eval``, ``exec``,
# ``__builtins__['eval']``, ``().__class__.__bases__``, ...). The helper still
# compiles and is pure AST, with no environment or filesystem touchpoints, so
# it is safe to keep around for the test suite even though no caller invokes
# it at runtime anymore.

_ALLOWED_CHECK_IMPORTS = {"build123d", "cadquery", "math", "numpy"}
# Blocked identifiers: any AST Name or Attribute attr matching this set is
# rejected. The validator also bans chained accesses through ``__builtins__``,
# ``__class__``, ``__bases__``, ``__mro__``, ``__subclasses__``, ``__globals__``,
# ``globals``, ``locals``, and ``vars`` so the validator cannot be tricked into
# evaluating ``getattr``/``__builtins__['eval']`` style escapes.
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
            self.fail(f"check -c does not allow accessing '{root}' attributes.")
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        slice_value = _subscript_constant(node)
        if slice_value and slice_value in _BLOCKED_CHECK_NAMES:
            self.fail(f"check -c does not allow the name '{slice_value}'.")
        root = _check_root_identifier(node)
        if root in _BANNED_ATTRIBUTE_ROOTS:
            self.fail(f"check -c does not allow subscripting '{root}'.")
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


def validate_check_code(code: str) -> None:
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


# --------------------------------------------------------------------------- #
#  Backwards-compatible aliases for tests + dynamic imports. These were the
#  private names exposed by the legacy ``terminal_tool`` module; keep them
#  around so existing imports keep working until they are migrated.
# --------------------------------------------------------------------------- #

_terminate = terminate
_stream_with_limit = stream_with_limit
_drain_remaining = drain_remaining
_validate_check_code = validate_check_code
_TimedOut = TimedOut
