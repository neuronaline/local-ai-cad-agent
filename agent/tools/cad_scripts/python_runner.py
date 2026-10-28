"""Sandbox-side restricted runner for user mathematical and engineering Python code."""
from __future__ import annotations

import ast
import builtins
import cmath
import collections
import decimal
import fractions
import functools
import io
import itertools
import json
import math
import numbers
import operator
import re
import statistics
import sys
import typing
from pathlib import Path
from typing import Any

import numpy as np
try:
    import sympy as sp
except ImportError:
    sp = None

ALLOWED_ROOTS = frozenset({
    "numpy",
    "np",
    "sympy",
    "sp",
    "math",
    "cmath",
    "fractions",
    "decimal",
    "statistics",
    "itertools",
    "functools",
    "collections",
    "json",
    "re",
    "numbers",
    "operator",
    "typing",
    "dataclasses",
    "copy",
    "random",
    # Python C-extension / internal standard helpers used by allowed modules:
    "_operator",
    "_collections",
    "_functools",
    "_json",
    "_sre",
    "_stat",
    "_statistics",
    "_decimal",
})

BLOCKED_SUBMODULES = frozenset({
    "numpy.distutils",
    "numpy.f2py",
    "numpy.testing",
    "sympy.utilities.autowrap",
    "sympy.external",
})

FORBIDDEN_NAMES = frozenset({
    "open",
    "eval",
    "exec",
    "compile",
    "input",
    "breakpoint",
    "getattr",
    "setattr",
    "delattr",
    "globals",
    "locals",
    "vars",
    "memoryview",
    "exit",
    "quit",
})

DANGEROUS_ATTRIBUTES = frozenset({
    "__class__",
    "__bases__",
    "__base__",
    "__subclasses__",
    "__mro__",
    "__globals__",
    "__code__",
    "__builtins__",
    "__dict__",
    "__closure__",
    "__func__",
    "__self__",
    "__module__",
    "f_globals",
    "f_locals",
    "f_code",
    "f_builtins",
    "gi_frame",
    "cr_frame",
    "ag_frame",
})


def _is_blocked_submodule(mod_name: str) -> bool:
    """Return True if ``mod_name`` matches or is a submodule of any entry in ``BLOCKED_SUBMODULES``."""
    return any(
        mod_name == blocked or mod_name.startswith(blocked + ".")
        for blocked in BLOCKED_SUBMODULES
    )


def _validate_ast(tree: ast.AST) -> None:
    """Enforce syntax-level security constraints before execution."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            attr = node.attr
            if attr in DANGEROUS_ATTRIBUTES:
                raise ValueError(
                    f"Access to restricted attribute '{attr}' is not permitted."
                )
        elif isinstance(node, ast.Name):
            name = node.id
            if name in FORBIDDEN_NAMES or name in DANGEROUS_ATTRIBUTES:
                raise ValueError(f"Use of '{name}' is restricted.")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                mod_name = alias.name
                root = mod_name.split(".")[0]
                if root not in ALLOWED_ROOTS or _is_blocked_submodule(mod_name):
                    raise ValueError(f"Importing module '{mod_name}' is restricted.")
        elif isinstance(node, ast.ImportFrom):
            if node.level != 0 or not node.module:
                raise ValueError("Relative imports are restricted.")
            mod_name = node.module
            root = mod_name.split(".")[0]
            if root not in ALLOWED_ROOTS or _is_blocked_submodule(mod_name):
                raise ValueError(f"Importing from module '{mod_name}' is restricted.")


_REAL_IMPORT = builtins.__import__


def _safe_import(
    name: str,
    globals: Any = None,
    locals: Any = None,
    fromlist: tuple = (),
    level: int = 0,
) -> Any:
    """Whitelisted __import__ implementation for restricted execution."""
    root = name.split(".")[0]
    if (
        root not in ALLOWED_ROOTS
        and not name.startswith("numpy")
        and not name.startswith("sympy")
    ):
        raise ImportError(f"Importing module '{name}' is restricted.")
    if _is_blocked_submodule(name):
        raise ImportError(f"Importing submodule '{name}' is restricted.")
    if fromlist:
        for item in fromlist:
            if isinstance(item, str) and _is_blocked_submodule(f"{name}.{item}"):
                raise ImportError(f"Importing submodule '{name}.{item}' is restricted.")
    return _REAL_IMPORT(name, globals, locals, fromlist, level)


_HIDDEN_BUILTINS = frozenset({
    "type",
    "dir",
    "hasattr",
    "help",
    "credits",
    "license",
    "copyright",
})


def _build_safe_builtins() -> dict[str, Any]:
    safe: dict[str, Any] = {
        k: v
        for k, v in builtins.__dict__.items()
        if k not in FORBIDDEN_NAMES
        and k not in _HIDDEN_BUILTINS
        and not (k.startswith("__") and k.endswith("__"))
    }
    safe["__import__"] = _safe_import
    safe["__build_class__"] = builtins.__build_class__
    safe["__name__"] = "__main__"
    return safe


def _serialize_value(val: Any) -> Any:
    """Format evaluation result into a JSON-compatible representation."""
    if val is None or isinstance(val, (bool, int)):
        return val
    if isinstance(val, float):
        if math.isnan(val):
            return "NaN"
        if math.isinf(val):
            return "Infinity" if val > 0 else "-Infinity"
        return val
    if isinstance(val, str):
        return val
    if isinstance(val, np.generic):
        return _serialize_value(val.item())
    if isinstance(val, np.ndarray):
        if val.size <= 2000:
            return _serialize_value(val.tolist())
        return f"ndarray(shape={val.shape}, dtype={val.dtype})"
    if sp is not None and isinstance(val, (sp.Basic, sp.MatrixBase)):
        return str(val)
    if isinstance(val, (list, tuple)):
        return [_serialize_value(item) for item in val]
    if isinstance(val, dict):
        return {str(k): _serialize_value(v) for k, v in val.items()}
    if isinstance(val, (set, frozenset)):
        return [_serialize_value(item) for item in sorted(val, key=str)]
    return str(val)


def execute_script(code_text: str) -> dict[str, Any]:
    """Parse, validate, and execute Python code in the restricted environment."""
    if not code_text or not code_text.strip():
        return {"ok": True, "stdout": "", "result": None}

    try:
        tree = ast.parse(code_text, filename="<agent_python>", mode="exec")
        _validate_ast(tree)
    except Exception as err:
        return {"ok": False, "error": f"Security/Syntax error: {err}", "stdout": ""}

    has_final_expr = bool(tree.body and isinstance(tree.body[-1], ast.Expr))
    if has_final_expr:
        exec_body = tree.body[:-1]
        final_expr = tree.body[-1].value
        module_code = compile(
            ast.Module(body=exec_body, type_ignores=[]),
            "<agent_python>",
            "exec",
        )
        expr_code = compile(
            ast.Expression(body=final_expr),
            "<agent_python>",
            "eval",
        )
    else:
        module_code = compile(tree, "<agent_python>", "exec")
        expr_code = None

    env: dict[str, Any] = {
        "__builtins__": _build_safe_builtins(),
        "np": np,
        "numpy": np,
        "math": math,
        "cmath": cmath,
        "Fraction": fractions.Fraction,
        "Decimal": decimal.Decimal,
        "itertools": itertools,
        "functools": functools,
        "collections": collections,
        "statistics": statistics,
        "json": json,
        "re": re,
    }
    if sp is not None:
        env["sp"] = sp
        env["sympy"] = sp

    stdout_buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = stdout_buf
    try:
        exec(module_code, env)
        result_raw = eval(expr_code, env) if expr_code is not None else None
        captured_stdout = stdout_buf.getvalue()
        return {
            "ok": True,
            "stdout": captured_stdout.strip(),
            "result": _serialize_value(result_raw),
        }
    except Exception as err:
        captured_stdout = stdout_buf.getvalue()
        return {
            "ok": False,
            "error": f"{type(err).__name__}: {err}",
            "stdout": captured_stdout.strip(),
        }
    finally:
        sys.stdout = old_stdout


def main() -> None:
    script_path = Path.cwd() / "script.py"
    if not script_path.is_file():
        result = {"ok": False, "error": "script.py was not found in workspace."}
    else:
        try:
            code_text = script_path.read_text(encoding="utf-8")
            result = execute_script(code_text)
        except Exception as err:
            result = {"ok": False, "error": f"Failed to execute: {err}"}
    sys.stdout.write(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

