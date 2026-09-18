"""Linux process sandbox for untrusted project Python code."""
from __future__ import annotations

import ctypes
import errno
import os
import shutil
import sys
from pathlib import Path

_BWRAP = shutil.which("bwrap")
_DENIED_SYSCALLS = (
    "ptrace",
    "process_vm_readv",
    "process_vm_writev",
    "mount",
    "umount2",
    "pivot_root",
    "open_by_handle_at",
    "bpf",
    "perf_event_open",
    "uselib",
)


def seccomp_filter_fd() -> int:
    """Return a libseccomp BPF fd that denies networking and host escape syscalls."""
    try:
        lib = ctypes.CDLL("libseccomp.so.2")
    except OSError as error:
        raise RuntimeError("libseccomp is required for CAD sandboxing.") from error

    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    lib.seccomp_rule_add.restype = ctypes.c_int
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_export_bpf.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.seccomp_export_bpf.restype = ctypes.c_int
    lib.seccomp_release.argtypes = [ctypes.c_void_p]

    allow = 0x7FFF0000
    deny = 0x00050000 | errno.EPERM
    context = lib.seccomp_init(allow)
    if not context:
        raise RuntimeError("Unable to initialize the CAD seccomp filter.")
    fd = os.memfd_create("cad-agent-seccomp", os.MFD_CLOEXEC)
    try:
        for name in _DENIED_SYSCALLS:
            syscall = lib.seccomp_syscall_resolve_name(name.encode())
            if syscall >= 0 and lib.seccomp_rule_add(context, deny, syscall, 0) != 0:
                raise RuntimeError(f"Unable to block sandbox syscall: {name}")
        if lib.seccomp_export_bpf(context, fd) != 0:
            raise RuntimeError("Unable to export the CAD seccomp filter.")
        os.lseek(fd, 0, os.SEEK_SET)
        return fd
    except Exception:
        os.close(fd)
        raise
    finally:
        lib.seccomp_release(context)


def command(
    workspace: Path,
    arguments: list[str],
    *,
    writable: bool,
    timeout_seconds: int,
    writable_tmp: bool = True,
) -> tuple[list[str], int]:
    """Build a fail-closed Bubblewrap command and its inherited seccomp fd."""
    if not _BWRAP:
        raise RuntimeError("bubblewrap (bwrap) is required for CAD sandboxing.")
    workspace = workspace.resolve()
    python_root = Path(sys.prefix).resolve()
    bind_mode = "--bind" if writable else "--ro-bind"
    tmp_mount = ["--tmpfs", "/tmp"]
    if not writable_tmp:
        tmp_mount.extend(["--remount-ro", "/tmp"])
    seccomp_fd = seccomp_filter_fd()
    path_dirs = ["/venv/bin", "/usr/local/bin", "/usr/bin", "/bin"]
    extra_binds: list[str] = [
        "--ro-bind-try", "/opt", "/opt",
    ]
    openscad_bin = shutil.which("openscad")
    if openscad_bin:
        scad_real = Path(openscad_bin).resolve()
        scad_dir = scad_real.parent
        try:
            scad_dir.relative_to(python_root)
        except ValueError:
            try:
                scad_dir.relative_to("/usr")
            except ValueError:
                extra_binds.extend(["--ro-bind-try", str(scad_dir), str(scad_dir)])
                if str(scad_dir) not in path_dirs:
                    path_dirs.append(str(scad_dir))
        scad_runtime = scad_dir.parent / "openscad"
        if scad_runtime.is_dir():
            extra_binds.extend([
                "--ro-bind-try", str(scad_runtime), str(scad_runtime),
                "--ro-bind-try", str(scad_runtime), "/venv/openscad",
            ])

    repo_scad = Path(__file__).resolve().parent.parent / ".venv" / "openscad"
    if repo_scad.is_dir():
        extra_binds.extend([
            "--ro-bind-try", str(repo_scad), str(repo_scad),
            "--ro-bind-try", str(repo_scad), "/venv/openscad",
        ])
        repo_scad_bin = repo_scad.parent / "bin"
        if repo_scad_bin.is_dir() and str(repo_scad_bin) not in path_dirs:
            extra_binds.extend(["--ro-bind-try", str(repo_scad_bin), str(repo_scad_bin)])
            path_dirs.append(str(repo_scad_bin))

    sandbox = [
        _BWRAP,
        "--die-with-parent",
        "--new-session",
        "--unshare-user",
        "--unshare-pid",
        "--unshare-net",
        "--unshare-ipc",
        "--unshare-uts",
        "--disable-userns",
        "--proc", "/proc",
        "--dev", "/dev",
        *tmp_mount,
        *extra_binds,
        "--ro-bind", "/usr", "/usr",
        "--symlink", "usr/bin", "/bin",
        "--symlink", "usr/lib", "/lib",
        "--symlink", "usr/lib64", "/lib64",
        "--ro-bind", "/etc", "/etc",
        "--ro-bind", str(python_root), "/venv",
        bind_mode, str(workspace), "/workspace",
        "--chdir", "/workspace",
        "--clearenv",
        "--setenv", "HOME", "/tmp",
        "--setenv", "PATH", ":".join(path_dirs),
        "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
        "--setenv", "TMPDIR", "/tmp",
        "--setenv", "LANG", "C.UTF-8",
        "--seccomp", str(seccomp_fd),
        "--",
        "/usr/bin/prlimit",
        f"--cpu={timeout_seconds + 5}",
        "--fsize=536870912",
        "--nofile=512",
        "--nproc=64",
        "--as=8589934592",
        "--",
        "/venv/bin/python",
        *arguments,
    ]
    return sandbox, seccomp_fd
