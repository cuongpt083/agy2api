"""Optional OS-level sandbox (bubblewrap) for agy processes.

agy's agent runs real built-in tools (run_command/view_file) with the daemon user's rights, so a
prompt can make it read the project source, `.env` or other conversations. With AGY_SANDBOX=bwrap
the agent only sees a read-only system, its own workdir and ~/.gemini + the agy binaries, in its own
PID namespace, with a cleared environment. Network stays available (agy must reach the model API).
"""

from __future__ import annotations

import logging
import os
import shutil

logger = logging.getLogger(__name__)

_MODE = os.environ.get("AGY_SANDBOX", "").strip().lower()
_warned = False


def sandbox_enabled() -> bool:
    return _MODE == "bwrap"


def wrap_cmd(cmd: list[str], workdir: str, extra_ro_dirs: list[str] | None = None) -> list[str]:
    """Return `cmd` wrapped in bwrap when AGY_SANDBOX=bwrap (and bwrap exists), else `cmd`."""
    global _warned
    if not sandbox_enabled():
        return cmd
    bwrap = shutil.which("bwrap")
    if not bwrap:
        if not _warned:
            logger.warning("AGY_SANDBOX=bwrap but bwrap not found; running agy UNSANDBOXED")
            _warned = True
        return cmd

    home = os.path.expanduser("~")
    uid = os.getuid()
    args = [
        bwrap, "--unshare-pid", "--unshare-ipc", "--unshare-uts", "--die-with-parent",
        "--ro-bind", "/usr", "/usr", "--ro-bind", "/etc", "/etc",
        "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib",
        "--symlink", "usr/lib64", "/lib64", "--symlink", "usr/sbin", "/sbin",
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--tmpfs", home,
        "--ro-bind", f"{home}/.local/bin", f"{home}/.local/bin",
        "--bind", f"{home}/.gemini", f"{home}/.gemini",
        "--bind", workdir, workdir, "--chdir", workdir,
    ]
    bus = f"/run/user/{uid}/bus"
    if os.path.exists(bus):
        args += ["--ro-bind", bus, bus, "--setenv", "DBUS_SESSION_BUS_ADDRESS", f"unix:path={bus}"]
    for d in extra_ro_dirs or []:
        if os.path.isdir(d):
            args += ["--ro-bind", d, d]
    args += [
        "--clearenv", "--setenv", "HOME", home,
        "--setenv", "PATH", "/usr/local/bin:/usr/bin:/bin",
        "--setenv", "AGY_IS_API_CALL", "1", "--setenv", "LANG", "C.UTF-8",
        "--",
    ]
    # Use the absolute agy wrapper so PATH inside the sandbox does not matter.
    agy = os.path.join(home, ".local", "bin", "agy") if cmd and cmd[0] == "agy" else cmd[0]
    return args + [agy] + cmd[1:]
