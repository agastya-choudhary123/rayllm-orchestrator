"""Kernel-awareness layer: the --kernel-profile low-latency knob.

For latency-sensitive workloads (e.g. a trading signal model behind a serving
endpoint) tail latency is dominated by scheduler jitter, NUMA-remote memory and
noisy-neighbor CPU contention. This layer applies the standard mitigations:

  * CPU pinning        -- taskset to bind the process to dedicated cores.
  * NUMA locality      -- numactl to keep memory on the local node.
  * cgroup limits      -- a cpuset cgroup so nothing else steals those cores.
  * real-time sched     -- SCHED_FIFO via chrt for deterministic wakeups.

All of this is Linux-only and requires privileges. On macOS / unprivileged
environments we detect that and *explain what would happen* instead of failing.
That keeps the profile honest and demoable everywhere.
"""

from __future__ import annotations

import os
import platform
import subprocess

from .util import have_cmd, log


def _is_linux() -> bool:
    return platform.system() == "Linux"


def _reserved_cores(n: int = 4) -> str:
    """Pick the top-N cores to dedicate to this process (leave 0..k for the OS)."""
    total = os.cpu_count() or 4
    start = max(0, total - n)
    return f"{start}-{total - 1}"


def apply_profile(profile: str) -> None:
    if profile == "default":
        return
    if profile != "low-latency":
        log(f"Unknown kernel profile '{profile}', ignoring.")
        return

    log("Applying kernel profile: low-latency")
    cores = _reserved_cores()

    if not _is_linux():
        log(f"  (non-Linux host) Would pin to cores {cores}, set SCHED_FIFO,")
        log(f"  bind NUMA-local memory, and isolate via a cpuset cgroup.")
        log("  Run on a Linux node with CAP_SYS_NICE to take effect.")
        return

    pid = os.getpid()

    # 1) CPU pinning (taskset) -- keep the hot path on dedicated cores.
    if have_cmd("taskset"):
        _try(["taskset", "-cp", cores, str(pid)],
             f"pinned pid {pid} to cores {cores}")
    else:
        log("  taskset not found -- skipping CPU pinning.")

    # 2) Real-time scheduling (chrt SCHED_FIFO) -- deterministic wakeups.
    if have_cmd("chrt"):
        _try(["chrt", "-f", "-p", "50", str(pid)],
             f"set SCHED_FIFO prio 50 on pid {pid}")
    else:
        log("  chrt not found -- skipping real-time scheduling.")

    # 3) cgroup cpuset isolation -- fence off the cores from other tenants.
    _try_cgroup(cores)

    # 4) NUMA note -- best applied at launch (`numactl --membind`). We surface it.
    if have_cmd("numactl"):
        log("  Tip: launch under `numactl --cpunodebind=0 --membind=0` for NUMA locality.")


def _try(cmd: list[str], ok_msg: str) -> None:
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        log(f"  {ok_msg}")
    except (subprocess.CalledProcessError, PermissionError) as e:
        log(f"  '{' '.join(cmd)}' failed ({e}); need privileges. Skipped.")


def _try_cgroup(cores: str) -> None:
    """Create a cpuset cgroup (v1 path) and move ourselves into it."""
    base = "/sys/fs/cgroup/cpuset/rayllm"
    try:
        os.makedirs(base, exist_ok=True)
        with open(f"{base}/cpuset.cpus", "w") as f:
            f.write(cores)
        with open(f"{base}/cpuset.mems", "w") as f:
            f.write("0")
        with open(f"{base}/tasks", "w") as f:
            f.write(str(os.getpid()))
        log(f"  cpuset cgroup 'rayllm' -> cores {cores}")
    except (PermissionError, FileNotFoundError, OSError):
        log("  cgroup cpuset unavailable (needs root / cgroup v1). Skipped.")
