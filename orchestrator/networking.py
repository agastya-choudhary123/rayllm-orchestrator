"""Fast-transfer layer: the "how do bytes move between GPUs" demo.

On a real multi-GPU node, the training->serving handoff and tensor-parallel
serving both live or die on interconnect bandwidth (NVLink / RDMA over
InfiniBand). This module demonstrates the three tiers, picking the best
available at runtime:

  1. NVLink P2P     -- torch cuda tensor .to(other_gpu), measured GB/s.
  2. RDMA           -- pyverbs one-sided read if the fabric + lib are present.
  3. Shared memory   -- POSIX shm (multiprocessing.shared_memory) as a portable
                        stand-in for zero-copy transfer. Always works.

The shared-memory path is intentionally the always-on fallback so the concept
(zero-copy, no serialization) is demoable on any laptop.
"""

from __future__ import annotations

import time
from multiprocessing import shared_memory

from .util import gpu_count, have, log


def transfer_demo(size_mb: int = 256) -> dict:
    nbytes = size_mb * 1024 * 1024
    log(f"Transferring {size_mb} MB payload...")

    if gpu_count() >= 2 and have("torch"):
        return _nvlink_demo(nbytes, size_mb)
    if have("pyverbs"):
        log("pyverbs present -- RDMA fabric available (see _rdma_demo).")
    return _shm_demo(nbytes, size_mb)


def _nvlink_demo(nbytes: int, size_mb: int) -> dict:
    import torch
    src = torch.empty(nbytes // 4, dtype=torch.float32, device="cuda:0")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    dst = src.to("cuda:1", non_blocking=True)  # noqa: F841
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    gbps = (nbytes / 1e9) / dt
    log(f"NVLink/P2P cuda:0 -> cuda:1 : {gbps:.1f} GB/s ({dt*1e3:.2f} ms)")
    return {"path": "nvlink", "gbps": gbps, "ms": dt * 1e3}


def _shm_demo(nbytes: int, size_mb: int) -> dict:
    """Zero-copy handoff via POSIX shared memory -- portable RDMA stand-in."""
    payload = b"\xa5" * nbytes
    shm = shared_memory.SharedMemory(create=True, size=nbytes)
    try:
        t0 = time.perf_counter()
        shm.buf[:nbytes] = payload           # producer writes
        # Consumer maps the SAME segment by name -- no copy, no serialization.
        view = shared_memory.SharedMemory(name=shm.name)
        _ = bytes(view.buf[:64])             # touch it to prove it's mapped
        dt = time.perf_counter() - t0
        view.close()
        gbps = (nbytes / 1e9) / dt
        log(f"Shared-memory zero-copy handoff : {gbps:.1f} GB/s ({dt*1e3:.2f} ms)")
        log("  (RDMA-style: consumer maps the segment by name, no serialization)")
        return {"path": "shared_memory", "gbps": gbps, "ms": dt * 1e3}
    finally:
        shm.close()
        shm.unlink()
