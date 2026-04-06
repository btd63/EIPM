from __future__ import annotations

import subprocess
import atexit
import os
from pathlib import Path
from typing import Dict, List

import torch


_LOCK_DIR = Path("/tmp/cbeipm_gpu_locks")
_ACTIVE_LOCKS: List[Path] = []


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _cleanup_stale_locks() -> None:
    _LOCK_DIR.mkdir(parents=True, exist_ok=True)
    for p in _LOCK_DIR.glob("gpu*.lock"):
        try:
            txt = p.read_text().strip()
            pid = int(txt) if txt else -1
        except Exception:
            pid = -1
        if not _pid_alive(pid):
            try:
                p.unlink()
            except Exception:
                pass


def _register_lock_cleanup() -> None:
    def _cleanup() -> None:
        for p in list(_ACTIVE_LOCKS):
            try:
                txt = p.read_text().strip()
                if txt == str(os.getpid()):
                    p.unlink()
            except Exception:
                pass

    atexit.register(_cleanup)


def _try_reserve_gpu(gpu_idx: int) -> bool:
    _LOCK_DIR.mkdir(parents=True, exist_ok=True)
    lock = _LOCK_DIR / f"gpu{int(gpu_idx)}.lock"
    try:
        # O_EXCL makes one-process-per-GPU reservation (best-effort).
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as f:
            f.write(str(os.getpid()))
        _ACTIVE_LOCKS.append(lock)
        return True
    except FileExistsError:
        return False
    except Exception:
        return False


def _gpu_status_from_nvidia_smi() -> List[Dict[str, int]]:
    """
    Return list of GPU status dicts without touching CUDA runtime contexts.
    Keys: idx, util, mem_used_mib, mem_free_mib, mem_total_mib
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []

    infos: List[Dict[str, int]] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        try:
            infos.append(
                {
                    "idx": int(parts[0]),
                    "util": int(parts[1]),
                    "mem_used_mib": int(parts[2]),
                    "mem_free_mib": int(parts[3]),
                    "mem_total_mib": int(parts[4]),
                }
            )
        except Exception:
            continue
    return infos


def _gpu_process_counts_from_nvidia_smi() -> Dict[int, int]:
    """
    Return compute-process count per GPU index.
    """
    counts: Dict[int, int] = {}
    try:
        map_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        uuid_to_idx: Dict[str, int] = {}
        for line in map_out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 2:
                continue
            try:
                uuid_to_idx[parts[1]] = int(parts[0])
            except Exception:
                continue

        app_out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        for line in app_out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            idx = uuid_to_idx.get(parts[0], None)
            if idx is None:
                continue
            counts[idx] = counts.get(idx, 0) + 1
    except Exception:
        return {}
    return counts


def _print_gpu_status_report(statuses: List[Dict[str, int]], proc_counts: Dict[int, int]) -> None:
    if not statuses:
        return
    print("[GPU-AUTO] status: idx util% mem_used/mem_total(MiB) procs locked")
    for s in statuses:
        idx = int(s["idx"])
        lock = _LOCK_DIR / f"gpu{idx}.lock"
        locked = lock.exists()
        pcount = int(proc_counts.get(idx, 0))
        print(
            f"[GPU-AUTO] {idx}: {int(s['util']):3d}% "
            f"{int(s['mem_used_mib']):5d}/{int(s['mem_total_mib']):5d} "
            f"procs={pcount:2d} lock={'Y' if locked else 'N'}"
        )


def select_device(device_arg: str) -> torch.device:
    """
    Select the best available CUDA device by free memory when device_arg == "auto".
    Otherwise, return the requested device.
    """
    if device_arg and device_arg != "auto":
        return torch.device(device_arg)

    if not torch.cuda.is_available():
        return torch.device("cpu")

    _cleanup_stale_locks()
    _register_lock_cleanup()

    gpu_status = _gpu_status_from_nvidia_smi()
    proc_counts = _gpu_process_counts_from_nvidia_smi()
    _print_gpu_status_report(gpu_status, proc_counts)

    if not gpu_status:
        # fallback (may create CUDA context on queried device)
        gpu_infos = []
        try:
            for i in range(torch.cuda.device_count()):
                free, _ = torch.cuda.mem_get_info(i)
                gpu_infos.append((int(i), int(free)))
        except Exception:
            return torch.device("cuda:0")
        ordered = sorted(gpu_infos, key=lambda x: x[1], reverse=True)
        for idx, _ in ordered:
            if _try_reserve_gpu(idx):
                print(f"[GPU-AUTO] selected cuda:{idx} (fallback/free-mem)")
                return torch.device(f"cuda:{idx}")
        idx = int(ordered[0][0])
        print(f"[GPU-AUTO] selected cuda:{idx} (fallback/all-locked)")
        return torch.device(f"cuda:{idx}")

    # Prefer unreserved + low-util + low-proc GPUs first.
    ordered = sorted(
        gpu_status,
        key=lambda s: (
            int(((_LOCK_DIR / f"gpu{int(s['idx'])}.lock").exists())),
            int(s["util"] > 15),  # strongly prefer truly idle GPUs
            int(proc_counts.get(int(s["idx"]), 0) > 0),
            int(s["util"]),
            int(s["mem_used_mib"]),
        ),
    )
    for s in ordered:
        idx = int(s["idx"])
        if _try_reserve_gpu(idx):
            print(
                f"[GPU-AUTO] selected cuda:{idx} "
                f"(util={int(s['util'])}%, mem_used={int(s['mem_used_mib'])}MiB, "
                f"procs={int(proc_counts.get(idx, 0))})"
            )
            return torch.device(f"cuda:{idx}")

    # If all GPUs are reserved, fall back to least-loaded GPU by same ordering.
    idx = int(ordered[0]["idx"])
    print(
        f"[GPU-AUTO] all reserved; fallback cuda:{idx} "
        f"(util={int(ordered[0]['util'])}%, mem_used={int(ordered[0]['mem_used_mib'])}MiB)"
    )
    return torch.device(f"cuda:{idx}")
