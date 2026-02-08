from __future__ import annotations

import torch


def select_device(device_arg: str) -> torch.device:
    """
    Select the best available CUDA device by free memory when device_arg == "auto".
    Otherwise, return the requested device.
    """
    if device_arg and device_arg != "auto":
        return torch.device(device_arg)

    if not torch.cuda.is_available():
        return torch.device("cpu")

    best_idx = 0
    best_free = -1
    try:
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                free, _ = torch.cuda.mem_get_info()
            if int(free) > best_free:
                best_free = int(free)
                best_idx = int(i)
    except Exception:
        return torch.device("cuda:0")

    return torch.device(f"cuda:{best_idx}")
