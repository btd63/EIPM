from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="./datasets")
    p.add_argument("--pattern", type=str, default="sim_*.npz")
    return p.parse_args()


def _fmt_value(arr: np.ndarray, max_items: int = 5) -> str:
    if arr.ndim == 0:
        try:
            return f"value={float(arr):.6g}"
        except Exception:
            return f"value={arr.item()}"
    flat = arr.ravel()
    n = min(max_items, flat.size)
    if np.issubdtype(flat.dtype, np.number):
        sample = ", ".join(f"{float(x):.6g}" for x in flat[:n])
    else:
        sample = ", ".join(str(x) for x in flat[:n])
    suffix = "" if flat.size <= n else ", ..."
    return f"shape={arr.shape}, sample=[{sample}]{suffix}"


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    files = sorted(glob.glob(str(data_dir / args.pattern)))

    if not files:
        raise FileNotFoundError(f"No npz files found in {data_dir} with pattern {args.pattern}")

    npz_path = files[0]
    print(f"[FILE] {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    print(f"[KEYS] {len(data.files)} keys")

    for key in data.files:
        arr = np.array(data[key])
        info = _fmt_value(arr)
        print(f"- {key}: {info}, dtype={arr.dtype}")

    if "Ttilde_train" in data.files:
        tt = np.array(data["Ttilde_train"])
        flat = tt.reshape(-1)
        uniq, counts = np.unique(flat, return_counts=True)
        max_count = int(counts.max()) if counts.size else 0
        num_dups = int((counts > 1).sum())
        print(f"[Ttilde_train] total={flat.size}, unique={uniq.size}, dup_values={num_dups}, max_count={max_count}")
        if num_dups > 0:
            top_idx = np.argsort(counts)[::-1]
            top_k = min(5, top_idx.size)
            print("[Ttilde_train] top_duplicates:")
            for i in range(top_k):
                idx = int(top_idx[i])
                print(f"  value={uniq[idx]:.6g}, count={int(counts[idx])}")
        flat32 = flat.astype(np.float32)
        uniq32, counts32 = np.unique(flat32, return_counts=True)
        max_count32 = int(counts32.max()) if counts32.size else 0
        num_dups32 = int((counts32 > 1).sum())
        print(f"[Ttilde_train float32] total={flat32.size}, unique={uniq32.size}, dup_values={num_dups32}, max_count={max_count32}")
        if num_dups32 > 0:
            top_idx32 = np.argsort(counts32)[::-1]
            top_k32 = min(5, top_idx32.size)
            print("[Ttilde_train float32] top_duplicates:")
            for i in range(top_k32):
                idx = int(top_idx32[i])
                print(f"  value={uniq32[idx]:.6g}, count={int(counts32[idx])}")
    else:
        print("[Ttilde_train] key not found")


if __name__ == "__main__":
    main()
