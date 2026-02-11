#!/usr/bin/env python3
"""
Plot observed Y vs T with true ADRF curve from an npz dataset.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--npz", type=str, required=True, help="Path to dataset npz")
    p.add_argument("--rep", type=int, default=0)
    p.add_argument("--out", type=str, default="./plots/adrf_overlay.png")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    npz_path = Path(args.npz)
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    data = np.load(npz_path, allow_pickle=True)
    rep = int(args.rep)

    T = np.array(data["T_train"][rep]).reshape(-1)
    Y = np.array(data["Y_train"][rep]).reshape(-1)
    T_eval_all = np.array(data["T_eval"])
    mu_eval_all = np.array(data["mu_eval"])
    if T_eval_all.ndim == 2:
        T_eval = np.array(T_eval_all[rep]).reshape(-1)
        mu_eval = np.array(mu_eval_all[rep]).reshape(-1)
    else:
        T_eval = np.array(T_eval_all).reshape(-1)
        mu_eval = np.array(mu_eval_all).reshape(-1)

    order = np.argsort(T_eval)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for plotting.") from exc

    plt.figure(figsize=(7, 5))
    plt.scatter(T, Y, s=8, alpha=0.25, color="gray", label="Y observed")
    plt.plot(T_eval[order], mu_eval[order], color="red", linewidth=2, label="True ADRF")
    plt.xlabel("T")
    plt.ylabel("Y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
