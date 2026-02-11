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
    p.add_argument("--nn", type=float, default=0.7, help="Nearest-neighbor fraction for conditional mean.")
    p.add_argument("--degree", type=int, choices=[0, 1], default=1, help="Local polynomial degree (0=constant, 1=linear).")
    p.add_argument("--t_transform", type=str, choices=["identity", "log1p"], default="identity")
    return p.parse_args()


def _apply_t_transform(x: np.ndarray, kind: str) -> np.ndarray:
    if kind in (None, "identity"):
        return np.asarray(x, dtype=np.float64)
    if kind == "log1p":
        return np.log1p(np.asarray(x, dtype=np.float64))
    raise ValueError(f"Unknown t_transform: {kind}")


def estimate_conditional_mean(
    T_obs: np.ndarray,
    Y_obs: np.ndarray,
    t_grid: np.ndarray,
    *,
    degree: int = 1,
    nn: float = 0.7,
    t_transform: str = "identity",
) -> np.ndarray:
    T_raw = np.asarray(T_obs, dtype=np.float64).reshape(-1)
    Y = np.asarray(Y_obs, dtype=np.float64).reshape(-1)
    t_grid_raw = np.asarray(t_grid, dtype=np.float64).reshape(-1)
    if not (0.0 < float(nn) <= 1.0):
        raise ValueError("--nn must be in (0,1].")

    T = _apply_t_transform(T_raw, t_transform)
    t_grid_t = _apply_t_transform(t_grid_raw, t_transform)

    n = int(T.shape[0])
    k = int(np.ceil(float(nn) * n))
    k = max(2, min(k, n))

    diff = (T[:, None] - t_grid_t[None, :]).astype(np.float64)
    dist = np.abs(diff)
    h = np.partition(dist, kth=k - 1, axis=0)[k - 1, :]
    h = np.maximum(h, 1e-8)

    u = diff / h[None, :]
    K = np.exp(-0.5 * (u ** 2))
    W = K

    if int(degree) == 0:
        num = np.sum(W * Y[:, None], axis=0)
        den = np.sum(W, axis=0)
        return num / (den + 1e-12)

    S0 = np.sum(W, axis=0)
    S1 = np.sum(W * diff, axis=0)
    S2 = np.sum(W * diff * diff, axis=0)
    T0 = np.sum(W * Y[:, None], axis=0)
    T1 = np.sum(W * diff * Y[:, None], axis=0)
    denom = S0 * S2 - S1 * S1
    return (S2 * T0 - S1 * T1) / (denom + 1e-12)


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
    cond_mean = estimate_conditional_mean(
        T_obs=T,
        Y_obs=Y,
        t_grid=T_eval,
        degree=int(args.degree),
        nn=float(args.nn),
        t_transform=str(args.t_transform),
    )

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
    plt.plot(T_eval[order], cond_mean[order], color="blue", linewidth=2, label="E[Y|A=t] (local poly)")
    plt.xlabel("T")
    plt.ylabel("Y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
