#!/usr/bin/env python3
"""
Approximate true weights W(x,t) = p(t) / p(t|x) for nonlinear DGP via Monte Carlo.
Uses MC samples of Z to approximate p(t) and a kNN-kernel approximation of p(t|x).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out


def z_to_x(Z: np.ndarray) -> np.ndarray:
    n, d_X = Z.shape
    if d_X % 5 != 0:
        raise ValueError("d_X/5 must be integer.")
    X = np.zeros((n, d_X), dtype=np.float64)
    n_blocks = d_X // 5
    for j in range(n_blocks):
        base = 5 * j
        z1 = Z[:, base + 0]
        z2 = Z[:, base + 1]
        z3 = Z[:, base + 2]
        z4 = Z[:, base + 3]
        z5 = Z[:, base + 4]
        X[:, base + 0] = np.exp(z1 / 2.0)
        X[:, base + 1] = z1 / (1.0 + np.exp(z2)) + 10.0
        X[:, base + 2] = ((z1 * z3 + 15.0) / 25.0) ** 3
        X[:, base + 3] = (z2 + z4 + 20.0) ** 2
        X[:, base + 4] = (z3 + z5 > 0.0).astype(float)
    return X


def lognormal_pdf(t: float, meanlog: np.ndarray) -> np.ndarray:
    if t <= 0:
        return np.zeros_like(meanlog, dtype=np.float64)
    logt = math.log(float(t))
    diff = (logt - meanlog)
    return np.exp(-0.5 * diff * diff) / (t * math.sqrt(2.0 * math.pi))


def parse_mc_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def knn_kernel_weights(X_mc: np.ndarray, x: np.ndarray, k: int) -> np.ndarray:
    diff = X_mc - x.reshape(1, -1)
    dist2 = np.sum(diff * diff, axis=1)
    k = max(1, min(int(k), dist2.shape[0]))
    idx = np.argpartition(dist2, k - 1)[:k]
    d2 = dist2[idx]
    h = math.sqrt(max(float(np.max(d2)), 1e-12))
    w = np.exp(-d2 / (2.0 * h * h))
    w_sum = float(np.sum(w))
    if w_sum <= 0:
        w = np.ones_like(w) / float(len(w))
    else:
        w = w / w_sum
    return idx, w


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="./datasets")
    p.add_argument(
        "--pattern",
        type=str,
        default="sim_nonlinear_dx5_ntr1000_nev10000_rpt100_tk5_ok5_pi0.0_seed42.npz",
    )
    p.add_argument("--rep", type=int, default=0)
    p.add_argument("--mc", type=str, default="20,100,1000")
    p.add_argument("--k_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    npz_path = Path(args.data_dir) / args.pattern
    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    scenario = str(np.array(data["scenario"]).reshape(-1)[0])
    if scenario != "nonlinear":
        raise ValueError(f"Only nonlinear supported. Got scenario={scenario}")

    rep = int(args.rep)
    X_train = np.array(data["X_train"][rep], dtype=np.float64)
    T_train = np.array(data["T_train"][rep], dtype=np.float64).reshape(-1)

    beta_T0 = float(np.array(data["beta_T0"][rep]).reshape(-1)[0])
    beta_T = np.array(data["beta_T"][rep], dtype=np.float64).reshape(-1)
    beta_pi0 = float(np.array(data["beta_pi0"][rep]).reshape(-1)[0])
    beta_pi = np.array(data["beta_pi"][rep], dtype=np.float64).reshape(-1)
    pi_0 = float(np.array(data["pi_0"]).reshape(-1)[0])

    n = X_train.shape[0]
    d_X = X_train.shape[1]

    mc_list = parse_mc_list(args.mc)
    np.set_printoptions(threshold=np.inf, linewidth=200, precision=6, suppress=True)

    for M in mc_list:
        rng = np.random.default_rng(int(args.seed) + int(M))
        Z_mc = rng.normal(0.0, 1.0, size=(int(M), d_X))
        X_mc = z_to_x(Z_mc)

        meanlog = beta_T0 + Z_mc @ beta_T
        if pi_0 > 0.0:
            p0 = _sigmoid(beta_pi0 + Z_mc[:, : beta_pi.shape[0]] @ beta_pi)
        else:
            p0 = np.zeros(M, dtype=np.float64)

        k = max(5, int(math.ceil(float(args.k_frac) * float(M))))
        k = min(k, M)

        weights = np.zeros(n, dtype=np.float64)
        for i in range(n):
            t_i = float(T_train[i])
            x_i = X_train[i]

            if t_i <= 0.0:
                p_t = float(np.mean(p0))
                idx, w = knn_kernel_weights(X_mc, x_i, k)
                p_t_given_x = float(np.sum(w * p0[idx]))
            else:
                pdf_all = lognormal_pdf(t_i, meanlog)
                if pi_0 > 0.0:
                    pdf_all = (1.0 - p0) * pdf_all
                p_t = float(np.mean(pdf_all))
                idx, w = knn_kernel_weights(X_mc, x_i, k)
                p_t_given_x = float(np.sum(w * pdf_all[idx]))

            denom = max(p_t_given_x, 1e-12)
            weights[i] = p_t / denom

        print(f"[MC={M}] W (n={n}):")
        print(weights)


if __name__ == "__main__":
    main()
