from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Literal

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="./datasets")
    p.add_argument(
        "--npz",
        type=str,
        default="sim_nonlinear_dx50_ntr1000_nev10000_rpt100_tk50_ok50_pi0.0_seed42.npz",
    )
    p.add_argument("--huling_dir", type=str, default="./models/huling")
    p.add_argument("--max_reps", type=int, default=100)
    p.add_argument("--only_rep", type=int, default=-1)
    p.add_argument("--degree", type=int, choices=[0, 1], default=1, help="Local polynomial degree (0=constant, 1=linear).")
    p.add_argument("--a_h", type=float, default=1.0, help="Bandwidth scale factor.")
    p.add_argument("--alpha", type=float, default=0.05, help="Exponent for bandwidth base.")
    p.add_argument("--t_transform", type=str, choices=["identity", "log1p"], default="identity")
    return p.parse_args()


def load_huling_weights(csv_path: Path) -> np.ndarray:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    w = np.loadtxt(str(csv_path), delimiter=",", skiprows=1)
    return np.asarray(w, dtype=np.float64).reshape(-1)


def _apply_t_transform(x: np.ndarray, kind: Literal["identity", "log1p"]) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if kind == "identity":
        return x
    if kind == "log1p":
        return np.log1p(x)
    raise ValueError(f"Unknown t_transform: {kind}")


def _nonzero_quantile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x_nz = x[x > 0]
    if x_nz.size == 0:
        return 0.0
    return float(np.quantile(x_nz, q))


def _pairwise_nonzero_quantile(T: np.ndarray, q: float) -> float:
    T = np.asarray(T, dtype=np.float64).reshape(-1)
    n = int(T.shape[0])
    if n <= 1:
        return 0.0
    # n=1000 => 1e6 distances OK; for bigger n, sample pairs.
    if n <= 3000:
        dist = np.abs(T[:, None] - T[None, :]).reshape(-1)
        dist = dist[dist > 0]
        if dist.size == 0:
            return 0.0
        return float(np.quantile(dist, q))
    rng = np.random.default_rng(0)
    m = 2_000_000
    i = rng.integers(0, n, size=m, dtype=np.int64)
    j = rng.integers(0, n, size=m, dtype=np.int64)
    mask = i != j
    d = np.abs(T[i[mask]] - T[j[mask]])
    d = d[d > 0]
    if d.size == 0:
        return 0.0
    return float(np.quantile(d, q))


def estimate_adrf_huling_local_poly_quantile(
    T_obs_raw: np.ndarray,
    Y_obs: np.ndarray,
    weights: np.ndarray,
    t_grid_raw: np.ndarray,
    *,
    a_h: float,
    alpha: float,
    degree: int,
    t_transform: Literal["identity", "log1p"],
) -> np.ndarray:
    """
    Pure-Python local polynomial ADRF estimator aligned with evaluate_eipm.py style:
      h(t) = a_h * (a(t) + q_global)^alpha
      a(t) = nonzero 0.1-quantile of |T_i - t|
      q_global = nonzero 0.1-quantile of |T_i - T_j|, i != j

    weights enter multiplicatively (in log-space for stability).
    """
    T_obs_raw = np.asarray(T_obs_raw, dtype=np.float64).reshape(-1)
    Y_obs = np.asarray(Y_obs, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    t_grid_raw = np.asarray(t_grid_raw, dtype=np.float64).reshape(-1)

    if T_obs_raw.shape[0] != Y_obs.shape[0] or T_obs_raw.shape[0] != w.shape[0]:
        raise ValueError("T_obs, Y_obs, weights must have the same length.")

    if np.all(w == 0.0) or (np.mean(w == 0.0) > 0.95):
        return np.full_like(t_grid_raw, fill_value=np.nan, dtype=np.float64)

    if np.any(w < 0):
        raise ValueError("weights must be nonnegative.")

    T_obs = _apply_t_transform(T_obs_raw, t_transform).reshape(-1)
    t_grid = _apply_t_transform(t_grid_raw, t_transform).reshape(-1)

    q_global = _pairwise_nonzero_quantile(T_obs, 0.1)

    eps = 1e-300  # keep log defined; extremely small so it doesn't change positive weights
    logw = np.log(w + eps)

    preds = []
    for t0 in t_grid:
        diff = T_obs - float(t0)
        dist = np.abs(diff)
        a_t = _nonzero_quantile(dist, 0.1)
        base = float(a_t + q_global)
        if base <= 0.0:
            base = 1e-8
        h_t = float(a_h) * float(base ** float(alpha))
        if h_t <= 1e-8:
            h_t = 1e-8

        u = diff / h_t
        logk = -0.5 * (u ** 2)
        logw_eff = logw + logk
        max_log = np.max(logw_eff)
        w_eff = np.exp(logw_eff - max_log)
        s = float(np.sum(w_eff))
        if np.isfinite(s) and s > 0.0:
            w_eff = w_eff / s

        if int(degree) == 0:
            preds.append(float(np.sum(w_eff * Y_obs)))
            continue

        # local linear with normalized weights
        s0 = float(np.sum(w_eff))
        s1 = float(np.sum(w_eff * diff))
        s2 = float(np.sum(w_eff * diff * diff))
        t0v = float(np.sum(w_eff * Y_obs))
        t1v = float(np.sum(w_eff * diff * Y_obs))
        denom = s0 * s2 - s1 * s1
        preds.append(float((s2 * t0v - s1 * t1v) / (denom + 1e-12)))

    return np.array(preds, dtype=np.float64)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    npz_path = data_dir / args.npz
    if not npz_path.exists():
        files = sorted(glob.glob(str(data_dir / "sim_*linear*.npz")))
        if not files:
            raise FileNotFoundError(f"Dataset not found: {npz_path}")
        npz_path = Path(files[0])

    data = np.load(npz_path, allow_pickle=True)
    if "T_eval" not in data.files or "mu_eval" not in data.files:
        raise KeyError("T_eval or mu_eval not found in dataset")

    T_eval_all = np.array(data["T_eval"])
    mu_eval_all = np.array(data["mu_eval"])

    T_all = np.array(data["T_train"])
    Y_all = np.array(data["Y_train"])

    n_rpt = int(T_all.shape[0])
    n_run = min(int(args.max_reps), n_rpt)

    stem = npz_path.stem
    huling_dir = Path(args.huling_dir)

    mse_list = []
    mae_list = []

    print(f"[INFO] Dataset: {npz_path.name}")
    print(f"[INFO] Using huling_dir: {huling_dir}")
    print(f"[INFO] Replications: {n_rpt} (running {n_run})")
    print(f"[INFO] local_poly: degree={args.degree} a_h={args.a_h} alpha={args.alpha} t_transform={args.t_transform}")

    for r in range(n_run):
        if args.only_rep >= 0 and int(r) != int(args.only_rep):
            continue

        w_path = huling_dir / f"{stem}_rep{r:03d}_huling_weights.csv"
        if not w_path.exists():
            print(f"[WARN] missing weights: {w_path}")
            continue

        w = load_huling_weights(w_path)
        if w.shape[0] != int(len(T_all[r])):
            raise ValueError(f"weights length mismatch: {w.shape[0]} != {len(T_all[r])} for rep {r}")

        if T_eval_all.ndim == 2:
            T_eval = np.array(T_eval_all[r]).reshape(-1)
            mu_eval = np.array(mu_eval_all[r]).reshape(-1)
        else:
            T_eval = np.array(T_eval_all).reshape(-1)
            mu_eval = np.array(mu_eval_all).reshape(-1)

        pred = estimate_adrf_huling_local_poly_quantile(
            T_obs_raw=np.asarray(T_all[r], dtype=np.float64),
            Y_obs=np.asarray(Y_all[r], dtype=np.float64),
            weights=np.asarray(w, dtype=np.float64),
            t_grid_raw=np.asarray(T_eval, dtype=np.float64),
            a_h=float(args.a_h),
            alpha=float(args.alpha),
            degree=int(args.degree),
            t_transform=str(args.t_transform),
        )

        pred_t = torch.tensor(pred, dtype=torch.float32)
        mu_eval_t = torch.tensor(mu_eval, dtype=torch.float32)
        valid = torch.isfinite(pred_t) & torch.isfinite(mu_eval_t)
        if int(valid.sum().item()) == 0:
            print(f"[WARN] rep={r:03d} no valid predictions")
            continue

        mse = torch.mean((pred_t[valid] - mu_eval_t[valid]) ** 2).item()
        mae = torch.mean(torch.abs(pred_t[valid] - mu_eval_t[valid])).item()
        mse_list.append(float(mse))
        mae_list.append(float(mae))
        print(f"[HULING_V2] rep={r:03d} MSE={mse:.6g} MAE={mae:.6g}")

    mse_rms = float(np.sqrt(np.mean(np.square(np.array(mse_list))))) if mse_list else float("nan")
    mae_mean = float(np.mean(np.array(mae_list))) if mae_list else float("nan")
    print(f"[SUMMARY] MSE_mean={mse_rms:.6g} MAE_mean={mae_mean:.6g}")


if __name__ == "__main__":
    main()

