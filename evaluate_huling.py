from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="./datasets")
    p.add_argument("--npz", type=str, default="sim_nonlinear_dx50_ntr1000_nev10000_rpt100_tk50_ok50_pi0.0_seed42.npz")
    p.add_argument("--huling_dir", type=str, default="./models/huling")
    p.add_argument("--max_reps", type=int, default=100)
    p.add_argument("--only_rep", type=int, default=-1)
    p.add_argument("--degree", type=int, choices=[0, 1], default=1, help="Local polynomial degree (0=constant, 1=linear).")
    p.add_argument("--nn", type=float, default=0.7, help="Nearest-neighbor fraction for adaptive bandwidth (0,1].")
    p.add_argument("--t_transform", type=str, choices=["identity", "log1p"], default="identity")
    return p.parse_args()


def load_huling_weights(csv_path: Path) -> np.ndarray:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    w = np.loadtxt(str(csv_path), delimiter=",", skiprows=1)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    return w


def _apply_t_transform(x: np.ndarray, kind: Literal["identity", "log1p"]) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if kind == "identity":
        return x
    if kind == "log1p":
        return np.log1p(x)
    raise ValueError(f"Unknown t_transform: {kind}")


def estimate_adrf_huling_local_poly(
    T_obs: np.ndarray,
    Y_obs: np.ndarray,
    weights: np.ndarray,
    t_grid: np.ndarray,
    *,
    degree: int = 1,
    nn: float = 0.7,
    t_transform: Literal["identity", "log1p"] = "identity",
) -> np.ndarray:
    """
    Pure-Python (NumPy) local polynomial regression with adaptive (kNN) bandwidth.

    This is meant to replace R locfit usage for evaluation.
    It is not a byte-for-byte reproduction of locfit defaults, but preserves:
      - local polynomial smoothing (degree 0 or 1)
      - nonnegative kernel weights
      - adaptive bandwidth via nearest-neighbor fraction.
    """
    T_raw = np.asarray(T_obs, dtype=np.float64).reshape(-1)
    Y = np.asarray(Y_obs, dtype=np.float64).reshape(-1)
    w_base = np.asarray(weights, dtype=np.float64).reshape(-1)
    t_grid_raw = np.asarray(t_grid, dtype=np.float64).reshape(-1)

    if T_raw.shape[0] != Y.shape[0] or T_raw.shape[0] != w_base.shape[0]:
        raise ValueError("T_obs, Y_obs, weights must have the same length.")

    # weight degeneracy guard (copied idea from Huling/utils.R)
    if np.all(w_base == 0.0) or (np.mean(w_base == 0.0) > 0.95):
        return np.full_like(t_grid_raw, fill_value=np.nan, dtype=np.float64)

    if not (0.0 < float(nn) <= 1.0):
        raise ValueError("--nn must be in (0,1].")

    T = _apply_t_transform(T_raw, t_transform)
    t_grid_t = _apply_t_transform(t_grid_raw, t_transform)

    n = int(T.shape[0])
    k = int(np.ceil(float(nn) * n))
    k = max(2, min(k, n))

    # (n,m) distances and adaptive bandwidth h(t)=kth NN distance
    diff = (T[:, None] - t_grid_t[None, :]).astype(np.float64)
    dist = np.abs(diff)
    h = np.partition(dist, kth=k - 1, axis=0)[k - 1, :]
    h = np.maximum(h, 1e-8)

    # Gaussian kernel on transformed scale
    u = diff / h[None, :]
    K = np.exp(-0.5 * (u ** 2))
    W = (w_base[:, None] * K).astype(np.float64)

    if int(degree) == 0:
        num = np.sum(W * Y[:, None], axis=0)
        den = np.sum(W, axis=0)
        return num / (den + 1e-12)

    # local linear in closed form (avoids per-t matrix inversion)
    S0 = np.sum(W, axis=0)
    S1 = np.sum(W * diff, axis=0)
    S2 = np.sum(W * diff * diff, axis=0)
    T0 = np.sum(W * Y[:, None], axis=0)
    T1 = np.sum(W * diff * Y[:, None], axis=0)
    denom = S0 * S2 - S1 * S1
    return (S2 * T0 - S1 * T1) / (denom + 1e-12)


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

    X_all = np.array(data["X_train"])
    T_all = np.array(data["T_train"])
    Y_all = np.array(data["Y_train"])

    n_rpt = X_all.shape[0]
    n_run = min(int(args.max_reps), n_rpt)

    stem = npz_path.stem
    huling_dir = Path(args.huling_dir)

    mse_list = []
    mae_list = []
    bias_list = []

    print(f"[INFO] Dataset: {npz_path.name}")
    print(f"[INFO] Using huling_dir: {huling_dir}")
    print(f"[INFO] Replications: {n_rpt} (running {n_run})")

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
            T_eval = torch.tensor(T_eval_all[r], dtype=torch.float32)
            mu_eval = torch.tensor(mu_eval_all[r], dtype=torch.float32)
        else:
            T_eval = torch.tensor(T_eval_all, dtype=torch.float32)
            mu_eval = torch.tensor(mu_eval_all, dtype=torch.float32)

        pred = estimate_adrf_huling_local_poly(
            T_obs=np.asarray(T_all[r], dtype=np.float64),
            Y_obs=np.asarray(Y_all[r], dtype=np.float64),
            weights=np.asarray(w, dtype=np.float64),
            t_grid=np.asarray(T_eval, dtype=np.float64),
            degree=int(args.degree),
            nn=float(args.nn),
            t_transform=str(args.t_transform),
        )
        pred_t = torch.tensor(pred, dtype=torch.float32)
        mu_eval_t = torch.tensor(mu_eval, dtype=torch.float32)
        valid = torch.isfinite(pred_t) & torch.isfinite(mu_eval_t)
        if valid.sum() == 0:
            print(f"[WARN] rep={r:03d} no valid predictions")
            continue

        if r == 15:
            sample_idx = torch.linspace(0, T_eval.numel() - 1, steps=5).long()
            t_s = T_eval[sample_idx].detach().cpu().numpy()
            p_s = pred[sample_idx]
            m_s = mu_eval[sample_idx]
            print("[SAMPLE] t, mu_hat, mu_true")
            for t_val, p_val, m_val in zip(t_s, p_s, m_s):
                print(f"{float(t_val):.6g}, {float(p_val):.6g}, {float(m_val):.6g}")

        diff = pred_t[valid] - mu_eval_t[valid]
        mse = torch.mean(diff ** 2).item()
        mae = torch.mean(torch.abs(diff)).item()
        bias = torch.mean(diff).item()

        mse_list.append(mse)
        mae_list.append(mae)
        bias_list.append(bias)

        print(f"[HULING] rep={r:03d} MSE={mse:.6g} MAE={mae:.6g} BIAS={bias:.6g}")

    mse_rms = float(np.sqrt(np.mean(np.square(np.array(mse_list))))) if mse_list else float("nan")
    mae_mean = float(np.mean(mae_list)) if mae_list else float("nan")
    bias_mean = float(np.mean(bias_list)) if bias_list else float("nan")
    print(f"[SUMMARY] MSE_mean={mse_rms:.6g} MAE_mean={mae_mean:.6g} BIAS_mean={bias_mean:.6g}")


if __name__ == "__main__":
    main()
