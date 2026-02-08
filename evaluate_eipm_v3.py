from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import torch

from device_utils import select_device
from train_eipm import load_replications_from_npz, EIPM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="./datasets")
    p.add_argument("--pattern", type=str, default="sim_*nonlinear*.npz")
    p.add_argument("--ckpt_dir", type=str, default="./models/eipm_single_v3")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--eval_n", type=int, default=0)
    p.add_argument("--max_reps", type=int, default=100)
    p.add_argument("--organize_ckpts", action="store_true", help="Move mixed checkpoints into dataset subfolders.")
    p.add_argument("--plot_h", action="store_true", help="Save h(t) curve plot for a rep.")
    p.add_argument("--plot_rep", type=int, default=-1, help="Replication index for h(t) plot.")
    p.add_argument("--plot_h_n", type=int, default=200, help="Number of grid points for h(t) plot.")
    p.add_argument("--only_rep", type=int, default=-1, help="Evaluate only this replication index.")
    p.add_argument("--top_k_errors", type=int, default=0, help="Print top-k |mu_hat-mu_true| rows.")
    p.add_argument("--debug_topk", action="store_true", help="Print local-linear diagnostics for top-k errors.")
    p.add_argument("--plot_train_hist", action="store_true", help="Save histogram of T_train for a rep.")
    p.add_argument("--hist_rep", type=int, default=-1, help="Replication index for T_train histogram.")
    p.add_argument("--hist_bins", type=int, default=50, help="Number of bins for T_train histogram.")
    return p.parse_args()


def _nonzero_quantile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x_nz = x[x > 0]
    if x_nz.size == 0:
        return 0.0
    return float(np.quantile(x_nz, q))


def _apply_t_transform(x: np.ndarray, kind: str) -> np.ndarray:
    if kind in (None, "identity"):
        return np.asarray(x, dtype=np.float64)
    if kind == "log1p":
        return np.log1p(np.asarray(x, dtype=np.float64))
    if kind == "cdf_sigmoid":
        raise ValueError("cdf_sigmoid requires tstar_params; use transform_t_to_star().")
    raise ValueError(f"Unknown t_transform: {kind}")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(u: np.ndarray) -> np.ndarray:
    return np.log(u / (1.0 - u))


def transform_t_to_star(t: np.ndarray, params: dict) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    T_sorted = np.asarray(params["T_sorted"], dtype=np.float64)
    u_grid = np.asarray(params["u_grid"], dtype=np.float64)
    Tmin = float(T_sorted[0])
    Tmax = float(T_sorted[-1])
    u0 = float(params["u0"])
    u1 = float(params["u1"])
    x0 = float(params["x0"])
    x1 = float(params["x1"])
    s_left = float(params["s_left"])
    s_right = float(params["s_right"])

    u = np.empty_like(t, dtype=np.float64)
    mask_mid = (t >= Tmin) & (t <= Tmax)
    if np.any(mask_mid):
        u[mask_mid] = np.interp(t[mask_mid], T_sorted, u_grid)
    mask_left = t < Tmin
    if np.any(mask_left):
        u[mask_left] = _sigmoid(x0 + (t[mask_left] - Tmin) * s_left)
    mask_right = t > Tmax
    if np.any(mask_right):
        u[mask_right] = _sigmoid(x1 + (t[mask_right] - Tmax) * s_right)

    eps = 1e-8
    u = np.clip(u, eps, 1.0 - eps)
    return 2.0 * u - 1.0


def transform_star_to_t(t_star: np.ndarray, params: dict) -> np.ndarray:
    t_star = np.asarray(t_star, dtype=np.float64)
    T_sorted = np.asarray(params["T_sorted"], dtype=np.float64)
    u_grid = np.asarray(params["u_grid"], dtype=np.float64)
    Tmin = float(T_sorted[0])
    Tmax = float(T_sorted[-1])
    u0 = float(params["u0"])
    u1 = float(params["u1"])
    x0 = float(params["x0"])
    x1 = float(params["x1"])
    s_left = float(params["s_left"])
    s_right = float(params["s_right"])

    u = (t_star + 1.0) * 0.5
    eps = 1e-8
    u = np.clip(u, eps, 1.0 - eps)

    t = np.empty_like(u, dtype=np.float64)
    mask_mid = (u >= u0) & (u <= u1)
    if np.any(mask_mid):
        t[mask_mid] = np.interp(u[mask_mid], u_grid, T_sorted)
    mask_left = u < u0
    if np.any(mask_left):
        t[mask_left] = Tmin + (_logit(u[mask_left]) - x0) / max(s_left, eps)
    mask_right = u > u1
    if np.any(mask_right):
        t[mask_right] = Tmax + (_logit(u[mask_right]) - x1) / max(s_right, eps)
    return t


def _pairwise_nonzero_quantile(T: np.ndarray, q: float) -> float:
    T = np.asarray(T, dtype=np.float64).reshape(-1)
    dist = np.abs(T[:, None] - T[None, :]).reshape(-1)
    dist = dist[dist > 0]
    if dist.size == 0:
        return 0.0
    return float(np.quantile(dist, q))


def _knn_bandwidth(T_obs: np.ndarray, t_grid: np.ndarray, nn: float) -> np.ndarray:
    T_obs = np.asarray(T_obs, dtype=np.float64).reshape(-1)
    t_grid = np.asarray(t_grid, dtype=np.float64).reshape(-1)
    n = int(T_obs.shape[0])
    if n == 0:
        return np.full_like(t_grid, fill_value=1e-8, dtype=np.float64)
    k = int(np.ceil(float(nn) * float(n)))
    k = max(2, min(k, n))
    dist = np.abs(T_obs[:, None] - t_grid[None, :])
    h = np.partition(dist, kth=k - 1, axis=0)[k - 1, :]
    return np.maximum(h, 1e-8)


def estimate_adrf_local_poly(
    T_obs: np.ndarray,
    Y_obs: np.ndarray,
    log_weights: np.ndarray,
    t_grid: np.ndarray,
    a_h: float,
    alpha: float,
) -> np.ndarray:
    preds = []
    T_obs = np.asarray(T_obs, dtype=np.float64).reshape(-1)
    Y_obs = np.asarray(Y_obs, dtype=np.float64).reshape(-1)
    log_weights = np.asarray(log_weights, dtype=np.float64).reshape(-1)
    global_q = _pairwise_nonzero_quantile(T_obs, 0.1)

    for t_val in t_grid:
        t0 = float(t_val)
        diff = T_obs - t0
        dist = np.abs(diff)
        dist_q = _nonzero_quantile(dist, 0.1)
        base = float(dist_q + global_q)
        if base <= 0.0:
            base = 1e-8
        h_t = float(a_h) * float(base ** float(alpha))
        if h_t <= 1e-8:
            h_t = 1e-8
        u = diff / h_t
        logk = -0.5 * (u ** 2)
        logw_eff = log_weights + logk
        max_log = np.max(logw_eff)
        w_eff = np.exp(logw_eff - max_log)
        s = float(np.sum(w_eff))
        if np.isfinite(s) and s > 0.0:
            w_eff = w_eff / s
        X_lp = np.stack([np.ones_like(diff), diff], axis=1)  # p=1
        W = w_eff[:, None]
        XW = X_lp * W
        xtwx = X_lp.T @ XW
        xtwy = XW.T @ Y_obs
        beta = np.linalg.pinv(xtwx) @ xtwy
        preds.append(float(beta[0]))

    return np.array(preds, dtype=np.float64)


def estimate_adrf_local_poly_knn(
    T_obs: np.ndarray,
    Y_obs: np.ndarray,
    log_weights: np.ndarray,
    t_grid: np.ndarray,
    nn: float,
    a_h: float,
) -> np.ndarray:
    preds = []
    T_obs = np.asarray(T_obs, dtype=np.float64).reshape(-1)
    Y_obs = np.asarray(Y_obs, dtype=np.float64).reshape(-1)
    log_weights = np.asarray(log_weights, dtype=np.float64).reshape(-1)
    t_grid = np.asarray(t_grid, dtype=np.float64).reshape(-1)

    h_all = _knn_bandwidth(T_obs, t_grid, nn=float(nn))
    for t0, h_t in zip(t_grid, h_all):
        diff = T_obs - float(t0)
        h_t = float(a_h) * float(h_t)
        if h_t <= 1e-8:
            h_t = 1e-8
        u = diff / h_t
        logk = -0.5 * (u ** 2)
        logw_eff = log_weights + logk
        max_log = np.max(logw_eff)
        w_eff = np.exp(logw_eff - max_log)
        s = float(np.sum(w_eff))
        if np.isfinite(s) and s > 0.0:
            w_eff = w_eff / s
        X_lp = np.stack([np.ones_like(diff), diff], axis=1)
        W = w_eff[:, None]
        XW = X_lp * W
        xtwx = X_lp.T @ XW
        xtwy = XW.T @ Y_obs
        beta = np.linalg.pinv(xtwx) @ xtwy
        preds.append(float(beta[0]))

    return np.array(preds, dtype=np.float64)


def _h_curve(
    T_obs: np.ndarray,
    t_grid: np.ndarray,
    a_h: float,
    alpha: float,
) -> np.ndarray:
    T_obs = np.asarray(T_obs, dtype=np.float64).reshape(-1)
    t_grid = np.asarray(t_grid, dtype=np.float64).reshape(-1)
    global_q = _pairwise_nonzero_quantile(T_obs, 0.1)
    h_vals = []
    for t_val in t_grid:
        diff = T_obs - float(t_val)
        dist_q = _nonzero_quantile(np.abs(diff), 0.1)
        base = float(dist_q + global_q)
        if base <= 0.0:
            base = 1e-8
        h_t = float(a_h) * float(base ** float(alpha))
        if h_t <= 1e-8:
            h_t = 1e-8
        h_vals.append(h_t)
    return np.array(h_vals, dtype=np.float64)


def _h_curve_knn(T_obs: np.ndarray, t_grid: np.ndarray, nn: float, a_h: float) -> np.ndarray:
    return _knn_bandwidth(T_obs, t_grid, nn=float(nn)) * float(a_h)


def _local_linear_debug(
    T_obs: np.ndarray,
    Y_obs: np.ndarray,
    logw: np.ndarray,
    t0: float,
    a_h: float,
    alpha: float,
) -> dict:
    T = np.asarray(T_obs, dtype=np.float64).reshape(-1)
    Y = np.asarray(Y_obs, dtype=np.float64).reshape(-1)
    logw = np.asarray(logw, dtype=np.float64).reshape(-1)

    diff = T - float(t0)
    dist = np.abs(diff)
    dist_q = _nonzero_quantile(dist, 0.1)
    global_q = _pairwise_nonzero_quantile(T, 0.1)
    base = float(dist_q + global_q)
    if base <= 0.0:
        base = 1e-8
    h = float(a_h) * float(base ** float(alpha))
    if h <= 1e-8:
        h = 1e-8

    u = diff / h
    logk = -0.5 * (u ** 2)
    logw_eff = logw + logk
    max_log = np.max(logw_eff)
    w = np.exp(logw_eff - max_log)
    s = float(np.sum(w))
    if np.isfinite(s) and s > 0.0:
        w = w / s

    s0 = float(np.sum(w))
    s1 = float(np.sum(w * diff))
    s2 = float(np.sum(w * diff * diff))
    denom = (s0 * s2 - s1 * s1)

    l = w * (s2 - diff * s1) / (denom + 1e-12)
    mu_hat = float(np.sum(l * Y))
    ess = float(1.0 / np.sum(w * w)) if np.any(w) else 0.0
    X_lp = np.stack([np.ones_like(diff), diff], axis=1)
    xtwx = X_lp.T @ (X_lp * w[:, None])
    try:
        cond = float(np.linalg.cond(xtwx))
    except Exception:
        cond = float("inf")

    return {
        "t": float(t0),
        "h": float(h),
        "denom": float(denom),
        "min_l": float(np.min(l)),
        "max_l": float(np.max(l)),
        "neg_cnt": int(np.sum(l < 0)),
        "ess": float(ess),
        "cond": float(cond),
        "mu_hat": float(mu_hat),
    }


def _local_linear_debug_knn(
    T_obs: np.ndarray,
    Y_obs: np.ndarray,
    logw: np.ndarray,
    t0: float,
    nn: float,
    a_h: float,
) -> dict:
    T = np.asarray(T_obs, dtype=np.float64).reshape(-1)
    Y = np.asarray(Y_obs, dtype=np.float64).reshape(-1)
    logw = np.asarray(logw, dtype=np.float64).reshape(-1)

    h = float(_knn_bandwidth(T, np.array([t0], dtype=np.float64), nn=float(nn))[0])
    h = float(a_h) * float(h)
    if h <= 1e-8:
        h = 1e-8

    diff = T - float(t0)
    u = diff / h
    logk = -0.5 * (u ** 2)
    logw_eff = logw + logk
    max_log = np.max(logw_eff)
    w = np.exp(logw_eff - max_log)
    s = float(np.sum(w))
    if np.isfinite(s) and s > 0.0:
        w = w / s

    s0 = float(np.sum(w))
    s1 = float(np.sum(w * diff))
    s2 = float(np.sum(w * diff * diff))
    denom = (s0 * s2 - s1 * s1)

    l = w * (s2 - diff * s1) / (denom + 1e-12)
    mu_hat = float(np.sum(l * Y))
    ess = float(1.0 / np.sum(w * w)) if np.any(w) else 0.0
    X_lp = np.stack([np.ones_like(diff), diff], axis=1)
    xtwx = X_lp.T @ (X_lp * w[:, None])
    try:
        cond = float(np.linalg.cond(xtwx))
    except Exception:
        cond = float("inf")

    return {
        "t": float(t0),
        "h": float(h),
        "denom": float(denom),
        "min_l": float(np.min(l)),
        "max_l": float(np.max(l)),
        "neg_cnt": int(np.sum(l < 0)),
        "ess": float(ess),
        "cond": float(cond),
        "mu_hat": float(mu_hat),
    }

def _save_h_plot(t_grid: np.ndarray, h_vals: np.ndarray, out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for plotting h(t).") from exc

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t_grid, h_vals, color="black", linewidth=1.5)
    ax.set_xlabel("t")
    ax.set_ylabel("h(t)")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_hist_plot(t_vals: np.ndarray, out_path: Path, bins: int, rep_idx: int, *, title: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for plotting histogram.") from exc

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(t_vals, bins=int(bins), color="steelblue", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel("T")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
def _organize_checkpoints(ckpt_root: Path) -> None:
    ckpt_root.mkdir(parents=True, exist_ok=True)
    moved = 0
    for ckpt_path in ckpt_root.glob("*.pth"):
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
        except Exception:
            continue
        npz_path = ckpt.get("npz_path")
        if not npz_path:
            continue
        dataset_tag = Path(npz_path).stem
        target_dir = ckpt_root / dataset_tag
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / ckpt_path.name
        if target_path == ckpt_path:
            continue
        os.replace(ckpt_path, target_path)
        moved += 1
    if moved > 0:
        print(f"[INFO] Organized {moved} checkpoint(s) under {ckpt_root}")


def main() -> None:
    args = parse_args()
    device = select_device(args.device)

    files = sorted(glob.glob(str(Path(args.data_dir) / args.pattern)))
    if not files:
        raise FileNotFoundError(f"No npz files found in {args.data_dir} with pattern {args.pattern}")

    npz_path = files[0]
    ckpt_root = Path(args.ckpt_dir)
    if args.organize_ckpts:
        _organize_checkpoints(ckpt_root)
    dataset_tag = Path(npz_path).stem
    ckpt_dir = ckpt_root / dataset_tag if (ckpt_root / dataset_tag).exists() else ckpt_root
    reps = load_replications_from_npz(npz_path)
    data = np.load(npz_path, allow_pickle=True)
    if "T_eval" not in data.files or "mu_eval" not in data.files:
        raise KeyError("T_eval or mu_eval not found in dataset")
    T_eval_all = np.array(data["T_eval"])
    mu_eval_all = np.array(data["mu_eval"])

    n_reps = min(int(args.max_reps), len(reps))
    mse_list = []
    mae_list = []
    bias_list = []

    for rep_idx in range(n_reps):
        rep = reps[int(rep_idx)]
        if args.only_rep >= 0 and int(rep.rep_idx) != int(args.only_rep):
            continue
        ckpt_path = ckpt_dir / f"eipm_single_nonlinear_rep{rep.rep_idx:03d}.pth"
        if not ckpt_path.exists():
            print(f"[WARN] missing checkpoint: {ckpt_path}")
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu")
        ckpt_npz = ckpt.get("npz_path")
        if ckpt_npz is not None:
            if Path(ckpt_npz).name != Path(npz_path).name:
                print(f"[WARN] ckpt dataset mismatch: {ckpt_npz} != {npz_path}")
                continue
        best_params = ckpt.get("best_params")
        if best_params is None:
            raise KeyError(f"best_params not found in checkpoint: {ckpt_path}")
        fixed_params = ckpt.get("fixed_params", {})
        std = ckpt.get("standardize")
        if std is None:
            raise KeyError(f"standardize stats not found in checkpoint: {ckpt_path}")
        t_transform = ckpt.get("t_transform", "identity")
        tstar_params = ckpt.get("tstar_params") if t_transform == "cdf_sigmoid" else None
        bandwidth = ckpt.get("bandwidth", {})
        h_type = str(bandwidth.get("type", "quantile"))
        if h_type == "knn":
            nn_val = float(bandwidth.get("nn", fixed_params.get("nn", 0.7)))
        else:
            nn_val = None

        script_args = ckpt.get("script_args", {})
        depth = int(script_args.get("depth", 2))
        width = int(script_args.get("width", 128))

        expected_in = int(rep.d_X) + 1
        ckpt_in = int(ckpt["model_state"]["net.0.weight"].shape[1])
        if ckpt_in != expected_in:
            print(f"[WARN] ckpt input_dim mismatch: {ckpt_in} != {expected_in}")
            continue
        model = EIPM(input_dim=expected_in, hidden=width, n_layers=depth)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        model.eval()

        X = torch.tensor(rep.X, dtype=torch.float32, device=device)
        T_raw = torch.tensor(rep.T, dtype=torch.float32, device=device)
        if t_transform == "cdf_sigmoid":
            if tstar_params is None:
                raise KeyError("tstar_params missing for cdf_sigmoid transform.")
            T_star = transform_t_to_star(rep.T, tstar_params)
            T = torch.tensor(T_star, dtype=torch.float32, device=device)
        elif t_transform == "log1p":
            T = torch.log1p(T_raw)
        elif t_transform in (None, "identity"):
            T = T_raw
        else:
            raise ValueError(f"Unknown t_transform in checkpoint: {t_transform}")

        x_mean = torch.tensor(np.asarray(std["x_mean"]), dtype=torch.float32, device=device).view(1, -1)
        x_std = torch.tensor(np.asarray(std["x_std"]), dtype=torch.float32, device=device).view(1, -1)
        t_mean = torch.tensor(float(std["t_mean"]), dtype=torch.float32, device=device)
        t_std = torch.tensor(float(std["t_std"]), dtype=torch.float32, device=device)

        X_std = (X - x_mean) / x_std
        X_scaled = X_std / np.sqrt(float(rep.d_X))
        T_scaled = (T.view(-1) - t_mean) / t_std

        with torch.no_grad():
            logw = model(X_scaled, T_scaled).detach().cpu().numpy().reshape(-1)

        if T_eval_all.ndim == 2:
            T_eval = np.array(T_eval_all[rep_idx]).reshape(-1)
            mu_eval = np.array(mu_eval_all[rep_idx]).reshape(-1)
        else:
            T_eval = np.array(T_eval_all).reshape(-1)
            mu_eval = np.array(mu_eval_all).reshape(-1)

        if args.eval_n > 0 and args.eval_n < T_eval.shape[0]:
            T_eval = T_eval[:args.eval_n]
            mu_eval = mu_eval[:args.eval_n]

        if h_type == "knn":
            a_h = float(np.exp(best_params.get("log_a_h", 0.0)))
            alpha = None
        else:
            a_h = float(np.exp(best_params["log_a_h"]))
            alpha = float(fixed_params.get("alpha", best_params.get("alpha", 0.05)))

        if t_transform == "cdf_sigmoid":
            T_obs_np = transform_t_to_star(rep.T, tstar_params)
            T_eval_np = transform_t_to_star(T_eval, tstar_params)
        else:
            T_obs_np = _apply_t_transform(rep.T, t_transform)
            T_eval_np = _apply_t_transform(T_eval, t_transform)
        t_mean_np = float(std["t_mean"])
        t_std_np = float(std["t_std"])
        T_obs_scaled = (T_obs_np.reshape(-1) - t_mean_np) / t_std_np
        T_eval_scaled = (T_eval_np.reshape(-1) - t_mean_np) / t_std_np

        if args.plot_h and (args.plot_rep < 0 or int(rep.rep_idx) == int(args.plot_rep)):
            t_min = float(np.min(T_eval))
            t_max = float(np.max(T_eval))
            n_plot = int(max(10, args.plot_h_n))
            t_grid = np.linspace(t_min, t_max, n_plot)
            if t_transform == "cdf_sigmoid":
                t_grid_star = transform_t_to_star(t_grid, tstar_params)
                t_grid_scaled = (t_grid_star.reshape(-1) - t_mean_np) / t_std_np
            else:
                t_grid_scaled = (_apply_t_transform(t_grid, t_transform).reshape(-1) - t_mean_np) / t_std_np
            if h_type == "knn":
                h_vals = _h_curve_knn(
                    T_obs=np.asarray(T_obs_scaled, dtype=np.float64),
                    t_grid=np.asarray(t_grid_scaled, dtype=np.float64),
                    nn=float(nn_val),
                    a_h=float(a_h),
                )
            else:
                h_vals = _h_curve(
                    T_obs=np.asarray(T_obs_scaled, dtype=np.float64),
                    t_grid=np.asarray(t_grid_scaled, dtype=np.float64),
                    a_h=a_h,
                    alpha=alpha,
                )
            out_path = ckpt_dir / f"h_curve_rep{rep.rep_idx:03d}.png"
            _save_h_plot(t_grid, h_vals, out_path)
            print(f"[INFO] Saved h(t) plot: {out_path}")

        if args.plot_train_hist and (args.hist_rep < 0 or int(rep.rep_idx) == int(args.hist_rep)):
            out_path = ckpt_dir / f"hist_T_train_rep{rep.rep_idx:03d}.png"
            _save_hist_plot(
                t_vals=np.asarray(rep.T, dtype=np.float64),
                out_path=out_path,
                bins=int(args.hist_bins),
                rep_idx=int(rep.rep_idx),
                title=f"T_train histogram (rep {rep.rep_idx})",
            )
            print(f"[INFO] Saved T_train hist: {out_path}")

            out_path_log = ckpt_dir / f"hist_log1p_T_train_rep{rep.rep_idx:03d}.png"
            _save_hist_plot(
                t_vals=np.log1p(np.asarray(rep.T, dtype=np.float64)),
                out_path=out_path_log,
                bins=int(args.hist_bins),
                rep_idx=int(rep.rep_idx),
                title=f"log1p(T_train) histogram (rep {rep.rep_idx})",
            )
            print(f"[INFO] Saved log1p(T_train) hist: {out_path_log}")

        if h_type == "knn":
            pred = estimate_adrf_local_poly_knn(
                T_obs=np.asarray(T_obs_scaled, dtype=np.float64),
                Y_obs=np.asarray(rep.Y, dtype=np.float64),
                log_weights=np.asarray(logw, dtype=np.float64),
                t_grid=np.asarray(T_eval_scaled, dtype=np.float64),
                nn=float(nn_val),
                a_h=float(a_h),
            )
        else:
            pred = estimate_adrf_local_poly(
                T_obs=np.asarray(T_obs_scaled, dtype=np.float64),
                Y_obs=np.asarray(rep.Y, dtype=np.float64),
                log_weights=np.asarray(logw, dtype=np.float64),
                t_grid=np.asarray(T_eval_scaled, dtype=np.float64),
                a_h=a_h,
                alpha=alpha,
            )

        diff = pred - mu_eval
        mse = float(np.mean(diff ** 2))
        mae = float(np.mean(np.abs(diff)))
        bias = float(np.mean(diff))

        if int(rep.rep_idx) == 15:
            sample_idx = np.linspace(0, len(T_eval) - 1, 5).astype(int)
            print("[SAMPLE] t, mu_hat, mu_true")
            for i in sample_idx:
                print(f"{float(T_eval[i]):.6g}, {float(pred[i]):.6g}, {float(mu_eval[i]):.6g}")

        mse_list.append(mse)
        mae_list.append(mae)
        bias_list.append(bias)

        print(f"[EIPM] rep={rep.rep_idx:03d} MSE={mse:.6g} MAE={mae:.6g} BIAS={bias:.6g}")

        if args.top_k_errors and int(args.top_k_errors) > 0:
            k = int(args.top_k_errors)
            abs_err = np.abs(pred - mu_eval)
            k = min(k, abs_err.shape[0])
            idx = np.argsort(abs_err)[-k:][::-1]
            print("[TOPK] t, mu_hat, mu_true, abs_err")
            for i in idx:
                print(
                    f"{float(T_eval[i]):.6g}, {float(pred[i]):.6g}, "
                    f"{float(mu_eval[i]):.6g}, {float(abs_err[i]):.6g}"
                )
            if args.debug_topk:
                print("[DBG] t, h, denom, min_l, max_l, neg_cnt, ess, cond, mu_hat_dbg")
                for i in idx:
                    if h_type == "knn":
                        dbg = _local_linear_debug_knn(
                            T_obs=np.asarray(T_obs_scaled, dtype=np.float64),
                            Y_obs=np.asarray(rep.Y, dtype=np.float64),
                            logw=np.asarray(logw, dtype=np.float64),
                            t0=float(T_eval_scaled[i]),
                            nn=float(nn_val),
                            a_h=float(a_h),
                        )
                    else:
                        dbg = _local_linear_debug(
                            T_obs=np.asarray(T_obs_scaled, dtype=np.float64),
                            Y_obs=np.asarray(rep.Y, dtype=np.float64),
                            logw=np.asarray(logw, dtype=np.float64),
                            t0=float(T_eval_scaled[i]),
                            a_h=a_h,
                            alpha=alpha,
                        )
                    print(
                        f"{dbg['t']:.6g}, {dbg['h']:.6g}, {dbg['denom']:.6g}, "
                        f"{dbg['min_l']:.6g}, {dbg['max_l']:.6g}, {dbg['neg_cnt']}, "
                        f"{dbg['ess']:.6g}, {dbg['cond']:.6g}, {dbg['mu_hat']:.6g}"
                    )

    mse_rms = float(np.sqrt(np.mean(np.square(np.array(mse_list))))) if mse_list else float("nan")
    mae_mean = float(np.mean(np.array(mae_list))) if mae_list else float("nan")
    bias_mean = float(np.mean(np.array(bias_list))) if bias_list else float("nan")
    print(f"[SUMMARY] MSE_mean={mse_rms:.6g} MAE_mean={mae_mean:.6g} BIAS_mean={bias_mean:.6g}")


if __name__ == "__main__":
    main()
