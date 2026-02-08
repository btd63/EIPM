from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import torch

from device_utils import select_device
from train_eipm import load_replications_from_npz, EIPM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="./datasets")
    p.add_argument(
        "--pattern",
        type=str,
        default="sim_nonlinear_dx5_ntr1000_nev10000_rpt100_tk5_ok5_pi0.0_seed42.npz",
    )
    p.add_argument("--ckpt_dir", type=str, default="./models/eipm_single")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--eval_n", type=int, default=0)
    p.add_argument("--max_reps", type=int, default=100)
    p.add_argument("--only_rep", type=int, default=-1)
    return p.parse_args()


def _nonzero_quantile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x_nz = x[x > 0]
    if x_nz.size == 0:
        return 0.0
    return float(np.quantile(x_nz, q))


def _pairwise_nonzero_quantile(T: np.ndarray, q: float) -> float:
    T = np.asarray(T, dtype=np.float64).reshape(-1)
    dist = np.abs(T[:, None] - T[None, :]).reshape(-1)
    dist = dist[dist > 0]
    if dist.size == 0:
        return 0.0
    return float(np.quantile(dist, q))


def _apply_t_transform(x: np.ndarray, kind: str) -> np.ndarray:
    if kind in (None, "identity"):
        return np.asarray(x, dtype=np.float64)
    if kind == "log1p":
        return np.log1p(np.asarray(x, dtype=np.float64))
    raise ValueError(f"Unknown t_transform: {kind}")

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


def main() -> None:
    args = parse_args()
    device = select_device(args.device)

    files = sorted(glob.glob(str(Path(args.data_dir) / args.pattern)))
    if not files:
        raise FileNotFoundError(f"No npz files found in {args.data_dir} with pattern {args.pattern}")

    npz_path = files[0]
    print(f"[INFO] Dataset path: {Path(npz_path).resolve()}")
    dataset_tag = Path(npz_path).stem
    ckpt_root = Path(args.ckpt_dir)
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
                msg = f"[WARN] ckpt dataset mismatch: {ckpt_npz} != {npz_path}"
                if args.only_rep >= 0:
                    raise RuntimeError(msg)
                print(msg)
                continue
        best_params = ckpt.get("best_params")
        if best_params is None:
            raise KeyError(f"best_params not found in checkpoint: {ckpt_path}")
        fixed_params = ckpt.get("fixed_params", {})
        t_transform = ckpt.get("t_transform", "identity")
        std = ckpt.get("standardize")
        if std is None:
            raise KeyError(f"standardize stats not found in checkpoint: {ckpt_path}")

        script_args = ckpt.get("script_args", {})
        depth = int(script_args.get("depth", 2))
        width = int(script_args.get("width", 128))

        expected_in = int(rep.d_X) + 1
        ckpt_in = int(ckpt["model_state"]["net.0.weight"].shape[1])
        if ckpt_in != expected_in:
            msg = f"[WARN] ckpt input_dim mismatch: {ckpt_in} != {expected_in}"
            if args.only_rep >= 0:
                raise RuntimeError(msg)
            print(msg)
            continue

        model = EIPM(input_dim=expected_in, hidden=width, n_layers=depth)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        model.eval()

        X = torch.tensor(rep.X, dtype=torch.float32, device=device)
        T_raw = torch.tensor(rep.T, dtype=torch.float32, device=device)
        if t_transform == "log1p":
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

        a_h = float(np.exp(best_params["log_a_h"]))
        alpha = float(fixed_params.get("alpha", best_params.get("alpha", 0.05)))

        T_obs_np = _apply_t_transform(rep.T, t_transform)
        T_eval_np = _apply_t_transform(T_eval, t_transform)
        t_mean_np = float(std["t_mean"])
        t_std_np = float(std["t_std"])
        T_obs_scaled = (T_obs_np.reshape(-1) - t_mean_np) / t_std_np
        T_eval_scaled = (T_eval_np.reshape(-1) - t_mean_np) / t_std_np

        pred = estimate_adrf_local_poly(
            T_obs=np.asarray(T_obs_scaled, dtype=np.float64),
            Y_obs=np.asarray(rep.Y, dtype=np.float64),
            log_weights=np.asarray(logw, dtype=np.float64),
            t_grid=np.asarray(T_eval_scaled, dtype=np.float64),
            a_h=a_h,
            alpha=alpha,
        )

        mse = float(np.mean((pred - mu_eval) ** 2))
        mae = float(np.mean(np.abs(pred - mu_eval)))
        mse_list.append(mse)
        mae_list.append(mae)

        print(f"[EIPM] rep={rep.rep_idx:03d} MSE={mse:.6g} MAE={mae:.6g}")

        if args.only_rep >= 0 and int(rep.rep_idx) == int(args.only_rep):
            print("[PRED] t, mu_hat, mu_true")
            for t_val, p_val, m_val in zip(T_eval, pred, mu_eval):
                print(f"{float(t_val):.6g}, {float(p_val):.6g}, {float(m_val):.6g}")
            break

    mse_mean = float(np.mean(np.array(mse_list))) if mse_list else float("nan")
    mae_mean = float(np.mean(np.array(mae_list))) if mae_list else float("nan")
    print(f"[SUMMARY] MSE_mean={mse_mean:.6g} MAE_mean={mae_mean:.6g}")


if __name__ == "__main__":
    main()
