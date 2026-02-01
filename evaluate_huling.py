from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Tuple

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="./datasets")
    p.add_argument("--npz", type=str, default="sim_nonlinear_dx50_ntr1000_nev10000_rpt100_tk50_ok50_pi0.0_seed42.npz")
    p.add_argument("--huling_dir", type=str, default="./models/huling")
    p.add_argument("--max_reps", type=int, default=100)
    return p.parse_args()


def load_huling_weights(csv_path: Path) -> np.ndarray:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    w = np.loadtxt(str(csv_path), delimiter=",", skiprows=1)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    return w


def estimate_adrf_huling_locfit(
    T_obs: np.ndarray,
    Y_obs: np.ndarray,
    weights: np.ndarray,
    t_grid: np.ndarray,
) -> np.ndarray:
    try:
        from rpy2 import robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects import packages as rpackages
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("rpy2 is required to call locfit from Python.") from exc

    if not rpackages.isinstalled("locfit"):
        raise RuntimeError("R package 'locfit' is required.")

    numpy2ri.activate()

    locfit = rpackages.importr("locfit")
    base = rpackages.importr("base")

    dfx = ro.DataFrame({"Y": ro.FloatVector(Y_obs), "TRT": ro.FloatVector(T_obs)})
    w = ro.FloatVector(weights)

    # use locfit default smoothing (nearest-neighbor based)
    fit = locfit.locfit(ro.Formula("Y ~ lp(TRT)"), weights=w, data=dfx)
    preds = base.predict(fit, newdata=ro.FloatVector(t_grid), where="fitp")
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

    X_all = np.array(data["X_train"])
    T_all = np.array(data["T_train"])
    Y_all = np.array(data["Y_train"])

    n_rpt = X_all.shape[0]
    n_run = min(int(args.max_reps), n_rpt)

    stem = npz_path.stem
    huling_dir = Path(args.huling_dir)

    mse_list = []
    mae_list = []

    print(f"[INFO] Dataset: {npz_path.name}")
    print(f"[INFO] Using huling_dir: {huling_dir}")
    print(f"[INFO] Replications: {n_rpt} (running {n_run})")

    for r in range(n_run):
        w_path = huling_dir / f"{stem}_rep{r:03d}_huling_weights.csv"
        if not w_path.exists():
            print(f"[WARN] missing weights: {w_path}")
            continue

        w = load_huling_weights(w_path)
        T = torch.tensor(T_all[r], dtype=torch.float32)
        Y = torch.tensor(Y_all[r], dtype=torch.float32)
        w_t = torch.tensor(w, dtype=torch.float32)

        if T_eval_all.ndim == 2:
            T_eval = torch.tensor(T_eval_all[r], dtype=torch.float32)
            mu_eval = torch.tensor(mu_eval_all[r], dtype=torch.float32)
        else:
            T_eval = torch.tensor(T_eval_all, dtype=torch.float32)
            mu_eval = torch.tensor(mu_eval_all, dtype=torch.float32)

        pred = estimate_adrf_huling_locfit(
            T_obs=np.asarray(T_all[r], dtype=np.float64),
            Y_obs=np.asarray(Y_all[r], dtype=np.float64),
            weights=np.asarray(w, dtype=np.float64),
            t_grid=np.asarray(T_eval, dtype=np.float64),
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

        mse = torch.mean((pred_t[valid] - mu_eval_t[valid]) ** 2).item()
        mae = torch.mean(torch.abs(pred_t[valid] - mu_eval_t[valid])).item()

        mse_list.append(mse)
        mae_list.append(mae)

        print(f"[HULING] rep={r:03d} MSE={mse:.6g} MAE={mae:.6g}")

    mse_rms = float(np.sqrt(np.mean(np.square(np.array(mse_list))))) if mse_list else float("nan")
    mae_mean = float(np.mean(mae_list)) if mae_list else float("nan")
    print(f"[SUMMARY] MSE_mean={mse_rms:.6g} MAE_mean={mae_mean:.6g}")


if __name__ == "__main__":
    main()
