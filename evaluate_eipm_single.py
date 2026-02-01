from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import torch

from train_eipm import load_replications_from_npz, EIPM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="./datasets")
    p.add_argument("--pattern", type=str, default="sim_*nonlinear*.npz")
    p.add_argument("--ckpt_dir", type=str, default="./models/eipm_single")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--eval_n", type=int, default=0)
    p.add_argument("--max_reps", type=int, default=100)
    return p.parse_args()


def estimate_adrf_locfit(
    T_obs: np.ndarray,
    Y_obs: np.ndarray,
    weights: np.ndarray,
    t_grid: np.ndarray,
    a_h: float,
    alpha: float,
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

    preds = []
    for t_val in t_grid:
        dist = np.abs(T_obs - float(t_val))
        dist_q = np.quantile(dist, 0.1)
        h_t = float(a_h) * float(dist_q ** float(alpha))
        if h_t <= 1e-8:
            h_t = 1e-8
        fit = locfit.locfit(ro.Formula("Y ~ lp(TRT, h=%f)" % h_t), weights=w, data=dfx)
        pred = base.predict(fit, newdata=ro.FloatVector([float(t_val)]), where="fitp")
        preds.append(float(pred[0]))

    return np.array(preds, dtype=np.float64)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    files = sorted(glob.glob(str(Path(args.data_dir) / args.pattern)))
    if not files:
        raise FileNotFoundError(f"No npz files found in {args.data_dir} with pattern {args.pattern}")

    npz_path = files[0]
    print(f"[INFO] Dataset path: {Path(npz_path).resolve()}")

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
        ckpt_path = Path(args.ckpt_dir) / f"eipm_single_nonlinear_rep{rep.rep_idx:03d}.pth"
        if not ckpt_path.exists():
            print(f"[WARN] missing checkpoint: {ckpt_path}")
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu")
        best_params = ckpt.get("best_params")
        if best_params is None:
            raise KeyError(f"best_params not found in checkpoint: {ckpt_path}")
        std = ckpt.get("standardize")
        if std is None:
            raise KeyError(f"standardize stats not found in checkpoint: {ckpt_path}")

        script_args = ckpt.get("script_args", {})
        depth = int(script_args.get("depth", 2))
        width = int(script_args.get("width", 128))

        model = EIPM(input_dim=int(rep.d_X) + 1, hidden=width, n_layers=depth)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        model.eval()

        X = torch.tensor(rep.X, dtype=torch.float32, device=device)
        T = torch.tensor(rep.T, dtype=torch.float32, device=device)

        x_mean = torch.tensor(np.asarray(std["x_mean"]), dtype=torch.float32, device=device).view(1, -1)
        x_std = torch.tensor(np.asarray(std["x_std"]), dtype=torch.float32, device=device).view(1, -1)
        t_mean = torch.tensor(float(std["t_mean"]), dtype=torch.float32, device=device)
        t_std = torch.tensor(float(std["t_std"]), dtype=torch.float32, device=device)

        X_std = (X - x_mean) / x_std
        X_scaled = X_std / np.sqrt(float(rep.d_X))
        T_scaled = (T.view(-1) - t_mean) / t_std

        with torch.no_grad():
            w = torch.exp(model(X_scaled, T_scaled)).detach().cpu().numpy().reshape(-1)
        w = w / np.mean(w)

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
        alpha = float(best_params["alpha"])

        pred = estimate_adrf_locfit(
            T_obs=np.asarray(rep.T, dtype=np.float64),
            Y_obs=np.asarray(rep.Y, dtype=np.float64),
            weights=np.asarray(w, dtype=np.float64),
            t_grid=np.asarray(T_eval, dtype=np.float64),
            a_h=a_h,
            alpha=alpha,
        )

        mse = float(np.mean((pred - mu_eval) ** 2))
        mae = float(np.mean(np.abs(pred - mu_eval)))
        mse_list.append(mse)
        mae_list.append(mae)

        print(f"[EIPM] rep={rep.rep_idx:03d} MSE={mse:.6g} MAE={mae:.6g}")

    mse_mean = float(np.mean(np.array(mse_list))) if mse_list else float("nan")
    mae_mean = float(np.mean(np.array(mae_list))) if mae_list else float("nan")
    print(f"[SUMMARY] MSE_mean={mse_mean:.6g} MAE_mean={mae_mean:.6g}")


if __name__ == "__main__":
    main()
