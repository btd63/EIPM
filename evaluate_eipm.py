from __future__ import annotations

import argparse
import glob
import math
import time
from pathlib import Path

import numpy as np
import torch

from train_eipm import load_replications_from_npz, mu_hat_at_t, h0_t, EIPM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="./datasets")
    p.add_argument("--pattern", type=str, default="sim_*nonlinear*.npz")
    p.add_argument("--ckpt_dir", type=str, default="./models/eipm_single")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--hq_chunk", type=int, default=500)
    p.add_argument("--eval_n", type=int, default=0)
    p.add_argument("--max_reps", type=int, default=100)
    return p.parse_args()


def compute_hq_chunked(
    T_train_cpu: torch.Tensor,
    T_eval_cpu: torch.Tensor,
    a_h: float,
    k_nn: int,
    chunk: int,
) -> torch.Tensor:
    hqs = []
    n = T_eval_cpu.numel()
    for i in range(0, n, int(chunk)):
        t_chunk = T_eval_cpu[i:i + int(chunk)]
        hq_chunk = h0_t(T_train=T_train_cpu, t=t_chunk, a_h=float(a_h), k=int(k_nn))
        hqs.append(hq_chunk.view(-1).cpu())
    return torch.cat(hqs, dim=0)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    files = sorted(glob.glob(str(Path(args.data_dir) / args.pattern)))
    if not files:
        raise FileNotFoundError(f"No npz files found in {args.data_dir} with pattern {args.pattern}")

    npz_path = files[0]
    reps = load_replications_from_npz(npz_path)
    data = np.load(npz_path, allow_pickle=True)
    if "T_eval" not in data.files or "mu_eval" not in data.files:
        raise KeyError("T_eval or mu_eval not found in dataset")
    T_eval_all = np.array(data["T_eval"])
    mu_eval_all = np.array(data["mu_eval"])

    n_reps = min(int(args.max_reps), len(reps))
    mse_list = []
    mae_list = []
    done_reps = 0

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
        Y = torch.tensor(rep.Y, dtype=torch.float32, device=device)

        x_mean = torch.tensor(np.asarray(std["x_mean"]), dtype=torch.float32, device=device).view(1, -1)
        x_std = torch.tensor(np.asarray(std["x_std"]), dtype=torch.float32, device=device).view(1, -1)
        t_mean = torch.tensor(float(std["t_mean"]), dtype=torch.float32, device=device)
        t_std = torch.tensor(float(std["t_std"]), dtype=torch.float32, device=device)

        X_std = (X - x_mean) / x_std
        T_scaled = (T.view(-1) - t_mean) / t_std
        X_scaled = X_std / math.sqrt(float(rep.d_X))

        if T_eval_all.ndim == 2:
            T_eval = np.array(T_eval_all[rep_idx]).reshape(-1)
            mu_eval = np.array(mu_eval_all[rep_idx]).reshape(-1)
        else:
            T_eval = np.array(T_eval_all).reshape(-1)
            mu_eval = np.array(mu_eval_all).reshape(-1)

        if args.eval_n > 0 and args.eval_n < T_eval.shape[0]:
            T_eval = T_eval[:args.eval_n]
            mu_eval = mu_eval[:args.eval_n]

        a_h = math.exp(float(best_params["log_a_h"]))
        k_nn = int(best_params["k_nn"])

        T_eval_t = torch.tensor(T_eval, dtype=torch.float32, device=device)
        T_eval_t = (T_eval_t - t_mean) / t_std

        T_train_cpu = T_scaled.detach().cpu()
        T_eval_cpu = T_eval_t.detach().cpu()
        hq = compute_hq_chunked(
            T_train_cpu,
            T_eval_cpu,
            a_h=float(a_h),
            k_nn=int(k_nn),
            chunk=int(args.hq_chunk),
        ).to(device)

        preds = []
        for i in range(T_eval_t.numel()):
            preds.append(mu_hat_at_t(model, X_scaled, T_scaled, Y, T_eval_t[i], float(hq[i].item())))
        pred = torch.stack(preds).view(-1).detach().cpu().numpy()

        mse = float(np.mean((pred - mu_eval) ** 2))
        mae = float(np.mean(np.abs(pred - mu_eval)))

        mse_list.append(mse)
        mae_list.append(mae)

        print(f"[DATA] {Path(npz_path).name} rep={rep.rep_idx}")
        print(f"[EVAL] mse={mse:.6g} mae={mae:.6g}")
        done_reps += 1

    mse_rms = float(np.sqrt(np.mean(np.square(np.array(mse_list))))) if mse_list else float("nan")
    mae_mean = float(np.mean(np.array(mae_list))) if mae_list else float("nan")
    print(f"[SUMMARY] mse_rms={mse_rms:.6g} mae_mean={mae_mean:.6g} reps={done_reps}")


if __name__ == "__main__":
    main()
