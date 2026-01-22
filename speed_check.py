from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path

import numpy as np
import optuna
import torch

from train_eipm import load_replications_from_npz, objective_cv_mse, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="./datasets")
    p.add_argument("--pattern", type=str, default="sim_*.npz")
    p.add_argument("--rep_idx", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--n_trials", type=int, default=25)
    p.add_argument("--k_folds", type=int, default=5)
    p.add_argument("--tune_steps", type=int, default=50)

    p.add_argument("--fast_n_trials", type=int, default=5)
    p.add_argument("--fast_k_folds", type=int, default=3)
    p.add_argument("--fast_tune_steps", type=int, default=10)
    return p.parse_args()


def run_tuning(
    X_scaled: torch.Tensor,
    T_scaled: torch.Tensor,
    Y: torch.Tensor,
    input_dim: int,
    device: torch.device,
    depth: int,
    width: int,
    n_trials: int,
    k_folds: int,
    tune_steps: int,
    seed: int,
) -> tuple[float, dict]:
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def _obj(trial):
        return objective_cv_mse(
            trial=trial,
            X_scaled=X_scaled,
            T_scaled=T_scaled,
            Y=Y,
            input_dim=input_dim,
            device=device,
            depth=depth,
            width=width,
            tune_steps=tune_steps,
            n_splits=k_folds,
            seed=seed,
        )

    study.optimize(_obj, n_trials=int(n_trials), show_progress_bar=False)
    return float(study.best_value), dict(study.best_params)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    files = sorted(glob.glob(str(Path(args.data_dir) / args.pattern)))
    if not files:
        raise FileNotFoundError(f"No npz files found in {args.data_dir} with pattern {args.pattern}")

    npz_path = files[0]
    reps = load_replications_from_npz(npz_path)
    if args.rep_idx < 0 or args.rep_idx >= len(reps):
        raise IndexError(f"rep_idx out of range: {args.rep_idx}, total={len(reps)}")

    rep = reps[int(args.rep_idx)]
    device = torch.device(args.device)

    X = torch.tensor(rep.Xtilde, dtype=torch.float32, device=device)
    T = torch.tensor(rep.tildeT, dtype=torch.float32, device=device)
    Y = torch.tensor(rep.Y, dtype=torch.float32, device=device)

    X_scaled = X / math.sqrt(float(rep.d_X))
    T_scaled = T.view(-1)

    input_dim = int(rep.d_X) + 1

    print(f"[FILE] {Path(npz_path).name}")
    print(f"[REP] {rep.rep_idx}")

    full_val, full_params = run_tuning(
        X_scaled=X_scaled,
        T_scaled=T_scaled,
        Y=Y,
        input_dim=input_dim,
        device=device,
        depth=int(args.depth),
        width=int(args.width),
        n_trials=int(args.n_trials),
        k_folds=int(args.k_folds),
        tune_steps=int(args.tune_steps),
        seed=int(args.seed),
    )

    fast_val, fast_params = run_tuning(
        X_scaled=X_scaled,
        T_scaled=T_scaled,
        Y=Y,
        input_dim=input_dim,
        device=device,
        depth=int(args.depth),
        width=int(args.width),
        n_trials=int(args.fast_n_trials),
        k_folds=int(args.fast_k_folds),
        tune_steps=int(args.fast_tune_steps),
        seed=int(args.seed),
    )

    delta = fast_val - full_val
    rel = delta / full_val if full_val != 0 else float("nan")

    print(f"[FULL] best_cv_mse={full_val:.6g} params={full_params}")
    print(f"[FAST] best_cv_mse={fast_val:.6g} params={fast_params}")
    print(f"[DIFF] delta={delta:.6g} rel={rel:.6g}")


if __name__ == "__main__":
    main()
