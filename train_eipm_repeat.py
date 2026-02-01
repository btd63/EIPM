from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path

import numpy as np
import torch
import optuna

from train_eipm import (
    set_seed,
    load_replications_from_npz,
    objective_cv_mse,
    train_folds_for_params,
    atomic_torch_save,
)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--data_dir", type=str, default="./datasets")
    p.add_argument("--out_dir", type=str, default="./models/eipm_single")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # model / training
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--epochs", type=int, default=300)

    # tuning (keep small)
    p.add_argument("--n_trials", type=int, default=20)
    p.add_argument("--k_folds", type=int, default=5)
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--min_delta", type=float, default=1e-8)

    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # baseline setting with ntr variations
    ntr_list = [250, 500, 1000, 2000]
    for ntr in ntr_list:
        pattern = f"sim_nonlinear_dx50_ntr{ntr}_nev10000_rpt100_tk50_ok50_pi0.0_seed42.npz"
        files = sorted(glob.glob(str(Path(args.data_dir) / pattern)))
        if len(files) == 0:
            print(f"[WARN] Dataset not found for ntr={ntr}: {pattern}")
            continue

        npz_path = files[0]
        print(f"[INFO] Using dataset: {Path(npz_path).name}")
        print(f"[INFO] Dataset path: {Path(npz_path).resolve()}")

        reps = load_replications_from_npz(npz_path)
        reps_to_run = reps[:100]
        for rep in reps_to_run:
            print(
                f"[INFO] Scenario={rep.scenario}, "
                f"rep_idx={rep.rep_idx}, "
                f"d_X={rep.d_X}, "
                f"n_train={rep.n_train}"
            )

            # ------------------------------------------------------------
            # 2. prepare tensors
            # ------------------------------------------------------------
            X = torch.tensor(rep.X, dtype=torch.float32, device=device)
            T = torch.tensor(rep.T, dtype=torch.float32, device=device)
            Y = torch.tensor(rep.Y, dtype=torch.float32, device=device)

            X_scaled = X / math.sqrt(float(rep.d_X))
            T_scaled = T.view(-1)

            input_dim = int(rep.d_X) + 1

            # ------------------------------------------------------------
            # 3. hyperparameter tuning (single run)
            # ------------------------------------------------------------
            def _obj(trial):
                return objective_cv_mse(
                    trial=trial,
                    X_scaled=X_scaled,
                    T_scaled=T_scaled,
                    Y=Y,
                    input_dim=input_dim,
                    device=device,
                    depth=args.depth,
                    width=args.width,
                    max_steps=args.max_steps,
                    patience=args.patience,
                    min_delta=args.min_delta,
                    n_splits=args.k_folds,
                    seed=args.seed,
                )

            print("[INFO] Start hyperparameter tuning...")
            study = optuna.create_study(direction="minimize")
            study.optimize(_obj, n_trials=args.n_trials, show_progress_bar=True)

            best_params = study.best_params
            best_cv_mse = float(study.best_value)
            best_trial = int(study.best_trial.number)

            print("[INFO] Best CV MSE:", best_cv_mse)
            print("[INFO] Best params:", best_params)

            # ------------------------------------------------------------
            # 4. train fold models using best params, then average
            # ------------------------------------------------------------
            print("[INFO] Train fold models with best params...")
            fold_states, fold_mse = train_folds_for_params(
                X_scaled=X_scaled,
                T_scaled=T_scaled,
                Y=Y,
                input_dim=input_dim,
                device=device,
                depth=args.depth,
                width=args.width,
                params=best_params,
                max_steps=args.max_steps,
                patience=args.patience,
                min_delta=args.min_delta,
                n_splits=args.k_folds,
                seed=args.seed,
            )
            avg_state = {}
            keys = fold_states[0].keys()
            for k in keys:
                stacked = torch.stack([fs[k] for fs in fold_states], dim=0)
                avg_state[k] = torch.mean(stacked, dim=0)
            train_stats = {
                "best_eipm_loss": float("nan"),
                "sigma": float("nan"),
                "h_median": float("nan"),
                "lr": float(best_params.get("lr", float("nan"))),
                "weight_decay": float(best_params.get("weight_decay", float("nan"))),
                "epochs": float(args.epochs),
            }

            # ------------------------------------------------------------
            # 5. save checkpoint
            # ------------------------------------------------------------
            ckpt_path = out_dir / f"eipm_single_nonlinear_rep{rep.rep_idx:03d}.pth"
            atomic_torch_save(
                ckpt_path,
                {
                    "model_state": avg_state,
                    "ensemble_fold_states": fold_states,
                    "ensemble_fold_mse": fold_mse,
                    "ensemble_best_trial": best_trial,
                    "best_params": best_params,
                    "best_cv_mse": best_cv_mse,
                    "train_stats": train_stats,
                    "rep_meta": rep.meta,
                    "npz_path": npz_path,
                    "script_args": vars(args),
                },
            )

            print(f"[DONE] Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
