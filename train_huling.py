#!/usr/bin/env python
# train_huling_fixed.py
# Train Huling-style exact OSQP weights on DGP datasets (sim_*.npz),
# saving checkpoints in a structure aligned with train_eipm.py.
#
# Key design choice:
# - This file does NOT store any bandwidth (h_med / h_median). Bandwidth is an evaluation-time choice.

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

import math
import scipy.sparse as sparse
import osqp
from scipy.spatial.distance import cdist


# ============================================================
# 0. Small utilities
# ============================================================

def atomic_write_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def atomic_torch_save(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _fmt_hms(sec: float) -> str:
    s = int(max(0.0, sec))
    h = s // 3600
    m = (s % 3600) // 60
    ss = s % 60
    return f"{h:02d}:{m:02d}:{ss:02d}"


# ============================================================
# 1. Huling-style OSQP exact solver (original logic preserved)
# ============================================================

def solve_huling_osqp_exact(X: np.ndarray, A: np.ndarray, lambda_reg: float = 0.0) -> np.ndarray:
    """
    Exact Energy Balancing Weights using OSQP (Huling-style).

    Inputs
      X: (n, pX) numpy
      A: (n,) or (n, pA) numpy

    Constraints
      w_i >= 0, sum_i w_i = n
    """
    n, p = X.shape
    gamma = 1.0

    # Distance matrices
    Xdist = cdist(X, X, metric="euclidean")
    if A.ndim == 1:
        Adist = cdist(A.reshape(-1, 1), A.reshape(-1, 1), metric="euclidean")
    elif A.ndim == 2:
        Adist = cdist(A, A, metric="euclidean")
    else:
        raise ValueError("A must be 1D or 2D array")

    # Energy terms
    Q_energy_A = -Adist / (n ** 2)
    aa_energy_A = np.sum(Adist, axis=1) / (n ** 2)

    Q_energy_X = -Xdist / (n ** 2)
    aa_energy_X = np.sum(Xdist, axis=1) / (n ** 2)

    # Centering for Distance Covariance
    Xmeans = np.mean(Xdist, axis=1)
    Xgrand = np.mean(Xmeans)
    XA = Xdist + Xgrand - (Xmeans[:, None] + Xmeans[None, :])

    Ameans = np.mean(Adist, axis=1)
    Agrand = np.mean(Ameans)
    AA = Adist + Agrand - (Ameans[:, None] + Ameans[None, :])

    # Product of centered matrices
    P_mat = (XA * AA) / (n ** 2)

    # Scaling factors (preserve original file's scaling logic)
    Q_A_adj = 1.0 / math.sqrt(p)
    Q_X_adj = 1.0
    sum_adj = Q_A_adj + Q_X_adj
    Q_A_adj /= sum_adj
    Q_X_adj /= sum_adj

    # Quadratic Matrix (QM)
    QM = P_mat + gamma * (Q_energy_A * Q_A_adj + Q_energy_X * Q_X_adj)

    # Regularization
    if lambda_reg > 0.0:
        np.fill_diagonal(QM, np.diag(QM) + lambda_reg / (n ** 2))

    # OSQP form: (1/2) w^T P w + q^T w
    P_osqp = sparse.csc_matrix(2.0 * QM)

    # IMPORTANT: linear term must be a vector; use aa_energy_X (not Q_energy_X)
    q_osqp = 2.0 * gamma * (aa_energy_A * Q_A_adj + aa_energy_X * Q_X_adj)

    # Constraints: w >= 0, sum(w) = n
    A_osqp = sparse.vstack([sparse.eye(n), np.ones((1, n))], format="csc")
    l = np.hstack([np.zeros(n), np.array([n], dtype=np.float64)])
    u = np.hstack([np.full(n, np.inf), np.array([n], dtype=np.float64)])

    prob = osqp.OSQP()
    prob.setup(
        P=P_osqp, q=q_osqp, A=A_osqp, l=l, u=u,
        verbose=False, polish=True,
        eps_abs=1e-6, eps_rel=1e-6, max_iter=200000
    )
    res = prob.solve()

    if res.info.status != "solved":
        # fallback: uniform weights summing to n
        return np.ones(n, dtype=np.float64)

    return np.maximum(res.x.astype(np.float64), 0.0)


def build_A_from_T(T: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    DGP-compatible A:
      A_i = [ I(T_i=0), log(1+T_i) ]
    then standardize columnwise.
    """
    T = np.asarray(T, dtype=np.float64).reshape(-1)
    is_zero = (T == 0.0).astype(np.float64).reshape(-1, 1)
    T_star = np.log1p(T).reshape(-1, 1)
    A = np.hstack([is_zero, T_star])  # (n, 2)

    A_mean = A.mean(axis=0)
    A_std = A.std(axis=0) + 1e-6
    A_norm = (A - A_mean) / A_std

    stats = {
        "A_mean": A_mean.astype(np.float64),
        "A_std": A_std.astype(np.float64),
    }
    return A_norm.astype(np.float64), stats


def compute_ess(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    sw = np.sum(w)
    if sw <= 0:
        return 0.0
    w_norm = w / sw
    return float(1.0 / np.sum(w_norm ** 2))


# ============================================================
# 2. Data loading (DGP npz)
# ============================================================

@dataclass
class ReplicationData:
    scenario: str
    d_X: int
    n_train: int
    seed: int
    rep_idx: int
    Xtilde: np.ndarray   # (n_train, d_X)
    T: np.ndarray        # (n_train,) raw T (nonnegative)
    tildeT: np.ndarray   # (n_train,) standardized T
    Y: np.ndarray        # (n_train,)
    meta: Dict


def load_replications_from_npz(npz_path: str) -> List[ReplicationData]:
    """
    Load replications from DGP-generated .npz (stacked over replications):
      Xtilde_train: (n_rpt, n_train, d_X)
      T_train:      (n_rpt, n_train)
      tildeT_train: (n_rpt, n_train)
      Y_train:      (n_rpt, n_train)
    """
    data = np.load(npz_path, allow_pickle=True)

    scenario = str(np.array(data["scenario"]).item())
    d_X = int(np.array(data["d_X"]).item())
    n_train = int(np.array(data["n_train"]).item())
    seed = int(np.array(data["seed"]).item())
    n_rpt = int(np.array(data["n_rpt"]).item())

    Xtilde_all = np.asarray(data["Xtilde_train"])
    T_all = np.asarray(data["T_train"])
    tildeT_all = np.asarray(data["tildeT_train"])
    Y_all = np.asarray(data["Y_train"])

    if Xtilde_all.shape[0] != n_rpt:
        raise ValueError(f"{npz_path}: expected n_rpt={n_rpt}, got Xtilde_train.shape[0]={Xtilde_all.shape[0]}")

    meta = {
        "scenario": scenario,
        "d_X": d_X,
        "n_train": n_train,
        "n_rpt": n_rpt,
        "seed": seed,
        "treatment_k": int(np.array(data["treatment_k"]).item()),
        "outcome_k": int(np.array(data["outcome_k"]).item()),
    }

    reps: List[ReplicationData] = []
    for i in range(n_rpt):
        reps.append(
            ReplicationData(
                scenario=scenario,
                d_X=d_X,
                n_train=n_train,
                seed=seed,
                rep_idx=i,
                Xtilde=np.asarray(Xtilde_all[i], dtype=np.float64),
                T=np.asarray(T_all[i], dtype=np.float64).reshape(-1),
                tildeT=np.asarray(tildeT_all[i], dtype=np.float64).reshape(-1),
                Y=np.asarray(Y_all[i], dtype=np.float64).reshape(-1),
                meta=meta,
            )
        )
    return reps


# ============================================================
# 3. Paths + summary schema (match train_eipm.py)
# ============================================================

def make_ckpt_path(out_dir: Path, npz_path: str, rep_idx: int, depth: int, width: int) -> Path:
    stem = Path(npz_path).stem
    ckpt_name = stem + f"_rep{rep_idx:03d}_d{depth}_w{width}.pth"
    return out_dir / ckpt_name


def make_hp_path(out_dir: Path, npz_path: str, depth: int, width: int) -> Path:
    return out_dir / (Path(npz_path).stem + f"_tuned_depth{depth}_width{width}.json")


SUMMARY_FIELDS = [
    "npz",
    "rep_idx",
    "scenario",
    "d_X",
    "n_train",
    "depth",
    "width",
    "best_cv_mse",
    "best_eipm_loss",
    "sigma",
    "h_median",
    "k_nn",
    "lr",
    "weight_decay",
    "epochs",
    "ckpt_path",
    "hp_path",
]


def append_summary_row(summary_path: Path, row: Dict) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = summary_path.exists()
    with open(summary_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ============================================================
# 4. Main
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--data_dir", type=str, default="./datasets")
    p.add_argument("--pattern", type=str, default="sim_*.npz")
    p.add_argument("--out_dir", type=str, default="./models/huling")

    # Keep interface compatible with train_eipm.py (not used by Huling)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--k_folds", type=int, default=5)
    p.add_argument("--tune_rep_idx", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)

    # Huling-specific
    p.add_argument("--lambda_reg", type=float, default=0.0)

    # Runtime controls
    p.add_argument("--max_files", type=int, default=None)
    p.add_argument("--max_reps", type=int, default=None)

    # Resume / overwrite
    p.add_argument("--overwrite_existing", action="store_true", default=False)
    p.add_argument("--skip_existing", action="store_true", default=False)  # backward-compat alias

    # Reproducibility
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resume behavior: default skip if exists
    overwrite = bool(args.overwrite_existing)
    if bool(args.skip_existing):
        overwrite = False

    npz_paths = sorted([str(p) for p in data_dir.glob(args.pattern)])
    if args.max_files is not None:
        npz_paths = npz_paths[: int(args.max_files)]

    summary_path = out_dir / "train_summary.csv"

    print(f"[HULING-TRAIN] data_dir={data_dir} pattern={args.pattern} files={len(npz_paths)} out_dir={out_dir}")
    print(f"[HULING-TRAIN] overwrite_existing={overwrite} lambda_reg={args.lambda_reg}")

    for npz_path in npz_paths:
        reps = load_replications_from_npz(npz_path)
        n_rpt = len(reps)

        rep_indices = list(range(n_rpt))
        if args.max_reps is not None:
            rep_indices = rep_indices[: int(args.max_reps)]

        for rep_idx in rep_indices:
            rep = reps[rep_idx]

            depth = int(args.depth)
            width = int(args.width)

            ckpt_path = make_ckpt_path(out_dir, npz_path, rep_idx, depth, width)
            hp_path = make_hp_path(out_dir, npz_path, depth, width)

            if ckpt_path.exists() and (not overwrite):
                # still ensure hp json exists
                if not hp_path.exists():
                    atomic_write_json(hp_path, {"lambda_reg": float(args.lambda_reg)})
                continue

            t0 = time.time()

            # Xtilde is already standardized by DGP
            X = rep.Xtilde
            # Build A from RAW T (nonnegative): DO NOT use tildeT here.
            A, A_stats = build_A_from_T(rep.T)

            # Solve weights
            w = solve_huling_osqp_exact(X, A, lambda_reg=float(args.lambda_reg))

            elapsed = time.time() - t0
            ess = compute_ess(w)
            max_w = float(np.max(w)) if w.size else float("nan")

            # Save hp json (for parity with train_eipm)
            atomic_write_json(
                hp_path,
                {
                    "lambda_reg": float(args.lambda_reg),
                },
            )

            # Checkpoint: aligned top-level keys with train_eipm.py, but model_state stores weights.
            ckpt = {
                "model_state": {
                    "weights": torch.tensor(w, dtype=torch.float32),
                    "A_stats": {
                        "A_mean": torch.tensor(A_stats["A_mean"], dtype=torch.float32),
                        "A_std": torch.tensor(A_stats["A_std"], dtype=torch.float32),
                    },
                },
                "best_params": {
                    "lambda_reg": float(args.lambda_reg),
                },
                "best_cv_mse": float("nan"),
                "train_stats": {
                    "ess": float(ess),
                    "max_w": float(max_w),
                    "elapsed_sec": float(elapsed),
                },
                "rep_meta": rep.meta,
                "script_args": vars(args),
            }

            atomic_torch_save(ckpt_path, ckpt)

            # Summary row (schema aligned with train_eipm.py)
            row = {
                "npz": Path(npz_path).name,
                "rep_idx": int(rep.rep_idx),
                "scenario": rep.scenario,
                "d_X": int(rep.d_X),
                "n_train": int(rep.n_train),
                "depth": int(depth),
                "width": int(width),

                # Not applicable for Huling (keep schema)
                "best_cv_mse": float("nan"),
                "best_eipm_loss": float("nan"),
                "sigma": float("nan"),
                "h_median": float("nan"),
                "k_nn": float("nan"),
                "lr": float("nan"),
                "weight_decay": float("nan"),

                "epochs": int(args.epochs),
                "ckpt_path": str(ckpt_path),
                "hp_path": str(hp_path),
            }
            append_summary_row(summary_path, row)

            print(
                f"[OK] {Path(npz_path).name} rep={rep_idx:03d} "
                f"n={rep.n_train} d_X={rep.d_X} ess={ess:.1f} max_w={max_w:.3f} "
                f"elapsed={_fmt_hms(elapsed)} -> {ckpt_path.name}"
            )


if __name__ == "__main__":
    main()
