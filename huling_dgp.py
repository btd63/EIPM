#!/usr/bin/env python3
"""
Python port of Huling simulate_data.R.
Generates datasets in the same npz format as my_dgp.py (X_train, T_train, Y_train, T_eval, mu_eval).
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, Any

import numpy as np


def simulate_one_rep(
    rng: np.random.Generator,
    nobs: int,
    n_eval: int,
    MX1: float,
    MX2: float,
    MX3: float,
    A_effect: bool,
    eval_mode: str,
) -> Dict[str, Any]:
    X1 = rng.normal(loc=MX1, scale=1.0, size=nobs)
    X2 = rng.normal(loc=MX2, scale=1.0, size=nobs)
    X3 = rng.normal(loc=0.0, scale=1.0, size=nobs)
    X4 = rng.normal(loc=MX2, scale=1.0, size=nobs)
    X5 = rng.binomial(n=1, p=MX3, size=nobs).astype(float)

    Z1 = np.exp(X1 / 2.0)
    Z2 = (X2 / (1.0 + np.exp(X1))) + 10.0
    Z3 = (X1 * X3 / 25.0) + 0.6
    Z4 = (X4 - MX2) ** 2
    Z5 = X5

    muA = 5.0 * np.abs(X1) + 6.0 * np.abs(X2) + 3.0 * np.abs(X5) + np.abs(X4)
    A = rng.noncentral_chisquare(df=3.0, nonc=muA, size=nobs)

    if A_effect:
        Cnum = ((MX1 + 3.0) ** 2 + 1.0) + 2.0 * ((MX2 - 25.0) ** 2 + 1.0)
        Y = (
            -0.15 * A ** 2
            + A * (X1 ** 2 + X2 ** 2)
            - 15.0
            + (X1 + 3.0) ** 2
            + 2.0 * (X2 - 25.0) ** 2
            + X3
            - Cnum
            + rng.normal(loc=0.0, scale=1.0, size=nobs)
        )
        Y = Y / 50.0
    else:
        Y = (
            X1
            + X1 ** 2
            + X2
            + X2 ** 2
            + X1 * X2
            + X5
            + rng.normal(loc=0.0, scale=1.0, size=nobs)
        )

    def true_adrf(a: np.ndarray) -> np.ndarray:
        if A_effect:
            truth = -0.15 * a ** 2 + a * (2.0 + MX1 ** 2 + MX2 ** 2) - 15.0
            truth = truth / 50.0
            return truth
        return np.full_like(a, MX1 + (MX1 ** 2 + 1.0) + MX2 + (MX2 ** 2 + 1.0) + MX1 * MX2 + MX3)

    if eval_mode == "grid":
        min_A = float(np.min(A))
        max_A = float(np.max(A))
        range_A = max_A - min_A
        t_eval = np.linspace(min_A - 0.05 * range_A, max_A + 0.05 * range_A, int(n_eval))
        mu_eval = true_adrf(t_eval)
    elif eval_mode == "sample":
        X1e = rng.normal(loc=MX1, scale=1.0, size=int(n_eval))
        X2e = rng.normal(loc=MX2, scale=1.0, size=int(n_eval))
        X3e = rng.normal(loc=0.0, scale=1.0, size=int(n_eval))
        X4e = rng.normal(loc=MX2, scale=1.0, size=int(n_eval))
        X5e = rng.binomial(n=1, p=MX3, size=int(n_eval)).astype(float)
        muA_eval = 5.0 * np.abs(X1e) + 6.0 * np.abs(X2e) + 3.0 * np.abs(X5e) + np.abs(X4e)
        t_eval = rng.noncentral_chisquare(df=3.0, nonc=muA_eval, size=int(n_eval))
        mu_eval = true_adrf(t_eval)
    elif eval_mode == "train":
        t_eval = A.copy()
        mu_eval = true_adrf(t_eval)
    else:
        raise ValueError(f"Unknown eval_mode: {eval_mode}")

    X = np.stack([Z1, Z2, Z3, Z4, Z5], axis=1)
    return {
        "X_train": X,
        "T_train": A,
        "Y_train": Y,
        "T_eval": t_eval,
        "mu_eval": mu_eval,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n_train", type=int, default=1000)
    p.add_argument("--n_eval", type=int, default=500)
    p.add_argument("--n_rpt", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--MX1", type=float, default=-0.5)
    p.add_argument("--MX2", type=float, default=1.0)
    p.add_argument("--MX3", type=float, default=0.3)
    p.add_argument("--A_effect", type=int, default=1)
    p.add_argument("--eval_mode", type=str, choices=["grid", "sample", "train"], default="grid")
    p.add_argument("--out_dir", type=str, default="./datasets_huling")
    p.add_argument("--save_csv", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_name = (
        f"sim_huling_ntr{args.n_train}_nev{args.n_eval}_rpt{args.n_rpt}"
        f"_mx1{args.MX1}_mx2{args.MX2}_mx3{args.MX3}"
        f"_ae{args.A_effect}_seed{args.seed}"
    )
    if str(args.eval_mode) != "grid":
        out_name = f"{out_name}_eval{args.eval_mode}"
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    reps = []
    for r in range(int(args.n_rpt)):
        rng = np.random.default_rng(int(args.seed) + int(r))
        rep = simulate_one_rep(
            rng=rng,
            nobs=int(args.n_train),
            n_eval=int(args.n_eval),
            MX1=float(args.MX1),
            MX2=float(args.MX2),
            MX3=float(args.MX3),
            A_effect=bool(args.A_effect),
            eval_mode=str(args.eval_mode),
        )
        reps.append(rep)

    payload: Dict[str, Any] = {
        "scenario": np.array("huling"),
        "d_X": np.array(5),
        "n_train": np.array(int(args.n_train)),
        "n_eval": np.array(int(args.n_eval)),
        "n_rpt": np.array(int(args.n_rpt)),
        "seed": np.array(int(args.seed)),
        "MX1": np.array(float(args.MX1)),
        "MX2": np.array(float(args.MX2)),
        "MX3": np.array(float(args.MX3)),
        "A_effect": np.array(int(args.A_effect)),
        "eval_mode": np.array(str(args.eval_mode)),
    }

    keys = reps[0].keys()
    for k in keys:
        v0 = reps[0][k]
        if isinstance(v0, np.ndarray):
            payload[k] = np.stack([r[k] for r in reps], axis=0)

    npz_path = os.path.join(out_dir, f"{out_name}.npz")
    np.savez_compressed(npz_path, **payload)
    print(f"Saved: {npz_path}")

    if int(args.save_csv) == 1:
        csv_dir = os.path.join(out_dir, "csv", out_name)
        os.makedirs(csv_dir, exist_ok=True)
        for r in range(int(args.n_rpt)):
            X = payload["X_train"][r]
            T = payload["T_train"][r].reshape(-1, 1)
            Y = payload["Y_train"][r].reshape(-1, 1)
            rows = np.hstack([X, T, Y])
            header = ",".join([f"X{i+1}" for i in range(X.shape[1])] + ["T", "Y"])
            np.savetxt(os.path.join(csv_dir, f"rep{r:03d}.csv"), rows, delimiter=",", header=header, comments="")


if __name__ == "__main__":
    main()
