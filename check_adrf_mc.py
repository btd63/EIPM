# check_adrf_mc.py
# Purpose:
#   Before running full DGP jobs, check whether ADRF Monte Carlo size m=100 is stable enough.
#
# What it does (focus on worst case = nonlinear):
#   - Sample one (or multiple) replications of (beta_T, beta_Y_nn)
#   - Generate T_eval of size n_eval
#   - Compute ADRF_ref using a large MC size m_ref
#   - For each m in m_list:
#       * compute ADRF_m with seed1 and seed2
#       * report stability (seed1 vs seed2) : mean/max abs diff
#       * report deviation from ADRF_ref     : mean/max abs diff
#
# Usage example:
#   python check_adrf_mc.py --dx 50 --n_eval 300 --n_rep_check 3 --m_list 25 50 100 200 --m_ref 2000 --seed 42
#
# NOTE:
#   This script imports DGP.py as a module. Put this file in the same directory as DGP.py.

from __future__ import annotations

import argparse
import time
from typing import Dict, List, Tuple

import numpy as np

import DGP  # your DGP.py


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dx", type=int, default=50)
    p.add_argument("--n_eval", type=int, default=300)
    p.add_argument("--n_rep_check", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--m_ref", type=int, default=5000)
    p.add_argument("--m_list", type=int, nargs="+", default=[10, 100, 500])

    p.add_argument("--t_batch", type=int, default=200, help="Batch size over t_eval for NN ADRF computation")
    return p.parse_args()


def lambda_nn_batch(
    Z: np.ndarray,
    t: np.ndarray,
    beta_Y_nn: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    Same architecture as in DGP.py (ReLU MLP), but vectorized.
    Z: (N, dx)
    t: (N,)
    returns: (N,)
    """
    x = np.concatenate([Z, t.reshape(-1, 1)], axis=1)
    h1 = np.maximum(x @ beta_Y_nn["W1"] + beta_Y_nn["b1"][None, :], 0.0)
    h2 = np.maximum(h1 @ beta_Y_nn["W2"] + beta_Y_nn["b2"][None, :], 0.0)
    out = h2 @ beta_Y_nn["W3"] + beta_Y_nn["b3"][None, :]
    return out.reshape(-1)


def adrf_nonlinear_mc(
    rng: np.random.Generator,
    dx: int,
    T_eval: np.ndarray,
    ET: float,
    VT: float,
    beta_Y_nn: Dict[str, np.ndarray],
    m: int,
    t_batch: int,
) -> np.ndarray:
    """
    ADRF(t) = E_Z[ lambda_NN(Z, tilde t; beta_Y) ] approximated by MC with size m.
    """
    Z_mc = rng.normal(0.0, 1.0, size=(m, dx))
    tildeT = (T_eval - ET) / np.sqrt(VT)

    n_eval = tildeT.shape[0]
    out = np.zeros(n_eval, dtype=float)

    for start in range(0, n_eval, t_batch):
        end = min(start + t_batch, n_eval)
        t_b = tildeT[start:end]
        B = t_b.shape[0]

        Z_rep = np.repeat(Z_mc, B, axis=0)      # (m*B, dx)
        t_rep = np.tile(t_b, m)                 # (m*B,)

        mu = lambda_nn_batch(Z_rep, t_rep, beta_Y_nn)  # (m*B,)
        mu = mu.reshape(m, B)
        out[start:end] = np.mean(mu, axis=0)

    return out


def summarize_diff(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    d = np.abs(a - b)
    return float(np.mean(d)), float(np.max(d))


def main() -> None:
    args = parse_args()

    dx = args.dx
    n_eval = args.n_eval
    n_rep_check = args.n_rep_check
    seed = args.seed

    m_ref = args.m_ref
    m_list = args.m_list
    t_batch = args.t_batch

    # We check the worst-case runtime path: nonlinear scenario.
    scenario = "nonlinear"

    print("=== ADRF MC sanity check (nonlinear) ===")
    print(f"dx={dx}, n_eval={n_eval}, n_rep_check={n_rep_check}, seed={seed}")
    print(f"m_ref={m_ref}, m_list={m_list}, t_batch={t_batch}")
    print()

    t0_all = time.time()

    # EX,VX only needed if you later want linear checks; keep for consistency.
    EX, VX = DGP.moments_of_X_components(dx=dx)

    # Accumulators for across-rep summary
    rows: List[Dict[str, float]] = []

    for r in range(n_rep_check):
        rep_seed = seed + 10_000 * r
        rng_data = np.random.default_rng(rep_seed)

        # Sample treatment beta
        beta_T0, beta_T = DGP.sample_beta_T(rng=rng_data, dx=dx, treatment_k=dx)

        # Nonlinear: ET,VT are exact (fast)
        ET, VT = DGP.moments_of_T_given_betas_nonlinear_exact(beta_T0, beta_T)

        # Sample NN outcome parameters (nonlinear outcome)
        beta_Y_nn = DGP.sample_beta_Y_nn(rng=rng_data, dx=dx)

        # Generate T_eval (nonlinear: depends on Z)
        Z_eval = rng_data.normal(0.0, 1.0, size=(n_eval, dx))
        # create dummy Xtilde arg (ignored in nonlinear branch if we call DGP.generate_logT)
        X_dummy = np.zeros((n_eval, dx), dtype=float)
        _, T_eval, _ = DGP.generate_logT(
            rng=rng_data,
            scenario=scenario,
            Xtilde=X_dummy,
            Z=Z_eval,
            beta_T0=beta_T0,
            beta_T=beta_T,
        )

        # Reference ADRF with large m_ref
        rng_ref = np.random.default_rng(rep_seed + 111_111)
        ADRF_ref = adrf_nonlinear_mc(
            rng=rng_ref,
            dx=dx,
            T_eval=T_eval,
            ET=ET,
            VT=VT,
            beta_Y_nn=beta_Y_nn,
            m=m_ref,
            t_batch=t_batch,
        )

        print(f"[rep {r+1}/{n_rep_check}] rep_seed={rep_seed}")
        for m in m_list:
            rng1 = np.random.default_rng(rep_seed + 1_000_000 + 13 * m)
            rng2 = np.random.default_rng(rep_seed + 2_000_000 + 29 * m)

            ADRF_1 = adrf_nonlinear_mc(rng1, dx, T_eval, ET, VT, beta_Y_nn, m=m, t_batch=t_batch)
            ADRF_2 = adrf_nonlinear_mc(rng2, dx, T_eval, ET, VT, beta_Y_nn, m=m, t_batch=t_batch)

            stab_mean, stab_max = summarize_diff(ADRF_1, ADRF_2)
            ref_mean, ref_max = summarize_diff(ADRF_1, ADRF_ref)

            rows.append(
                {
                    "rep": float(r),
                    "m": float(m),
                    "stability_mean_abs": stab_mean,
                    "stability_max_abs": stab_max,
                    "vs_ref_mean_abs": ref_mean,
                    "vs_ref_max_abs": ref_max,
                }
            )

            print(
                f"  m={m:>4d} | "
                f"stability mean/max={stab_mean:.4g}/{stab_max:.4g} | "
                f"vs_ref mean/max={ref_mean:.4g}/{ref_max:.4g}"
            )
        print()

    # Aggregate summary across reps
    print("=== Aggregate over reps (mean across replications) ===")
    for m in m_list:
        sel = [x for x in rows if int(x["m"]) == int(m)]
        stab_mean = float(np.mean([x["stability_mean_abs"] for x in sel]))
        stab_max = float(np.mean([x["stability_max_abs"] for x in sel]))
        ref_mean = float(np.mean([x["vs_ref_mean_abs"] for x in sel]))
        ref_max = float(np.mean([x["vs_ref_max_abs"] for x in sel]))

        print(
            f"m={m:>4d} | "
            f"stability(mean of mean/max)={stab_mean:.4g}/{stab_max:.4g} | "
            f"vs_ref(mean of mean/max)={ref_mean:.4g}/{ref_max:.4g}"
        )

    t1_all = time.time()
    print()
    print(f"Done. Total wall time: {t1_all - t0_all:.2f} sec")


if __name__ == "__main__":
    main()


# === Aggregate over reps (mean across replications) ===
# m=  25 | stability(mean of mean/max)=0.03741/0.07303 | vs_ref(mean of mean/max)=0.02023/0.04236
# m=  50 | stability(mean of mean/max)=0.05015/0.07186 | vs_ref(mean of mean/max)=0.03486/0.03888
# m= 100 | stability(mean of mean/max)=0.01956/0.02057 | vs_ref(mean of mean/max)=0.009759/0.01498
# m= 200 | stability(mean of mean/max)=0.04128/0.04782 | vs_ref(mean of mean/max)=0.0152/0.02797
