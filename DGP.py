# DGP.py
# Baseline DGP for simulation study (linear / nonlinear scenarios)
#
# Key points:
# - Z_i ~ N(0, I_d_X)
# - X_i is generated via Kang–Schafer-style nonlinear transforms in 5-d blocks
# - Standardize X using E[X_k], Var[X_k] under the DGP (closed-form where possible)
# - Treatment:
#   * linear:    log(T_i) | (Xtilde_i, beta_T) ~ N(beta_T0 + Xtilde_i^T beta_T, 1)
#   * nonlinear: log(T_i) | (Z_i,      beta_T) ~ N(beta_T0 + Z_i^T      beta_T, 1)
# - Standardize treatment using T (NOT log T):
#     Ttilde_i = (T_i - E[T|beta_T]) / sqrt(Var(T|beta_T))
#   where E[T|beta_T], Var(T|beta_T) are computed by MC over the covariate DGP.
# - Outcome:
#   * linear:
#       Y_i | (Xtilde_i, Ttilde_i, beta_Y-linear) ~ N(
#         beta_Y0 + Ttilde_i*beta_YT + Xtilde_i^T beta_YX + Ttilde_i * Xtilde_i^T beta_YXT, 1)
#   * nonlinear:
#       Y_i | (Z_i, Ttilde_i, beta_Y-NN) ~ N( lambda_NN(Z_i, Ttilde_i; beta_Y), 1)
#       where lambda_NN is a 3-layer feed-forward NN with ELU(alpha=1/2) activations.
#
# Saved outputs:
# - train: Z_train, X_train, Xtilde_train, logT_train, T_train, Ttilde_train, Y_train
# - eval:  T_eval and ADRF evaluated at those T's (mu_eval)
# - coefficients and standardization moments (EX,VX, ET,VT)
#
# Usage example:
#   python DGP.py --scenario linear --d_X 50 --n_train 1000 --n_eval 1000 --n_rpt 100 --seed 42
#   python DGP.py --scenario nonlinear --d_X 50 --n_train 1000 --n_eval 1000 --n_rpt 100 --seed 42

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Tuple
import argparse
import json
import os
import numpy as np

# -----------------------------
# Utilities
# -----------------------------
def _elu(x: np.ndarray) -> np.ndarray:
    """
    ELU(alpha=1/2):
      x                    if x > 0
      (1/2) * (exp(x) - 1)  if x <= 0
    """
    return np.where(x > 0.0, x, np.exp(x) - 1.0)


def _E_fZ(f) -> float:
    """
    Compute E[f(Z)] for Z ~ N(0,1) using Gauss-Hermite quadrature.

    If {x_i,w_i} are GH nodes/weights for exp(-x^2), then
      E[f(Z)] = (1/sqrt(pi)) * sum_i w_i * f(sqrt(2)*x_i).
    """
    xs, ws = np.polynomial.hermite.hermgauss(deg=50)
    z = np.sqrt(2.0) * xs
    vals = f(z)
    return float(np.sum(ws * vals) / np.sqrt(np.pi))


def save(payload: Dict[str, Any], out_name: str) -> Tuple[str, str]:
    """
    Save arrays into a compressed NPZ and metadata into a JSON.
    """
    out_dir = "./datasets"
    os.makedirs(out_dir, exist_ok=True)

    npz_name = os.path.join(out_dir, f"{out_name}.npz")
    meta_name = os.path.join(out_dir, f"{out_name}_meta.json")

    np.savez_compressed(npz_name, **payload)

    def _json_default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

    # make scenario a plain string in meta
    scenario_val = payload.get("scenario", None)
    if isinstance(scenario_val, np.ndarray) and scenario_val.shape == ():
        scenario_val = scenario_val.item()

    meta = {
        "scenario": scenario_val,
        "d_X": int(np.array(payload["d_X"]).item()),
        "n_train": int(np.array(payload["n_train"]).item()),
        "n_eval": int(np.array(payload["n_eval"]).item()),
        "n_rpt": int(np.array(payload["n_rpt"]).item()),
        "pi_0": float(np.array(payload["pi_0"]).item()),
        "seed": int(np.array(payload["seed"]).item()),
        "treatment_k": int(np.array(payload["treatment_k"]).item()),
        "outcome_k": int(np.array(payload["outcome_k"]).item()),
        "keys_in_npz": sorted(list(payload.keys())),
        "shapes": {k: list(v.shape) for k, v in payload.items() if isinstance(v, np.ndarray)},
    }

    with open(meta_name, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=_json_default)

    return npz_name, meta_name


# -----------------------------
# Covariate DGP: Z -> X
# -----------------------------
def generate_Z(rng: np.random.Generator, n: int, d_X: int) -> np.ndarray:
    return rng.normal(loc=0.0, scale=1.0, size=(n, d_X))


def Z_to_X(Z: np.ndarray) -> np.ndarray:
    """
    Apply the 5-d block nonlinear transformations.
    For d_X not multiple of 5, apply transformations for the available indices.
    """
    n, d_X = Z.shape
    X = np.zeros((n, d_X), dtype=float)

    n_full_blocks = d_X // 5
    for j in range(n_full_blocks):
        base = 5 * j
        z1 = Z[:, base + 0]
        z2 = Z[:, base + 1]
        z3 = Z[:, base + 2]
        z4 = Z[:, base + 3]
        z5 = Z[:, base + 4]

        X[:, base + 0] = np.exp(z1 / 2.0)
        X[:, base + 1] = z1 / (1.0 + np.exp(z2)) + 10.0
        X[:, base + 2] = ((z1 * z3 + 15.0) / 25.0) ** 3
        X[:, base + 3] = (z2 + z4 + 20.0) ** 2
        X[:, base + 4] = (z3 + z5 > 0.0).astype(float)

    rem = d_X - 5 * n_full_blocks
    if rem > 0:
        base = 5 * n_full_blocks
        if rem >= 1:
            z1 = Z[:, base + 0]
            X[:, base + 0] = np.exp(z1 / 2.0)
        if rem >= 2:
            z1 = Z[:, base + 0]
            z2 = Z[:, base + 1]
            X[:, base + 1] = z1 / (1.0 + np.exp(z2)) + 10.0
        if rem >= 3:
            z1 = Z[:, base + 0]
            z3 = Z[:, base + 2]
            X[:, base + 2] = ((z1 * z3 + 15.0) / 25.0) ** 3
        if rem >= 4:
            z2 = Z[:, base + 1]
            z4 = Z[:, base + 3]
            X[:, base + 3] = (z2 + z4 + 20.0) ** 2

    return X


def moments_X(d_X: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute E[X_k] and Var(X_k) under the covariate DGP, componentwise.
    Uses closed form when available; uses GH for Var of X_{5j+2}.
    """
    EX = np.zeros(d_X, dtype=float)
    VX = np.zeros(d_X, dtype=float)

    # (1) X1 = exp(Z1/2)
    ex1 = np.exp(1.0 / 8.0)
    vx1 = np.exp(1.0 / 2.0) - np.exp(1.0 / 4.0)

    # (2) X2 = Z1/(1+exp(Z2)) + 10
    ex2 = 10.0

    def f_var_x2(z):
        return (1.0 + np.exp(z)) ** (-2.0)

    e_inv_sq = _E_fZ(f_var_x2)
    vx2 = e_inv_sq  # since E[Z1^2]=1

    # (3) X3 = ((Z1*Z3+15)/25)^3
    ex3 = (2.0**2) * (3.0**2) * 19.0 / (5.0**5)
    vx3 = (2.0**3) * (3.0**2) * 269.0 / (5.0**10)

    # (4) X4 = (Z2+Z4+20)^2 where Z2+Z4 ~ N(0,2)
    ex4 = 402.0
    vx4 = 3208.0

    # (5) X5 = 1(Z3+Z5>0) -> Bern(1/2)
    ex5 = 0.5
    vx5 = 0.25

    n_full_blocks = d_X // 5
    for j in range(n_full_blocks):
        base = 5 * j
        EX[base + 0], VX[base + 0] = ex1, vx1
        EX[base + 1], VX[base + 1] = ex2, vx2
        EX[base + 2], VX[base + 2] = ex3, vx3
        EX[base + 3], VX[base + 3] = ex4, vx4
        EX[base + 4], VX[base + 4] = ex5, vx5

    rem = d_X - 5 * n_full_blocks
    if rem > 0:
        base = 5 * n_full_blocks
        if rem >= 1:
            EX[base + 0], VX[base + 0] = ex1, vx1
        if rem >= 2:
            EX[base + 1], VX[base + 1] = ex2, vx2
        if rem >= 3:
            EX[base + 2], VX[base + 2] = ex3, vx3
        if rem >= 4:
            EX[base + 3], VX[base + 3] = ex4, vx4

    return EX, VX


def standardize_X(X: np.ndarray, EX: np.ndarray, VX: np.ndarray) -> np.ndarray:
    return (X - EX[None, :]) / np.sqrt(VX[None, :])

# -----------------------------
# Treatment DGP
# -----------------------------
def sample_beta_T(
    rng: np.random.Generator,
    d_X: int,
    treatment_k: int,
) -> Tuple[float, np.ndarray]:
    """
    beta_T0 ~ N(0,1)
    beta_T  ~ N(0, (1/d_X) I_d_X), optionally sparse on first treatment_k coords.

    If treatment_k < d_X :
      scale nonzero entries by sqrt(d_X/treatment_k) so that E||beta||^2 matches dense case.
    """
    beta_T0 = float(rng.normal(0.0, 1.0))
    beta_T = rng.normal(0.0, 1.0 / np.sqrt(d_X), size=(d_X,))

    if treatment_k < d_X:
        mask = np.zeros(d_X, dtype=bool)
        mask[:treatment_k] = True
        beta_T = beta_T * mask.astype(float) * np.sqrt(d_X / float(treatment_k))
    return beta_T0, beta_T

def generate_logT(
    rng: np.random.Generator,
    scenario: str,
    Xtilde: np.ndarray,
    Z: np.ndarray,
    beta_T0: float,
    beta_T: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (logT, T, meanlog) with sd=1 in both scenarios.
    """
    if scenario == "linear":
        meanlog = beta_T0 + Xtilde @ beta_T
    elif scenario == "nonlinear":
        meanlog = beta_T0 + Z @ beta_T
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    eps = rng.normal(0.0, 1.0, size=meanlog.shape)
    logT = meanlog + eps
    return logT

def moments_of_T_given_betas_mc(
    rng: np.random.Generator,
    scenario: str,
    d_X: int,
    beta_T0: float,
    beta_T: np.ndarray,
    EX: np.ndarray,
    VX: np.ndarray,
) -> Tuple[float, float]:
    """
    Approximate E[T | beta_T] and Var(T | beta_T) by Monte Carlo over the covariate DGP.

    Conditional on covariates:
      logT | covariates ~ N(mu, 1)
      => E[T | cov]   = exp(mu + 1/2)
         E[T^2 | cov] = exp(2 mu + 2)

    Then:
      E[T|beta]   = E_cov[ exp(mu(cov) + 1/2) ]
      E[T^2|beta] = E_cov[ exp(2 mu(cov) + 2) ]
      Var         = E[T^2|beta] - (E[T|beta])^2
    """
    Zmc = generate_Z(rng, n=20000, d_X)

    if scenario == "linear":
        Xmc = Z_to_X(Zmc)
        Xmc_tilde = standardize_X(Xmc, EX, VX)
        mu = beta_T0 + Xmc_tilde @ beta_T
    elif scenario == "nonlinear":
        mu = beta_T0 + Zmc @ beta_T
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    m1 = float(np.mean(np.exp(mu + 0.5)))
    m2 = float(np.mean(np.exp(2.0 * mu + 2.0)))
    v = m2 - (m1**2)
    if v <= 0.0:
        v = float(max(v, 1e-12))
    return m1, v


# -----------------------------
# Outcome DGP
# -----------------------------
def sample_beta_Y_linear(
    rng: np.random.Generator,
    d_X: int,
    outcome_k: int,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    beta_{Y,0} ~ N(0,1)
    beta_{Y,T} ~ N(0,1)
    beta_{Y,X}  ~ N(0, (1/d_X) I_d_X), optionally sparse
    beta_{Y,XT} ~ N(0, (1/d_X) I_d_X), optionally sparse
    """
    beta_Y0 = float(rng.normal(0.0, 1.0))
    beta_YT = float(rng.normal(0.0, 1.0))

    beta_YX = rng.normal(0.0, 1.0 / np.sqrt(d_X), size=(d_X,))
    beta_YXT = rng.normal(0.0, 1.0 / np.sqrt(d_X), size=(d_X,))

    if outcome_k < d_X:
        mask = np.zeros(d_X, dtype=bool)
        mask[:outcome_k] = True
        beta_YX = beta_YX * mask.astype(float) * np.sqrt(d_X / float(outcome_k))
        beta_YXT = beta_YXT * mask.astype(float) * np.sqrt(d_X / float(outcome_k))
    return beta_Y0, beta_YT, beta_YX, beta_YXT


def sample_beta_Y_nonlinear(
    rng: np.random.Generator,
    d_X: int,
    h1: int = 32,
    h2: int = 16,
) -> Dict[str, np.ndarray]:
    """
    Sample parameters for a 3-layer feed-forward NN:
      input (d_X+1) -> h1 -> h2 -> 1

    We sample weights with N(0, 1/sqrt(fan_in)) scaling for stability.
    """
    din = d_X + 1

    W1 = rng.normal(0.0, 1.0 / np.sqrt(din), size=(din, h1))
    b1 = rng.normal(0.0, 1.0, size=(h1,))

    W2 = rng.normal(0.0, 1.0 / np.sqrt(h1), size=(h1, h2))
    b2 = rng.normal(0.0, 1.0, size=(h2,))

    W3 = rng.normal(0.0, 1.0 / np.sqrt(h2), size=(h2, 1))
    b3 = rng.normal(0.0, 1.0, size=(1,))

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}


def lambda_NN(
    Z: np.ndarray,
    Ttilde: np.ndarray,
    beta_Y_nonlinear: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    Compute lambda_NN(Z_i, Ttilde_i; beta_Y).
    Z: (n, d_X), Ttilde: (n,)
    returns: (n,)
    """
    x = np.concatenate([Z, Ttilde.reshape(-1, 1)], axis=1)  # (n, d_X+1)

    h1 = _elu(x @ beta_Y_nonlinear["W1"] + beta_Y_nonlinear["b1"][None, :])
    h2 = _elu(h1 @ beta_Y_nonlinear["W2"] + beta_Y_nonlinear["b2"][None, :])
    out = h2 @ beta_Y_nonlinear["W3"] + beta_Y_nonlinear["b3"][None, :]  # (n, 1)
    return out.reshape(-1)

def generate_Y(
    rng: np.random.Generator,
    scenario: str,
    Xtilde: np.ndarray,
    Z: np.ndarray,
    Ttilde: np.ndarray,
    beta_Y_linear: Tuple[float, float, np.ndarray, np.ndarray],
    beta_Y_nonlinear: Dict[str, np.ndarray] | None,
) -> np.ndarray:
    """
    Generate outcome with unit noise variance: Var = 1.
    """
    eps = rng.normal(0.0, 1.0, size=(Z.shape[0],))

    if scenario == "linear":
        beta_Y0, beta_YT, beta_YX, beta_YXT = beta_Y_linear
        mu = (
            beta_Y0
            + Ttilde * beta_YT
            + Xtilde @ beta_YX
            + Ttilde * (Xtilde @ beta_YXT)
        )
        return mu + eps

    if scenario == "nonlinear":
        if beta_Y_nonlinear is None:
            raise ValueError("beta_Y_nonlinear must be provided for nonlinear scenario.")
        mu = lambda_NN(Z, Ttilde, beta_Y_nonlinear)
        return mu + eps

    raise ValueError(f"Unknown scenario: {scenario}")


# -----------------------------
# ADRF at t: mu(t) = E[Y^t | betas]
# -----------------------------
def adrf_linear_at_t(
    t: np.ndarray,
    beta_Y0: float,
    beta_YT: float,
    ET: float,
    VT: float,
) -> np.ndarray:
    """
    For the linear outcome model, because E[Xtilde]=0 exactly under true standardization,
      mu(t) = beta_Y0 + beta_YT * ((t - ET)/sqrt(VT)).
    """
    Ttilde = (t - ET) / np.sqrt(VT)
    return beta_Y0 + beta_YT * Ttilde


def adrf_nonlinear_mc_at_t(
    rng: np.random.Generator,
    t: np.ndarray,
    d_X: int,
    beta_Y_nonlinear: Dict[str, np.ndarray],
    ET: float,
    VT: float,
    m: int,
    batch_size: int = 200,
) -> np.ndarray:
    """
    For the nonlinear outcome model,
      mu(t) = E_Z[ lambda_NN(Z, Ttilde(t); beta_Y_nonlinear) ],
    where Z ~ N(0, I_d_X).

    Compute by MC with m samples of Z, per t.
    """
    Ttilde = (t - ET) / np.sqrt(VT)  # (n,)
    n = t.shape[0]
    out = np.zeros(n, dtype=float)
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        tb = Ttilde[start:end]  # (b,)
        mu_b = np.zeros(end - start, dtype=float)
        for _ in range(m):
            Zm = rng.normal(0.0, 1.0, size=(end - start, d_X))
            mu_b += lambda_NN(Zm, tb, beta_Y_nonlinear)
        mu_b /= float(m)
        out[start:end] = mu_b
    return out


def adrf_mc_stability_check(
    rng: np.random.Generator,
    scenario: str,
    t: np.ndarray,
    d_X: int,
    beta_Y0: float,
    beta_YT: float,
    beta_Y_nonlinear: Dict[str, np.ndarray] | None,
    ET: float,
    VT: float,
) -> Dict[str, np.ndarray]:
    """
    Compute mu_MC with m=50 and m=100 (or analytic for linear),
    and return diagnostics and the chosen mu (m=100).
    """
    if scenario == "linear":
        mu = adrf_linear_at_t(t=t, beta_Y0=beta_Y0, beta_YT=beta_YT, ET=ET, VT=VT)
        mu_50 = mu.copy()
        mu_100 = mu.copy()
        diff = np.zeros_like(mu)
    elif scenario == "nonlinear":
        if beta_Y_nonlinear is None:
            raise ValueError("beta_Y_nonlinear must be provided for nonlinear scenario.")
        rng_50 = np.random.default_rng(rng.integers(0, 2**32 - 1))
        rng_100 = np.random.default_rng(rng.integers(0, 2**32 - 1))
        mu_50 = adrf_nonlinear_mc_at_t(
            rng=rng_50, t=t, d_X=d_X, beta_Y_nonlinear=beta_Y_nonlinear, ET=ET, VT=VT, m=50
        )
        mu_100 = adrf_nonlinear_mc_at_t(
            rng=rng_100, t=t, d_X=d_X, beta_Y_nonlinear=beta_Y_nonlinear, ET=ET, VT=VT, m=100
        )
        diff = np.abs(mu_100 - mu_50)
        mu = mu_100
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    # Diagnostics as scalar arrays (for easy saving)
    return {
        "mu_eval_m50": mu_50,
        "mu_eval_m100": mu_100,
        "mu_eval_absdiff_50_100": diff,
        "mu_eval_absdiff_50_100_mean": np.array([float(np.mean(diff))], dtype=float),
        "mu_eval_absdiff_50_100_max": np.array([float(np.max(diff))], dtype=float),
        "mu_eval": mu,
    }


# -----------------------------
# Top-level dataset generator
# -----------------------------
@dataclass
class Config:
    scenario: str  # "linear" or "nonlinear"
    d_X: int
    n_train: int
    n_eval: int
    n_rpt: int
    pi_0: float
    seed: int
    treatment_k: int
    outcome_k: int

def generate_one_replication(
    cfg: Config,
    rng: np.random.Generator,
    EX: np.ndarray,
    VX: np.ndarray,
) -> Dict[str, Any]:
    """
    Generate one replication: train set + eval set (T_eval and ADRF only) + betas.
    """
    d_X = cfg.d_X

    # --------
    # Train covariates
    # --------
    Z_train = generate_Z(rng, cfg.n_train, d_X)
    X_train = Z_to_X(Z_train)
    Xtilde_train = standardize_X(X_train, EX, VX)

    # --------
    # Eval covariates (only to generate T_eval; not saved)
    # --------
    Z_eval = generate_Z(rng, cfg.n_eval, d_X)
    X_eval = Z_to_X(Z_eval)
    Xtilde_eval = standardize_X(X_eval, EX, VX)

    # --------
    # Treatment coefficients and treatment draws
    # --------
    beta_T0, beta_T = sample_beta_T(
        rng=rng,
        d_X=d_X,
        treatment_k=cfg.treatment_k,
    )

    logT_train = generate_logT(
        rng=rng,
        scenario=cfg.scenario,
        Xtilde=Xtilde_train,
        Z=Z_train,
        beta_T0=beta_T0,
        beta_T=beta_T,
    )
    logT_eval = generate_logT(
        rng=rng,
        scenario=cfg.scenario,
        Xtilde=Xtilde_eval,
        Z=Z_eval,
        beta_T0=beta_T0,
        beta_T=beta_T,
    )

    # --------
    # Standardize T using moments conditional on beta_T only
    # --------
    ET, VT = moments_of_T_given_betas_mc(
        rng=rng,
        scenario=cfg.scenario,
        d_X=d_X,
        beta_T0=beta_T0,
        beta_T=beta_T,
        EX=EX,
        VX=VX,
    )
    Ttilde_train = (T_train - ET) / np.sqrt(VT)

    # --------
    # Outcome coefficients and Y_train
    # --------
    beta_Y_linear = None
    beta_Y_nonlinear = None
    if cfg.scenario == "linear":
        beta_Y0, beta_YT, beta_YX, beta_YXT = sample_beta_Y_linear(
            rng=rng,
            d_X=d_X,
            outcome_k=cfg.outcome_k,
        )
        beta_Y_linear = (beta_Y0, beta_YT, beta_YX, beta_YXT)
    if cfg.scenario == "nonlinear":
        beta_Y_nonlinear = sample_beta_Y_nonlinear(rng=rng, d_X=d_X)
    Y_train = generate_Y(
        rng=rng,
        scenario=cfg.scenario,
        Xtilde=Xtilde_train,
        Z=Z_train,
        Ttilde=Ttilde_train,
        beta_Y_linear=beta_Y_linear,
        beta_Y_nonlinear=beta_Y_nonlinear,
    )
    if cfg.scenario == 'linear':
        mu_eval = adrf_linear_at_t(t=T_eval, beta_Y0=beta_Y0, beta_YT=beta_YT, ET=ET, VT=VT)
    if cfg.scenario == 'nonlinear':
        mu_eval = adrf_nonlinear_mc_at_t(
            rng=rng, t=T_eval, d_X=d_X, beta_Y_nonlinear=beta_Y_nonlinear, ET=ET, VT=VT, m=100
        )
    # --------
    # ADRF on eval T's (store ONLY T_eval and mu_eval for eval data)
    # plus stability check (m=50 vs m=100)
    # --------
    # adrf_pack = adrf_mc_stability_check(
    #     rng=rng,
    #     scenario=cfg.scenario,
    #     t=T_eval,
    #     d_X=d_X,
    #     beta_Y0=beta_Y0,
    #     beta_YT=beta_YT,
    #     beta_Y_nonlinear=beta_Y_nonlinear,
    #     ET=ET,
    #     VT=VT,
    # )
    out: Dict[str, Any] = {
        # moments for X standardization
        "EX": EX,
        "VX": VX,
        # moments for T standardization
        "ET": np.array([ET], dtype=float),
        "VT": np.array([VT], dtype=float),
        # train data
        "Z_train": Z_train,
        "X_train": X_train,
        "Xtilde_train": Xtilde_train,
        "logT_train": logT_train,
        "T_train": T_train,
        "Ttilde_train": Ttilde_train,
        "Y_train": Y_train,
        # eval data: only T and ADRF (and stability diagnostics)
        "T_eval": T_eval,
        "mu_eval": mu_eval,
        "beta_pi0": np.array([beta_pi0], dtype=float),
        "beta_pi": beta_pi,
        "beta_T0": np.array([beta_T0], dtype=float),
        "beta_T": beta_T,
        # "mu_eval_m50": adrf_pack["mu_eval_m50"],
        # "mu_eval_m100": adrf_pack["mu_eval_m100"],
        # "mu_eval_absdiff_50_100": adrf_pack["mu_eval_absdiff_50_100"],
        # "mu_eval_absdiff_50_100_mean": adrf_pack["mu_eval_absdiff_50_100_mean"],
        # "mu_eval_absdiff_50_100_max": adrf_pack["mu_eval_absdiff_50_100_max"],
        # coefficients
    }
    if beta_Y_linear is not None:
        out["beta_Y0"]=np.array([beta_Y0], dtype=float)
        out["beta_YT"]=np.array([beta_YT], dtype=float)
        out["beta_YX"] = beta_YX
        out["beta_YXT"]=beta_YXT        
    if beta_Y_nonlinear is not None:
        for k, v in beta_Y_nonlinear.items():
            out[f"beta_Y_nonlinear_{k}"] = v
    return out


def generate_dataset(cfg: Config) -> Dict[str, Any]:
    """
    Generate n_rpt replications and stack them along axis=0.
    EX,VX are computed ONCE (not per replication).
    """
    rng = np.random.default_rng(cfg.seed)

    EX, VX = moments_X(d_X=cfg.d_X)

    reps = [generate_one_replication(cfg, rng, EX=EX, VX=VX) for _ in range(cfg.n_rpt)]

    payload: Dict[str, Any] = {
        "scenario": np.array(cfg.scenario),
        "d_X": np.array(cfg.d_X),
        "n_train": np.array(cfg.n_train),
        "n_eval": np.array(cfg.n_eval),
        "n_rpt": np.array(cfg.n_rpt),
        "pi_0": np.array(cfg.pi_0),
        "seed": np.array(cfg.seed),
        "treatment_k": np.array(cfg.treatment_k),
        "outcome_k": np.array(cfg.outcome_k),
        # record fixed numerical settings for reproducibility
    }

    keys = reps[0].keys()
    for k in keys:
        if k in payload:
            continue
        v0 = reps[0][k]
        if isinstance(v0, np.ndarray):
            payload[k] = np.stack([r[k] for r in reps], axis=0)

    return payload


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", type=str, required=True, choices=["linear", "nonlinear"])
    p.add_argument("--d_X", type=int, default=50)
    p.add_argument("--n_train", type=int, default=1000)
    p.add_argument("--n_eval", type=int, default=1000)   # changed to 1000
    p.add_argument("--n_rpt", type=int, default=100)
    p.add_argument("--pi_0", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)

    # sparsity controls (optional)
    p.add_argument("--treatment_k", type=int, default=None)
    p.add_argument("--outcome_k", type=int, default=None)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    treatment_k = args.treatment_k if args.treatment_k is not None else args.d_X
    outcome_k = args.outcome_k if args.outcome_k is not None else args.d_X

    out_name = (
        f"sim_{args.scenario}_d_X{args.d_X}"
        f"_ntr{args.n_train}_nev{args.n_eval}_rpt{args.n_rpt}"
        f"_tk{treatment_k}_ok{outcome_k}"
        f"_pi{args.pi_0}"
        f"_seed{args.seed}"
    )

    cfg = Config(
        scenario=args.scenario,
        d_X=args.d_X,
        n_train=args.n_train,
        n_eval=args.n_eval,
        n_rpt=args.n_rpt,
        pi_0=args.pi_0,
        seed=args.seed,
        treatment_k=treatment_k,
        outcome_k=outcome_k,
    )

    payload = generate_dataset(cfg)
    npz_name, meta_name = save(payload, out_name)
    print(f"Saved: {npz_name}")
    print(f"Meta : {meta_name}")


if __name__ == "__main__":
    main()
