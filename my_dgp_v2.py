# my_dgp_v2.py

# import
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Tuple
import argparse
import json
import os
import numpy as np
import time

# functions

def _sigmoid(x):
    x = np.asarray(x)
    out = np.empty_like(x, dtype=np.float64)

    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))

    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)

    return out

def _elu(x: np.ndarray) -> np.ndarray: # ELU function with alpha = 1
    return np.where(x > 0.0, x, 0.5 * (np.exp(x) - 1.0))

def _E_fZ(f) -> float: # E[f(Z)]의 근삿값
    xs, ws = np.polynomial.hermite.hermgauss(deg=50)
    z = np.sqrt(2.0) * xs
    vals = f(z)
    return float(np.sum(ws * vals) / np.sqrt(np.pi))

def save(payload: Dict[str, Any], out_name: str, save_csv: bool = True) -> Tuple[str]: # payload를 out_name의 이름으로 저장
    out_dir = "./datasets_v2"
    os.makedirs(out_dir, exist_ok=True)
    npz_name = os.path.join(out_dir, f"{out_name}.npz")
    np.savez_compressed(npz_name, **payload)

    if save_csv:
        csv_dir = os.path.join(out_dir, "csv", out_name)
        os.makedirs(csv_dir, exist_ok=True)
        X_all = payload.get("X_train")
        T_all = payload.get("T_train")
        Y_all = payload.get("Y_train")
        if isinstance(X_all, np.ndarray) and isinstance(T_all, np.ndarray) and isinstance(Y_all, np.ndarray):
            n_rpt = X_all.shape[0]
            d_x = X_all.shape[2]
            header = ",".join([f"X{i+1}" for i in range(d_x)] + ["T", "Y"])
            for r in range(n_rpt):
                rep_path = os.path.join(csv_dir, f"rep{r:03d}.csv")
                if os.path.exists(rep_path):
                    continue
                Xr = X_all[r]
                Tr = T_all[r].reshape(-1, 1)
                Yr = Y_all[r].reshape(-1, 1)
                rows = np.hstack([Xr, Tr, Yr])
                np.savetxt(rep_path, rows, delimiter=",", header=header, comments="")
    return npz_name

def _logit(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64)
    return np.log(u / (1.0 - u))

def _choose_tail_k(n: int, tail_k: int) -> int:
    if int(tail_k) > 0:
        return max(2, min(int(tail_k), n))
    return max(2, min(int(round(n ** 0.8)), n))

def fit_tstar_transform(T_train: np.ndarray, tail_k: int) -> Dict[str, np.ndarray]:
    T_sorted = np.sort(np.asarray(T_train, dtype=np.float64).reshape(-1))
    n = int(T_sorted.size)
    if n < 2:
        raise ValueError("T_train must have at least 2 samples for T* transform.")

    u_grid = (np.arange(n) + 0.5) / n
    u0 = float(u_grid[0])
    u1 = float(u_grid[-1])
    x0 = float(_logit(np.array(u0)))
    x1 = float(_logit(np.array(u1)))

    k = _choose_tail_k(n, int(tail_k))
    eps = 1e-12
    left_span = max(float(T_sorted[k - 1] - T_sorted[0]), eps)
    right_span = max(float(T_sorted[-1] - T_sorted[-k]), eps)
    f_left = float((k - 1) / (n * left_span))
    f_right = float((k - 1) / (n * right_span))
    denom_left = max(u0 * (1.0 - u0), eps)
    denom_right = max(u1 * (1.0 - u1), eps)
    s_left = float(f_left / denom_left)
    s_right = float(f_right / denom_right)

    return {
        "T_sorted": T_sorted,
        "u_grid": u_grid,
        "u0": np.array(u0, dtype=np.float64),
        "u1": np.array(u1, dtype=np.float64),
        "x0": np.array(x0, dtype=np.float64),
        "x1": np.array(x1, dtype=np.float64),
        "s_left": np.array(s_left, dtype=np.float64),
        "s_right": np.array(s_right, dtype=np.float64),
        "k_tail": np.array(k, dtype=np.int64),
    }

def transform_t_to_star(t: np.ndarray, params: Dict[str, np.ndarray]) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    T_sorted = params["T_sorted"]
    u_grid = params["u_grid"]
    Tmin = float(T_sorted[0])
    Tmax = float(T_sorted[-1])
    u0 = float(params["u0"])
    u1 = float(params["u1"])
    x0 = float(params["x0"])
    x1 = float(params["x1"])
    s_left = float(params["s_left"])
    s_right = float(params["s_right"])

    u = np.empty_like(t, dtype=np.float64)
    mask_mid = (t >= Tmin) & (t <= Tmax)
    if np.any(mask_mid):
        u[mask_mid] = np.interp(t[mask_mid], T_sorted, u_grid)
    mask_left = t < Tmin
    if np.any(mask_left):
        u[mask_left] = _sigmoid(x0 + (t[mask_left] - Tmin) * s_left)
    mask_right = t > Tmax
    if np.any(mask_right):
        u[mask_right] = _sigmoid(x1 + (t[mask_right] - Tmax) * s_right)

    eps = 1e-8
    u = np.clip(u, eps, 1.0 - eps)
    return 2.0 * u - 1.0

def generate_Z(rng: np.random.Generator, n: int, d_X: int) -> np.ndarray: # Z ~ N(0, I_d_X) 생성.
    return rng.normal(loc=0.0, scale=1.0, size=(n, d_X))

def Z_to_X(Z: np.ndarray) -> np.ndarray: # Z를 X로 변환.
    n, d_X = Z.shape
    if d_X % 5 != 0:
      raise TypeError("d_X/5 must be integer.")    
    X = np.zeros((n, d_X), dtype=float)
    n_blocks = d_X // 5
    for j in range(n_blocks):
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
    return X

def moments_X(d_X: int) -> Tuple[np.ndarray, np.ndarray]: # EX, VX 계산 코드
    EX = np.zeros(d_X, dtype=float)
    VX = np.zeros(d_X, dtype=float)

    # (1) X1 = exp(Z1/2)
    ex1 = np.exp(1/8)
    vx1 = np.exp(1/2) - np.exp(1/4)

    # (2) X2 = Z1/(1+exp(Z2)) + 10
    ex2 = 10.0
    vx2 = _E_fZ(f = lambda z: _sigmoid(-z) ** 2)

    # (3) X3 = ((Z1*Z3+15)/25)^3
    ex3 = (2.0**2) * (3.0**2) * 19.0 / (5.0**5)
    vx3 = (2.0**3) * (3.0**2) * 269.0 / (5.0**10)

    # (4) X4 = (Z2+Z4+20)^2 where Z2+Z4 ~ N(0,2)
    ex4 = 402.0
    vx4 = 3208.0

    # (5) X5 = 1(Z3+Z5>0) -> Bern(1/2)
    ex5 = 0.5
    vx5 = 0.25

    n_blocks = d_X // 5
    for j in range(n_blocks):
        base = 5 * j
        EX[base + 0], VX[base + 0] = ex1, vx1
        EX[base + 1], VX[base + 1] = ex2, vx2
        EX[base + 2], VX[base + 2] = ex3, vx3
        EX[base + 3], VX[base + 3] = ex4, vx4
        EX[base + 4], VX[base + 4] = ex5, vx5
    return EX, VX
    

def compute_beta_pi0(pi_0: float) -> float: # solve int sigmoid(beta+z) phi(z) dz = pi_0.
    if pi_0 < 1e-6:
        return 0.0
    max_iter = 10000
    beta = np.log(pi_0 / (1 - pi_0))
    for i in range(max_iter):
        f_val = _E_fZ(lambda z: _sigmoid(z+beta)) - pi_0
        df_val = _E_fZ(lambda z: _sigmoid(z+beta) * (1-_sigmoid(z+beta)))
        if abs(df_val) < 1e-12: # 도함수가 0에 가까우면 중단
            raise TypeError('Computation Failure : beta_pi0')
        beta_new = beta - f_val / df_val        
        if abs(beta_new - beta) < 1e-6:
            return beta_new            
        beta = beta_new
    if i == (max_iter - 1):
        raise TypeError('Computation Failure : beta_pi0')
    return beta

def sample_beta_pi(rng: np.random.Generator, d_X: int, treatment_k: int) -> Tuple[np.ndarray]:
    beta_pi = np.zeros(d_X, dtype=float)
    p = rng.dirichlet(alpha=np.ones(treatment_k))          # shape (treatment_k,)
    s = rng.choice(np.array([-1.0, 1.0]), size=treatment_k)  # shape (treatment_k,)
    beta_pi[:treatment_k] = s * np.sqrt(p)                      
    return beta_pi  # shape (d_X,)

def sample_beta_T(rng: np.random.Generator, d_X: int, treatment_k: int) -> Tuple[float, np.ndarray]:
    beta_T0 = float(rng.normal(0.0, 1.0))
    beta_T = rng.normal(0.0, 1.0 / np.sqrt(treatment_k), size=(d_X,))
    beta_T[treatment_k:] = 0.0
    return beta_T0, beta_T

def generate_T(
    rng: np.random.Generator,
    scenario: str,
    Xtilde: np.ndarray,
    Z: np.ndarray,
    beta_T0: float,
    beta_T: np.ndarray,
    beta_pi0: float,
    beta_pi: np.ndarray,
    pi_0: float,
) -> Tuple[np.ndarray]:
    if scenario == "linear":
        meanlogT = beta_T0 + Xtilde @ beta_T
    elif scenario == "nonlinear":
        meanlogT = beta_T0 + Z @ beta_T
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    eps = rng.normal(0.0, 1.0, size=meanlogT.shape)
    T = np.exp(meanlogT + eps)
    if pi_0 != 0.0:
        if scenario == "linear":
            mask = (rng.random(len(T)) >= pi_0) # 
        elif scenario == "nonlinear":
            p = _sigmoid(beta_pi0 + Z[:,:len(beta_pi)] @ beta_pi)
            mask = (rng.random(len(p)) >= p)
        T = T * mask
    return T

def moments_T(
    rng: np.random.Generator,
    scenario: str,
    d_X: int,
    beta_T0: float,
    beta_T: np.ndarray,
    beta_pi0: float,
    beta_pi: np.ndarray,
    EX: np.ndarray,
    VX: np.ndarray,
    pi_0: float,
) -> Tuple[float, float]:
    Z_ = generate_Z(rng=rng, n=10000, d_X=d_X)
    if scenario == "linear":
        X_ = Z_to_X(Z_)
        X_tilde = (X_ - EX.reshape(1,-1)) / np.sqrt(VX.reshape(1,-1))
        U = X_tilde @ beta_T
        ET = (1-pi_0) * np.exp(beta_T0 + 1/2) * (np.exp(U)).mean()
        VT = (1-pi_0) * np.exp(2 * beta_T0 + 2) * (np.exp(2*U)).mean() - (ET) ** 2
        # ET_blk = (1-pi_0) * np.exp(beta_T0 + 1/2) * np.array([(np.exp(U[i::10])).mean() for i in range(10)])
        # VT_blk = (1-pi_0) * np.exp(2 * beta_T0 + 2) * np.array([(np.exp(2*U)).mean() for i in range(10)]) - ET_blk ** 2
        # print("ET block dev / sqrt(VT):", np.std(ET_blk) / np.sqrt(VT), " | sqrt(VT) block dev / sqrt(VT):", np.std(np.sqrt(VT_blk)) / np.sqrt(VT))
    if scenario == "nonlinear":
        if pi_0<1e-6:
            ET = np.exp(beta_T0 + 1/2 + (beta_T**2).sum()/2)
            VT = np.exp(2*beta_T0 + 2 + 2*(beta_T**2).sum()) - (ET)**2
        else:
            U_ = Z_ @ beta_pi
            V_ = Z_ @ beta_T
            ET = np.exp(beta_T0 + 1/2) * ((1-_sigmoid(beta_pi0 + U_)) * np.exp(V_)).mean()
            VT = np.exp(2 * beta_T0 + 2) * ((1-_sigmoid(beta_pi0 + U_))*np.exp(2*V_)).mean() - (ET) ** 2
            # ET_blk = np.array([(np.exp(beta_T0 + 1/2) * ((1-_sigmoid(beta_pi0 + U_[i::10])) * np.exp(V_[i::10]))).mean() for i in range(10)])
            # VT_blk = np.array([(np.exp(2*beta_T0 + 2) * ((1-_sigmoid(beta_pi0 + U_[i::10])) * np.exp(2*V_[i::10]))).mean() for i in range(10)]) - ET_blk ** 2
            # print("ET block dev / sqrt(VT):", np.std(ET_blk) / np.sqrt(VT), " | sqrt(VT) block dev / sqrt(VT):", np.std(np.sqrt(VT_blk)) / np.sqrt(VT))
    return ET, VT

def sample_beta_Y_linear(
    rng: np.random.Generator,
    d_X: int,
    outcome_k: int,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    beta_Y0 = float(rng.normal(0.0, 1.0))
    beta_YT = float(rng.normal(0.0, 1.0))

    beta_YX = np.zeros(d_X, dtype=float)
    beta_YXT = np.zeros(d_X, dtype=float)
    beta_YX[:outcome_k] = rng.normal(0.0, 1.0 / np.sqrt(outcome_k), size=(outcome_k,))
    beta_YXT[:outcome_k] = rng.normal(0.0, 1.0 / np.sqrt(outcome_k), size=(outcome_k,))
    return beta_Y0, beta_YT, beta_YX, beta_YXT

def sample_beta_Y_nonlinear(
    rng: np.random.Generator,
    d_X: int,
    outcome_k: int,
    h1: int = 32,
    h2: int = 16,
) -> Dict[str, np.ndarray]:
    W1 = np.zeros(shape=(d_X+1, h1), dtype=float)
    W1[:outcome_k,:] = rng.normal(0.0, 1.0 / np.sqrt(2*outcome_k), size=(outcome_k, h1))
    W1[-1,:] = rng.normal(0.0, 1.0   / np.sqrt(2), size=h1)
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
    x = np.concatenate([Z, Ttilde.reshape(-1, 1)], axis=1)  # (n, d_X+1)
    h1 = _elu(x @ beta_Y_nonlinear["W1"] + beta_Y_nonlinear["b1"][None, :])
    h2 = _elu(h1 @ beta_Y_nonlinear["W2"] + beta_Y_nonlinear["b2"][None, :])
    out = h2 @ beta_Y_nonlinear["W3"] + beta_Y_nonlinear["b3"][None, :]  # (n, 1)
    return out.reshape(-1) # n차원 벡터

def generate_Y(
    rng: np.random.Generator,
    scenario: str,
    Xtilde: np.ndarray,
    Z: np.ndarray,
    Ttilde: np.ndarray,
    beta_Y_linear: Tuple[float, float, np.ndarray, np.ndarray],
    beta_Y_nonlinear: Dict[str, np.ndarray] | None,
) -> np.ndarray:
    eps = rng.normal(0.0, 1.0, size=(Z.shape[0],))
    if scenario == "linear":
        beta_Y0, beta_YT, beta_YX, beta_YXT = beta_Y_linear
        return beta_Y0 + Ttilde * beta_YT + Xtilde @ beta_YX + Ttilde * (Xtilde @ beta_YXT) + eps
    if scenario == "nonlinear":
        return lambda_NN(Z, Ttilde, beta_Y_nonlinear) + eps
    
def adrf_linear_at_t(
    t: np.ndarray,
    beta_Y0: float,
    beta_YT: float,
    ET: float,
    VT: float,
) -> np.ndarray:
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
    batch_size: int = 500,
) -> np.ndarray:
    Ttilde = (t - ET) / np.sqrt(VT)  # (n,)
    n = t.shape[0]
    out = np.zeros(n, dtype=float)
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        tb = Ttilde[start:end]  # (b,)
        Z_ = rng.normal(0.0, 1.0, size=(m*(end - start), d_X))
        mu_ = lambda_NN(Z_, np.repeat(tb, m), beta_Y_nonlinear).reshape(m,end-start)
        out[start:end] = mu_.mean(axis=0)
    return out

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
    tail_k: int

def generate_one_replication(
    cfg: Config,
    rng: np.random.Generator,
    EX: np.ndarray,
    VX: np.ndarray,
) -> Dict[str, Any]:
    d_X = cfg.d_X

    Z_train = generate_Z(rng, cfg.n_train, d_X)
    X_train = Z_to_X(Z_train)
    Xtilde_train = (X_train - EX.reshape(1,-1)) / np.sqrt(VX.reshape(1,-1))

    Z_eval = generate_Z(rng, cfg.n_eval, d_X)
    X_eval = Z_to_X(Z_eval)
    Xtilde_eval =  (X_eval - EX.reshape(1,-1)) / np.sqrt(VX.reshape(1,-1))
    beta_pi0 = compute_beta_pi0(cfg.pi_0)
    beta_pi = sample_beta_pi(rng=rng, d_X=d_X, treatment_k=cfg.treatment_k)
    beta_T0, beta_T = sample_beta_T(rng=rng, d_X=d_X, treatment_k=cfg.treatment_k)
    T_train = generate_T(
        rng=rng,
        scenario=cfg.scenario,
        Xtilde=Xtilde_train,
        Z=Z_train,
        beta_T0=beta_T0,
        beta_T=beta_T,
        beta_pi0=beta_pi0,
        beta_pi=beta_pi,
        pi_0=cfg.pi_0
    )
    T_eval = generate_T(
        rng=rng,
        scenario=cfg.scenario,
        Xtilde=Xtilde_eval,
        Z=Z_eval,
        beta_T0=beta_T0,
        beta_T=beta_T,
        beta_pi0=beta_pi0,
        beta_pi=beta_pi,
        pi_0=cfg.pi_0
    )

    ET, VT =  moments_T(
        rng=rng, 
        scenario=cfg.scenario,
        d_X=d_X,
        beta_T0=beta_T0,
        beta_T=beta_T,
        beta_pi0=beta_pi0,
        beta_pi=beta_pi,
        EX=EX,
        VX=VX,
        pi_0=cfg.pi_0
    )
    Ttilde_train = (T_train - ET) / np.sqrt(VT)
    
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
        beta_Y_nonlinear = sample_beta_Y_nonlinear(rng=rng, d_X=d_X, outcome_k=cfg.outcome_k)
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
            rng=rng, t=T_eval, d_X=d_X, beta_Y_nonlinear=beta_Y_nonlinear, ET=ET, VT=VT, m=200
        )

    tstar_params = fit_tstar_transform(T_train, tail_k=int(cfg.tail_k))
    T_train_star = transform_t_to_star(T_train, tstar_params)
    T_eval_star = transform_t_to_star(T_eval, tstar_params)
    out: Dict[str, Any] = {
        # observed data
        "X_train": X_train,
        "T_train": T_train_star,
        "Y_train": Y_train,
        # moments for X standardization
        "EX": EX,
        "VX": VX,
        # moments for T standardization
        "ET": np.array([ET], dtype=float),
        "VT": np.array([VT], dtype=float),
        # latent variables
        "Z_train": Z_train,
        "Xtilde_train": Xtilde_train,
        "Ttilde_train": Ttilde_train,
        "beta_pi0": np.array([beta_pi0], dtype=float),
        "beta_pi": beta_pi,
        "beta_T0": np.array([beta_T0], dtype=float),
        "beta_T": beta_T,
        # eval data: only T and ADRF (and stability diagnostics)
        "T_eval": T_eval_star,
        "mu_eval": mu_eval,
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

    # xt = payload["Xtilde_train"]  # (n_rpt, n_train, d_X)

    # # rep별/차원별 mean, var
    # rep_dim_mean = xt.mean(axis=1)  # (n_rpt, d_X)
    # rep_dim_var  = xt.var(axis=1)   # (n_rpt, d_X)

    # # 차원별로 replication들을 요약
    # mean_of_means_j = rep_dim_mean.mean(axis=0)  # (d_X,)
    # min_mean_j      = rep_dim_mean.min(axis=0)
    # max_mean_j      = rep_dim_mean.max(axis=0)

    # mean_of_vars_j  = rep_dim_var.mean(axis=0)   # (d_X,)
    # min_var_j       = rep_dim_var.min(axis=0)
    # max_var_j       = rep_dim_var.max(axis=0)

    # # 전체적으로 얼마나 벗어나는지 (차원 축소는 "진단"만)
    # max_abs_mean_of_means = float(np.max(np.abs(mean_of_means_j)))
    # max_abs_mean_over_reps = float(np.max(np.maximum(np.abs(min_mean_j), np.abs(max_mean_j))))
    # min_var_overall = float(np.min(min_var_j))
    # max_var_overall = float(np.max(max_var_j))

    # print("[STDCHK] Xtilde_train per-dimension across replications (target: mean≈0, var≈1)")
    # print(f"  max_j |mean_of_means_j| = {max_abs_mean_of_means:.6g}")
    # print(f"  max_j max_rep |mean_rep_j| = {max_abs_mean_over_reps:.6g}")
    # print(f"  var range over all (rep,j): min={min_var_overall:.6g}, max={max_var_overall:.6g}")

    # # 차원별 상세 출력 (d_X줄)
    # print("  j : mean_of_means_j   [min_mean_j, max_mean_j]      mean_of_vars_j   [min_var_j,  max_var_j]")
    # for j in range(cfg.d_X):
    #     print(
    #         f" {j:3d}: {mean_of_means_j[j]: .6g} "
    #         f"[{min_mean_j[j]: .6g}, {max_mean_j[j]: .6g}]   "
    #         f"{mean_of_vars_j[j]: .6g} "
    #         f"[{min_var_j[j]: .6g}, {max_var_j[j]: .6g}]"
    #     )

    # tt = payload["Ttilde_train"]  # shape (n_rpt, n_train)
    # print(tt.shape)
    # rep_mean = tt.mean(axis=1)
    # rep_var = tt.var(axis=1)
    # print("[STDCHK] Ttilde_train across replications (target mean=0, var=1)")
    # print(f"  mean_of_means = {float(rep_mean.mean()):.6g},  min/max_mean = ({float(rep_mean.min()):.6g}, {float(rep_mean.max()):.6g})")
    # print(f"  mean_of_vars  = {float(rep_var.mean()):.6g},  min/max_var  = ({float(rep_var.min()):.6g}, {float(rep_var.max()):.6g})")
    return payload

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", type=str, required=True, choices=["linear", "nonlinear"])
    p.add_argument("--d_X", type=int, default=50)
    p.add_argument("--n_train", type=int, default=1000)
    p.add_argument("--n_eval", type=int, default=1000)   # changed to 1000
    p.add_argument("--n_rpt", type=int, default=100)
    p.add_argument("--pi_0", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_csv", type=int, default=1)
    p.add_argument("--tail_k", type=int, default=0)

    # sparsity controls (optional)
    p.add_argument("--treatment_k", type=int, default=None)
    p.add_argument("--outcome_k", type=int, default=None)

    return p.parse_args()

def main() -> None:
    args = parse_args()

    treatment_k = args.treatment_k if args.treatment_k is not None else args.d_X
    outcome_k = args.outcome_k if args.outcome_k is not None else args.d_X

    out_name = (
        f"sim_{args.scenario}_dx{args.d_X}"
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
        tail_k=args.tail_k,
    )

    payload = generate_dataset(cfg)
    npz_name = save(payload, out_name, save_csv=bool(args.save_csv))
    print(f"Saved: {npz_name}")

if __name__ == "__main__":
    main()
