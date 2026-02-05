# train_eipm.py
# Train EIPM (Expected Integral Probability Metrics) weighting model on TRAIN data only.
#
# This script is designed to work with datasets produced by my_dgp.py in this project.
# my_dgp.py saves (stacked over replications):
#   - train (observed only): X_train, T_train, Y_train
#   - eval:  T_eval, mu_eval (ADRF)   <-- MUST NOT be used here (per user's instruction).
# See my_dgp.py "Saved outputs" description.  (We intentionally ignore eval keys.)

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold, StratifiedKFold
from torch import Tensor

try:
    from zoneinfo import ZoneInfo
    _KST = ZoneInfo("Asia/Seoul")
except Exception:
    _KST = None


# ============================================================
# 0. Utility
# ============================================================

def _locfit_modules():
    try:
        from rpy2 import robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects import packages as rpackages
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("rpy2 is required to use locfit in CV.") from exc

    if not rpackages.isinstalled("locfit"):
        raise RuntimeError("R package 'locfit' is required.")

    numpy2ri.activate()
    locfit = rpackages.importr("locfit")
    base = rpackages.importr("base")
    return ro, locfit, base


def locfit_mu_at_t(
    X_tr: Tensor,
    T_tr: Tensor,
    Y_tr: Tensor,
    model: nn.Module,
    t_val: Tensor,
    a_h: float,
    alpha: float,
) -> Tensor:
    # w_i = exp(s(X_i, t)) with t fixed, then weighted locfit at t
    t_fixed = t_val.view(1).repeat(X_tr.shape[0]).view(-1, 1)
    with torch.no_grad():
        w = torch.exp(model(X_tr, t_fixed)).view(-1).cpu().numpy()

    T_np = T_tr.detach().cpu().numpy()
    Y_np = Y_tr.detach().cpu().numpy()

    # h(t) = a_h * dist(t)^alpha, dist(t)=0.1-quantile of |T_i - t|
    with torch.no_grad():
        dist = torch.abs(T_tr.view(-1) - t_val.view(()))
        dist_q = torch.quantile(dist, 0.1)
        h_t = float(a_h) * float(dist_q.item() ** float(alpha))
        if h_t <= 1e-8:
            h_t = 1e-8

    ro, locfit, base = _locfit_modules()
    dfx = ro.DataFrame({"Y": ro.FloatVector(Y_np), "TRT": ro.FloatVector(T_np)})
    w_r = ro.FloatVector(w)
    fit = locfit.locfit(ro.Formula("Y ~ lp(TRT, h=%f)" % h_t), weights=w_r, data=dfx)
    pred = base.predict(fit, newdata=ro.FloatVector([float(t_val.item())]), where="fitp")
    return torch.tensor(float(pred[0]), device=X_tr.device)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _now_str() -> str:
    if _KST is None:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return datetime.now(_KST).strftime("%Y-%m-%d %H:%M:%S KST")


def _fmt_hms(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    s = int(round(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    ss = s % 60
    return f"{h:02d}:{m:02d}:{ss:02d}"


def _safe_mean(xs: List[float]) -> Optional[float]:
    if len(xs) == 0:
        return None
    return float(mean(xs))


def atomic_write_json(path: Path, obj: Dict) -> None:  #
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def atomic_torch_save(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def standardize_train(X: Tensor, T: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Standardize observed train data using sample mean/variance.
    Returns standardized X, standardized T, X mean/std, and T mean/std (for later use).
    """
    T_flat = T.view(-1)

    x_mean = X.mean(dim=0, keepdim=True)
    x_var = X.var(dim=0, unbiased=False, keepdim=True)
    x_std = torch.sqrt(x_var).clamp_min(1e-8)
    X_std = (X - x_mean) / x_std

    t_mean = T_flat.mean()
    t_var = T_flat.var(unbiased=False)
    t_std = torch.sqrt(t_var).clamp_min(1e-8)
    T_std = (T_flat - t_mean) / t_std

    return X_std, T_std, x_mean, x_std, t_mean, t_std


def load_json_(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"{path} must contain a JSON object (dict), got {type(obj).__name__}")
    return obj


# ============================================================
# 1. Model: score network s_theta(x, t)
# ============================================================

class EIPM(nn.Module):
    """
    Simple MLP score model s_theta(x, t).
    Input = concat([x, t]).
    """
    def __init__(self, input_dim: int, hidden: int = 128, n_layers: int = 2):
        super().__init__()
        layers: List[nn.Module] = []
        d_in = input_dim

        for _ in range(n_layers):
            layers.append(nn.Linear(d_in, hidden))
            layers.append(nn.ELU(alpha=0.5))
            d_in = hidden

        layers.append(nn.Linear(d_in, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, X: Tensor, T: Tensor) -> Tensor:
        if T.ndim == 1:
            T_in = T.view(-1, 1)
        else:
            T_in = T
        inp = torch.cat([X, T_in], dim=1)
        return self.net(inp).view(-1)


# ============================================================
# 2. Kernels
# ============================================================

def rbf(X: Tensor, sigma: float) -> Tensor:
    # X = X.contiguous()
    K = torch.exp(-torch.cdist(X, X) ** 2 / (2.0 * (sigma ** 2)))
    return K


@torch.no_grad()
def get_med(x: Tensor, max_n: int = 500) -> float:
    # x: (n,) or (n,d)
    if x.ndim == 1:
        x = x.view(-1, 1).contiguous()

    n = x.shape[0]
    if n > max_n:
        idx = torch.randperm(n, device=x.device)[:max_n]
        x = x[idx]

    d = torch.cdist(x, x).flatten()
    d = d[d > 0]
    return float(torch.median(d).item())


# ============================================================
# 3. EIPM loss
# ============================================================

# def stable_exp_scores(s: Tensor) -> Tensor:
#     """
#     Numerically-stable exp(s):
#       exp(s - max(s)).
#     This prevents overflow when s is large (and keeps weights unchanged up to normalization).
#     """
#     s_max = torch.max(s)
#     return torch.exp(s - s_max)


def h0_t(
    T_train: Tensor,
    t: Tensor,
    *,
    a_h: float,
    alpha: float,
) -> Tensor:
    T_tr = T_train.view(-1, 1)
    t_q = t.view(1, -1)
    dist = torch.abs(T_tr - t_q)  # (n,m)
    dist_q = torch.quantile(dist, 0.1, dim=0)
    h = float(a_h) * (dist_q ** float(alpha))
    return h.clamp_min(1e-8).view(-1, 1)


def compute_eipm_loss(model: nn.Module, X: Tensor, T: Tensor, a_sigma: float, a_h: float, alpha: float) -> Tensor:
    r"""
    EIPM loss (empirical MMD^2 averaged over target indices i):
    For each target i (with t_i = T_i), define weights over j:
      w_{ij} =  K_T(t_i, T_j; h_loss) * exp(s_theta(X_j, T_j)) / \sum_{l} K_T(t_i, T_l; h_loss) * exp(s_theta(X_l, T_l)).
    """
    s_val = model(X, T)                    # (n,)
    s_max = torch.max(s_val)
    W_s = torch.exp(s_val-s_max)            # (n,)

    T_flat = T.view(-1, 1)
    dist_sq_T = (T_flat - T_flat.t()) ** 2  # (n,n)

    h_T_vec = h0_t(T_train=T, t=T, a_h=float(a_h), alpha=float(alpha))
    h_T_vec = h_T_vec.view(-1, 1)
    H = (h_T_vec @ h_T_vec.t()).clamp_min(1e-8)  # (n,n)
    K_T = torch.exp(-0.5 * dist_sq_T / H)        # (n,n)

    d_med = get_med(X)
    sigma = a_sigma * d_med
    K_X = rbf(X, sigma)

    denom = K_T @ W_s.view(-1, 1)  # (n,1)
    W_mat = (K_T * W_s.view(1, -1)) / (denom.view(-1, 1) + 1e-8)  # (n,n)

    term1 = torch.sum((W_mat @ K_X) * W_mat, dim=1)  # (n,)
    term2 = torch.mean(K_X)
    term3 = 2.0 * torch.mean(W_mat @ K_X, dim=1)  # (n,)

    loss = torch.mean(term1 - term3 + term2)
    return loss


# ============================================================
# 4. CV objective: minimize MSE between mu_hat(T_i) and Y_i (train only)
# ============================================================

def objective_cv_mse(
    trial,
    X_scaled: Tensor,
    T_scaled: Tensor,
    Y: Tensor,
    input_dim: int,
    device: torch.device,
    depth: int,
    width: int,
    max_steps: int = 500,
    patience: int = 3,
    min_delta: float = 1e-3,
    n_splits: int = 5,
    seed: int = 42,
    log_a_h_low: float = -1.0,
    log_a_h_high: float = 1.0,
) -> float:
    """
    Tune hyperparameters by K-fold CV on the TRAIN set only.

    Key point for bandwidth:
      - We use h(t) = a_h * dist(t)^alpha with dist(t)=0.1-quantile of |T_i - t|.
      - The same h(t) is used both in K_T and in locfit during CV.
      - We do not use any eval data (T_eval, mu_eval) from DGP.
    """
    log_a_sigma = trial.suggest_float("log_a_sigma", -2.0, 2.0)
    log_a_h = trial.suggest_float("log_a_h", float(log_a_h_low), float(log_a_h_high))
    alpha = trial.suggest_float("alpha", 0.05, 0.95)

    a_sigma = math.exp(float(log_a_sigma))
    a_h = math.exp(float(log_a_h))

    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    with torch.no_grad():
        y_strat = (T_scaled.detach().cpu().numpy() == 0.0).astype(int)
        use_strat = (np.unique(y_strat).size >= 2)

    if use_strat:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(np.zeros_like(y_strat), y_strat)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(np.arange(X_scaled.shape[0]))

    eval_every = 10
    fold_mse: List[float] = []

    for fold_i, (tr_idx, va_idx) in enumerate(split_iter, start=1):
        tr = torch.as_tensor(tr_idx, device=device, dtype=torch.long)
        va = torch.as_tensor(va_idx, device=device, dtype=torch.long)

        X_tr, T_tr, Y_tr = X_scaled[tr], T_scaled[tr], Y[tr]
        X_va, T_va, Y_va = X_scaled[va], T_scaled[va], Y[va]

        # inner training (short, for tuning)
        model = EIPM(input_dim=input_dim, hidden=width, n_layers=depth).to(device)
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_mse = float("inf")
        no_improve = 0
        for it in range(int(max_steps)):
            opt.zero_grad(set_to_none=True)
            loss = compute_eipm_loss(model, X_tr, T_tr, a_sigma, a_h, alpha)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    "EIPM loss is not finite: "
                    f"trial={getattr(trial, 'number', 'NA')} it={it} "
                    f"a_sigma={a_sigma:.3g} a_h={a_h:.3g} alpha={alpha:.3g} "
                    f"lr={lr:.3g} wd={weight_decay:.3g} "
                )
            loss.backward()
            opt.step()
            if (it + 1) % eval_every == 0:
                model.eval()
                preds = []
                for i in range(T_va.numel()):
                    mu_i = locfit_mu_at_t(model=model, X_tr=X_tr, T_tr=T_tr, Y_tr=Y_tr, t_val=T_va[i], a_h=a_h, alpha=alpha)
                    preds.append(mu_i)
                pred = torch.stack(preds).view(-1)
                mse = torch.mean((pred - Y_va.view(-1)) ** 2).item()

                if mse < best_mse - float(min_delta):
                    best_mse = float(mse)
                    no_improve = 0
                else:
                    no_improve += 1
                model.train()
                if no_improve >= int(patience):
                    break
        if best_mse == float("inf"):
            model.eval()
            preds = []
            for i in range(T_va.numel()):
                mu_i = locfit_mu_at_t(model=model, X_tr=X_tr, T_tr=T_tr, Y_tr=Y_tr, t_val=T_va[i], a_h=a_h, alpha=alpha)
                preds.append(mu_i)
            pred = torch.stack(preds).view(-1)
            best_mse = float(torch.mean((pred - Y_va.view(-1)) ** 2).item())

        fold_mse.append(float(best_mse))

    result = float(np.mean(fold_mse))
    return result


def train_folds_for_params(
    X_scaled: Tensor,
    T_scaled: Tensor,
    Y: Tensor,
    input_dim: int,
    device: torch.device,
    depth: int,
    width: int,
    params: Dict,
    *,
    max_steps: int,
    patience: int,
    min_delta: float,
    n_splits: int,
    seed: int,
) -> Tuple[List[Dict[str, Tensor]], List[float]]:
    a_sigma = math.exp(float(params["log_a_sigma"]))
    a_h = math.exp(float(params["log_a_h"]))
    alpha = float(params["alpha"])
    lr = float(params["lr"])
    weight_decay = float(params["weight_decay"])

    with torch.no_grad():
        y_strat = (T_scaled.detach().cpu().numpy() == 0.0).astype(int)
        use_strat = (np.unique(y_strat).size >= 2)

    if use_strat:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(np.zeros_like(y_strat), y_strat)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(np.arange(X_scaled.shape[0]))

    eval_every = 10
    fold_states: List[Dict[str, Tensor]] = []
    fold_mse: List[float] = []

    for tr_idx, va_idx in split_iter:
        tr = torch.as_tensor(tr_idx, device=device, dtype=torch.long)
        va = torch.as_tensor(va_idx, device=device, dtype=torch.long)

        X_tr, T_tr, Y_tr = X_scaled[tr], T_scaled[tr], Y[tr]
        X_va, T_va, Y_va = X_scaled[va], T_scaled[va], Y[va]

        model = EIPM(input_dim=input_dim, hidden=width, n_layers=depth).to(device)
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_mse = float("inf")
        no_improve = 0
        for it in range(int(max_steps)):
            opt.zero_grad(set_to_none=True)
            loss = compute_eipm_loss(model, X_tr, T_tr, a_sigma, a_h, alpha)
            loss.backward()
            opt.step()

            if (it + 1) % eval_every == 0:
                model.eval()
                preds = []
                for i in range(T_va.numel()):
                    preds.append(locfit_mu_at_t(model=model, X_tr=X_tr, T_tr=T_tr, Y_tr=Y_tr, t_val=T_va[i], a_h=a_h, alpha=alpha))
                pred = torch.stack(preds).view(-1)
                mse = torch.mean((pred - Y_va.view(-1)) ** 2).item()
                if mse < best_mse - float(min_delta):
                    best_mse = float(mse)
                    no_improve = 0
                else:
                    no_improve += 1
                model.train()
                if no_improve >= int(patience):
                    break

        if best_mse == float("inf"):
            model.eval()
            preds = []
            for i in range(T_va.numel()):
                preds.append(locfit_mu_at_t(model=model, X_tr=X_tr, T_tr=T_tr, Y_tr=Y_tr, t_val=T_va[i], a_h=a_h, alpha=alpha))
            pred = torch.stack(preds).view(-1)
            best_mse = float(torch.mean((pred - Y_va.view(-1)) ** 2).item())

        fold_states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        fold_mse.append(float(best_mse))

    return fold_states, fold_mse


# ============================================================
# 5. Final training (full train set, with tuned params)
# ============================================================

def train_final_model_full(
    X_scaled: Tensor,
    T_scaled: Tensor,
    Y: Tensor,
    input_dim: int,
    device: torch.device,
    best_params: Dict,
    depth: int,
    width: int,
    k_folds: int,
    seed: int = 42,
    epochs: int = 300,
) -> Tuple[nn.Module, Dict[str, float]]:
    set_seed(seed)

    log_a_sigma = float(best_params["log_a_sigma"])
    lr = float(best_params["lr"])
    weight_decay = float(best_params["weight_decay"])

    a_sigma = math.exp(log_a_sigma)
    a_h = math.exp(float(best_params["log_a_h"]))
    alpha = float(best_params["alpha"])

    with torch.no_grad():
        d_med = get_med(X_scaled)
        sigma = float(a_sigma) * float(d_med)

    n = int(X_scaled.shape[0])
    splitter = KFold(n_splits=int(k_folds), shuffle=True, random_state=seed)

    model = EIPM(input_dim=input_dim, hidden=width, n_layers=depth).to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_loss = float("inf")
    best_state = None

    for ep in range(int(epochs)):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = compute_eipm_loss(model, X_scaled, T_scaled, a_sigma, a_h, alpha)
        loss.backward()
        opt.step()

        lval = float(loss.item())
        if lval < best_loss:
            best_loss = lval
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # per-epoch diagnostics (val EIPM + val MSE via KFold)
        model.eval()
        with torch.no_grad():
            val_eipm_sum = 0.0
            val_mse_sum = 0.0
            n_folds = 0
            for tr_idx, va_idx in splitter.split(np.arange(n)):
                tr = torch.as_tensor(tr_idx, device=device, dtype=torch.long)
                va = torch.as_tensor(va_idx, device=device, dtype=torch.long)
                X_tr, T_tr, Y_tr = X_scaled[tr], T_scaled[tr], Y[tr]
                X_va, T_va, Y_va = X_scaled[va], T_scaled[va], Y[va]

                val_eipm = compute_eipm_loss(model, X_va, T_va, a_sigma, a_h, alpha)
                preds = []
                for i in range(T_va.numel()):
                    preds.append(locfit_mu_at_t(model=model, X_tr=X_tr, T_tr=T_tr, Y_tr=Y_tr, t_val=T_va[i], a_h=a_h, alpha=alpha))
                pred = torch.stack(preds).view(-1)
                val_mse = torch.mean((pred - Y_va.view(-1)) ** 2)

                val_eipm_sum += float(val_eipm.item())
                val_mse_sum += float(val_mse.item())
                n_folds += 1

            val_eipm_avg = val_eipm_sum / max(1, n_folds)
            val_mse_avg = val_mse_sum / max(1, n_folds)
        print(
            f"[EPOCH {ep + 1}] "
            f"train_eipm={lval:.6g} "
            f"val_eipm={val_eipm_avg:.6g} "
            f"val_mse={val_mse_avg:.6g}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    train_stats = {
        "best_eipm_loss": float(best_loss),
        "sigma": float(sigma),
        "alpha": float(alpha),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "epochs": float(epochs),
    }
    return model, train_stats


# ============================================================
# 6. Data loading (DGP npz)
# ============================================================

@dataclass
class ReplicationData:
    scenario: str
    d_X: int
    n_train: int
    seed: int
    rep_idx: int
    X: np.ndarray  # (n_train, d_X)
    T: np.ndarray  # (n_train,)
    Y: np.ndarray       # (n_train,)
    meta: Dict          # extra scalars/arrays for bookkeeping


def load_replications_from_npz(npz_path: str) -> List[ReplicationData]:
    """
    Load replications from a DGP-generated .npz.

    DGP stacks arrays across replications along axis=0.
    For example:
      X_train: (n_rpt, n_train, d_X)
      T_train: (n_rpt, n_train)
      Y_train:      (n_rpt, n_train)

    We IGNORE eval keys (T_eval, mu_eval, ...).
    """
    data = np.load(npz_path, allow_pickle=True)

    scenario = str(np.array(data["scenario"]).item())
    d_X = int(np.array(data["d_X"]).item())
    n_train = int(np.array(data["n_train"]).item())
    n_rpt = int(np.array(data["n_rpt"]).item())
    seed = int(np.array(data["seed"]).item())

    if "X_train" not in data.files or "T_train" not in data.files or "Y_train" not in data.files:
        raise KeyError("Expected observed keys X_train, T_train, Y_train in dataset")

    X_all = data["X_train"]               # (n_rpt, n_train, d_X)
    T_all = data["T_train"]               # (n_rpt, n_train)
    Y_all = data["Y_train"]                 # (n_rpt, n_train)

    reps: List[ReplicationData] = []
    for r in range(n_rpt):
        meta = {
            "npz_path": npz_path,
            "rep_idx": int(r),
        }

        for k in ["ET", "VT", "beta_T0", "beta_T", "beta_Y0", "beta_YT", "beta_YX", "beta_YXT"]:
            if k in data.files:
                arr = data[k]
                meta[k] = arr[r] if (isinstance(arr, np.ndarray) and arr.ndim >= 1 and arr.shape[0] == n_rpt) else arr

        for k in ["EX", "VX"]:
            if k in data.files:
                meta[k] = data[k]

        reps.append(
            ReplicationData(
                scenario=scenario,
                d_X=d_X,
                n_train=n_train,
                seed=seed,
                rep_idx=r,
                X=np.asarray(X_all[r]),
                T=np.asarray(T_all[r]),
                Y=np.asarray(Y_all[r]),
                meta=meta,
            )
        )

    return reps


# ============================================================
# 7. Resume + logging helpers
# ============================================================

def make_ckpt_path(out_dir: Path, npz_path: str, rep_idx: int, depth: int, width: int) -> Path:
    stem = Path(npz_path).stem
    ckpt_name = stem + f"_rep{rep_idx:03d}_d{depth}_w{width}.pth"
    return out_dir / ckpt_name


def make_hp_path(out_dir: Path, npz_path: str, rep_idx: int, depth: int, width: int) -> Path:
    stem = Path(npz_path).stem
    hp_name = stem + f"_rep{rep_idx:03d}_tuned_depth{depth}_width{width}.json"
    return out_dir / hp_name


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
    "alpha",
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
        writer.writerow({k: row.get(k, "") for k in SUMMARY_FIELDS})


def hp_is_valid(hp_obj: Optional[Dict]) -> bool:
    if not isinstance(hp_obj, dict):
        return False
    if "best_params" not in hp_obj or "best_cv_mse" not in hp_obj:
        return False
    bp = hp_obj.get("best_params")
    if not isinstance(bp, dict):
        return False
    needed = ["log_a_sigma", "log_a_h", "alpha", "lr", "weight_decay"]
    return all(k in bp for k in needed)


def count_tasks(
    files: List[str],
    out_dir: Path,
    depth: int,
    width: int,
    n_rpt: int,
) -> Tuple[int, int, int]:
    """
    Returns (total_files, total_tune_tasks, total_train_tasks) considering resume/overwrite.
    - Train task: each rep whose checkpoint does not exist.
    - Tune task: each rep needing tuning because hp json missing/invalid.
    Note: assumes all files share the same n_rpt.
    """
    total_tune = 0
    total_train = 0

    for npz_path in files:
        for r in range(n_rpt):
            ckpt = make_ckpt_path(out_dir, npz_path, r, depth, width)
            if ckpt.exists():
                continue

            total_train += 1

            hp_path = make_hp_path(out_dir, npz_path, r, depth, width)
            hp_obj = load_json_(hp_path) if (hp_path.exists()) else None
            if not hp_is_valid(hp_obj):
                total_tune += 1

    return len(files), total_tune, total_train


def print_eta(
    *,
    t_start_all: float,
    done_tune: int,
    total_tune: int,
    done_train: int,
    total_train: int,
    tune_times: List[float],
    train_times: List[float],
    cur_npz: str,
    rep_idx_1based: int,
    n_reps_in_file: int,
    file_i_1based: int,
    n_files_total: int,
) -> None:
    elapsed = time.time() - t_start_all

    rem_tune = max(0, total_tune - done_tune)
    rem_train = max(0, total_train - done_train)

    avg_tune = _safe_mean(tune_times)
    avg_train = _safe_mean(train_times)

    if avg_tune is None:
        avg_tune = 0.0
    if avg_train is None:
        avg_train = 0.0

    eta_sec = avg_tune * rem_tune + avg_train * rem_train

    if _KST is None:
        finish_dt = datetime.now() + timedelta(seconds=eta_sec)
        finish_str = finish_dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        finish_dt = datetime.now(_KST) + timedelta(seconds=eta_sec)
        finish_str = finish_dt.strftime("%Y-%m-%d %H:%M:%S KST")

    print(
        "\n"
        f"[PROGRESS @ {_now_str()}]\n"
        f"  - File: {file_i_1based}/{n_files_total} | Current: {Path(cur_npz).name}\n"
        f"  - Rep : {rep_idx_1based}/{n_reps_in_file}\n"
        f"  - Done (tune):  {done_tune}/{total_tune} | Remaining: {rem_tune}\n"
        f"  - Done (train): {done_train}/{total_train} | Remaining: {rem_train}\n"
        f"  - Elapsed: {_fmt_hms(elapsed)}\n"
        f"  - Avg tune/rep executed: {('N/A' if len(tune_times)==0 else _fmt_hms(avg_tune))}\n"
        f"  - Avg train/rep executed: {('N/A' if len(train_times)==0 else _fmt_hms(avg_train))}\n"
        f"  - ETA remaining: {_fmt_hms(eta_sec)}\n"
        f"  - Expected finish: {finish_str}\n"
    )


# ============================================================
# 8. Main
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--data_dir", type=str, default="./datasets")
    p.add_argument("--pattern", type=str, default="sim_*.npz")
    p.add_argument("--out_dir", type=str, default="./models/eipm")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Model / training
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--epochs", type=int, default=300)

    # Tuning
    p.add_argument("--n_trials", type=int, default=25)
    p.add_argument("--k_folds", type=int, default=5)
    p.add_argument("--tune_rep_idx", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--min_delta", type=float, default=1e-6)

    # Reproducibility
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    files = sorted(glob.glob(str(data_dir / args.pattern)))

    if len(files) == 0:
        raise FileNotFoundError(f"No npz files found in {data_dir} with pattern {args.pattern}")

    # assume same n_rpt across all files; read once
    with np.load(files[0], allow_pickle=True) as d0:
        n_rpt_global = int(np.array(d0["n_rpt"]).item())

    t_start_all = time.time()
    plan_t0 = time.time()
    n_files_total, total_tune, total_train = count_tasks(
        files=files,
        out_dir=out_dir,
        depth=int(args.depth),
        width=int(args.width),
        n_rpt=int(n_rpt_global),
    )
    plan_dt = time.time() - plan_t0
    print(f"[PLAN @ {_now_str()}] files_found={n_files_total}, tune_tasks={total_tune}, train_tasks={total_train}, plan_build={_fmt_hms(plan_dt)}")
    print(f"[ARGS] depth={args.depth}, width={args.width}, epochs={args.epochs}, n_trials={args.n_trials}, k_folds={args.k_folds}")

    summary_path = out_dir / "train_summary.csv"

    done_tune = 0
    done_train = 0
    tune_times: List[float] = []
    train_times: List[float] = []

    for file_idx_1based, npz_path in enumerate(files, start=1):
        reps = load_replications_from_npz(npz_path)

        if len(reps) == 0:
            continue

        depth = int(args.depth)
        width = int(args.width)

        reps_to_run: List[ReplicationData] = []
        for rep in reps:
            ckpt_path = make_ckpt_path(out_dir, npz_path, rep.rep_idx, depth, width)
            if ckpt_path.exists():
                continue
            reps_to_run.append(rep)

        if len(reps_to_run) == 0:
            continue

        for rep in reps_to_run:
            ckpt_path = make_ckpt_path(out_dir, npz_path, rep.rep_idx, depth, width)

            hp_path = make_hp_path(out_dir, npz_path, rep.rep_idx, depth, width)
            hp_obj = load_json_(hp_path) if (hp_path.exists()) else None

            best_params = None
            best_value = None

            if hp_is_valid(hp_obj):
                best_params = hp_obj["best_params"]
                best_value = float(hp_obj["best_cv_mse"])

            input_dim = int(rep.d_X) + 1

            if best_params is None or best_value is None:
                X_tune = torch.tensor(rep.X, dtype=torch.float32, device=device)
                T_tune = torch.tensor(rep.T, dtype=torch.float32, device=device)
                Y_tune = torch.tensor(rep.Y, dtype=torch.float32, device=device)

                X_std_tune, T_std_tune, _, _, _, _ = standardize_train(X_tune, T_tune)
                X_scaled_tune = X_std_tune / math.sqrt(float(rep.d_X))
                T_scaled_tune = T_std_tune.view(-1)

                def _boundary_tol(n_trials: int, width: float) -> float:
                    rel = max(0.05, 1.0 / (2.0 * math.sqrt(max(1, int(n_trials)))))
                    return rel * float(width)

                t_tune0 = time.time()
                log_a_h_low = -1.0
                log_a_h_high = 1.0
                expand_factor = 4.0
                max_expand = 5

                best_params = None
                best_value = float("inf")
                best_trial_num = -1

                for expand_i in range(max_expand + 1):
                    def _obj_local(trial, low=log_a_h_low, high=log_a_h_high):
                        return objective_cv_mse(
                            trial=trial,
                            X_scaled=X_scaled_tune,
                            T_scaled=T_scaled_tune,
                            Y=Y_tune,
                            input_dim=input_dim,
                            device=device,
                            depth=depth,
                            width=width,
                            max_steps=int(args.max_steps),
                            patience=int(args.patience),
                            min_delta=float(args.min_delta),
                            n_splits=int(args.k_folds),
                            seed=int(args.seed),
                            log_a_h_low=low,
                            log_a_h_high=high,
                        )

                    study = optuna.create_study(direction="minimize")
                    study.optimize(_obj_local, n_trials=int(args.n_trials), show_progress_bar=False)

                    if float(study.best_value) < float(best_value):
                        best_params = study.best_params
                        best_value = float(study.best_value)
                        best_trial_num = int(study.best_trial.number)

                    best_log = float(study.best_params["log_a_h"])
                    width = log_a_h_high - log_a_h_low
                    tol = _boundary_tol(int(args.n_trials), width)
                    at_low = best_log <= (log_a_h_low + tol)
                    at_high = best_log >= (log_a_h_high - tol)
                    if not (at_low or at_high):
                        break
                    if expand_i == max_expand:
                        break
                    width = log_a_h_high - log_a_h_low
                    center = (log_a_h_low + log_a_h_high) * 0.5
                    new_half = 0.5 * float(expand_factor) * width
                    if at_high and not at_low:
                        log_a_h_low = center
                        log_a_h_high = center + new_half
                    elif at_low and not at_high:
                        log_a_h_low = center - new_half
                        log_a_h_high = center
                    else:
                        log_a_h_low = center - new_half
                        log_a_h_high = center + new_half
                    print(
                        f"[TUNE] expand log_a_h to [{log_a_h_low:.3g}, {log_a_h_high:.3g}] "
                        f"(best_log={best_log:.3g}, tol={tol:.3g})"
                    )

                t_tune1 = time.time()

                atomic_write_json(
                    hp_path,
                    {
                        "npz_path": npz_path,
                        "rep_idx": int(rep.rep_idx),
                        "best_cv_mse": best_value,
                        "best_params": best_params,
                        "best_trial": best_trial_num,
                    },
                )
                hp_obj = {
                    "best_cv_mse": best_value,
                    "best_params": best_params,
                    "best_trial": best_trial_num,
                }

                X_tune = torch.tensor(rep.X, dtype=torch.float32, device=device)
                T_tune = torch.tensor(rep.T, dtype=torch.float32, device=device)
                Y_tune = torch.tensor(rep.Y, dtype=torch.float32, device=device)
                X_std_tune, T_std_tune, _, _, _, _ = standardize_train(X_tune, T_tune)
                X_scaled_tune = X_std_tune / math.sqrt(float(rep.d_X))
                T_scaled_tune = T_std_tune.view(-1)

                fold_states, fold_mse = train_folds_for_params(
                    X_scaled=X_scaled_tune,
                    T_scaled=T_scaled_tune,
                    Y=Y_tune,
                    input_dim=input_dim,
                    device=device,
                    depth=depth,
                    width=width,
                    params=best_params,
                    max_steps=int(args.max_steps),
                    patience=int(args.patience),
                    min_delta=float(args.min_delta),
                    n_splits=int(args.k_folds),
                    seed=int(args.seed),
                )

                fold_dir = out_dir / "tune_folds"
                fold_dir.mkdir(parents=True, exist_ok=True)
                fold_tag = f"{Path(npz_path).stem}_rep{rep.rep_idx:03d}_d{depth}_w{width}"
                fold_path = fold_dir / f"{fold_tag}_best_trial{best_trial_num:03d}.pth"
                atomic_torch_save(
                    fold_path,
                    {
                        "trial": best_trial_num,
                        "fold_states": fold_states,
                        "fold_mse": fold_mse,
                    },
                )

                tune_times.append(t_tune1 - t_tune0)
                done_tune += 1

            fold_tag = f"{Path(npz_path).stem}_rep{rep.rep_idx:03d}_d{depth}_w{width}"
            fold_dir = out_dir / "tune_folds"
            best_trial_num = None
            if hp_obj is not None and isinstance(hp_obj, dict):
                best_trial_num = hp_obj.get("best_trial")
            if isinstance(best_trial_num, int):
                fold_path = fold_dir / f"{fold_tag}_best_trial{best_trial_num:03d}.pth"
            else:
                fold_path = None

            if fold_path is not None and fold_path.exists():
                fold_payload = torch.load(fold_path, map_location="cpu")
                fold_states = fold_payload.get("fold_states")
                if isinstance(fold_states, list) and len(fold_states) > 0:
                    avg_state = {}
                    keys = fold_states[0].keys()
                    for k in keys:
                        stacked = torch.stack([fs[k] for fs in fold_states], dim=0)
                        avg_state[k] = torch.mean(stacked, dim=0)
                    model_state = avg_state
                else:
                    model_state = None
                t_train0 = time.time()
                t_train1 = t_train0
                train_stats = {
                    "best_eipm_loss": float("nan"),
                    "sigma": float("nan"),
                    "alpha": float(best_params.get("alpha", float("nan"))),
                    "lr": float(best_params.get("lr", float("nan"))),
                    "weight_decay": float(best_params.get("weight_decay", float("nan"))),
                    "epochs": float(args.epochs),
                }
            else:
                X = torch.tensor(rep.X, dtype=torch.float32, device=device)
                T = torch.tensor(rep.T, dtype=torch.float32, device=device)
                Y = torch.tensor(rep.Y, dtype=torch.float32, device=device)

                X_std, T_std, _, _, _, _ = standardize_train(X, T)
                X_scaled = X_std / math.sqrt(float(rep.d_X))
                T_scaled = T_std.view(-1)

                t_train0 = time.time()
                model, train_stats = train_final_model_full(
                    X_scaled=X_scaled,
                    T_scaled=T_scaled,
                    Y=Y,
                    input_dim=input_dim,
                    device=device,
                    best_params=best_params,
                    depth=depth,
                    width=width,
                    k_folds=int(args.k_folds),
                    seed=int(args.seed),
                    epochs=int(args.epochs),
                )
                t_train1 = time.time()
                model_state = model.state_dict()

            X_raw = torch.tensor(rep.X, dtype=torch.float32, device=device)
            T_raw = torch.tensor(rep.T, dtype=torch.float32, device=device)
            _, _, x_mean_t, x_std_t, t_mean_t, t_std_t = standardize_train(X_raw, T_raw)

            atomic_torch_save(
                ckpt_path,
                {
                    "model_state": model_state,
                    "ensemble_fold_states": fold_payload.get("fold_states") if fold_path is not None and fold_path.exists() else None,
                    "ensemble_fold_mse": fold_payload.get("fold_mse") if fold_path is not None and fold_path.exists() else None,
                    "ensemble_best_trial": best_trial_num,
                    "best_params": best_params,
                    "best_cv_mse": float(best_value),
                    "train_stats": train_stats,
                    "standardize": {
                        "x_mean": x_mean_t.view(-1).detach().cpu().numpy(),
                        "x_std": x_std_t.view(-1).detach().cpu().numpy(),
                        "t_mean": float(t_mean_t),
                        "t_std": float(t_std_t),
                    },
                    "rep_meta": rep.meta,
                    "script_args": vars(args),
                },
            )

            row = {
                "npz": Path(npz_path).name,
                "rep_idx": int(rep.rep_idx),
                "scenario": rep.scenario,
                "d_X": int(rep.d_X),
                "n_train": int(rep.n_train),
                "depth": int(depth),
                "width": int(width),
                "best_cv_mse": float(best_value),
                "best_eipm_loss": float(train_stats["best_eipm_loss"]),
                "sigma": float(train_stats["sigma"]),
                "alpha": float(train_stats.get("alpha", float("nan"))),
                "lr": float(train_stats["lr"]),
                "weight_decay": float(train_stats["weight_decay"]),
                "epochs": float(train_stats["epochs"]),
                "ckpt_path": str(ckpt_path),
                "hp_path": str(hp_path),
            }
            append_summary_row(summary_path, row)

            train_times.append(t_train1 - t_train0)
            done_train += 1

            print_eta(
                t_start_all=t_start_all,
                done_tune=done_tune,
                total_tune=total_tune,
                done_train=done_train,
                total_train=total_train,
                tune_times=tune_times,
                train_times=train_times,
                cur_npz=npz_path,
                rep_idx_1based=int(rep.rep_idx) + 1,
                n_reps_in_file=len(reps),
                file_i_1based=file_idx_1based,
                n_files_total=n_files_total,
            )


if __name__ == "__main__":
    main()
