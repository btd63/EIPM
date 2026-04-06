# train.py

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold, StratifiedKFold
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
LIB_DIR = ROOT_DIR / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from gpu import select_device
_T_TRANSFORM = "cdf_sigmoid"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(u: np.ndarray) -> np.ndarray:
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


def transform_star_to_t(t_star: np.ndarray, params: Dict[str, np.ndarray]) -> np.ndarray:
    t_star = np.asarray(t_star, dtype=np.float64)
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

    u = (t_star + 1.0) * 0.5
    eps = 1e-8
    u = np.clip(u, eps, 1.0 - eps)

    t = np.empty_like(u, dtype=np.float64)
    mask_mid = (u >= u0) & (u <= u1)
    if np.any(mask_mid):
        t[mask_mid] = np.interp(u[mask_mid], u_grid, T_sorted)
    mask_left = u < u0
    if np.any(mask_left):
        t[mask_left] = Tmin + (_logit(u[mask_left]) - x0) / max(s_left, eps)
    mask_right = u > u1
    if np.any(mask_right):
        t[mask_right] = Tmax + (_logit(u[mask_right]) - x1) / max(s_right, eps)
    return t

def nw_mu_at_t(
    X_tr: Tensor,
    T_tr: Tensor,
    Y_tr: Tensor,
    model: nn.Module,
    t_val: Tensor,
    nn: float,
) -> Tensor:
    t_fixed = t_val.view(1).repeat(X_tr.shape[0]).view(-1, 1)
    with torch.no_grad():
        logw = model(X_tr, t_fixed).view(-1)

    with torch.no_grad():
        t0 = t_val.view(())
        diff = T_tr.view(-1) - t0
        h_knn = _knn_bandwidth(T_tr.view(-1), t_val.view(1), nn=float(nn))[0]
        h_t = float(h_knn.item())
        if h_t <= 1e-8:
            h_t = 1e-8
        u = diff / h_t
        logk = -0.5 * (u ** 2)
        logw_eff = logw + logk
        max_log = torch.max(logw_eff)
        w_eff = torch.exp(logw_eff - max_log)
        s = torch.sum(w_eff)
        if torch.isfinite(s) and float(s) > 0.0:
            w_eff = w_eff / s

    return torch.sum(w_eff * Y_tr.view(-1))


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def parse_sigma_scales(s: str) -> List[float]:
    vals: List[float] = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok:
            continue
        v = float(tok)
        if v <= 0.0:
            raise ValueError(f"sigma scale must be > 0, got {v}")
        vals.append(v)
    if len(vals) == 0:
        vals = [1.0]
    return vals


def parse_log_a_sigma_grid(s: str) -> List[float]:
    vals: List[float] = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(float(tok))
    return vals


def build_optimizer(
    model: nn.Module,
    *,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    adam_eps: float,
    sgd_momentum: float,
) -> optim.Optimizer:
    name = str(optimizer_name).lower()
    if name == "adam":
        return optim.Adam(
            model.parameters(),
            lr=float(lr),
            weight_decay=float(weight_decay),
            eps=float(adam_eps),
        )
    if name == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=float(lr),
            weight_decay=float(weight_decay),
            momentum=float(sgd_momentum),
        )
    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def atomic_torch_save(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def standardize_train(X: Tensor, T: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
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


class EIPM(nn.Module):
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


def rbf(X: Tensor, sigma: float) -> Tensor:
    return torch.exp(-torch.cdist(X, X) ** 2 / (2.0 * (sigma ** 2)))


@torch.no_grad()
def get_med(x: Tensor, max_n: int = 500) -> float:
    if x.ndim == 1:
        x = x.view(-1, 1).contiguous()

    n = x.shape[0]
    if n > max_n:
        idx = torch.randperm(n, device=x.device)[:max_n]
        x = x[idx]

    d = torch.cdist(x, x).flatten()
    d = d[d > 0]
    return float(torch.median(d).item())


@torch.no_grad()
def _knn_bandwidth(T_train: Tensor, t: Tensor, nn: float) -> Tensor:
    T_train = T_train.view(-1)
    t = t.view(-1)
    n = int(T_train.numel())
    if n == 0:
        return t.new_full(t.shape, 1e-8)
    k = int(math.ceil(float(nn) * float(n)))
    k = max(2, min(k, n))
    dist = torch.abs(T_train.view(-1, 1) - t.view(1, -1))
    h, _ = torch.kthvalue(dist, k, dim=0)
    return h.clamp_min(1e-8)


def h0_t(
    T_train: Tensor,
    t: Tensor,
    *,
    nn: float,
) -> Tensor:
    h_knn = _knn_bandwidth(T_train, t, nn=float(nn))
    return h_knn.clamp_min(1e-8).view(-1, 1)


def _sample_anchor_indices(T: Tensor, n_anchors: int) -> Tensor:
    n = int(T.numel())
    if n <= 0:
        return torch.zeros((0,), device=T.device, dtype=torch.long)
    n_anchor_use = int(n_anchors)
    if n_anchor_use > 0 and n_anchor_use < n:
        return torch.randint(0, n, (n_anchor_use,), device=T.device)
    return torch.arange(n, device=T.device, dtype=torch.long)


def _compute_logw_anchor_chunked(
    model: nn.Module,
    X_cond: Tensor,
    t_anchor: Tensor,
    chunk_size: int = 8,
) -> Tensor:
    # Returns logw with shape [m_anchor, n_cond]
    n_cond = int(X_cond.shape[0])
    m_anchor = int(t_anchor.shape[0])
    d_x = int(X_cond.shape[1])
    chunk_size = max(1, int(chunk_size))
    chunks: List[Tensor] = []
    for s in range(0, m_anchor, chunk_size):
        e = min(m_anchor, s + chunk_size)
        t_chunk = t_anchor[s:e]  # [c, 1]
        c = int(t_chunk.shape[0])
        X_rep = X_cond.unsqueeze(0).expand(c, -1, -1).reshape(c * n_cond, d_x)
        t_rep = t_chunk.expand(-1, n_cond).reshape(-1)
        logw_chunk = model(X_rep, t_rep).view(c, n_cond)
        chunks.append(logw_chunk)
    return torch.cat(chunks, dim=0)


def _build_kx(
    X_a: Tensor,
    X_b: Tensor,
    sigma_scales: List[float] | Tuple[float, ...],
    sigma_base: float,
) -> Tensor:
    K = None
    n_sig = 0
    for sc in sigma_scales:
        sigma = max(float(sc) * float(sigma_base), 1e-12)
        if X_a.data_ptr() == X_b.data_ptr() and X_a.shape == X_b.shape:
            Ki = rbf(X_a, sigma)
        else:
            Ki = torch.exp(-torch.cdist(X_a, X_b) ** 2 / (2.0 * (sigma ** 2)))
        K = Ki if K is None else (K + Ki)
        n_sig += 1
    return K / float(max(1, n_sig))


def compute_eipm_loss(
    model: nn.Module,
    X: Tensor,
    T: Tensor,
    a_sigma: float,
    nn: float,
    d_med_fixed: float,
    sigma_scales: List[float] | Tuple[float, ...] = (1.0,),
    leave_one_out_conditional: bool = False,
    n_anchors: int = 0,
    return_aux: bool = False,
    anchor_chunk_size: int = 8,
    anchor_idx: Tensor | None = None,
) -> Tensor | Tuple[Tensor, Dict[str, float]]:
    n_cond = int(T.numel())
    if n_cond <= 0:
        z = torch.zeros((), dtype=X.dtype, device=X.device)
        return z

    idx_anchor = anchor_idx if anchor_idx is not None else _sample_anchor_indices(T, n_anchors)
    idx_anchor = idx_anchor.to(device=T.device, dtype=torch.long)
    n_anchor = int(idx_anchor.numel())
    t_anchor = T.view(-1)[idx_anchor].view(-1, 1)  # [n_anchor,1]

    # w(X, t_anchor): evaluate model with fixed t per anchor in chunks.
    logw = _compute_logw_anchor_chunked(
        model=model,
        X_cond=X,
        t_anchor=t_anchor,
        chunk_size=int(anchor_chunk_size),
    )  # [n_anchor, n_cond]
    logw = logw - torch.max(logw, dim=1, keepdim=True).values
    W_anchor = torch.exp(logw)

    # K_t(T_j, t_anchor_k) with local bandwidth at each anchor.
    h_anchor = h0_t(T_train=T, t=t_anchor.view(-1), nn=float(nn)).view(-1, 1).clamp_min(1e-8)  # [m,1]
    diff_sq = (T.view(1, -1) - t_anchor) ** 2  # [m,n]
    K_T_anchor = torch.exp(-0.5 * diff_sq / (h_anchor ** 2))

    # LOO at conditional anchor: when anchor is exactly T[idx_anchor[k]], remove self-contribution j=idx_anchor[k].
    if bool(leave_one_out_conditional):
        K_T_anchor = K_T_anchor.clone()
        K_T_anchor[torch.arange(n_anchor, device=T.device), idx_anchor] = 0.0

    sigma_base = float(a_sigma) * float(d_med_fixed)
    sigma_base = max(sigma_base, 1e-12)
    K_X = _build_kx(X, X, sigma_scales=sigma_scales, sigma_base=sigma_base)

    A_num = K_T_anchor * W_anchor
    A_den = torch.sum(A_num, dim=1, keepdim=True).clamp_min(1e-8)
    A = A_num / A_den

    term1 = torch.sum((A @ K_X) * A, dim=1)
    term2 = torch.mean(K_X)
    term3 = 2.0 * torch.mean(A @ K_X, dim=1)

    loss = torch.mean(term1 - term3 + term2)
    if bool(return_aux):
        ess_anchor = 1.0 / torch.sum(A * A, dim=1).clamp_min(1e-12)
        w_mean1 = W_anchor / torch.mean(W_anchor, dim=1, keepdim=True).clamp_min(1e-12)
        aux = {
            "n_anchor": float(n_anchor),
            "ess_mean": float(torch.mean(ess_anchor).item()),
            "w_max": float(torch.max(w_mean1).item()),
        }
        return loss, aux
    return loss


def compute_eipm_val_mmd_cross(
    model: nn.Module,
    X_tr: Tensor,
    T_tr: Tensor,
    X_va: Tensor,
    T_va: Tensor,
    a_sigma: float,
    nn: float,
    d_med_fixed: float,
    sigma_scales: List[float] | Tuple[float, ...] = (1.0,),
    leave_one_out_conditional: bool = False,
    n_anchors: int = 0,
    return_aux: bool = False,
    anchor_chunk_size: int = 8,
    anchor_idx: Tensor | None = None,
) -> Tensor | Tuple[Tensor, Dict[str, float]]:
    n_tr = int(T_tr.numel())
    n_va = int(T_va.numel())
    if n_tr <= 0 or n_va <= 0:
        z = torch.zeros((), dtype=X_tr.dtype, device=X_tr.device)
        return z

    idx_anchor_va = anchor_idx if anchor_idx is not None else _sample_anchor_indices(T_va, n_anchors)
    idx_anchor_va = idx_anchor_va.to(device=T_va.device, dtype=torch.long)
    n_anchor = int(idx_anchor_va.numel())
    t_anchor = T_va.view(-1)[idx_anchor_va].view(-1, 1)  # [n_anchor,1]

    logw = _compute_logw_anchor_chunked(
        model=model,
        X_cond=X_tr,
        t_anchor=t_anchor,
        chunk_size=int(anchor_chunk_size),
    )  # [n_anchor, n_tr]
    logw = logw - torch.max(logw, dim=1, keepdim=True).values
    W_anchor = torch.exp(logw)

    h_anchor = h0_t(T_train=T_tr, t=t_anchor.view(-1), nn=float(nn)).view(-1, 1).clamp_min(1e-8)
    diff_sq = (T_tr.view(1, -1) - t_anchor) ** 2  # [n_anchor, n_tr]
    K_T_anchor = torch.exp(-0.5 * diff_sq / (h_anchor ** 2))

    if bool(leave_one_out_conditional):
        # For cross-split evaluation, remove exact matches only.
        match = T_tr.view(1, -1) == t_anchor
        K_T_anchor = K_T_anchor.masked_fill(match, 0.0)

    sigma_base = float(a_sigma) * float(d_med_fixed)
    sigma_base = max(sigma_base, 1e-12)
    K_trtr = _build_kx(X_tr, X_tr, sigma_scales=sigma_scales, sigma_base=sigma_base)
    K_vava = _build_kx(X_va, X_va, sigma_scales=sigma_scales, sigma_base=sigma_base)
    K_trva = _build_kx(X_tr, X_va, sigma_scales=sigma_scales, sigma_base=sigma_base)

    A_num = K_T_anchor * W_anchor
    A_den = torch.sum(A_num, dim=1, keepdim=True).clamp_min(1e-8)
    A = A_num / A_den

    term1 = torch.sum((A @ K_trtr) * A, dim=1)  # E_cond,cond
    term2 = torch.mean(K_vava)  # E_target,target
    term3 = 2.0 * torch.mean(A @ torch.mean(K_trva, dim=1), dim=0)  # 2 E_cond,target
    loss = torch.mean(term1 - term3 + term2)

    if bool(return_aux):
        ess_anchor = 1.0 / torch.sum(A * A, dim=1).clamp_min(1e-12)
        w_mean1 = W_anchor / torch.mean(W_anchor, dim=1, keepdim=True).clamp_min(1e-12)
        aux = {
            "n_anchor": float(n_anchor),
            "ess_mean": float(torch.mean(ess_anchor).item()),
            "w_max": float(torch.max(w_mean1).item()),
        }
        return loss, aux
    return loss


def compute_weight_regularizer(
    model: nn.Module,
    X: Tensor,
    T: Tensor,
    reg_type: str = "none",
    n_anchors: int = 0,
    anchor_chunk_size: int = 8,
    anchor_idx: Tensor | None = None,
) -> Tensor:
    rtype = str(reg_type).lower()
    if rtype == "none":
        return torch.zeros((), device=X.device, dtype=X.dtype)

    idx_anchor = anchor_idx if anchor_idx is not None else _sample_anchor_indices(T, n_anchors)
    idx_anchor = idx_anchor.to(device=T.device, dtype=torch.long)
    if idx_anchor.numel() == 0:
        return torch.zeros((), device=X.device, dtype=X.dtype)
    t_anchor = T.view(-1)[idx_anchor].view(-1, 1)
    logw = _compute_logw_anchor_chunked(
        model=model,
        X_cond=X,
        t_anchor=t_anchor,
        chunk_size=int(anchor_chunk_size),
    )
    logw = logw - torch.max(logw, dim=1, keepdim=True).values
    w = torch.exp(logw)
    w = w / (torch.mean(w, dim=1, keepdim=True) + 1e-8)  # [m,n], mean-normalized per anchor

    if rtype == "l2_w":
        # Penalize deviation from constant weights (w=1 after normalization)
        return torch.mean((w - 1.0) ** 2)
    if rtype == "var_w":
        return torch.mean(torch.var(w, dim=1, unbiased=False))
    if rtype == "ess_inv":
        # mean(w^2)-1 is 0 at uniform weights and increases with concentration
        return torch.mean(w ** 2) - 1.0

    raise ValueError(f"Unknown reg_type: {reg_type}")


@torch.no_grad()
def compute_h_curve(
    T_scaled: Tensor,
    t_grid_scaled: Tensor,
    nn: float,
) -> Tensor:
    h_vals = h0_t(T_train=T_scaled, t=t_grid_scaled, nn=nn).view(-1)
    return h_vals


def plot_h_curve(t_grid_raw: np.ndarray, h_raw: np.ndarray, out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for plotting h(t).") from exc

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t_grid_raw, h_raw, color="black", linewidth=1.5)
    ax.set_xlabel("t")
    ax.set_ylabel("h(t)")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def objective_cv_mse(
    trial,
    X_raw: Tensor,
    T_raw_np: np.ndarray,
    Y_raw: Tensor,
    input_dim: int,
    device: torch.device,
    depth: int,
    width: int,
    max_steps: int = 500,
    patience: int = 3,
    min_delta: float = 1e-3,
    n_splits: int = 5,
    seed: int = 42,
    eval_every: int = 10,
    fixed_lr: float = 0.003,
    fixed_weight_decay: float = 1.0e-6,
    fixed_nn: float = 0.7,
    fixed_optimizer: str = "adam",
    fixed_adam_eps: float = 1.0e-8,
    fixed_sgd_momentum: float = 0.0,
    fixed_grad_clip: float = 0.0,
    fixed_reg_type: str = "none",
    fixed_reg_lambda: float = 0.0,
    fixed_loo_conditional: bool = False,
    fixed_n_anchors: int = 0,
    tuning_trace: bool = False,
    early_stop_min_steps: int = 120,
    fixed_sigma_scales: List[float] | None = None,
) -> float:
    log_a_sigma = trial.suggest_float("log_a_sigma", -2.0, 2.0)

    a_sigma = math.exp(float(log_a_sigma))

    lr = float(fixed_lr)
    weight_decay = float(fixed_weight_decay)
    nn_val = float(fixed_nn)
    optimizer_name = str(fixed_optimizer).lower()
    adam_eps = float(fixed_adam_eps)
    sgd_momentum = float(fixed_sgd_momentum)
    grad_clip = float(fixed_grad_clip)
    reg_type = str(fixed_reg_type)
    reg_lambda = float(fixed_reg_lambda)
    loo_conditional = bool(fixed_loo_conditional)
    n_anchors = int(fixed_n_anchors)
    sigma_scales = [1.0] if fixed_sigma_scales is None else list(fixed_sigma_scales)

    T_raw_np = np.asarray(T_raw_np, dtype=np.float64).reshape(-1)
    with torch.no_grad():
        y_strat = (T_raw_np <= 0.0).astype(int)
        use_strat = (np.unique(y_strat).size >= 2)

    if use_strat:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(np.zeros_like(y_strat), y_strat)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(np.arange(X_raw.shape[0]))

    eval_every = int(max(1, int(eval_every)))

    # Train all folds in lockstep, and early-stop by mean validation MSE across folds.
    fold_objs = []
    for _, (tr_idx, va_idx) in enumerate(split_iter, start=1):
        tr = torch.as_tensor(tr_idx, device=device, dtype=torch.long)
        va = torch.as_tensor(va_idx, device=device, dtype=torch.long)

        # Fit fold-specific T transform and standardization on train fold only.
        tstar_params_fold = fit_tstar_transform(T_raw_np[tr_idx], tail_k=0)
        T_tr_star = transform_t_to_star(T_raw_np[tr_idx], tstar_params_fold)
        T_va_star = transform_t_to_star(T_raw_np[va_idx], tstar_params_fold)

        X_tr_raw = X_raw[tr]
        X_va_raw = X_raw[va]
        Y_tr = Y_raw[tr]
        Y_va = Y_raw[va]

        x_mean = X_tr_raw.mean(dim=0, keepdim=True)
        x_var = X_tr_raw.var(dim=0, unbiased=False, keepdim=True)
        x_std = torch.sqrt(x_var).clamp_min(1e-8)
        X_tr_std = (X_tr_raw - x_mean) / x_std
        X_va_std = (X_va_raw - x_mean) / x_std

        T_tr_t = torch.tensor(T_tr_star, dtype=torch.float32, device=device).view(-1)
        T_va_t = torch.tensor(T_va_star, dtype=torch.float32, device=device).view(-1)
        t_mean = T_tr_t.mean()
        t_var = T_tr_t.var(unbiased=False)
        t_std = torch.sqrt(t_var).clamp_min(1e-8)
        T_tr = (T_tr_t - t_mean) / t_std
        T_va = (T_va_t - t_mean) / t_std

        X_tr = X_tr_std / math.sqrt(float(input_dim - 1))
        X_va = X_va_std / math.sqrt(float(input_dim - 1))
        d_med_fold = get_med(X_tr, max_n=int(X_tr.shape[0]))
        model = EIPM(input_dim=input_dim, hidden=width, n_layers=depth).to(device)
        opt = build_optimizer(
            model,
            optimizer_name=optimizer_name,
            lr=lr,
            weight_decay=weight_decay,
            adam_eps=adam_eps,
            sgd_momentum=sgd_momentum,
        )
        fold_objs.append(
            {
                "X_tr": X_tr,
                "T_tr": T_tr,
                "Y_tr": Y_tr,
                "X_va": X_va,
                "T_va": T_va,
                "Y_va": Y_va,
                "d_med": d_med_fold,
                "model": model,
                "opt": opt,
            }
        )

    best_mean_mse = float("inf")
    best_mean_mmd = float("inf")
    mse_at_best_mmd = float("inf")
    no_improve = 0
    printed_debug = False
    for it in range(int(max_steps)):
        # one training step per fold
        cur_train_total_losses: List[float] = []
        cur_train_mmd_losses: List[float] = []
        cur_train_reg_losses: List[float] = []
        for fo in fold_objs:
            model = fo["model"]
            opt = fo["opt"]
            opt.zero_grad(set_to_none=True)
            train_mmd = compute_eipm_loss(
                model=model,
                X=fo["X_tr"],
                T=fo["T_tr"],
                a_sigma=a_sigma,
                nn=nn_val,
                d_med_fixed=fo["d_med"],
                sigma_scales=sigma_scales,
                leave_one_out_conditional=loo_conditional,
                n_anchors=n_anchors,
            )
            reg_loss = compute_weight_regularizer(
                model=model,
                X=fo["X_tr"],
                T=fo["T_tr"],
                reg_type=reg_type,
                n_anchors=n_anchors,
            )
            loss = train_mmd + reg_lambda * reg_loss
            if not torch.isfinite(loss):
                raise RuntimeError(
                    "EIPM loss is not finite: "
                    f"trial={getattr(trial, 'number', 'NA')} it={it} "
                    f"a_sigma={a_sigma:.3g} "
                    f"lr={lr:.3g} wd={weight_decay:.3g} "
                )
            loss.backward()
            if grad_clip > 0.0:
                clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            opt.step()
            cur_train_total_losses.append(float(loss.item()))
            cur_train_mmd_losses.append(float(train_mmd.item()))
            cur_train_reg_losses.append(float(reg_loss.item()))

        if (it + 1) % eval_every == 0:
            cur_fold_mse: List[float] = []
            cur_fold_mmd: List[float] = []
            for fo in fold_objs:
                model = fo["model"]
                model.eval()
                preds = []
                for i in range(fo["T_va"].numel()):
                    mu_i = nw_mu_at_t(
                        model=model,
                        X_tr=fo["X_tr"],
                        T_tr=fo["T_tr"],
                        Y_tr=fo["Y_tr"],
                        t_val=fo["T_va"][i],
                        nn=nn_val,
                    )
                    preds.append(mu_i)
                pred = torch.stack(preds).view(-1)
                mse = torch.mean((pred - fo["Y_va"].view(-1)) ** 2).item()
                cur_fold_mse.append(float(mse))
                mmd_val = float(
                    compute_eipm_val_mmd_cross(
                        model=model,
                        X_tr=fo["X_tr"],
                        T_tr=fo["T_tr"],
                        X_va=fo["X_va"],
                        T_va=fo["T_va"],
                        a_sigma=a_sigma,
                        nn=nn_val,
                        d_med_fixed=fo["d_med"],
                        sigma_scales=sigma_scales,
                        leave_one_out_conditional=loo_conditional,
                        n_anchors=n_anchors,
                    ).item()
                )
                cur_fold_mmd.append(mmd_val)
                model.train()

            cur_mean_mse = float(np.mean(cur_fold_mse))
            cur_mean_mmd = float(np.mean(cur_fold_mmd))
            cur_mean_train_total = float(np.mean(cur_train_total_losses)) if len(cur_train_total_losses) > 0 else float("nan")
            cur_mean_train_mmd = float(np.mean(cur_train_mmd_losses)) if len(cur_train_mmd_losses) > 0 else float("nan")
            cur_mean_train_reg = float(np.mean(cur_train_reg_losses)) if len(cur_train_reg_losses) > 0 else float("nan")
            if cur_mean_mse < best_mean_mse:
                best_mean_mse = cur_mean_mse
            if tuning_trace:
                print(
                    f"[TUNE] trial={getattr(trial, 'number', 'NA')} "
                    f"step={it+1:04d} log_a_sigma={log_a_sigma:.4f} "
                    f"mean_train_total={cur_mean_train_total:.6g} "
                    f"mean_train_mmd={cur_mean_train_mmd:.6g} "
                    f"mean_train_reg={cur_mean_train_reg:.6g} "
                    f"mean_val_mse={cur_mean_mse:.6g} best_mse={best_mean_mse:.6g} "
                    f"mean_val_mmd={cur_mean_mmd:.6g} best_mmd={best_mean_mmd:.6g}"
                )
            # Early stopping criterion: validation MMD improvement.
            if cur_mean_mmd < best_mean_mmd - float(min_delta):
                best_mean_mmd = cur_mean_mmd
                mse_at_best_mmd = cur_mean_mse
                no_improve = 0
            else:
                no_improve += 1

            if not printed_debug and len(fold_objs) > 0:
                fo = fold_objs[0]
                with torch.no_grad():
                    t_dbg = fo["T_va"][0].view(())
                    t_fixed = t_dbg.view(1).repeat(fo["X_tr"].shape[0]).view(-1, 1)
                    w_dbg = torch.exp(fo["model"](fo["X_tr"], t_fixed)).view(-1)
                    diff = fo["T_tr"].view(-1) - t_dbg
                    h_knn = _knn_bandwidth(fo["T_tr"].view(-1), t_dbg.view(1), nn=nn_val)[0]
                    h_t = float(h_knn.item())
                    if h_t <= 1e-8:
                        h_t = 1e-8
                    k = torch.exp(-0.5 * (diff / h_t) ** 2)
                    w_eff = w_dbg * k
                    print(
                        f"[DBG] w_std={float(w_dbg.std().item()):.6g} "
                        f"w_eff_sum={float(w_eff.sum().item()):.6g} "
                        f"h_t={float(h_t):.6g}"
                    )
                    printed_debug = True

            if (it + 1) >= int(early_stop_min_steps) and no_improve >= int(patience):
                break

    if best_mean_mse == float("inf"):
        # fallback: evaluate once at the end
        cur_fold_mse = []
        for fo in fold_objs:
            fo["model"].eval()
            preds = []
            for i in range(fo["T_va"].numel()):
                preds.append(
                    nw_mu_at_t(
                        model=fo["model"],
                        X_tr=fo["X_tr"],
                        T_tr=fo["T_tr"],
                        Y_tr=fo["Y_tr"],
                        t_val=fo["T_va"][i],
                        nn=nn_val,
                    )
                )
            pred = torch.stack(preds).view(-1)
            cur_fold_mse.append(float(torch.mean((pred - fo["Y_va"].view(-1)) ** 2).item()))
        best_mean_mse = float(np.mean(cur_fold_mse))
    if best_mean_mmd == float("inf"):
        cur_fold_mmd = []
        for fo in fold_objs:
            fo["model"].eval()
            cur_fold_mmd.append(
                float(
                    compute_eipm_val_mmd_cross(
                        model=fo["model"],
                        X_tr=fo["X_tr"],
                        T_tr=fo["T_tr"],
                        X_va=fo["X_va"],
                        T_va=fo["T_va"],
                        a_sigma=a_sigma,
                        nn=nn_val,
                        d_med_fixed=fo["d_med"],
                        sigma_scales=sigma_scales,
                        leave_one_out_conditional=loo_conditional,
                        n_anchors=n_anchors,
                    ).item()
                )
            )
        best_mean_mmd = float(np.mean(cur_fold_mmd))
    if mse_at_best_mmd == float("inf"):
        mse_at_best_mmd = best_mean_mse
    trial.set_user_attr("best_mean_val_mse", float(best_mean_mse))
    trial.set_user_attr("best_mean_val_mmd", float(best_mean_mmd))
    trial.set_user_attr("mse_at_best_mmd", float(mse_at_best_mmd))

    # tuning score: mean validation MSE measured at the step where mean validation MMD is best.
    return float(mse_at_best_mmd)


def select_best_steps_cv(
    X_scaled: Tensor,
    T_scaled: Tensor,
    Y: Tensor,
    T_raw_np: np.ndarray,
    input_dim: int,
    device: torch.device,
    depth: int,
    width: int,
    *,
    log_a_sigma: float,
    nn_val: float,
    max_steps: int,
    patience: int,
    min_delta: float,
    n_splits: int,
    seed: int,
    eval_every: int,
    lr: float,
    weight_decay: float,
    optimizer_name: str,
    adam_eps: float,
    sgd_momentum: float,
    grad_clip: float,
    reg_type: str,
    reg_lambda: float,
    leave_one_out_conditional: bool,
    n_anchors: int,
    sigma_scales: List[float] | None = None,
    early_stop_min_steps: int = 120,
    train_trace: bool = True,
) -> Tuple[int, float, List[Dict[str, Tensor]], List[float]]:
    if int(n_splits) < 2:
        raise ValueError("k_folds must be >= 2 for CV stopping.")

    a_sigma = math.exp(float(log_a_sigma))
    sigma_scales_ = [1.0] if sigma_scales is None else list(sigma_scales)
    T_raw_np = np.asarray(T_raw_np, dtype=np.float64).reshape(-1)
    y_strat = (T_raw_np <= 0.0).astype(int)
    use_strat = (np.unique(y_strat).size >= 2)
    if use_strat:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(np.zeros_like(y_strat), y_strat)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(np.arange(X_scaled.shape[0]))

    eval_every = int(max(1, int(eval_every)))
    fold_objs = []
    for tr_idx, va_idx in split_iter:
        tr = torch.as_tensor(tr_idx, device=device, dtype=torch.long)
        va = torch.as_tensor(va_idx, device=device, dtype=torch.long)
        X_tr = X_scaled[tr]
        T_tr = T_scaled[tr]
        X_va = X_scaled[va]
        T_va = T_scaled[va]
        d_med_fold = get_med(X_tr, max_n=int(X_tr.shape[0]))
        model = EIPM(input_dim=input_dim, hidden=width, n_layers=depth).to(device)
        opt = build_optimizer(
            model,
            optimizer_name=optimizer_name,
            lr=lr,
            weight_decay=weight_decay,
            adam_eps=adam_eps,
            sgd_momentum=sgd_momentum,
        )
        fold_objs.append(
            {
                "tr_idx": tr,
                "va_idx": va,
                "X_tr": X_tr,
                "T_tr": T_tr,
                "X_va": X_va,
                "T_va": T_va,
                "d_med": d_med_fold,
                "model": model,
                "opt": opt,
            }
        )

    best_step = eval_every
    best_val_mmd = float("inf")
    best_fold_states: List[Dict[str, Tensor]] | None = None
    best_fold_mse: List[float] | None = None
    no_improve = 0

    for it in range(int(max_steps)):
        for fo in fold_objs:
            fo["opt"].zero_grad(set_to_none=True)
            tr_mmd = compute_eipm_loss(
                model=fo["model"],
                X=fo["X_tr"],
                T=fo["T_tr"],
                a_sigma=a_sigma,
                nn=nn_val,
                d_med_fixed=fo["d_med"],
                sigma_scales=sigma_scales_,
                leave_one_out_conditional=leave_one_out_conditional,
                n_anchors=n_anchors,
            )
            tr_reg = compute_weight_regularizer(
                model=fo["model"],
                X=fo["X_tr"],
                T=fo["T_tr"],
                reg_type=reg_type,
                n_anchors=n_anchors,
            )
            loss = tr_mmd + float(reg_lambda) * tr_reg
            loss.backward()
            if float(grad_clip) > 0.0:
                clip_grad_norm_(fo["model"].parameters(), max_norm=float(grad_clip))
            fo["opt"].step()

        if (it + 1) % eval_every == 0:
            cur_mmd = []
            cur_mse = []
            for fo in fold_objs:
                fo["model"].eval()
                with torch.no_grad():
                    vm = float(
                        compute_eipm_val_mmd_cross(
                            model=fo["model"],
                            X_tr=fo["X_tr"],
                            T_tr=fo["T_tr"],
                            X_va=fo["X_va"],
                            T_va=fo["T_va"],
                            a_sigma=a_sigma,
                            nn=nn_val,
                            d_med_fixed=fo["d_med"],
                            sigma_scales=sigma_scales_,
                            leave_one_out_conditional=leave_one_out_conditional,
                            n_anchors=n_anchors,
                        ).item()
                    )
                    Y_tr = Y[fo["tr_idx"]]
                    Y_va = Y[fo["va_idx"]]
                    preds = []
                    for i in range(fo["T_va"].numel()):
                        preds.append(
                            nw_mu_at_t(
                                model=fo["model"],
                                X_tr=fo["X_tr"],
                                T_tr=fo["T_tr"],
                                Y_tr=Y_tr,
                                t_val=fo["T_va"][i],
                                nn=nn_val,
                            )
                        )
                    pred = torch.stack(preds).view(-1)
                    mse = float(torch.mean((pred - Y_va.view(-1)) ** 2).item())
                cur_mmd.append(vm)
                cur_mse.append(mse)
                fo["model"].train()
            cur_mean = float(np.mean(cur_mmd))
            if train_trace:
                print(
                    f"[CVSTOP] step={it+1:04d} "
                    f"mean_val_mmd={cur_mean:.6g} best_val_mmd={best_val_mmd:.6g}"
                )
            if cur_mean < best_val_mmd - float(min_delta):
                best_val_mmd = cur_mean
                best_step = it + 1
                best_fold_states = [{k: v.detach().cpu().clone() for k, v in fo["model"].state_dict().items()} for fo in fold_objs]
                best_fold_mse = [float(v) for v in cur_mse]
                no_improve = 0
            else:
                no_improve += 1
            if (it + 1) >= int(early_stop_min_steps) and no_improve >= int(patience):
                break

    if best_fold_states is None:
        best_fold_states = [{k: v.detach().cpu().clone() for k, v in fo["model"].state_dict().items()} for fo in fold_objs]
        best_fold_mse = []
        for fo in fold_objs:
            tr_idx = fo["tr_idx"]
            va_idx = fo["va_idx"]
            Y_tr = Y[tr_idx]
            Y_va = Y[va_idx]
            preds = []
            for i in range(fo["T_va"].numel()):
                preds.append(
                    nw_mu_at_t(
                        model=fo["model"],
                        X_tr=fo["X_tr"],
                        T_tr=fo["T_tr"],
                        Y_tr=Y_tr,
                        t_val=fo["T_va"][i],
                        nn=nn_val,
                    )
                )
            pred = torch.stack(preds).view(-1)
            mse = torch.mean((pred - Y_va.view(-1)) ** 2).item()
            best_fold_mse.append(float(mse))
    return int(best_step), float(best_val_mmd), best_fold_states, best_fold_mse


def train_full_with_cvstop(
    X_scaled: Tensor,
    T_scaled: Tensor,
    T_raw_np: np.ndarray,
    input_dim: int,
    device: torch.device,
    depth: int,
    width: int,
    *,
    log_a_sigma: float,
    nn: float,
    max_steps: int,
    eval_every: int,
    patience: int,
    min_delta: float,
    early_stop_min_steps: int,
    n_splits: int,
    seed: int,
    lr: float,
    weight_decay: float,
    optimizer_name: str,
    adam_eps: float,
    sgd_momentum: float,
    grad_clip: float,
    reg_type: str,
    reg_lambda: float,
    leave_one_out_conditional: bool,
    n_anchors: int,
    sigma_scales: List[float] | None = None,
    train_trace: bool = True,
) -> Tuple[Dict[str, Tensor], float, int, float]:
    a_sigma = math.exp(float(log_a_sigma))
    sigma_scales_ = [1.0] if sigma_scales is None else list(sigma_scales)
    eval_every = int(max(1, int(eval_every)))

    T_raw_np = np.asarray(T_raw_np, dtype=np.float64).reshape(-1)
    y_strat = (T_raw_np <= 0.0).astype(int)
    use_strat = (np.unique(y_strat).size >= 2)
    if use_strat:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(np.zeros_like(y_strat), y_strat)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(np.arange(X_scaled.shape[0]))

    fold_views = []
    for tr_idx, va_idx in split_iter:
        tr = torch.as_tensor(tr_idx, dtype=torch.long, device=device)
        va = torch.as_tensor(va_idx, dtype=torch.long, device=device)
        X_tr = X_scaled[tr]
        T_tr = T_scaled[tr]
        X_va = X_scaled[va]
        T_va = T_scaled[va]
        d_med_fold = get_med(X_tr, max_n=int(X_tr.shape[0]))
        fold_views.append(
            {
                "X_tr": X_tr,
                "T_tr": T_tr,
                "X_va": X_va,
                "T_va": T_va,
                "d_med": d_med_fold,
            }
        )

    d_med_full = get_med(X_scaled, max_n=int(X_scaled.shape[0]))
    model = EIPM(input_dim=input_dim, hidden=width, n_layers=depth).to(device)
    opt = build_optimizer(
        model,
        optimizer_name=optimizer_name,
        lr=float(lr),
        weight_decay=float(weight_decay),
        adam_eps=float(adam_eps),
        sgd_momentum=float(sgd_momentum),
    )

    best_cv_val_mmd = float("inf")
    best_step = 0
    no_improve = 0
    last_loss = float("nan")
    best_state: Dict[str, Tensor] | None = None

    for it in range(int(max_steps)):
        opt.zero_grad(set_to_none=True)
        tr_mmd = compute_eipm_loss(
            model=model,
            X=X_scaled,
            T=T_scaled,
            a_sigma=a_sigma,
            nn=float(nn),
            d_med_fixed=d_med_full,
            sigma_scales=sigma_scales_,
            leave_one_out_conditional=leave_one_out_conditional,
            n_anchors=n_anchors,
        )
        tr_reg = compute_weight_regularizer(
            model=model,
            X=X_scaled,
            T=T_scaled,
            reg_type=reg_type,
            n_anchors=n_anchors,
        )
        loss = tr_mmd + float(reg_lambda) * tr_reg
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss in full training at step={it+1}")
        loss.backward()
        if float(grad_clip) > 0.0:
            clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
        opt.step()
        last_loss = float(loss.item())

        if (it + 1) % eval_every == 0 or it == 0:
            with torch.no_grad():
                fold_vals = []
                for fo in fold_views:
                    vm = float(
                        compute_eipm_val_mmd_cross(
                            model=model,
                            X_tr=fo["X_tr"],
                            T_tr=fo["T_tr"],
                            X_va=fo["X_va"],
                            T_va=fo["T_va"],
                            a_sigma=a_sigma,
                            nn=float(nn),
                            d_med_fixed=fo["d_med"],
                            sigma_scales=sigma_scales_,
                            leave_one_out_conditional=leave_one_out_conditional,
                            n_anchors=n_anchors,
                        ).item()
                    )
                    fold_vals.append(vm)
                mean_cv = float(np.mean(fold_vals))

            if train_trace:
                print(
                    f"[TRAIN] step={it+1:04d} total={last_loss:.6g} "
                    f"mmd={float(tr_mmd.item()):.6g} reg={float(tr_reg.item()):.6g} "
                    f"cv_val_mmd={mean_cv:.6g} best_cv_val_mmd={best_cv_val_mmd:.6g}"
                )

            if mean_cv < best_cv_val_mmd - float(min_delta):
                best_cv_val_mmd = mean_cv
                best_step = it + 1
                no_improve = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1

            if (it + 1) >= int(early_stop_min_steps) and no_improve >= int(patience):
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_step = int(max_steps)
    return best_state, float(last_loss), int(best_step), float(best_cv_val_mmd)


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
    eval_every: int = 10,
    train_trace: bool = True,
    early_stop_min_steps: int = 120,
    sigma_scales: List[float] | None = None,
    optimizer_name: str = "adam",
    adam_eps: float = 1.0e-8,
    sgd_momentum: float = 0.0,
    grad_clip: float = 0.0,
    reg_type: str = "none",
    reg_lambda: float = 0.0,
    leave_one_out_conditional: bool = False,
    n_anchors: int = 0,
    use_early_stop: bool = True,
) -> Tuple[List[Dict[str, Tensor]], List[float]]:
    a_sigma = math.exp(float(params["log_a_sigma"]))
    nn_val = float(params["nn"])
    lr = float(params["lr"])
    weight_decay = float(params["weight_decay"])
    sigma_scales_ = [1.0] if sigma_scales is None else list(sigma_scales)

    with torch.no_grad():
        y_strat = (T_scaled.detach().cpu().numpy() == 0.0).astype(int)
        use_strat = (np.unique(y_strat).size >= 2)

    if use_strat:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(np.zeros_like(y_strat), y_strat)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(np.arange(X_scaled.shape[0]))

    eval_every = int(max(1, int(eval_every)))
    fold_states: List[Dict[str, Tensor]] = []
    fold_mse: List[float] = []

    for fold_idx, (tr_idx, va_idx) in enumerate(split_iter, start=1):
        tr = torch.as_tensor(tr_idx, device=device, dtype=torch.long)
        va = torch.as_tensor(va_idx, device=device, dtype=torch.long)

        X_tr, T_tr, Y_tr = X_scaled[tr], T_scaled[tr], Y[tr]
        X_va, T_va, Y_va = X_scaled[va], T_scaled[va], Y[va]
        # Fix sigma scale once per fold to avoid stochastic objective drift.
        d_med_fold = get_med(X_tr, max_n=int(X_tr.shape[0]))
        model = EIPM(input_dim=input_dim, hidden=width, n_layers=depth).to(device)
        opt = build_optimizer(
            model,
            optimizer_name=optimizer_name,
            lr=lr,
            weight_decay=weight_decay,
            adam_eps=adam_eps,
            sgd_momentum=sgd_momentum,
        )

        best_mse = float("inf")
        best_mmd = float("inf")
        best_state_by_mmd: Dict[str, Tensor] | None = None
        no_improve = 0
        for it in range(int(max_steps)):
            opt.zero_grad(set_to_none=True)
            train_mmd = compute_eipm_loss(
                model=model,
                X=X_tr,
                T=T_tr,
                a_sigma=a_sigma,
                nn=nn_val,
                d_med_fixed=d_med_fold,
                sigma_scales=sigma_scales_,
                leave_one_out_conditional=bool(leave_one_out_conditional),
                n_anchors=int(n_anchors),
            )
            reg_loss = compute_weight_regularizer(
                model=model,
                X=X_tr,
                T=T_tr,
                reg_type=reg_type,
                n_anchors=int(n_anchors),
            )
            loss = train_mmd + float(reg_lambda) * reg_loss
            loss.backward()
            if float(grad_clip) > 0.0:
                clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
            opt.step()

            if (it + 1) % eval_every == 0:
                model.eval()
                preds = []
                for i in range(T_va.numel()):
                    preds.append(
                        nw_mu_at_t(
                            model=model,
                            X_tr=X_tr,
                            T_tr=T_tr,
                            Y_tr=Y_tr,
                            t_val=T_va[i],
                            nn=nn_val,
                        )
                    )
                pred = torch.stack(preds).view(-1)
                mse = torch.mean((pred - Y_va.view(-1)) ** 2).item()
                mmd_val = float(
                    compute_eipm_val_mmd_cross(
                        model=model,
                        X_tr=X_tr,
                        T_tr=T_tr,
                        X_va=X_va,
                        T_va=T_va,
                        a_sigma=a_sigma,
                        nn=nn_val,
                        d_med_fixed=d_med_fold,
                        sigma_scales=sigma_scales_,
                        leave_one_out_conditional=bool(leave_one_out_conditional),
                        n_anchors=int(n_anchors),
                    ).item()
                )
                if train_trace:
                    print(
                        f"[TRAIN] fold={fold_idx}/{n_splits} "
                        f"step={it+1:04d} train_total={float(loss.item()):.6g} "
                        f"train_mmd={float(train_mmd.item()):.6g} "
                        f"train_reg={float(reg_loss.item()):.6g} "
                        f"val_mse={mse:.6g} best_mse={best_mse:.6g} "
                        f"val_mmd={mmd_val:.6g} best_mmd={best_mmd:.6g}"
                    )
                if mse < best_mse:
                    best_mse = float(mse)
                if mmd_val < best_mmd - float(min_delta):
                    best_mmd = float(mmd_val)
                    best_state_by_mmd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                model.train()
                if bool(use_early_stop) and (it + 1) >= int(early_stop_min_steps) and no_improve >= int(patience):
                    break

        if best_mse == float("inf"):
            model.eval()
            preds = []
            for i in range(T_va.numel()):
                preds.append(
                    nw_mu_at_t(
                        model=model,
                        X_tr=X_tr,
                        T_tr=T_tr,
                        Y_tr=Y_tr,
                        t_val=T_va[i],
                        nn=nn_val,
                    )
                )
            pred = torch.stack(preds).view(-1)
            best_mse = float(torch.mean((pred - Y_va.view(-1)) ** 2).item())
        if best_state_by_mmd is None:
            best_state_by_mmd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        fold_states.append(best_state_by_mmd)
        fold_mse.append(float(best_mse))
        if train_trace:
            print(f"[TRAIN] fold={fold_idx}/{n_splits} best_val_mse={best_mse:.6g} best_val_mmd={best_mmd:.6g}")

    return fold_states, fold_mse


def train_full_fixed(
    X_scaled: Tensor,
    T_scaled: Tensor,
    input_dim: int,
    device: torch.device,
    depth: int,
    width: int,
    *,
    max_steps: int,
    eval_every: int,
    lr: float,
    weight_decay: float,
    log_a_sigma: float,
    nn: float,
    sigma_scales: List[float] | None = None,
    optimizer_name: str = "adam",
    adam_eps: float = 1.0e-8,
    sgd_momentum: float = 0.0,
    grad_clip: float = 0.0,
    reg_type: str = "none",
    reg_lambda: float = 0.0,
    leave_one_out_conditional: bool = False,
    n_anchors: int = 0,
) -> Tuple[Dict[str, Tensor], float]:
    a_sigma = math.exp(float(log_a_sigma))
    d_med_full = get_med(X_scaled, max_n=int(X_scaled.shape[0]))
    sigma_scales_ = [1.0] if sigma_scales is None else list(sigma_scales)

    model = EIPM(input_dim=input_dim, hidden=width, n_layers=depth).to(device)
    opt = build_optimizer(
        model,
        optimizer_name=optimizer_name,
        lr=float(lr),
        weight_decay=float(weight_decay),
        adam_eps=float(adam_eps),
        sgd_momentum=float(sgd_momentum),
    )

    last_loss = float("nan")
    eval_every = int(max(1, int(eval_every)))
    for it in range(int(max_steps)):
        opt.zero_grad(set_to_none=True)
        train_mmd = compute_eipm_loss(
            model=model,
            X=X_scaled,
            T=T_scaled,
            a_sigma=a_sigma,
            nn=float(nn),
            d_med_fixed=d_med_full,
            sigma_scales=sigma_scales_,
            leave_one_out_conditional=bool(leave_one_out_conditional),
            n_anchors=int(n_anchors),
        )
        reg_loss = compute_weight_regularizer(
            model=model,
            X=X_scaled,
            T=T_scaled,
            reg_type=reg_type,
            n_anchors=int(n_anchors),
        )
        loss = train_mmd + float(reg_lambda) * reg_loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss in full fixed training at step={it+1}")
        loss.backward()
        if float(grad_clip) > 0.0:
            clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
        opt.step()
        last_loss = float(loss.item())
        if (it + 1) % eval_every == 0 or it == 0 or it == int(max_steps) - 1:
            print(
                f"[TRAIN] step={it+1:04d} total={last_loss:.6g} "
                f"mmd={float(train_mmd.item()):.6g} reg={float(reg_loss.item()):.6g}"
            )

    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return state, last_loss


@dataclass
class ReplicationData:
    scenario: str
    rep_idx: int
    d_X: int
    n_train: int
    X: np.ndarray
    T: np.ndarray
    Y: np.ndarray


def load_replications_from_npz(npz_path: str) -> List[ReplicationData]:
    data = np.load(npz_path, allow_pickle=True)
    X_all = data["X_train"]
    T_all = data["T_train"]
    Y_all = data["Y_train"]
    scenario = data["scenario"] if "scenario" in data.files else "unknown"

    reps: List[ReplicationData] = []
    for i in range(len(X_all)):
        X_i = np.array(X_all[i])
        T_i = np.array(T_all[i]).reshape(-1)
        Y_i = np.array(Y_all[i]).reshape(-1)
        reps.append(
            ReplicationData(
                scenario=str(scenario),
                rep_idx=int(i),
                d_X=int(X_i.shape[1]),
                n_train=int(X_i.shape[0]),
                X=X_i,
                T=T_i,
                Y=Y_i,
            )
        )
    return reps


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--data_dir", type=str, default=str(ROOT_DIR / "data"))
    p.add_argument(
        "--pattern",
        type=str,
        default="sim_nonlinear_dx50_ntr1000_nev10000_rpt100_tk50_ok50_pi0.0_seed42.npz",
        help="Dataset filename under data_dir.",
    )
    p.add_argument("--out_dir", type=str, default=str(ROOT_DIR / "out" / "models" / "eipm"))
    p.add_argument("--device", type=str, default="auto")

    # model / training
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--epochs", type=int, default=300)

    # tuning (keep small)
    p.add_argument("--n_trials", type=int, default=20)
    p.add_argument("--n_startup_trials", type=int, default=5)
    p.add_argument("--k_folds", type=int, default=5)
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--eval_every", type=int, default=30)
    p.add_argument("--fixed_lr", type=float, default=0.003)
    p.add_argument("--fixed_weight_decay", type=float, default=1.0e-6)
    p.add_argument("--optimizer", type=str, choices=["adam", "sgd"], default="adam")
    p.add_argument("--adam_eps", type=float, default=1.0e-8)
    p.add_argument("--sgd_momentum", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=0.0, help="Global grad-norm clip; 0 disables.")
    p.add_argument("--reg_type", type=str, choices=["none", "l2_w", "var_w", "ess_inv"], default="none")
    p.add_argument("--reg_lambda", type=float, default=0.0, help="Weight regularizer coefficient.")
    p.add_argument("--loo_conditional", type=int, default=1, help="1: use leave-one-out in K_T row weights.")
    p.add_argument("--n_anchors", type=int, default=64, help="Number of anchor t values per loss eval; <=0 uses all.")
    p.add_argument("--nn", type=float, default=0.05, help="Nearest-neighbor fraction for bandwidth (0,1].")
    p.add_argument("--sigma_scales", type=str, default="1.0", help="Comma-separated multipliers for base sigma.")
    p.add_argument("--tail_k", type=int, default=0, help="Tail k for CDF-sigmoid transform (0 => n**0.8).")
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--min_delta", type=float, default=1e-8)
    p.add_argument("--early_stop_min_steps", type=int, default=120, help="Do not early-stop before this many steps.")

    p.add_argument("--max_reps", type=int, default=100, help="Maximum number of reps to train (in order).")
    p.add_argument("--only_rep", type=int, default=-1, help="Train only this replication index.")
    p.add_argument("--overwrite", type=int, default=0, help="1: retrain even if checkpoint exists.")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot_h", action="store_true", help="Save h(t) curve plot per replication.")
    p.add_argument("--plot_h_n", type=int, default=200, help="Number of grid points for h(t) plot.")
    p.add_argument("--no_tuning", action="store_true", help="Skip Optuna and train with fixed hyperparameters.")
    p.add_argument("--fixed_log_a_sigma", type=float, default=0.0, help="Used when --no_tuning is set.")
    p.add_argument(
        "--log_a_sigma_grid",
        type=str,
        default="",
        help="Comma-separated fixed candidates for grid search (e.g. -1,-0.5,0,0.5,1).",
    )
    p.add_argument("--tuning_trace", type=int, default=1, help="1: print tuning progress, 0: silent.")
    p.add_argument("--train_trace", type=int, default=1, help="1: print final training progress, 0: silent.")

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = select_device(args.device)
    out_dir = Path(args.out_dir)
    sigma_scales = parse_sigma_scales(args.sigma_scales)
    log_a_sigma_grid = parse_log_a_sigma_grid(args.log_a_sigma_grid)

    # ------------------------------------------------------------
    # 1. pick dataset
    # ------------------------------------------------------------
    target_name = args.pattern
    npz_path = Path(args.data_dir) / target_name
    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset not found: {npz_path}")
    dataset_tag = Path(npz_path).stem
    out_dir = out_dir / dataset_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Using dataset: {Path(npz_path).name}")
    print(f"[INFO] Dataset path: {Path(npz_path).resolve()}")
    print(f"[INFO] Checkpoints dir: {out_dir.resolve()}")

    reps = load_replications_from_npz(npz_path)
    data = np.load(npz_path, allow_pickle=True)
    T_eval_all = np.array(data["T_eval"]) if "T_eval" in data.files else None
    if args.only_rep >= 0:
        reps_to_run = [rep for rep in reps if int(rep.rep_idx) == int(args.only_rep)]
        if len(reps_to_run) == 0:
            raise RuntimeError(f"rep_idx={args.only_rep} not found in dataset.")
    else:
        n_run = min(int(args.max_reps), len(reps))
        reps_to_run = reps[:n_run]
    for rep in reps_to_run:
        print(
            f"[INFO] Scenario={rep.scenario}, "
            f"rep_idx={rep.rep_idx}, "
            f"d_X={rep.d_X}, "
            f"n_train={rep.n_train}"
        )
        ckpt_path = out_dir / f"eipm_single_nonlinear_rep{rep.rep_idx:03d}.pth"
        if ckpt_path.exists() and int(args.overwrite) != 1:
            print(f"[SKIP] Checkpoint exists: {ckpt_path}")
            continue
        if ckpt_path.exists() and int(args.overwrite) == 1:
            print(f"[INFO] overwrite=1, retraining existing checkpoint: {ckpt_path}")

        # ------------------------------------------------------------
        # 2. prepare tensors
        # ------------------------------------------------------------
        X = torch.tensor(rep.X, dtype=torch.float32, device=device)
        tstar_params = fit_tstar_transform(rep.T, tail_k=int(args.tail_k))
        T_star = transform_t_to_star(rep.T, tstar_params)
        T = torch.tensor(T_star, dtype=torch.float32, device=device)
        Y = torch.tensor(rep.Y, dtype=torch.float32, device=device)

        X_std, T_std, x_mean_t, x_std_t, t_mean_t, t_std_t = standardize_train(X, T)
        X_scaled = X_std / math.sqrt(float(rep.d_X))
        T_scaled = T_std.view(-1)

        input_dim = int(rep.d_X) + 1

        # ------------------------------------------------------------
        # 3. hyperparameter selection
        # ------------------------------------------------------------
        if args.no_tuning:
            best_params = {"log_a_sigma": float(args.fixed_log_a_sigma)}
            best_cv_mse = float("nan")
            best_trial = -1
            print("[INFO] no_tuning=True. Use fixed params:", best_params)
        elif len(log_a_sigma_grid) > 0:
            print("[INFO] Start fixed-grid search for log_a_sigma:", log_a_sigma_grid)

            class _FixedTrial:
                def __init__(self, val: float, number: int):
                    self._val = float(val)
                    self.number = int(number)
                    self.user_attrs: Dict[str, float] = {}

                def suggest_float(self, _name: str, _low: float, _high: float) -> float:
                    return float(self._val)

                def set_user_attr(self, key: str, value: float) -> None:
                    self.user_attrs[str(key)] = float(value)

            best_score = float("inf")
            best_params = {"log_a_sigma": float(log_a_sigma_grid[0])}
            best_cv_mse = float("nan")
            best_trial = -1

            for i, cand in enumerate(log_a_sigma_grid):
                fixed_trial = _FixedTrial(float(cand), int(i))
                score = float(
                    objective_cv_mse(
                        trial=fixed_trial,
                        X_raw=X,
                        T_raw_np=np.asarray(rep.T, dtype=np.float64),
                        Y_raw=Y,
                        input_dim=input_dim,
                        device=device,
                        depth=args.depth,
                        width=args.width,
                        max_steps=args.max_steps,
                        patience=args.patience,
                        min_delta=args.min_delta,
                        n_splits=args.k_folds,
                        seed=args.seed,
                        eval_every=args.eval_every,
                        fixed_lr=args.fixed_lr,
                        fixed_weight_decay=args.fixed_weight_decay,
                        fixed_nn=args.nn,
                        fixed_optimizer=args.optimizer,
                        fixed_adam_eps=args.adam_eps,
                        fixed_sgd_momentum=args.sgd_momentum,
                        fixed_grad_clip=args.grad_clip,
                        fixed_reg_type=args.reg_type,
                        fixed_reg_lambda=args.reg_lambda,
                        fixed_loo_conditional=bool(int(args.loo_conditional)),
                        fixed_n_anchors=int(args.n_anchors),
                        tuning_trace=bool(int(args.tuning_trace)),
                        early_stop_min_steps=args.early_stop_min_steps,
                        fixed_sigma_scales=sigma_scales,
                    )
                )
                print(f"[GRID] idx={i} log_a_sigma={float(cand):.4f} score={score:.6g}")
                if score < best_score:
                    best_score = score
                    best_trial = int(i)
                    best_cv_mse = score
                    best_params = {"log_a_sigma": float(cand)}

            print("[INFO] Grid best CV score:", float(best_cv_mse))
            print("[INFO] Grid best params:", best_params)
        else:
            print("[INFO] Start hyperparameter tuning...")
            def _obj_local(trial):
                return objective_cv_mse(
                    trial=trial,
                    X_raw=X,
                    T_raw_np=np.asarray(rep.T, dtype=np.float64),
                    Y_raw=Y,
                    input_dim=input_dim,
                    device=device,
                    depth=args.depth,
                    width=args.width,
                    max_steps=args.max_steps,
                    patience=args.patience,
                    min_delta=args.min_delta,
                    n_splits=args.k_folds,
                    seed=args.seed,
                    eval_every=args.eval_every,
                    fixed_lr=args.fixed_lr,
                    fixed_weight_decay=args.fixed_weight_decay,
                    fixed_nn=args.nn,
                    fixed_optimizer=args.optimizer,
                    fixed_adam_eps=args.adam_eps,
                    fixed_sgd_momentum=args.sgd_momentum,
                    fixed_grad_clip=args.grad_clip,
                    fixed_reg_type=args.reg_type,
                    fixed_reg_lambda=args.reg_lambda,
                    fixed_loo_conditional=bool(int(args.loo_conditional)),
                    fixed_n_anchors=int(args.n_anchors),
                    tuning_trace=bool(int(args.tuning_trace)),
                    early_stop_min_steps=args.early_stop_min_steps,
                    fixed_sigma_scales=sigma_scales,
                )

            sampler = optuna.samplers.TPESampler(seed=int(args.seed), n_startup_trials=int(args.n_startup_trials))
            study = optuna.create_study(direction="minimize", sampler=sampler)
            study.optimize(_obj_local, n_trials=args.n_trials, show_progress_bar=True)

            best_params = study.best_params
            best_cv_mse = float(study.best_value)
            best_trial = int(study.best_trial.number)

            print("[INFO] Best CV MSE:", best_cv_mse)
            print(
                "[INFO] Best params:",
                {
                    "log_a_sigma": float(best_params["log_a_sigma"]),
                },
            )
        fixed_params = {
            "lr": float(args.fixed_lr),
            "weight_decay": float(args.fixed_weight_decay),
            "nn": float(args.nn),
            "optimizer": str(args.optimizer),
            "adam_eps": float(args.adam_eps),
            "sgd_momentum": float(args.sgd_momentum),
            "grad_clip": float(args.grad_clip),
            "reg_type": str(args.reg_type),
            "reg_lambda": float(args.reg_lambda),
            "loo_conditional": bool(int(args.loo_conditional)),
            "n_anchors": int(args.n_anchors),
        }

        if args.plot_h:
            if T_eval_all is None:
                print("[WARN] T_eval not found in dataset; skip h(t) plot.")
            else:
                if T_eval_all.ndim >= 2:
                    t_eval_rep = np.array(T_eval_all[int(rep.rep_idx)]).reshape(-1)
                else:
                    t_eval_rep = np.array(T_eval_all).reshape(-1)
                if t_eval_rep.size == 0:
                    print("[WARN] empty T_eval; skip h(t) plot.")
                else:
                    t_min = float(np.min(t_eval_rep))
                    t_max = float(np.max(t_eval_rep))
                    n_plot = int(max(10, args.plot_h_n))
                    t_grid_raw = np.linspace(t_min, t_max, n_plot)
                    t_grid_star = transform_t_to_star(t_grid_raw, tstar_params)
                    t_grid_scaled = (t_grid_star - float(t_mean_t)) / float(t_std_t)
                    t_grid_scaled_t = torch.tensor(t_grid_scaled, dtype=torch.float32, device=device)
                    h_scaled = compute_h_curve(
                        T_scaled,
                        t_grid_scaled_t,
                        nn=float(fixed_params["nn"]),
                    ).detach().cpu().numpy()
                    h_raw = h_scaled  # h is on transformed+standardized scale
                    out_path = out_dir / f"h_curve_rep{rep.rep_idx:03d}.png"
                    plot_h_curve(t_grid_raw, h_raw, out_path)
                    print(f"[INFO] Saved h(t) plot: {out_path}")

        # ------------------------------------------------------------
        # 4. train model(s)
        # ------------------------------------------------------------
        print("[INFO] Train folds once (lockstep) and keep fold states at best mean CV val_mmd...")
        selected_steps, selected_val_mmd, fold_states, fold_mse = select_best_steps_cv(
            X_scaled=X_scaled,
            T_scaled=T_scaled,
            Y=Y,
            T_raw_np=np.asarray(rep.T, dtype=np.float64),
            input_dim=input_dim,
            device=device,
            depth=args.depth,
            width=args.width,
            log_a_sigma=float(best_params["log_a_sigma"]),
            nn_val=float(fixed_params["nn"]),
            max_steps=int(args.max_steps),
            patience=int(args.patience),
            min_delta=float(args.min_delta),
            n_splits=int(args.k_folds),
            seed=int(args.seed),
            eval_every=int(args.eval_every),
            lr=float(fixed_params["lr"]),
            weight_decay=float(fixed_params["weight_decay"]),
            optimizer_name=str(fixed_params["optimizer"]),
            adam_eps=float(fixed_params["adam_eps"]),
            sgd_momentum=float(fixed_params["sgd_momentum"]),
            grad_clip=float(fixed_params["grad_clip"]),
            reg_type=str(fixed_params["reg_type"]),
            reg_lambda=float(fixed_params["reg_lambda"]),
            leave_one_out_conditional=bool(fixed_params["loo_conditional"]),
            n_anchors=int(fixed_params["n_anchors"]),
            sigma_scales=sigma_scales,
            early_stop_min_steps=int(args.early_stop_min_steps),
            train_trace=bool(int(args.train_trace)),
        )
        print(f"[INFO] selected_common_step={selected_steps} selected_cv_val_mmd={selected_val_mmd:.6g}")
        avg_state = {}
        keys = fold_states[0].keys()
        for k in keys:
            stacked = torch.stack([fs[k] for fs in fold_states], dim=0)
            avg_state[k] = torch.mean(stacked, dim=0)
        final_loss = float("nan")
        best_cv_mse = float("nan") if args.no_tuning else float(best_cv_mse)
        train_stats = {
            "best_eipm_loss": float("nan"),
            "sigma": float("nan"),
            "h_median": float("nan"),
            "lr": float(fixed_params["lr"]),
            "weight_decay": float(fixed_params["weight_decay"]),
            "optimizer": str(fixed_params["optimizer"]),
            "adam_eps": float(fixed_params["adam_eps"]),
            "sgd_momentum": float(fixed_params["sgd_momentum"]),
            "grad_clip": float(fixed_params["grad_clip"]),
            "reg_type": str(fixed_params["reg_type"]),
            "reg_lambda": float(fixed_params["reg_lambda"]),
            "loo_conditional": bool(fixed_params["loo_conditional"]),
            "n_anchors": int(fixed_params["n_anchors"]),
            "epochs": float(args.epochs),
            "selected_steps_cv": int(selected_steps),
            "selected_cv_val_mmd": float(selected_val_mmd),
            "final_full_loss": float(final_loss),
            "sigma_scales": [float(v) for v in sigma_scales],
        }

        # ------------------------------------------------------------
        # 5. save checkpoint
        # ------------------------------------------------------------
        atomic_torch_save(
            ckpt_path,
            {
                "model_state": avg_state,
                "ensemble_fold_states": fold_states,
                "ensemble_fold_mse": fold_mse,
                "ensemble_best_trial": best_trial,
                "best_params": best_params,
                "fixed_params": fixed_params,
                "best_cv_mse": best_cv_mse,
                "train_stats": train_stats,
                    "standardize": {
                        "x_mean": x_mean_t.view(-1).detach().cpu().numpy(),
                        "x_std": x_std_t.view(-1).detach().cpu().numpy(),
                        "t_mean": float(t_mean_t),
                        "t_std": float(t_std_t),
                    },
                    "t_transform": _T_TRANSFORM,
                    "tstar_params": tstar_params,
                    "bandwidth": {"type": "knn", "nn": float(args.nn)},
                    "npz_path": npz_path,
                    "script_args": vars(args),
                },
            )

        print(f"[DONE] Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
