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

def set_seed(seed: int) -> None: # seed fix하는 함수.
    torch.manual_seed(seed)
    np.random.seed(seed)

def _now_str() -> str: #현재 시각을 문자열로. 소요시간 tracking용.
    if _KST is None:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return datetime.now(_KST).strftime("%Y-%m-%d %H:%M:%S KST")

def _fmt_hms(seconds: float) -> str: # 초 단위를 시, 분, 초로 표기.
    seconds = max(0.0, float(seconds))
    s = int(round(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    ss = s % 60
    return f"{h:02d}:{m:02d}:{ss:02d}"

def _safe_mean(xs: List[float]) -> Optional[float]: # 없으면 오류대신 None 출력.
    if len(xs) == 0:
        return None
    return float(mean(xs))

def atomic_write_json(path: Path, obj: Dict) -> None: # 도중에 에러가 뜨면 불완전 저장되도록.
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

def load_json_(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"{path} must contain a JSON object (dict), got {type(obj).__name__}")
    return obj

# ============================================================
# 1. Model
# ============================================================

class EIPM(nn.Module): # s_theta([x, t])가 속할 class임. d_X + 1 -> 128 -> -> 128 -> 1
    def __init__(self, input_dim: int, hidden: int = 128, n_layers: int = 2):
        super().__init__()
        layers: List[nn.Module] = []
        d_in = input_dim

        for _ in range(n_layers):
            layers.append(nn.Linear(d_in, hidden))
            layers.append(nn.ELU(alpha=1))
            d_in = hidden

        layers.append(nn.Linear(d_in, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, X: Tensor, T: Tensor) -> Tensor: # T는 matrix 꼴로 만들어, X에 붙임.
        if T.ndim == 1:
            T_in = T.view(-1, 1)
        else:
            T_in = T
        inp = torch.cat([X, T_in], dim=1)
        return self.net(inp).view(-1)

# ============================================================
# 2. Kernels
# ============================================================

def rbf(X: Tensor, sigma: List[float]) -> Tensor: # RBF kernel 행렬 K n*n. (K_ij는 X_i, X_j에 대한 거리 기반 RBF kernel 값)
    X = X.contiguous()
    K = torch.exp(-torch.cdist(X, X)**2 / (2*sigma**2))
    return K

@torch.no_grad()
def get_med(x: Tensor, max_n: int) -> float:  # x((n,) or (n,d))를 넣으면, x_i, x_j 사이 거리의 median을 구해줌.
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
def h0_t( # h_0(t)
    T_train: torch.Tensor,          # (n_train,)
    t: torch.Tensor,                # (m,)
    *,
    a_h: float,
    k: int = 20,
) -> torch.Tensor: # (m,)
    T_tr = T_train.view(-1, 1)      # (n,1)
    T_q  = t.view(1, -1)      # (1,m)
    n = T_tr.shape[0]
    dist = torch.abs(T_tr - T_q)    # (n,m)

    # kNN radius (order statistic)
    dist_sorted, _ = torch.sort(dist, dim=0)  # (n,m)
    k_eff = min(k, n - 1)        # 1..n-1
    h_knn = a_h * dist_sorted[k_eff, :]             # (m,)
    return h_knn.view(-1, 1)


def compute_eipm_loss(model: nn.Module, X: Tensor, T: Tensor, a_sigma : float, a_h : float, k_nn : int) -> Tensor:
    # MMD_k(hat P_{X|T=t}^{W_s}, hat P_X)
    # P_{X|T=t}^{W_s} = \sum_i a_i delta_{X_i}
    h_T_vec = h_T_vec.view(-1, 1)          # (n,1)

    s_val = model(X, T)                    # (n,)
    s_max = torch.max(s_val)
    W_s = torch.exp(s_val-s_max)            # (n,)

    T_flat = T.view(-1, 1)
    dist_sq_T = (T_flat - T_flat.t()) ** 2  # (n,n)

    H = h_T_vec @ h_T_vec.t()              # (n,n), H_ij = h_i * h_j
    K_T = torch.exp(-0.5 * dist_sq_T / H)  # (n,n)

    d_med = get_med(X)
    sigma = a_sigma * d_med
    K_X = rbf(X, sigma)

    denom = K_T @ W_s.view(-1, 1)  # (n,1)
    W_mat = (K_T * W_s.view(1, -1)) / (denom.view(-1, 1) + 1e-8)  # (n,n)

    term1 = torch.sum((W_mat @ K_X) * W_mat, dim=1)  # (n,)
    term2 = torch.mean(K_X)
    term3 = 2*torch.mean(W_mat @ K_X, dim=1)  # (n,)

    loss = torch.mean(term1 - term3 + term2)
    return loss