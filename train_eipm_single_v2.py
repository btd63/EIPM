# train_eipm_single.py
#
# Train EIPM on a SINGLE dataset:
#   - first nonlinear npz (sorted order)
#   - first replication (rep_idx = 0)
#
# Purpose:
#   canonical train → evaluate pipeline bootstrap

from __future__ import annotations

import argparse
import glob
import math
import os
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

from device_utils import select_device
_T_TRANSFORM = "identity"


def _transform_t(T: Tensor) -> Tensor:
    return T

def local_poly_mu_at_t(
    X_tr: Tensor,
    T_tr: Tensor,
    Y_tr: Tensor,
    model: nn.Module,
    t_val: Tensor,
    a_h: float,
    nn: float,
) -> Tensor:
    t_fixed = t_val.view(1).repeat(X_tr.shape[0]).view(-1, 1)
    with torch.no_grad():
        logw = model(X_tr, t_fixed).view(-1)

    with torch.no_grad():
        t0 = t_val.view(())
        diff = T_tr.view(-1) - t0
        h_knn = _knn_bandwidth(T_tr.view(-1), t_val.view(1), nn=float(nn))[0]
        h_t = float(a_h) * float(h_knn.item())
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

    ones = torch.ones_like(diff)
    X_lp = torch.stack([ones, diff], dim=1)  # (n,2) for p=1
    W = w_eff.view(-1, 1)
    XW = X_lp * W
    xtwx = X_lp.t().mm(XW)
    xtwy = XW.t().mv(Y_tr.view(-1))
    beta = torch.linalg.pinv(xtwx).mv(xtwy)
    return beta[0]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


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
    a_h: float,
    nn: float,
) -> Tensor:
    h_knn = _knn_bandwidth(T_train, t, nn=float(nn))
    h = float(a_h) * h_knn
    return h.clamp_min(1e-8).view(-1, 1)


def compute_eipm_loss(
    model: nn.Module,
    X: Tensor,
    T: Tensor,
    a_sigma: float,
    a_h: float,
    nn: float,
) -> Tensor:
    s_val = model(X, T)
    s_max = torch.max(s_val)
    W_s = torch.exp(s_val - s_max)

    T_flat = T.view(-1, 1)
    dist_sq_T = (T_flat - T_flat.t()) ** 2

    h_T_vec = h0_t(
        T_train=T,
        t=T,
        a_h=float(a_h),
        nn=float(nn),
    )
    h_T_vec = h_T_vec.view(-1, 1)
    H = (h_T_vec @ h_T_vec.t()).clamp_min(1e-8)
    K_T = torch.exp(-0.5 * dist_sq_T / H)

    d_med = get_med(X)
    sigma = a_sigma * d_med
    K_X = rbf(X, sigma)

    denom = K_T @ W_s.view(-1, 1)
    W_mat = (K_T * W_s.view(1, -1)) / (denom.view(-1, 1) + 1e-8)

    term1 = torch.sum((W_mat @ K_X) * W_mat, dim=1)
    term2 = torch.mean(K_X)
    term3 = 2.0 * torch.mean(W_mat @ K_X, dim=1)

    loss = torch.mean(term1 - term3 + term2)
    return loss


@torch.no_grad()
def compute_h_curve(
    T_scaled: Tensor,
    t_grid_scaled: Tensor,
    a_h: float,
    nn: float,
) -> Tensor:
    h_vals = h0_t(T_train=T_scaled, t=t_grid_scaled, a_h=a_h, nn=nn).view(-1)
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
    eval_every: int = 10,
    fixed_lr: float = 0.003,
    fixed_weight_decay: float = 1.0e-6,
    fixed_nn: float = 0.7,
) -> float:
    log_a_sigma = trial.suggest_float("log_a_sigma", -2.0, 2.0)
    log_a_h = trial.suggest_float("log_a_h", -1.0, 1.0)

    a_sigma = math.exp(float(log_a_sigma))
    a_h = math.exp(float(log_a_h))

    lr = float(fixed_lr)
    weight_decay = float(fixed_weight_decay)
    nn_val = float(fixed_nn)

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
    fold_mse: List[float] = []
    printed_debug = False
    for _, (tr_idx, va_idx) in enumerate(split_iter, start=1):
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
            loss = compute_eipm_loss(model, X_tr, T_tr, a_sigma, a_h, nn_val)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    "EIPM loss is not finite: "
                    f"trial={getattr(trial, 'number', 'NA')} it={it} "
                    f"a_sigma={a_sigma:.3g} a_h={a_h:.3g} "
                    f"lr={lr:.3g} wd={weight_decay:.3g} "
                )
            loss.backward()
            opt.step()
            if (it + 1) % eval_every == 0:
                model.eval()
                preds = []
                for i in range(T_va.numel()):
                    mu_i = local_poly_mu_at_t(
                        model=model,
                        X_tr=X_tr,
                        T_tr=T_tr,
                        Y_tr=Y_tr,
                        t_val=T_va[i],
                        a_h=a_h,
                        nn=nn_val,
                    )
                    preds.append(mu_i)
                pred = torch.stack(preds).view(-1)
                mse = torch.mean((pred - Y_va.view(-1)) ** 2).item()
                if not printed_debug:
                    with torch.no_grad():
                        t_dbg = T_va[0].view(())
                        t_fixed = t_dbg.view(1).repeat(X_tr.shape[0]).view(-1, 1)
                        w_dbg = torch.exp(model(X_tr, t_fixed)).view(-1)
                        diff = T_tr.view(-1) - t_dbg
                        h_knn = _knn_bandwidth(T_tr.view(-1), t_dbg.view(1), nn=nn_val)[0]
                        h_t = float(a_h) * float(h_knn.item())
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
                mu_i = local_poly_mu_at_t(
                    model=model,
                    X_tr=X_tr,
                    T_tr=T_tr,
                    Y_tr=Y_tr,
                    t_val=T_va[i],
                    a_h=a_h,
                    nn=nn_val,
                )
                preds.append(mu_i)
            pred = torch.stack(preds).view(-1)
            best_mse = float(torch.mean((pred - Y_va.view(-1)) ** 2).item())

        fold_mse.append(float(best_mse))

    return float(np.mean(fold_mse))


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
) -> Tuple[List[Dict[str, Tensor]], List[float]]:
    a_sigma = math.exp(float(params["log_a_sigma"]))
    a_h = math.exp(float(params["log_a_h"]))
    nn_val = float(params["nn"])
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

    eval_every = int(max(1, int(eval_every)))
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
            loss = compute_eipm_loss(model, X_tr, T_tr, a_sigma, a_h, nn_val)
            loss.backward()
            opt.step()

            if (it + 1) % eval_every == 0:
                model.eval()
                preds = []
                for i in range(T_va.numel()):
                    preds.append(
                        local_poly_mu_at_t(
                            model=model,
                            X_tr=X_tr,
                            T_tr=T_tr,
                            Y_tr=Y_tr,
                            t_val=T_va[i],
                            a_h=a_h,
                            nn=nn_val,
                        )
                    )
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
                preds.append(
                    local_poly_mu_at_t(
                        model=model,
                        X_tr=X_tr,
                    T_tr=T_tr,
                    Y_tr=Y_tr,
                    t_val=T_va[i],
                    a_h=a_h,
                    nn=nn_val,
                )
            )
            pred = torch.stack(preds).view(-1)
            best_mse = float(torch.mean((pred - Y_va.view(-1)) ** 2).item())

        fold_states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        fold_mse.append(float(best_mse))

    return fold_states, fold_mse


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

    p.add_argument("--data_dir", type=str, default="./datasets")
    p.add_argument("--out_dir", type=str, default="./models/eipm_single_v2")
    p.add_argument("--device", type=str, default="auto")

    # model / training
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--epochs", type=int, default=300)

    # tuning (keep small)
    p.add_argument("--n_trials", type=int, default=50)
    p.add_argument("--n_startup_trials", type=int, default=10)
    p.add_argument("--k_folds", type=int, default=5)
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--eval_every", type=int, default=30)
    p.add_argument("--fixed_lr", type=float, default=0.003)
    p.add_argument("--fixed_weight_decay", type=float, default=1.0e-6)
    p.add_argument("--nn", type=float, default=0.7, help="Nearest-neighbor fraction for bandwidth (0,1].")
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--min_delta", type=float, default=1e-8)

    p.add_argument("--max_reps", type=int, default=100, help="Maximum number of reps to train (in order).")
    p.add_argument("--only_rep", type=int, default=-1, help="Train only this replication index.")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot_h", action="store_true", help="Save h(t) curve plot per replication.")
    p.add_argument("--plot_h_n", type=int, default=200, help="Number of grid points for h(t) plot.")

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = select_device(args.device)
    out_dir = Path(args.out_dir)

    # ------------------------------------------------------------
    # 1. pick fixed dataset (dx50, tk50, ok50)
    # ------------------------------------------------------------
    target_name = "sim_nonlinear_dx50_ntr1000_nev10000_rpt100_tk50_ok50_pi0.0_seed42.npz"
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

        # ------------------------------------------------------------
        # 2. prepare tensors
        # ------------------------------------------------------------
        X = torch.tensor(rep.X, dtype=torch.float32, device=device)
        T_raw = torch.tensor(rep.T, dtype=torch.float32, device=device)
        T = _transform_t(T_raw)
        Y = torch.tensor(rep.Y, dtype=torch.float32, device=device)

        X_std, T_std, x_mean_t, x_std_t, t_mean_t, t_std_t = standardize_train(X, T)
        X_scaled = X_std / math.sqrt(float(rep.d_X))
        T_scaled = T_std.view(-1)

        input_dim = int(rep.d_X) + 1

        # ------------------------------------------------------------
        # 3. hyperparameter tuning (only a_sigma)
        # ------------------------------------------------------------
        print("[INFO] Start hyperparameter tuning...")
        def _obj_local(trial):
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
                eval_every=args.eval_every,
                fixed_lr=args.fixed_lr,
                fixed_weight_decay=args.fixed_weight_decay,
                fixed_nn=args.nn,
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
                "log_a_h": float(best_params["log_a_h"]),
            },
        )
        fixed_params = {
            "lr": float(args.fixed_lr),
            "weight_decay": float(args.fixed_weight_decay),
            "nn": float(args.nn),
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
                    t_grid_scaled = (t_grid_raw - float(t_mean_t)) / float(t_std_t)
                    t_grid_scaled_t = torch.tensor(t_grid_scaled, dtype=torch.float32, device=device)
                    h_scaled = compute_h_curve(
                        T_scaled,
                        t_grid_scaled_t,
                        a_h=float(math.exp(best_params["log_a_h"])),
                        nn=float(fixed_params["nn"]),
                    ).detach().cpu().numpy()
                    h_raw = h_scaled  # h is on transformed+standardized scale
                    out_path = out_dir / f"h_curve_rep{rep.rep_idx:03d}.png"
                    plot_h_curve(t_grid_raw, h_raw, out_path)
                    print(f"[INFO] Saved h(t) plot: {out_path}")

        # ------------------------------------------------------------
        # 4. train fold models using best params, then average
        # ------------------------------------------------------------
        print("[INFO] Train fold models with best params...")
        train_params = dict(best_params)
        train_params.update(fixed_params)

        fold_states, fold_mse = train_folds_for_params(
            X_scaled=X_scaled,
            T_scaled=T_scaled,
            Y=Y,
            input_dim=input_dim,
            device=device,
            depth=args.depth,
            width=args.width,
            params=train_params,
            max_steps=args.max_steps,
            patience=args.patience,
            min_delta=args.min_delta,
            n_splits=args.k_folds,
            seed=args.seed,
            eval_every=args.eval_every,
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
                    "bandwidth": {"type": "knn", "nn": float(args.nn)},
                    "npz_path": npz_path,
                    "script_args": vars(args),
                },
            )

        print(f"[DONE] Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
