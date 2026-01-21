#!/usr/bin/env python
# evaluate_eipm_fixed.py
#
# Difference vs original:
# - Bandwidth h is computed at evaluation-time from training tildeT and best_params (a_h, k_nn).
# - For backward compatibility, if that fails we fall back to train_stats["h_median"].

import argparse
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = RESULTS_DIR / "eval_summary.csv"


def _fmt_hms(sec: float) -> str:
    s = int(max(0.0, sec))
    h = s // 3600
    m = (s % 3600) // 60
    ss = s % 60
    return f"{h:02d}:{m:02d}:{ss:02d}"


def get_local_bandwidths(T: torch.Tensor, k: int = 20, min_h: float = 1e-3, max_n: int = 4000) -> torch.Tensor:
    """
    Local bandwidth h_i based on kNN distance in 1D T (for each target i).
    Uses an O(n^2) distance matrix. For very large n, we subsample to estimate scale and then
    assign a global bandwidth.
    """
    T = T.view(-1, 1).contiguous()
    n = T.shape[0]

    if n > max_n:
        idx = torch.randperm(n, device=T.device)[:max_n]
        Ts = T[idx]
        dist = torch.cdist(Ts, Ts)
        dist_sorted, _ = torch.sort(dist, dim=1)
        k_eff = min(k, dist_sorted.shape[1] - 1)
        h_local = dist_sorted[:, k_eff].view(-1, 1)
        h_global = torch.median(h_local).clamp(min=min_h)
        return h_global.repeat(n, 1)

    dist = torch.cdist(T, T)
    dist_sorted, _ = torch.sort(dist, dim=1)
    k_eff = min(k, dist_sorted.shape[1] - 1)
    h_local = dist_sorted[:, k_eff].view(-1, 1)
    return h_local.clamp(min=min_h)


class ScoreNet(nn.Module):
    def __init__(self, d_X: int, depth: int = 2, width: int = 128):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = d_X + 1
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.ReLU())
            in_dim = width
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        if T.ndim == 1:
            T = T.view(-1, 1)
        inp = torch.cat([X, T], dim=1)
        return self.net(inp).view(-1)


def gaussian_kernel(d: torch.Tensor, h: float) -> torch.Tensor:
    return torch.exp(-0.5 * (d / h) ** 2)


@torch.no_grad()
def stable_exp_scores(model: ScoreNet, X: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    s = model(X, T)
    s = s - torch.max(s)
    return torch.exp(s)


@torch.no_grad()
def mu_hat_batch(
    model: ScoreNet,
    X_train: torch.Tensor,
    T_train: torch.Tensor,
    Y_train: torch.Tensor,
    T_query: torch.Tensor,
    h_query: float,
) -> torch.Tensor:
    base_w = stable_exp_scores(model, X_train, T_train)  # (n,)
    out = []
    for tq in T_query.view(-1):
        K = gaussian_kernel(T_train - tq, h_query)
        w = base_w * K
        denom = torch.sum(w) + 1e-12
        out.append(torch.sum(w * Y_train) / denom)
    return torch.stack(out, dim=0)


def mab_rmse(mu_hat: np.ndarray, mu_true: np.ndarray) -> Dict[str, float]:
    err = mu_hat - mu_true
    mab = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return {"MAB": mab, "RMSE": rmse}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="./datasets")
    p.add_argument("--pattern", type=str, default="sim_*.npz")
    p.add_argument("--ckpt_dir", type=str, default="./models/eipm")
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_files", type=int, default=None)
    p.add_argument("--max_reps", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    data_dir = Path(args.data_dir)
    ckpt_dir = Path(args.ckpt_dir)

    npz_paths = sorted([str(p) for p in data_dir.glob(args.pattern)])
    if args.max_files is not None:
        npz_paths = npz_paths[: int(args.max_files)]

    rows: List[Dict] = []
    t0_all = time.time()

    print(f"[EVAL-EIPM] files={len(npz_paths)} device={device} depth={args.depth} width={args.width}")
    print(f"[EVAL-EIPM] ckpt_dir={ckpt_dir}")

    for f_idx, npz_path in enumerate(npz_paths):
        data = np.load(npz_path, allow_pickle=True)

        scenario = str(np.array(data["scenario"]).item())
        d_X = int(np.array(data["d_X"]).item())
        n_train = int(np.array(data["n_train"]).item())
        n_rpt = int(np.array(data["n_rpt"]).item())
        treatment_k = int(np.array(data["treatment_k"]).item())
        outcome_k = int(np.array(data["outcome_k"]).item())

        Xtilde_train_all = np.asarray(data["Xtilde_train"])
        tildeT_train_all = np.asarray(data["tildeT_train"])
        Y_train_all = np.asarray(data["Y_train"])

        T_eval = np.asarray(data["T_eval"], dtype=np.float64).reshape(-1)
        mu_eval = np.asarray(data["mu_eval"], dtype=np.float64).reshape(-1)

        ET = float(np.asarray(data["ET"], dtype=np.float64).reshape(-1)[0])
        VT = float(np.asarray(data["VT"], dtype=np.float64).reshape(-1)[0])
        tildeT_eval = (T_eval - ET) / (np.sqrt(VT) + 1e-12)

        rep_indices = list(range(n_rpt))
        if args.max_reps is not None:
            rep_indices = rep_indices[: int(args.max_reps)]

        for rep_idx in rep_indices:
            stem = Path(npz_path).stem
            ckpt_path = ckpt_dir / (stem + f"_rep{rep_idx:03d}_d{args.depth}_w{args.width}.pth")
            if not ckpt_path.exists():
                continue

            Xtr = torch.tensor(Xtilde_train_all[rep_idx], dtype=torch.float32, device=device)
            Ttr = torch.tensor(tildeT_train_all[rep_idx], dtype=torch.float32, device=device)
            Ytr = torch.tensor(Y_train_all[rep_idx], dtype=torch.float32, device=device)
            Tq = torch.tensor(tildeT_eval, dtype=torch.float32, device=device)

            ckpt = torch.load(ckpt_path, map_location=device)
            model = ScoreNet(d_X=d_X, depth=args.depth, width=args.width).to(device)
            model.load_state_dict(ckpt["model_state"])
            model.eval()

            train_stats = ckpt.get("train_stats", {}) or {}
            best_params = ckpt.get("best_params", {}) or {}

            # Bandwidth is computed at evaluation-time from training T (no need to store it in checkpoints).
            # Preference order:
            #   1) use tuned (a_h, k_nn) from best_params if available
            #   2) fall back to train_stats["h_median"] for backward compatibility
            a_h = float(best_params.get("a_h", 1.0))
            k_nn = int(best_params.get("k_nn", 20))

            Ttilde_tmp = torch.tensor(tildeT_train_all[rep_idx], dtype=torch.float32)
            try:
                h_local = get_local_bandwidths(Ttilde_tmp, k=k_nn)
                h_query = float(a_h * torch.median(h_local).item())
            except Exception:
                h_query = float(train_stats.get("h_median", 0.0))

            h_query = max(h_query, 1e-3)

            mu_hat = mu_hat_batch(model, Xtr, Ttr, Ytr, Tq, h_query).detach().cpu().numpy()
            met = mab_rmse(mu_hat, mu_eval)

            rows.append({
                "npz": Path(npz_path).name,
                "rep_idx": int(rep_idx),
                "scenario": scenario,
                "d_X": int(d_X),
                "n_train": int(n_train),
                "treatment_k": int(treatment_k),
                "outcome_k": int(outcome_k),
                **met,
            })

        if (f_idx + 1) % 5 == 0:
            elapsed = time.time() - t0_all
            print(f"[PROGRESS] {f_idx+1}/{len(npz_paths)} files | rows={len(rows)} | elapsed={_fmt_hms(elapsed)}")

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"[DONE] wrote {len(df)} rows -> {OUTPUT_CSV}")
    else:
        print("[DONE] no rows written (missing checkpoints?)")


if __name__ == "__main__":
    main()
