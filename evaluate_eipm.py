# evaluate_eipm.py
# Run-only evaluation (no CLI args): loads DGP datasets + trained EIPM checkpoints,
# computes MAB / RMSE on stored (T_eval, mu_eval), and prints a settings table + saves CSV.

from __future__ import annotations

import math
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# =========================
# Fixed paths (run-only)
# =========================
DATA_DIR = Path("./datasets")
DATA_GLOB = "sim_*.npz"

MODELS_DIR = Path("./models/eipm")
RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = RESULTS_DIR / "eval_summary.csv"


# =========================
# Model (must match train_eipm.py)
# =========================
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

    def forward(self, X: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        if T.ndim == 1:
            T_in = T.view(-1, 1)
        else:
            T_in = T
        inp = torch.cat([X, T_in], dim=1)
        return self.net(inp).view(-1)


@torch.no_grad()
def stable_exp_scores(s: torch.Tensor) -> torch.Tensor:
    # exp(s - max(s)) for stability (matches train_eipm.py logic)
    s = s.view(-1)
    return torch.exp(s - torch.max(s))


@torch.no_grad()
def mu_hat_batch(
    model: nn.Module,
    X_train_scaled: torch.Tensor,   # (n_train, d_X), already / sqrt(d_X)
    T_train_tilde: torch.Tensor,    # (n_train,)
    Y_train: torch.Tensor,          # (n_train,)
    T_query_tilde: torch.Tensor,    # (n_eval,) in tildeT scale
    h_query: float,
    batch_size: int = 256,
) -> torch.Tensor:
    """
    Vectorized implementation of mu_hat_at_t over many t_query:
      \hat{\mu}(t) = sum_j W_j(t) Y_j,
      W_j(t) ∝ K_h(T_j - t) * exp(s_theta(X_j, T_j)).
    """
    device = X_train_scaled.device
    n_train = T_train_tilde.numel()
    n_eval = T_query_tilde.numel()

    # exp(s_theta) on train points once
    s_tr = model(X_train_scaled, T_train_tilde).view(-1)            # (n_train,)
    exp_s_tr = stable_exp_scores(s_tr).view(n_train, 1)             # (n_train,1)

    Y_col = Y_train.view(n_train, 1)

    h = float(max(h_query, 1e-3))
    out = torch.empty((n_eval,), dtype=torch.float32, device=device)

    T_tr_col = T_train_tilde.view(-1, 1)

    for start in range(0, n_eval, batch_size):
        end = min(start + batch_size, n_eval)
        t = T_query_tilde[start:end].view(1, -1)                    # (1,b)

        diff_sq = (T_tr_col - t) ** 2                               # (n_train,b)
        K = torch.exp(-0.5 * diff_sq / (h ** 2))                    # (n_train,b)

        w_unnorm = K * exp_s_tr                                     # (n_train,b)
        denom = torch.sum(w_unnorm, dim=0) + 1e-8                   # (b,)
        num = torch.sum(w_unnorm * Y_col, dim=0)                    # (b,)

        out[start:end] = num / denom

    return out


# =========================
# Helpers
# =========================
CKPT_RE = re.compile(r"^(?P<stem>.+)_rep(?P<rep>\d{3})_d(?P<d>\d+)_w(?P<w>\d+)\.pth$")


def fmt_hms(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def np_scalar(d: np.lib.npyio.NpzFile, key: str, default=None):
    if key not in d.files:
        return default
    x = np.array(d[key])
    if x.shape == ():
        return x.item()
    if x.size == 1:
        return x.reshape(-1)[0].item()
    return x


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def scan_tasks(npz_files: List[Path]) -> Tuple[int, Dict[Tuple[str, int, int], Dict]]:
    """
    Count total (rep-evaluation) tasks to provide ETA.
    Returns:
      total_tasks,
      index dict: (stem, depth, width) -> {"ckpts": {rep_idx: ckpt_path}, "npz_path": Path}
    """
    index: Dict[Tuple[str, int, int], Dict] = {}
    total = 0

    for npz_path in npz_files:
        stem = npz_path.stem
        ckpt_paths = list(MODELS_DIR.glob(f"{stem}_rep*_d*_w*.pth"))
        for p in ckpt_paths:
            m = CKPT_RE.match(p.name)
            if not m:
                continue
            rep = int(m.group("rep"))
            d = int(m.group("d"))
            w = int(m.group("w"))
            key = (stem, d, w)
            if key not in index:
                index[key] = {"npz_path": npz_path, "ckpts": {}}
            index[key]["ckpts"][rep] = p

    for key, obj in index.items():
        total += len(obj["ckpts"])

    return total, index


def main() -> None:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Missing datasets directory: {DATA_DIR.resolve()}")
    if not MODELS_DIR.exists():
        raise FileNotFoundError(f"Missing models directory: {MODELS_DIR.resolve()}")

    npz_files = sorted(DATA_DIR.glob(DATA_GLOB))
    if len(npz_files) == 0:
        print(f"No dataset files found: {DATA_DIR.resolve()}/{DATA_GLOB}")
        return

    total_tasks, task_index = scan_tasks(npz_files)
    if total_tasks == 0:
        print(f"No checkpoints found in: {MODELS_DIR.resolve()} (pattern: *_rep###_d*_w*.pth)")
        return

    device = get_device()
    print(f"Device: {device}")
    print(f"Datasets: {len(npz_files)} files")
    print(f"Checkpoints to evaluate (rep-level tasks): {total_tasks}")
    print("=== Evaluation starts ===")

    t_start = time.time()
    done = 0

    rows: List[Dict] = []

    # Iterate by dataset, then by (depth,width) combos for that dataset
    # Build mapping: npz_stem -> list of (depth,width) keys present
    stem_to_configs: Dict[str, List[Tuple[int, int]]] = {}
    for (stem, d, w) in task_index.keys():
        stem_to_configs.setdefault(stem, []).append((d, w))
    for stem in stem_to_configs:
        stem_to_configs[stem] = sorted(list(set(stem_to_configs[stem])))

    for file_i, npz_path in enumerate(npz_files, start=1):
        stem = npz_path.stem
        configs = stem_to_configs.get(stem, [])
        if len(configs) == 0:
            continue

        with np.load(npz_path, allow_pickle=True) as d:
            scenario = str(np_scalar(d, "scenario", "unknown"))
            d_X = int(np_scalar(d, "d_X", -1))
            n_train = int(np_scalar(d, "n_train", -1))
            n_eval = int(np_scalar(d, "n_eval", -1))
            n_rpt = int(np_scalar(d, "n_rpt", -1))
            seed = int(np_scalar(d, "seed", -1))
            treatment_k = int(np_scalar(d, "treatment_k", -1))
            outcome_k = int(np_scalar(d, "outcome_k", -1))

            Xtilde_train_all = d["Xtilde_train"]   # (n_rpt, n_train, d_X)
            tildeT_train_all = d["tildeT_train"]   # (n_rpt, n_train)
            Y_train_all      = d["Y_train"]        # (n_rpt, n_train)

            T_eval_all       = d["T_eval"]         # (n_rpt, n_eval) RAW T scale
            mu_eval_all      = d["mu_eval"]        # (n_rpt, n_eval)

            ET_all           = d["ET"]             # stacked across reps
            VT_all           = d["VT"]             # stacked across reps

        print(f"\n[{file_i}/{len(npz_files)}] Dataset: {npz_path.name} | scenario={scenario} | d_X={d_X} | n_train={n_train} | (T_k={treatment_k}, Y_k={outcome_k})")

        for (depth, width) in configs:
            key = (stem, depth, width)
            ckpts: Dict[int, Path] = task_index[key]["ckpts"]

            rep_abs_means: List[float] = []
            rep_sq_means: List[float] = []

            # Evaluate only reps that have checkpoints
            rep_indices = sorted([r for r in ckpts.keys() if 0 <= r < n_rpt])
            missing_in_range = n_rpt - len(rep_indices)

            print(f"  Config: depth={depth}, width={width} | reps_with_ckpt={len(rep_indices)}/{n_rpt}")

            for rep_idx in rep_indices:
                ckpt_path = ckpts[rep_idx]
                ckpt = torch.load(str(ckpt_path), map_location="cpu")

                train_stats = ckpt.get("train_stats", {}) or {}
                h_query = float(train_stats.get("h_median", 0.0))
                h_query = max(h_query, 1e-3)

                # Build model + load
                model = EIPM(input_dim=d_X + 1, hidden=width, n_layers=depth)
                model.load_state_dict(ckpt["model_state"])
                model.to(device)
                model.eval()

                # tensors
                Xtilde = torch.tensor(Xtilde_train_all[rep_idx], dtype=torch.float32, device=device)
                Ttilde = torch.tensor(tildeT_train_all[rep_idx], dtype=torch.float32, device=device).view(-1)
                Y      = torch.tensor(Y_train_all[rep_idx], dtype=torch.float32, device=device).view(-1)

                # training uses X / sqrt(d_X)
                X_scaled = Xtilde / math.sqrt(float(d_X))

                # eval: raw T -> tilde using ET,VT of that rep
                ET_r = float(np.array(ET_all[rep_idx]).item())
                VT_r = float(np.array(VT_all[rep_idx]).item())
                denom = math.sqrt(max(VT_r, 1e-12))

                T_eval_raw = torch.tensor(T_eval_all[rep_idx], dtype=torch.float32, device=device).view(-1)
                T_eval_tilde = (T_eval_raw - ET_r) / denom

                mu_true = torch.tensor(mu_eval_all[rep_idx], dtype=torch.float32, device=device).view(-1)

                # predict
                mu_pred = mu_hat_batch(
                    model=model,
                    X_train_scaled=X_scaled,
                    T_train_tilde=Ttilde,
                    Y_train=Y,
                    T_query_tilde=T_eval_tilde,
                    h_query=h_query,
                    batch_size=256,
                )

                err = (mu_pred - mu_true).view(-1)
                mab_r = float(torch.mean(torch.abs(err)).item())
                mse_r = float(torch.mean(err ** 2).item())

                rep_abs_means.append(mab_r)
                rep_sq_means.append(mse_r)

                done += 1
                elapsed = time.time() - t_start
                eta = (elapsed / done) * (total_tasks - done) if done > 0 else float("nan")

                print(
                    f"    [{done}/{total_tasks}] rep={rep_idx:03d} | MAB_r={mab_r:.6g} | MSE_r={mse_r:.6g} | elapsed={fmt_hms(elapsed)} | ETA={fmt_hms(eta)}"
                )

            if len(rep_abs_means) == 0:
                mab = float("nan")
                rmse = float("nan")
                n_used = 0
            else:
                mab = float(np.mean(rep_abs_means))
                rmse = float(math.sqrt(np.mean(rep_sq_means)))
                n_used = len(rep_abs_means)

            rows.append(
                {
                    "n_tr": int(n_train),
                    "d_X": int(d_X),
                    "scenario": scenario,
                    "treatment_k": int(treatment_k),
                    "outcome_k": int(outcome_k),
                    "depth": int(depth),
                    "width": int(width),
                    "n_eval": int(n_eval),
                    "n_rpt_in_file": int(n_rpt),
                    "n_reps_used": int(n_used),
                    "n_ckpt_missing_in_range": int(missing_in_range),
                    "MAB": mab,
                    "RMSE": rmse,
                    "npz": npz_path.name,
                }
            )

            print(f"  -> Summary (depth={depth}, width={width}): MAB={mab:.6g}, RMSE={rmse:.6g} (used reps={n_used}/{n_rpt})")

    if len(rows) == 0:
        print("No evaluations completed (no matching ckpts for found datasets).")
        return

    df = pd.DataFrame(rows)

    # Your requested table shape: settings columns then last two cols MAB, RMSE
    show_cols = ["n_tr", "d_X", "scenario", "treatment_k", "outcome_k", "MAB", "RMSE"]
    # If multiple (depth,width) exist, keep them too (otherwise ambiguity).
    if df[["depth", "width"]].drop_duplicates().shape[0] > 1:
        show_cols = ["n_tr", "d_X", "scenario", "treatment_k", "outcome_k", "depth", "width", "MAB", "RMSE"]

    # Sort for readability (matches DGP grid)
    sort_cols = [c for c in ["d_X", "scenario", "n_tr", "treatment_k", "outcome_k", "depth", "width"] if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda x: f"{x:.6g}")

    print("\n=== Final table ===")
    print(df[show_cols].to_string(index=False))

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved CSV: {OUTPUT_CSV.resolve()}")
    print(f"Total elapsed: {fmt_hms(time.time() - t_start)}")


if __name__ == "__main__":
    main()
