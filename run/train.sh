#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_SH="/usr/local/miniconda3/etc/profile.d/conda.sh"
if [[ -f "${CONDA_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate ten
fi
PYTHON_BIN="${PYTHON:-python}"

CODE_DIR="${ROOT}" "${PYTHON_BIN}" - "$@" <<'PY'
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ["CODE_DIR"])
sys.path.insert(0, str(ROOT / "lib"))

from core import make_repeated_subsample_indices, make_synthetic_outcome, predict_oracle_mu
from helpers import build_dgp_name, train_eipm_for_dataset


def parse_args():
    p = argparse.ArgumentParser(description="Train all simulation and NMES EIPM models in one run.")
    p.add_argument("--data_dir", type=str, default=str(ROOT / "data"))
    p.add_argument("--sim_models_dir", type=str, default=str(ROOT / "out" / "models" / "eipm_sweeps"))
    p.add_argument("--nmes_models_dir", type=str, default=str(ROOT / "out" / "models" / "eipm_nmes"))
    p.add_argument("--nmes_path", type=str, default=str(ROOT / "data" / "nmes_data.csv"))
    p.add_argument("--scenarios", type=str, default="linear,nonlinear")
    p.add_argument("--n_rpt", type=int, default=100)
    p.add_argument("--n_eval", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--beta_T_scale", type=float, default=0.2)
    p.add_argument("--beta_T0_scale", type=float, default=0.2)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--k_folds", type=int, default=5)
    p.add_argument("--eval_every", type=int, default=30)
    p.add_argument("--nn", type=float, default=0.7)
    p.add_argument("--overwrite", type=int, default=0)
    p.add_argument("--grid_n_pos", type=int, default=40)
    p.add_argument("--oracle_model", type=str, choices=["hgb", "rf"], default="hgb")
    p.add_argument("--nmes_sample_n", type=int, default=2000)
    p.add_argument("--nmes_n_rpt", type=int, default=5)
    return p.parse_args()


def build_grid(T: np.ndarray, n_pos: int) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).reshape(-1)
    pos = T[T > 0.0]
    if pos.size == 0:
        return np.array([0.0], dtype=np.float64)
    q = np.linspace(0.02, 0.98, max(5, int(n_pos)))
    gp = np.quantile(np.log1p(pos), q)
    gp = np.expm1(gp)
    gp = np.unique(np.asarray(gp, dtype=np.float64))
    return np.concatenate([np.array([0.0], dtype=np.float64), gp])


args = parse_args()
data_dir = Path(args.data_dir)
sim_models_dir = Path(args.sim_models_dir)
nmes_models_dir = Path(args.nmes_models_dir)
data_dir.mkdir(parents=True, exist_ok=True)
sim_models_dir.mkdir(parents=True, exist_ok=True)
nmes_models_dir.mkdir(parents=True, exist_ok=True)

# 1) Simulation training from pre-generated sweep datasets.
baseline = {"n_train": 1000, "d_x": 50, "tk": 5, "ok": 5, "pi0": 0.5}
n_train_list = [250, 500, 1000, 2000]
dims_list = [(5, 5, 5), (50, 5, 5), (50, 5, 50), (50, 50, 5), (50, 50, 50)]
pi0_list = [0.0, 0.2, 0.5, 0.8]
scenarios = [s.strip().lower() for s in str(args.scenarios).split(",") if s.strip()]
if not scenarios:
    raise ValueError("At least one scenario must be provided.")

seen = set()
configs = []
for scenario in scenarios:
    for ntr in n_train_list:
        key = (scenario, ntr, baseline["d_x"], baseline["tk"], baseline["ok"], baseline["pi0"])
        if key not in seen:
            seen.add(key)
            configs.append(key)
    for d_x, tk, ok in dims_list:
        key = (scenario, baseline["n_train"], d_x, tk, ok, baseline["pi0"])
        if key not in seen:
            seen.add(key)
            configs.append(key)
    for pi0 in pi0_list:
        key = (scenario, baseline["n_train"], baseline["d_x"], baseline["tk"], baseline["ok"], float(pi0))
        if key not in seen:
            seen.add(key)
            configs.append(key)

for scenario, n_train, d_x, tk, ok, pi0 in configs:
    ds_name = build_dgp_name(
        scenario=scenario,
        d_x=d_x,
        n_train=n_train,
        n_eval=int(args.n_eval),
        n_rpt=int(args.n_rpt),
        treatment_k=int(tk),
        outcome_k=int(ok),
        pi_0=float(pi0),
        seed=int(args.seed),
        beta_t0_scale=float(args.beta_T0_scale),
        beta_t_scale=float(args.beta_T_scale),
    )
    npz_path = data_dir / f"{ds_name}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing dataset {npz_path}. Run run/data_sim.sh first.")
    train_eipm_for_dataset(
        code_dir=ROOT,
        data_dir=data_dir,
        dataset_name=ds_name,
        models_dir=sim_models_dir,
        n_rpt=int(args.n_rpt),
        device=str(args.device),
        max_steps=int(args.max_steps),
        k_folds=int(args.k_folds),
        eval_every=int(args.eval_every),
        nn=float(args.nn),
        overwrite=int(args.overwrite),
    )
    print(f"[DONE] trained simulation {ds_name}")

# 2) NMES adapter creation + training.
nmes_path = Path(args.nmes_path)
if not nmes_path.exists():
    raise FileNotFoundError(f"NMES file not found: {nmes_path}")

df = pd.read_csv(nmes_path)
if "packyears" not in df.columns or "TOTALEXP" not in df.columns:
    raise KeyError("NMES CSV must include 'packyears' and 'TOTALEXP'.")

T = pd.to_numeric(df["packyears"], errors="coerce").to_numpy(dtype=np.float64)
Y = pd.to_numeric(df["TOTALEXP"], errors="coerce").to_numpy(dtype=np.float64)

x_cols = [c for c in df.columns if c not in ["packyears", "TOTALEXP"]]
X_df = df[x_cols].copy()
cat_cols = [c for c in X_df.columns if str(X_df[c].dtype) in ["object", "category"]]
X = pd.get_dummies(X_df, columns=cat_cols, drop_first=False).to_numpy(dtype=np.float64)

ok = np.isfinite(T) & np.isfinite(Y) & np.isfinite(X).all(axis=1)
X = X[ok]
T = T[ok]
Y = Y[ok]

Y_syn, oracle = make_synthetic_outcome(X, T, Y, model_kind=str(args.oracle_model), seed=int(args.seed))
grid = build_grid(T, n_pos=int(args.grid_n_pos))
mu_oracle = predict_oracle_mu(oracle, X, grid)
rep_indices = make_repeated_subsample_indices(
    X.shape[0],
    sample_n=int(args.nmes_sample_n),
    n_reps=int(args.nmes_n_rpt),
    seed=int(args.seed),
)

sample_n = int(args.nmes_sample_n) if int(args.nmes_sample_n) > 0 else int(X.shape[0])
ds_name = f"nmes_tmp_for_eipm_n{sample_n}_rpt{int(args.nmes_n_rpt)}"
X_reps = np.stack([X[idx] for idx in rep_indices], axis=0)
T_reps = np.stack([T[idx] for idx in rep_indices], axis=0)
Y_reps = np.stack([Y_syn[idx] for idx in rep_indices], axis=0)
T_eval = np.repeat(grid.reshape(1, -1), len(rep_indices), axis=0)
mu_eval = np.repeat(mu_oracle.reshape(1, -1), len(rep_indices), axis=0)

np.savez_compressed(
    data_dir / f"{ds_name}.npz",
    scenario=np.array("nmes"),
    d_X=np.array(X_reps.shape[2]),
    n_train=np.array(X_reps.shape[1]),
    n_eval=np.array(grid.shape[0]),
    n_rpt=np.array(len(rep_indices)),
    X_train=X_reps,
    T_train=T_reps,
    Y_train=Y_reps,
    T_eval=T_eval,
    mu_eval=mu_eval,
)

train_eipm_for_dataset(
    code_dir=ROOT,
    data_dir=data_dir,
    dataset_name=ds_name,
    models_dir=nmes_models_dir,
    n_rpt=len(rep_indices),
    device=str(args.device),
    max_steps=int(args.max_steps),
    k_folds=int(args.k_folds),
    eval_every=int(args.eval_every),
    nn=float(args.nn),
    overwrite=int(args.overwrite),
)
print(f"[DONE] trained NMES {ds_name}")
PY
