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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ["CODE_DIR"])
sys.path.insert(0, str(ROOT / "lib"))

from core import make_repeated_subsample_indices, make_synthetic_outcome, predict_oracle_mu
from helpers import (
    MetricSummary,
    build_dgp_name,
    clean_results_dir,
    estimate_semicont_curve_from_context,
    evaluate_sim_config_all_methods,
    filter_supported_methods,
    fit_method_eval_specs,
    normalize_method_list,
    prepare_eval_context_from_ckpt,
    results_csv_path,
    rmse_mae,
    summarize_metric_rows,
    upsert_results_rows,
)


@dataclass
class RepData:
    d_X: int
    X: np.ndarray
    T: np.ndarray
    Y: np.ndarray


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate simulation and NMES in one run and write results_*.csv.")
    p.add_argument("--results_dir", type=str, default=str(ROOT / "out"))
    p.add_argument("--data_dir", type=str, default=str(ROOT / "data"))
    p.add_argument("--sim_models_dir", type=str, default=str(ROOT / "out" / "models" / "eipm_sweeps"))
    p.add_argument("--nmes_models_dir", type=str, default=str(ROOT / "out" / "models" / "eipm_nmes"))
    p.add_argument("--nmes_path", type=str, default=str(ROOT / "data" / "nmes_data.csv"))
    p.add_argument("--methods", type=str, default="eipm,stabilized_gps,independence_weights")
    p.add_argument("--scenarios", type=str, default="linear,nonlinear")
    p.add_argument("--clip_max", type=float, default=1e4)
    p.add_argument("--n_rpt", type=int, default=100)
    p.add_argument("--n_eval", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--beta_T_scale", type=float, default=0.2)
    p.add_argument("--beta_T0_scale", type=float, default=0.2)
    p.add_argument("--grid_n_pos", type=int, default=40)
    p.add_argument("--oracle_model", type=str, choices=["hgb", "rf"], default="hgb")
    p.add_argument("--external_weights_dir", type=str, default="")
    p.add_argument("--nmes_sample_n", type=int, default=2000)
    p.add_argument("--nmes_n_rpt", type=int, default=5)
    p.add_argument("--no_clean", action="store_true")
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


def sim_row(dataset_id: str, scenario: str, sweep_type: str, factor_value: str, smoothing: str, m: MetricSummary):
    return {
        "dataset_id": dataset_id,
        "scenario": scenario,
        "sweep_type": sweep_type,
        "factor_value": factor_value,
        "smoothing": smoothing,
        "rmse_mean": m.rmse_mean,
        "rmse_se": m.rmse_se,
        "mae_mean": m.mae_mean,
        "mae_se": m.mae_se,
        "pseudo_rmse": "",
        "pseudo_mae": "",
    }


def nmes_row(dataset_id: str, smoothing: str, rmse_mean: float, rmse_se: float, mae_mean: float, mae_se: float):
    return {
        "dataset_id": dataset_id,
        "scenario": "nmes",
        "sweep_type": "nmes",
        "factor_value": "nmes",
        "smoothing": smoothing,
        "rmse_mean": rmse_mean,
        "rmse_se": rmse_se,
        "mae_mean": mae_mean,
        "mae_se": mae_se,
        "pseudo_rmse": rmse_mean,
        "pseudo_mae": mae_mean,
    }


args = parse_args()
methods, skipped = filter_supported_methods(normalize_method_list(args.methods))
for m, reason in skipped.items():
    print(f"[SKIP] method={m}: {reason}")

results_dir = Path(args.results_dir)
data_dir = Path(args.data_dir)
sim_models_dir = Path(args.sim_models_dir)
nmes_models_dir = Path(args.nmes_models_dir)
if not args.no_clean:
    clean_results_dir(results_dir)
results_dir.mkdir(parents=True, exist_ok=True)

# 1) Simulation evaluation.
baseline = {"n_train": 1000, "d_x": 50, "tk": 5, "ok": 5, "pi0": 0.5}
n_train_list = [250, 500, 1000, 2000]
dims_list = [(5, 5, 5), (50, 5, 5), (50, 5, 50), (50, 50, 5), (50, 50, 50)]
pi0_list = [0.0, 0.2, 0.5, 0.8]
scenarios = [s.strip().lower() for s in str(args.scenarios).split(",") if s.strip()]

cache = {}
def get_sim_metrics(scenario, n_train, d_x, tk, ok, pi0):
    key = (scenario, int(n_train), int(d_x), int(tk), int(ok), float(pi0))
    if key in cache:
        return cache[key]
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
    ckpt_dir = sim_models_dir / ds_name
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Missing checkpoints for {ds_name}. Run run/train.sh first.")
    by_degree = evaluate_sim_config_all_methods(
        npz_path=npz_path,
        ckpt_root=sim_models_dir,
        n_rpt=int(args.n_rpt),
        methods=methods,
        degrees=[0, 1],
        seed=int(args.seed),
        clip_max=float(args.clip_max),
        external_weights_dir=(Path(args.external_weights_dir) if str(args.external_weights_dir).strip() else None),
    )
    cache[key] = (ds_name, by_degree)
    return ds_name, by_degree

sim_rows_by_method = {m: [] for m in methods}
for scenario in scenarios:
    for ntr in n_train_list:
        ds_name, by_degree = get_sim_metrics(scenario, ntr, baseline["d_x"], baseline["tk"], baseline["ok"], baseline["pi0"])
        fv = str(int(ntr))
        for degree, smoothing in [(0, "NW"), (1, "LL")]:
            for m in methods:
                sim_rows_by_method[m].append(sim_row(ds_name, scenario, "n", fv, smoothing, by_degree[degree][m]))
    for d_x, tk, ok in dims_list:
        ds_name, by_degree = get_sim_metrics(scenario, baseline["n_train"], d_x, tk, ok, baseline["pi0"])
        fv = f"({int(d_x)},{int(tk)},{int(ok)})"
        for degree, smoothing in [(0, "NW"), (1, "LL")]:
            for m in methods:
                sim_rows_by_method[m].append(sim_row(ds_name, scenario, "dims", fv, smoothing, by_degree[degree][m]))
    for pi0 in pi0_list:
        ds_name, by_degree = get_sim_metrics(scenario, baseline["n_train"], baseline["d_x"], baseline["tk"], baseline["ok"], float(pi0))
        fv = f"{float(pi0):.1f}"
        for degree, smoothing in [(0, "NW"), (1, "LL")]:
            for m in methods:
                sim_rows_by_method[m].append(sim_row(ds_name, scenario, "pi0", fv, smoothing, by_degree[degree][m]))

# 2) NMES evaluation.
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
ckpt_dir = nmes_models_dir / ds_name
if not ckpt_dir.exists():
    raise FileNotFoundError(f"Missing checkpoints for {ds_name}. Run run/train.sh first.")

ext_dir = Path(args.external_weights_dir) if str(args.external_weights_dir).strip() else None
nmes_metric_rows = {m: {"NW": [], "LL": []} for m in methods}
for r, idx in enumerate(rep_indices):
    ckpt_path = ckpt_dir / f"eipm_single_nonlinear_rep{r:03d}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Expected checkpoint not found: {ckpt_path}")
    rep = RepData(d_X=int(X.shape[1]), X=X[idx], T=T[idx], Y=Y_syn[idx])
    ctx = prepare_eval_context_from_ckpt(rep, ckpt_path)
    method_specs = {}
    for m in methods:
        if m == "eipm":
            continue
        try:
            method_specs.update(
                fit_method_eval_specs(
                    rep.X,
                    rep.T,
                    [m],
                    seed=int(args.seed) + int(r),
                    clip_max=float(args.clip_max),
                    dataset_id=ds_name,
                    rep_index=r,
                    external_weights_dir=ext_dir,
                )
            )
        except Exception as e:
            if ext_dir is not None and m in {"cbgps", "independence_weights"}:
                raise
            print(f"[WARN] skip method={m} on NMES rep={r}: {e}")
    for degree, smoothing in [(0, "NW"), (1, "LL")]:
        for m in methods:
            try:
                if m != "eipm" and m not in method_specs:
                    raise ValueError(f"missing method spec for {m}")
                mu_hat, _ess, _maxw = estimate_semicont_curve_from_context(
                    ctx,
                    grid,
                    degree=int(degree),
                    method_spec=None if m == "eipm" else method_specs[m],
                )
                rmse_m, mae_m = rmse_mae(mu_hat, mu_oracle)
            except Exception as e:
                print(f"[WARN] evaluation failed method={m} rep={r} smoothing={smoothing}: {e}")
                rmse_m, mae_m = float("nan"), float("nan")
            nmes_metric_rows[m][smoothing].append((rmse_m, mae_m))

# 3) Write merged CSV outputs.
for m in methods:
    out_csv = results_csv_path(results_dir, m)
    rows = list(sim_rows_by_method[m])
    for smoothing in ["NW", "LL"]:
        summary = summarize_metric_rows(nmes_metric_rows[m][smoothing])
        rows.append(nmes_row(ds_name, smoothing, summary.rmse_mean, summary.rmse_se, summary.mae_mean, summary.mae_se))
    upsert_results_rows(out_csv, rows, key_fields=["dataset_id", "scenario", "sweep_type", "factor_value", "smoothing"])
    print(f"[DONE] wrote {out_csv}")
PY
