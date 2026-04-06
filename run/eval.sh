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

ROOT = Path(os.environ["CODE_DIR"])
sys.path.insert(0, str(ROOT / "lib"))

from helpers import (
    MetricSummary,
    build_dgp_name,
    clean_results_dir,
    evaluate_sim_config_all_methods,
    filter_supported_methods,
    normalize_method_list,
    results_csv_path,
    upsert_results_rows,
)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate simulation and NMES in one run and write results_*.csv.")
    p.add_argument("--results_dir", type=str, default=str(ROOT / "out"))
    p.add_argument("--data_dir", type=str, default=str(ROOT / "data"))
    p.add_argument("--sim_models_dir", type=str, default=str(ROOT / "out" / "models" / "eipm_sweeps"))
    p.add_argument("--nmes_models_dir", type=str, default=str(ROOT / "out" / "models" / "eipm_nmes"))
    p.add_argument("--methods", type=str, default="eipm,stabilized_gps,independence_weights")
    p.add_argument("--scenarios", type=str, default="linear,nonlinear")
    p.add_argument("--clip_max", type=float, default=1e4)
    p.add_argument("--n_rpt", type=int, default=100)
    p.add_argument("--n_eval", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--beta_T_scale", type=float, default=0.2)
    p.add_argument("--beta_T0_scale", type=float, default=0.2)
    p.add_argument("--external_weights_dir", type=str, default="")
    p.add_argument("--nmes_sample_n", type=int, default=2000)
    p.add_argument("--nmes_n_rpt", type=int, default=5)
    p.add_argument("--no_clean", action="store_true")
    return p.parse_args()


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


def nmes_row(dataset_id: str, smoothing: str, m: MetricSummary):
    return {
        "dataset_id": dataset_id,
        "scenario": "nmes",
        "sweep_type": "nmes",
        "factor_value": "nmes",
        "smoothing": smoothing,
        "rmse_mean": m.rmse_mean,
        "rmse_se": m.rmse_se,
        "mae_mean": m.mae_mean,
        "mae_se": m.mae_se,
        "pseudo_rmse": m.rmse_mean,
        "pseudo_mae": m.mae_mean,
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

external_weights_dir = Path(args.external_weights_dir) if str(args.external_weights_dir).strip() else None

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
        raise FileNotFoundError(f"Missing dataset {npz_path}. Run run/data.sh first.")
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
        external_weights_dir=external_weights_dir,
    )
    cache[key] = (ds_name, by_degree)
    return ds_name, by_degree


rows_by_method = {m: [] for m in methods}
for scenario in scenarios:
    for ntr in n_train_list:
        ds_name, by_degree = get_sim_metrics(scenario, ntr, baseline["d_x"], baseline["tk"], baseline["ok"], baseline["pi0"])
        fv = str(int(ntr))
        for degree, smoothing in [(0, "NW"), (1, "LL")]:
            for m in methods:
                rows_by_method[m].append(sim_row(ds_name, scenario, "n", fv, smoothing, by_degree[degree][m]))
    for d_x, tk, ok in dims_list:
        ds_name, by_degree = get_sim_metrics(scenario, baseline["n_train"], d_x, tk, ok, baseline["pi0"])
        fv = f"({int(d_x)},{int(tk)},{int(ok)})"
        for degree, smoothing in [(0, "NW"), (1, "LL")]:
            for m in methods:
                rows_by_method[m].append(sim_row(ds_name, scenario, "dims", fv, smoothing, by_degree[degree][m]))
    for pi0 in pi0_list:
        ds_name, by_degree = get_sim_metrics(scenario, baseline["n_train"], baseline["d_x"], baseline["tk"], baseline["ok"], float(pi0))
        fv = f"{float(pi0):.1f}"
        for degree, smoothing in [(0, "NW"), (1, "LL")]:
            for m in methods:
                rows_by_method[m].append(sim_row(ds_name, scenario, "pi0", fv, smoothing, by_degree[degree][m]))

# 2) NMES evaluation from pre-generated adapter dataset.
sample_n = int(args.nmes_sample_n)
ds_name = f"nmes_tmp_for_eipm_n{sample_n}_rpt{int(args.nmes_n_rpt)}"
npz_path = data_dir / f"{ds_name}.npz"
if not npz_path.exists():
    raise FileNotFoundError(f"Missing dataset {npz_path}. Run run/data.sh first.")
ckpt_dir = nmes_models_dir / ds_name
if not ckpt_dir.exists():
    raise FileNotFoundError(f"Missing checkpoints for {ds_name}. Run run/train.sh first.")

nmes_by_degree = evaluate_sim_config_all_methods(
    npz_path=npz_path,
    ckpt_root=nmes_models_dir,
    n_rpt=int(args.nmes_n_rpt),
    methods=methods,
    degrees=[0, 1],
    seed=int(args.seed),
    clip_max=float(args.clip_max),
    external_weights_dir=external_weights_dir,
)
for degree, smoothing in [(0, "NW"), (1, "LL")]:
    for m in methods:
        rows_by_method[m].append(nmes_row(ds_name, smoothing, nmes_by_degree[degree][m]))

for m in methods:
    out_csv = results_csv_path(results_dir, m)
    upsert_results_rows(out_csv, rows_by_method[m], key_fields=["dataset_id", "scenario", "sweep_type", "factor_value", "smoothing"])
    print(f"[DONE] wrote {out_csv}")
PY
