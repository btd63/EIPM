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

from helpers import generate_dgp_dataset


def parse_args():
    p = argparse.ArgumentParser(description="Generate simulation sweep datasets.")
    p.add_argument("--data_dir", type=str, default=str(ROOT / "data"))
    p.add_argument("--scenarios", type=str, default="linear,nonlinear")
    p.add_argument("--n_rpt", type=int, default=100)
    p.add_argument("--n_eval", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--beta_T_scale", type=float, default=0.2)
    p.add_argument("--beta_T0_scale", type=float, default=0.2)
    return p.parse_args()


args = parse_args()
data_dir = Path(args.data_dir)
data_dir.mkdir(parents=True, exist_ok=True)

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
    ds_name = generate_dgp_dataset(
        code_dir=ROOT,
        scenario=scenario,
        d_x=d_x,
        n_train=n_train,
        n_eval=int(args.n_eval),
        n_rpt=int(args.n_rpt),
        pi_0=float(pi0),
        seed=int(args.seed),
        treatment_k=int(tk),
        outcome_k=int(ok),
        beta_t0_scale=float(args.beta_T0_scale),
        beta_t_scale=float(args.beta_T_scale),
    )
    print(f"[DONE] generated {ds_name}")
PY
