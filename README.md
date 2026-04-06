# ADRF Benchmark Code

This public repository contains the executable scripts in `run/`, the import-only modules in `lib/`, and the public NMES input files in `data/`.

Public files included in the repository:

- `run/data.sh`
- `run/train.sh`
- `run/eval.sh`
- `lib/core.py`
- `lib/dgp_base.py`
- `lib/train_base.py`
- `lib/eval_base.py`
- `lib/helpers.py`
- `lib/gpu.py`
- `data/nmes_data.csv`
- `data/nmes_data_n2000_seed42.csv`

The public NMES CSV files are stored in a cleaned analysis schema. Their columns are:

- `packyears`
- `totalexp`
- `lastage`
- `male`
- `race3`
- `beltuse`
- `educate`
- `marital`
- `sregion`
- `povstalb`

Files that are **not** stored in the public repository and are created locally when you run the code:

- simulation `.npz` datasets under `data/`
- trained checkpoints under `out/models/`
- result CSV files under `out/`

## Replication Flow

Run all commands from this directory:

```bash
cd code
```

First, run `run/data.sh`. This script does the entire data-preparation stage. It generates the full simulation sweep and saves the simulation datasets as `.npz` files in `data/`. The built-in simulation sweep covers the two scenarios (`linear`, `nonlinear`) and the three configuration sweeps used in the study: `n_train`, `(d_X, |S_T|, |S_Y|)`, and `pi_0`. In the same data stage, it also reads the public NMES file `data/nmes_data.csv`, constructs the repeated-subsample NMES adapter dataset, generates the NMES synthetic outcome used as the pseudo-outcome benchmark, and saves that adapter dataset as `data/nmes_tmp_for_eipm_*.npz`. For NMES, the public CSV already contains only the cleaned analysis columns listed above, so treatment/outcome duplicates and derived leakage columns are not present in the repository data.

Next, run `run/train.sh`. This script no longer creates data. It reads the simulation `.npz` files already prepared in `data/` and trains the EIPM models for all simulation settings, storing the resulting checkpoints under `out/models/eipm_sweeps/`. It also reads the pre-generated NMES adapter dataset from `data/nmes_tmp_for_eipm_*.npz` and trains the corresponding NMES EIPM checkpoints under `out/models/eipm_nmes/`.

Finally, run `run/eval.sh`. This script reads the locally generated simulation datasets and the pre-generated NMES adapter dataset from `data/`, reads the locally trained checkpoints from `out/models/eipm_sweeps/` and `out/models/eipm_nmes/`, and writes the final benchmark result tables as per-method CSV files under `out/`. The default public evaluation path runs `eipm`, `stabilized_gps`, and `independence_weights`. If you explicitly add `unweighted` to `--methods`, the script will also write `out/results_unweighted.csv`.

In short, the public replication order is:

```bash
run/data.sh
run/train.sh
run/eval.sh
```

and the final outputs to inspect are the locally generated `out/results_*.csv` files.

## Notes

- `EIPM` is model-free in this code path: it directly learns `log w_theta(x,t)` and does not fit a propensity/GPS model.
- Exact `CBGPS` is not part of the default public workflow here, because it requires an external R bridge or externally supplied weights.
- The shell scripts try to activate the conda environment `ten` if `/usr/local/miniconda3/etc/profile.d/conda.sh` exists; otherwise they use the current `python` on `PATH`.
