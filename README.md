# Code (simulation + real data)

This folder contains the current EIPM pipeline for simulated semicontinuous treatments.

Notes
- `device_utils.py` is included to keep `train_eipm.py` / `evaluate_eipm.py` runnable.
- Comparator method implementations and benchmark runners are available:
  - `benchmark_core.py`
  - `run_simulation_benchmarks.py`
  - `run_nmes_benchmarks.py`
- NMES analysis code includes semicontinuous `T=0` handling and pseudo-truth benchmarking.
- Comparator status from attached sources:
  - implemented and runnable: `stabilized_gps`, `independence_hsic` (DCOW QP).
  - intentionally skipped in this environment: `cbgps_like` (requires R/CBPS), `koow_like` (exact source not attached).

High-level workflow
1. Generate datasets from the semicontinuous DGP.
2. Fit weights for each method.
   - EIPM is model-free: it directly learns `log w_theta(x,t)` and uses checkpoint model outputs `f_theta(x,t)` at evaluation time.
   - No separate treatment-assignment estimator is used in the EIPM path.
3. Estimate ADRF on a common evaluation grid (treat `t=0` separately when applicable).
4. Aggregate RMSE/MAE (simulation) or pseudo-RMSE/MAE (NMES) and export tables for the LaTeX templates.
