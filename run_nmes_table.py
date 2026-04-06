from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from core.benchmark_core import make_repeated_subsample_indices, make_synthetic_outcome, predict_oracle_mu
from core.table_helpers import (
    clean_results_dir,
    estimate_semicont_curve_from_context,
    filter_supported_methods,
    fit_method_eval_specs,
    normalize_method_list,
    prepare_eval_context_from_ckpt,
    results_csv_path,
    rmse_mae,
    summarize_metric_rows,
    train_eipm_for_dataset,
    upsert_results_rows,
)

ROOT = Path(__file__).resolve().parent


@dataclass
class RepData:
    d_X: int
    X: np.ndarray
    T: np.ndarray
    Y: np.ndarray


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run NMES (1-rep adapter) and append per-method NW/LL results CSV rows.")
    p.add_argument("--nmes_path", type=str, default="")
    p.add_argument("--results_dir", type=str, default=str(ROOT / "results"))
    p.add_argument("--data_dir", type=str, default=str(ROOT / "datasets"))
    p.add_argument("--models_dir", type=str, default=str(ROOT / "models" / "eipm_nmes"))
    p.add_argument("--methods", type=str, default="eipm,stabilized_gps,cbgps,independence_weights")
    p.add_argument("--clip_max", type=float, default=1e4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grid_n_pos", type=int, default=40)
    p.add_argument("--oracle_model", type=str, choices=["hgb", "rf"], default="hgb")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--k_folds", type=int, default=5)
    p.add_argument("--eval_every", type=int, default=30)
    p.add_argument("--nn", type=float, default=0.7)
    p.add_argument(
        "--eipm_max_n",
        type=int,
        default=0,
        help="If >0 and n is larger, subsample rows for faster debug runs.",
    )
    p.add_argument("--external_weights_dir", type=str, default="")
    p.add_argument("--nmes_sample_n", type=int, default=2000)
    p.add_argument("--nmes_n_rpt", type=int, default=5)
    p.add_argument("--overwrite", type=int, default=0)
    p.add_argument("--no_clean", action="store_true")
    return p.parse_args()


def _build_grid(T: np.ndarray, n_pos: int) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).reshape(-1)
    pos = T[T > 0.0]
    if pos.size == 0:
        return np.array([0.0], dtype=np.float64)
    q = np.linspace(0.02, 0.98, max(5, int(n_pos)))
    gp = np.quantile(np.log1p(pos), q)
    gp = np.expm1(gp)
    gp = np.unique(np.asarray(gp, dtype=np.float64))
    return np.concatenate([np.array([0.0], dtype=np.float64), gp])


def _to_row(dataset_id: str, smoothing: str, rmse_mean: float, rmse_se: float, mae_mean: float, mae_se: float) -> Dict:
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


def main() -> None:
    args = parse_args()
    methods_raw = normalize_method_list(args.methods)
    methods, skipped_methods = filter_supported_methods(methods_raw)
    for m, reason in skipped_methods.items():
        print(f"[SKIP] method={m}: {reason}")

    if not str(args.nmes_path).strip():
        print("[SKIP] --nmes_path not provided. NMES execution skipped.")
        return

    results_dir = Path(args.results_dir)
    if not args.no_clean:
        clean_results_dir(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

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
    X_enc = pd.get_dummies(X_df, columns=cat_cols, drop_first=False)
    X = X_enc.to_numpy(dtype=np.float64)

    ok = np.isfinite(T) & np.isfinite(Y) & np.isfinite(X).all(axis=1)
    X = X[ok]
    T = T[ok]
    Y = Y[ok]

    Y, oracle = make_synthetic_outcome(X, T, Y, model_kind=str(args.oracle_model), seed=int(args.seed))
    print("[INFO] NMES outcome replaced with synthetic Y = m_hat(X,T) + permuted residual.")
    # NMES is intentionally evaluated on repeated n=2000 subsamples because the
    # exact independence_weights / DCOW code materializes n x n distance
    # matrices and can exceed memory on the full NMES sample.
    sample_n = int(args.nmes_sample_n) if int(args.nmes_sample_n) > 0 else int(args.eipm_max_n)
    rep_indices = make_repeated_subsample_indices(
        X.shape[0],
        sample_n=int(sample_n),
        n_reps=int(args.nmes_n_rpt),
        seed=int(args.seed),
    )
    grid = _build_grid(T, n_pos=int(args.grid_n_pos))
    mu_oracle = predict_oracle_mu(oracle, X, grid)

    ds_name = f"nmes_tmp_for_eipm_n{int(sample_n) if int(sample_n) > 0 else int(X.shape[0])}_rpt{int(args.nmes_n_rpt)}"
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    ds_path = data_dir / f"{ds_name}.npz"
    X_reps = np.stack([X[idx] for idx in rep_indices], axis=0)
    T_reps = np.stack([T[idx] for idx in rep_indices], axis=0)
    Y_reps = np.stack([Y[idx] for idx in rep_indices], axis=0)
    T_eval = np.repeat(grid.reshape(1, -1), len(rep_indices), axis=0)
    mu_eval = np.repeat(mu_oracle.reshape(1, -1), len(rep_indices), axis=0)

    np.savez_compressed(
        ds_path,
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

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    train_eipm_for_dataset(
        code_dir=ROOT,
        data_dir=data_dir,
        dataset_name=ds_name,
        models_dir=models_dir,
        n_rpt=len(rep_indices),
        device=str(args.device),
        max_steps=int(args.max_steps),
        k_folds=int(args.k_folds),
        eval_every=int(args.eval_every),
        nn=float(args.nn),
        overwrite=int(args.overwrite),
    )

    ext_dir = Path(args.external_weights_dir) if str(args.external_weights_dir).strip() else None
    metric_rows: Dict[str, Dict[str, List[tuple[float, float]]]] = {
        m: {"NW": [], "LL": []} for m in methods
    }
    for r, idx in enumerate(rep_indices):
        ckpt_path = models_dir / ds_name / f"eipm_single_nonlinear_rep{r:03d}.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Expected checkpoint not found: {ckpt_path}")
        rep = RepData(d_X=int(X.shape[1]), X=X[idx], T=T[idx], Y=Y[idx])
        ctx = prepare_eval_context_from_ckpt(rep, ckpt_path)

        method_specs: Dict[str, object] = {}
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
                metric_rows[m][smoothing].append((rmse_m, mae_m))

    rows_by_method: Dict[str, List[Dict]] = {m: [] for m in methods}
    for m in methods:
        for smoothing in ["NW", "LL"]:
            summary = summarize_metric_rows(metric_rows[m][smoothing])
            rows_by_method[m].append(
                _to_row(
                    ds_name,
                    smoothing,
                    summary.rmse_mean,
                    summary.rmse_se,
                    summary.mae_mean,
                    summary.mae_se,
                )
            )

    for m in methods:
        out_csv = results_csv_path(results_dir, m)
        upsert_results_rows(out_csv, rows_by_method[m], key_fields=["dataset_id", "scenario", "sweep_type", "factor_value", "smoothing"])
        print(f"[DONE] wrote/updated {out_csv}")


if __name__ == "__main__":
    main()
