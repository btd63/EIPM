from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
LIB_DIR = ROOT / "lib"
RUN_DIR = ROOT / "run"
for p in [str(LIB_DIR), str(RUN_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from table_helpers import (
    MetricSummary,
    clean_results_dir,
    evaluate_sim_config_all_methods,
    filter_supported_methods,
    generate_dgp_dataset,
    normalize_method_list,
    results_csv_path,
    train_eipm_for_dataset,
    upsert_results_rows,
)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run simulation sweeps and store per-method results CSVs (NW/LL).")
    p.add_argument("--results_dir", type=str, default=str(ROOT / "out"))
    p.add_argument("--data_dir", type=str, default=str(ROOT / "data"))
    p.add_argument("--models_dir", type=str, default=str(ROOT / "out" / "models" / "eipm_sweeps"))
    p.add_argument("--methods", type=str, default="eipm,stabilized_gps,cbgps,independence_weights")
    p.add_argument("--scenarios", type=str, default="linear,nonlinear")
    p.add_argument("--clip_max", type=float, default=1e4)
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
    p.add_argument("--external_weights_dir", type=str, default="")
    p.add_argument("--no_clean", action="store_true")
    return p.parse_args()


def _cfg_key(scenario: str, n_train: int, d_x: int, tk: int, ok: int, pi0: float) -> Tuple:
    return (scenario, int(n_train), int(d_x), int(tk), int(ok), float(pi0))


def _factor_label_n(v: int) -> str:
    return str(int(v))


def _factor_label_dims(v: Tuple[int, int, int]) -> str:
    return f"({int(v[0])},{int(v[1])},{int(v[2])})"


def _factor_label_pi0(v: float) -> str:
    return f"{float(v):.1f}"


def _to_row(dataset_id: str, scenario: str, sweep_type: str, factor_value: str, smoothing: str, m: MetricSummary) -> Dict:
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


def main() -> None:
    args = parse_args()
    methods_raw = normalize_method_list(args.methods)
    methods, skipped_methods = filter_supported_methods(methods_raw)
    for m, reason in skipped_methods.items():
        print(f"[SKIP] method={m}: {reason}")
    scenarios = [s.strip().lower() for s in str(args.scenarios).split(",") if s.strip()]
    if not scenarios:
        raise ValueError("At least one scenario must be provided in --scenarios.")
    for s in scenarios:
        if s not in {"linear", "nonlinear"}:
            raise ValueError(f"Unsupported scenario '{s}'. Use linear/nonlinear.")

    results_dir = Path(args.results_dir)
    if not args.no_clean:
        clean_results_dir(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    baseline = {
        "n_train": 1000,
        "d_x": 50,
        "tk": 5,
        "ok": 5,
        "pi0": 0.5,
    }

    n_train_list = [250, 500, 1000, 2000]
    dims_list = [(5, 5, 5), (50, 5, 5), (50, 5, 50), (50, 50, 5), (50, 50, 50)]
    pi0_list = [0.0, 0.2, 0.5, 0.8]

    cache: Dict[Tuple, Tuple[str, Dict[int, Dict[str, MetricSummary]]]] = {}

    def get_metrics(scenario: str, n_train: int, d_x: int, tk: int, ok: int, pi0: float):
        key = _cfg_key(scenario, n_train, d_x, tk, ok, pi0)
        if key in cache:
            return cache[key]

        ds_name = generate_dgp_dataset(
            code_dir=ROOT,
            scenario=scenario,
            d_x=d_x,
            n_train=n_train,
            n_eval=int(args.n_eval),
            n_rpt=int(args.n_rpt),
            pi_0=float(pi0),
            seed=int(args.seed),
            treatment_k=tk,
            outcome_k=ok,
            beta_t0_scale=float(args.beta_T0_scale),
            beta_t_scale=float(args.beta_T_scale),
        )
        train_eipm_for_dataset(
            code_dir=ROOT,
            data_dir=data_dir,
            dataset_name=ds_name,
            models_dir=models_dir,
            n_rpt=int(args.n_rpt),
            device=str(args.device),
            max_steps=int(args.max_steps),
            k_folds=int(args.k_folds),
            eval_every=int(args.eval_every),
            nn=float(args.nn),
            overwrite=int(args.overwrite),
        )

        by_degree = evaluate_sim_config_all_methods(
            npz_path=data_dir / f"{ds_name}.npz",
            ckpt_root=models_dir,
            n_rpt=int(args.n_rpt),
            methods=methods,
            degrees=[0, 1],
            seed=int(args.seed),
            clip_max=float(args.clip_max),
            external_weights_dir=(Path(args.external_weights_dir) if str(args.external_weights_dir).strip() else None),
        )

        cache[key] = (ds_name, by_degree)
        return ds_name, by_degree

    rows_by_method: Dict[str, List[Dict]] = {m: [] for m in methods}

    for scenario in scenarios:
        for ntr in n_train_list:
            ds_name, by_degree = get_metrics(scenario, ntr, baseline["d_x"], baseline["tk"], baseline["ok"], baseline["pi0"])
            fv = _factor_label_n(ntr)
            for degree, smoothing in [(0, "NW"), (1, "LL")]:
                for m in methods:
                    rows_by_method[m].append(_to_row(ds_name, scenario, "n", fv, smoothing, by_degree[degree][m]))

        for d_x, tk, ok in dims_list:
            ds_name, by_degree = get_metrics(scenario, baseline["n_train"], d_x, tk, ok, baseline["pi0"])
            fv = _factor_label_dims((d_x, tk, ok))
            for degree, smoothing in [(0, "NW"), (1, "LL")]:
                for m in methods:
                    rows_by_method[m].append(_to_row(ds_name, scenario, "dims", fv, smoothing, by_degree[degree][m]))

        for pi0 in pi0_list:
            ds_name, by_degree = get_metrics(scenario, baseline["n_train"], baseline["d_x"], baseline["tk"], baseline["ok"], float(pi0))
            fv = _factor_label_pi0(float(pi0))
            for degree, smoothing in [(0, "NW"), (1, "LL")]:
                for m in methods:
                    rows_by_method[m].append(_to_row(ds_name, scenario, "pi0", fv, smoothing, by_degree[degree][m]))

    for m in methods:
        out_csv = results_csv_path(results_dir, m)
        upsert_results_rows(out_csv, rows_by_method[m], key_fields=["dataset_id", "scenario", "sweep_type", "factor_value", "smoothing"])
        print(f"[DONE] wrote {out_csv}")


if __name__ == "__main__":
    main()
