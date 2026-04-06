from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from core.benchmark_core import METHOD_DISPLAY_NAME
from core.table_helpers import (
    MetricSummary,
    ensure_tables_dir,
    metric_cell,
    read_results_csv,
    write_csv,
    write_nmes_table_tex_dynamic,
    write_sim_table_tex_dynamic,
)

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build tables from results_*.csv files only.")
    p.add_argument("--results_dir", type=str, default=str(ROOT / "results"))
    return p.parse_args()


def _f(x) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def _metric_from_row(r: Dict) -> MetricSummary:
    return MetricSummary(
        rmse_mean=_f(r.get("rmse_mean", np.nan)),
        rmse_se=_f(r.get("rmse_se", np.nan)),
        mae_mean=_f(r.get("mae_mean", np.nan)),
        mae_se=_f(r.get("mae_se", np.nan)),
    )


def _pseudo_from_row(r: Dict) -> Tuple[float, float]:
    pr = _f(r.get("pseudo_rmse", np.nan))
    pa = _f(r.get("pseudo_mae", np.nan))
    if not np.isfinite(pr):
        pr = _f(r.get("rmse_mean", np.nan))
    if not np.isfinite(pa):
        pa = _f(r.get("mae_mean", np.nan))
    return float(pr), float(pa)


def _index_rows(rows: List[Dict]) -> Dict[Tuple[str, str, str, str], Dict]:
    out: Dict[Tuple[str, str, str, str], Dict] = {}
    for r in rows:
        key = (
            str(r.get("scenario", "")),
            str(r.get("sweep_type", "")),
            str(r.get("factor_value", "")),
            str(r.get("smoothing", "")),
        )
        out[key] = r
    return out


def _display_name(method: str) -> str:
    m = str(method).lower()
    if m == "eipm":
        return "EIPM"
    if m == "unweighted":
        return "Unweighted"
    return METHOD_DISPLAY_NAME.get(m, m)


def _discover_methods(results_dir: Path) -> List[str]:
    files = sorted(results_dir.glob("results_*.csv"))
    if not files:
        raise FileNotFoundError("No results_*.csv files found. Run run_sweep_tables.py/run_nmes_table.py first.")
    discovered = []
    for p in files:
        name = p.name
        if not name.startswith("results_") or not name.endswith(".csv"):
            continue
        discovered.append(name[len("results_") : -len(".csv")].strip().lower())
    discovered = [m for m in discovered if m]
    preferred = [
        "eipm",
        "stabilized_gps",
        "cbgps",
        "independence_weights",
        "unweighted",
    ]
    ordered = [m for m in preferred if m in discovered]
    extras = sorted([m for m in discovered if m not in set(ordered)])
    return ordered + extras


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    tables_dir = ensure_tables_dir(results_dir)

    methods = _discover_methods(results_dir)
    rows_by_method: Dict[str, List[Dict]] = {}
    idx_by_method: Dict[str, Dict[Tuple[str, str, str, str], Dict]] = {}
    for m in methods:
        rows = read_results_csv(results_dir / f"results_{m}.csv")
        rows_by_method[m] = rows
        idx_by_method[m] = _index_rows(rows)

    factor_n = ["250", "500", "1000", "2000"]
    factor_dims = ["(5,5,5)", "(50,5,5)", "(50,5,50)", "(50,50,5)", "(50,50,50)"]
    factor_pi0 = ["0.0", "0.2", "0.5", "0.8"]

    table_specs = [
        ("linear", "n", "n_train", factor_n, "table_linear_n.tex"),
        ("nonlinear", "n", "n_train", factor_n, "table_nonlinear_n.tex"),
        ("linear", "dims", "(d_X, |S_T|, |S_Y|)", factor_dims, "table_linear_dims.tex"),
        ("nonlinear", "dims", "(d_X, |S_T|, |S_Y|)", factor_dims, "table_nonlinear_dims.tex"),
        ("linear", "pi0", "pi_0", factor_pi0, "table_linear_pi0.tex"),
        ("nonlinear", "pi0", "pi_0", factor_pi0, "table_nonlinear_pi0.tex"),
    ]

    written: List[Path] = []

    for scenario, sweep, factor_header, factors, tex_name in table_specs:
        tex_path = tables_dir / tex_name
        csv_path = tables_dir / tex_name.replace(".tex", ".csv")

        tex_rows: List[Dict] = []
        csv_header: List[str] = ["method_smoothing"]
        for fv in factors:
            fkey = str(fv).replace(" ", "")
            csv_header += [
                f"{fkey}_rmse_mean",
                f"{fkey}_rmse_se",
                f"{fkey}_mae_mean",
                f"{fkey}_mae_se",
                f"{fkey}_cell",
            ]

        csv_rows: List[List] = []
        for m in methods:
            dname = _display_name(m)
            for smoothing in ["NW", "LL"]:
                row_label = f"{dname}_{smoothing}"
                cells: List[str] = []
                row_vals: List = [row_label]
                for fv in factors:
                    met = _metric_from_row(idx_by_method[m].get((scenario, sweep, fv, smoothing), {}))
                    c = metric_cell(met)
                    cells.append(c)
                    row_vals += [met.rmse_mean, met.rmse_se, met.mae_mean, met.mae_se, c]

                tex_rows.append({"factor": row_label, "cells": cells})
                csv_rows.append(row_vals)

        write_sim_table_tex_dynamic(tex_path, "Method", factors, tex_rows)
        write_csv(csv_path, header=csv_header, rows=csv_rows)
        written.append(tex_path)

    nmes_tex = tables_dir / "table_nmes.tex"
    nmes_csv = tables_dir / "table_nmes.csv"
    method_rows = []
    nmes_csv_rows = []
    for m in methods:
        nw = _pseudo_from_row(idx_by_method[m].get(("nmes", "nmes", "nmes", "NW"), {}))
        ll = _pseudo_from_row(idx_by_method[m].get(("nmes", "nmes", "nmes", "LL"), {}))
        method_rows.append((_display_name(m), nw, ll))
        nmes_csv_rows.append([_display_name(m), nw[0], nw[1], ll[0], ll[1]])

    write_nmes_table_tex_dynamic(nmes_tex, method_rows)
    write_csv(
        nmes_csv,
        header=["method", "nw_pseudo_rmse", "nw_pseudo_mae", "ll_pseudo_rmse", "ll_pseudo_mae"],
        rows=nmes_csv_rows,
    )
    written.append(nmes_tex)

    print("[DONE] tables written from results CSV:")
    for p in written:
        print(f" - {p}")


if __name__ == "__main__":
    main()
