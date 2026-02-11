#!/usr/bin/env python3
"""
Evaluate all available datasets across methods and print/save matrices.

Rows: datasets (short names)
Cols: methods (huling, unweighted, eipm_v1..v5)
Cells: MSE/MAE/BIAS (three separate tables)
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


METHODS = ["huling", "unweighted", "eipm_v1", "eipm_v2", "eipm_v3", "eipm_v4", "eipm_v5"]


def _parse_summary(text: str) -> Dict[str, Optional[float]]:
    pat = re.compile(
        r"\[SUMMARY\]\s+MSE_mean=([0-9.eE+\-]+)\s+MAE_mean=([0-9.eE+\-]+)"
        r"(?:\s+BIAS_mean=([0-9.eE+\-]+))?"
    )
    matches = pat.findall(text)
    if not matches:
        return {"MSE_mean": None, "MAE_mean": None, "BIAS_mean": None}
    mse, mae, bias = matches[-1]
    return {
        "MSE_mean": float(mse),
        "MAE_mean": float(mae),
        "BIAS_mean": float(bias) if bias != "" else None,
    }


def _run_capture(cmd: List[str]) -> Tuple[int, str, str]:
    proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _short_int(x: str) -> str:
    n = int(x)
    if n % 1000 == 0 and n >= 1000:
        return f"{n//1000}k"
    return str(n)


def _short_float(x: str) -> str:
    if x.endswith(".0"):
        return x[:-2]
    return x


def short_name(stem: str) -> str:
    # Huling logA
    m = re.match(
        r"sim_huling_logA_ntr(\d+)_nev(\d+)_rpt(\d+)_mx1([-\d.]+)_mx2([-\d.]+)_mx3([-\d.]+)"
        r"_ae(\d+)_sig([-\d.]+)_off([-\d.]+)(?:_seed\d+)?(?:_eval(\w+))?",
        stem,
    )
    if m:
        ntr, nev, rpt, mx1, mx2, mx3, ae, sig, off, evalm = m.groups()
        name = f"HlogA_n{_short_int(ntr)}_e{_short_int(nev)}_ae{ae}_s{_short_float(sig)}_o{_short_float(off)}"
        # only append mx if non-default
        if not (mx1 == "-0.5" and mx2 == "1.0" and mx3 == "0.3"):
            name += f"_mx{_short_float(mx1)}_{_short_float(mx2)}_{_short_float(mx3)}"
        if evalm:
            name += f"_e{evalm[:4]}"
        return name

    # Huling base
    m = re.match(
        r"sim_huling_ntr(\d+)_nev(\d+)_rpt(\d+)_mx1([-\d.]+)_mx2([-\d.]+)_mx3([-\d.]+)"
        r"_ae(\d+)(?:_seed\d+)?(?:_eval(\w+))?",
        stem,
    )
    if m:
        ntr, nev, rpt, mx1, mx2, mx3, ae, evalm = m.groups()
        name = f"H_n{_short_int(ntr)}_e{_short_int(nev)}_ae{ae}"
        if not (mx1 == "-0.5" and mx2 == "1.0" and mx3 == "0.3"):
            name += f"_mx{_short_float(mx1)}_{_short_float(mx2)}_{_short_float(mx3)}"
        if evalm:
            name += f"_e{evalm[:4]}"
        return name

    # linear/nonlinear
    m = re.match(
        r"sim_(linear|nonlinear)_dx(\d+)_ntr(\d+)_nev(\d+)_rpt(\d+)_tk(\d+)_ok(\d+)_pi([0-9.]+)_seed\d+",
        stem,
    )
    if m:
        kind, dx, ntr, nev, rpt, tk, ok, pi = m.groups()
        k = "L" if kind == "linear" else "NL"
        return f"{k}_d{dx}_n{_short_int(ntr)}_e{_short_int(nev)}_tk{tk}_ok{ok}_p{_short_float(pi)}"

    # fallback
    s = stem
    if s.startswith("sim_"):
        s = s[4:]
    s = re.sub(r"_seed\d+", "", s)
    return s


def list_dataset_stems(model_root: Path) -> List[str]:
    stems = set()
    for v in [1, 2, 3, 4, 5]:
        base = model_root / f"eipm_single_v{v}"
        if not base.exists():
            continue
        for p in base.iterdir():
            if p.is_dir():
                stems.add(p.name)
    return sorted(stems)


def find_data_dir(stem: str, datasets_dir: Path, huling_dir: Path) -> Path:
    if stem.startswith("sim_huling") or stem.startswith("sim_huling_logA"):
        return huling_dir
    return datasets_dir


def find_equal_ckpt_dir(stem: str, model_root: Path) -> Optional[Path]:
    for v in [1, 2, 3, 4, 5]:
        base = model_root / f"eipm_single_v{v}" / stem
        if base.exists():
            return model_root / f"eipm_single_v{v}"
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets_dir", type=str, default="./datasets")
    p.add_argument("--datasets_huling_dir", type=str, default="./datasets_huling")
    p.add_argument("--model_root", type=str, default="./models")
    p.add_argument("--huling_dir", type=str, default="./models/huling")
    p.add_argument("--out_dir", type=str, default="./results")
    p.add_argument("--only_rep", type=int, default=None)
    p.add_argument("--max_reps", type=int, default=None)
    p.add_argument("--limit", type=int, default=None, help="Limit number of datasets")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    datasets_dir = Path(args.datasets_dir)
    huling_dir = Path(args.datasets_huling_dir)
    model_root = Path(args.model_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stems = list_dataset_stems(model_root)
    if args.limit is not None:
        stems = stems[: int(args.limit)]
    if not stems:
        print("No dataset stems found under models/eipm_single_v*")
        sys.exit(1)

    # Matrices: metric -> {row_key: {method: value}}
    results: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {
        "MSE_mean": {},
        "MAE_mean": {},
        "BIAS_mean": {},
    }

    py = sys.executable

    for stem in stems:
        data_dir = find_data_dir(stem, datasets_dir, huling_dir)
        npz = f"{stem}.npz"
        row_key = short_name(stem)
        for metric in results:
            results[metric].setdefault(row_key, {})

        # Huling
        cmd = [py, "evaluate_huling.py", "--data_dir", str(data_dir), "--npz", npz, "--huling_dir", str(args.huling_dir)]
        if args.only_rep is not None:
            cmd += ["--only_rep", str(args.only_rep)]
        if args.max_reps is not None:
            cmd += ["--max_reps", str(args.max_reps)]
        rc, out, err = _run_capture(cmd)
        summ = _parse_summary(out)
        for metric in results:
            results[metric][row_key]["huling"] = summ[metric]

        # Unweighted (Huling-style)
        eq_ckpt_dir = find_equal_ckpt_dir(stem, model_root)
        if eq_ckpt_dir is None:
            for metric in results:
                results[metric][row_key]["unweighted"] = None
        else:
            cmd = [
                py,
                "evaluate_equal_weight.py",
                "--data_dir",
                str(data_dir),
                "--ckpt_dir",
                str(eq_ckpt_dir),
                "--out_dir",
                str(model_root / "equal_weight"),
                "--pattern",
                npz,
            ]
            if args.only_rep is not None:
                cmd += ["--only_rep", str(args.only_rep)]
            if args.max_reps is not None:
                cmd += ["--max_reps", str(args.max_reps)]
            rc, out, err = _run_capture(cmd)
            summ = _parse_summary(out)
            for metric in results:
                results[metric][row_key]["unweighted"] = summ[metric]

        # EIPM v1~v5
        for v in [1, 2, 3, 4, 5]:
            method = f"eipm_v{v}"
            ckpt_dir = model_root / f"eipm_single_v{v}"
            cmd = [
                py,
                f"evaluate_eipm_v{v}.py",
                "--data_dir",
                str(data_dir),
                "--ckpt_dir",
                str(ckpt_dir),
                "--pattern",
                npz,
            ]
            if args.only_rep is not None:
                cmd += ["--only_rep", str(args.only_rep)]
            if args.max_reps is not None:
                cmd += ["--max_reps", str(args.max_reps)]
            rc, out, err = _run_capture(cmd)
            summ = _parse_summary(out)
            for metric in results:
                results[metric][row_key][method] = summ[metric]

    # Print and save tables
    def fmt(v: Optional[float]) -> str:
        return "NA" if v is None else f"{v:.6g}"

    for metric, table in results.items():
        print(f"\n[{metric}]")
        # header
        header = ["dataset"] + METHODS
        rows = []
        for row_key in sorted(table.keys()):
            row = [row_key] + [fmt(table[row_key].get(m)) for m in METHODS]
            rows.append(row)

        # print as Markdown table
        print("| " + " | ".join(header) + " |")
        print("|" + "|".join(["---"] * len(header)) + "|")
        for row in rows:
            print("| " + " | ".join(row) + " |")

        # save CSV
        csv_path = out_dir / f"matrix_{metric}.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

        # save R data.frame
        r_path = out_dir / f"matrix_{metric}.R"
        r_lines = ["df <- data.frame("]
        r_lines.append(f"  dataset = c({', '.join([repr(r[0]) for r in rows])}),")
        for i, m in enumerate(METHODS):
            vals = []
            for r in rows:
                v = r[i + 1]
                vals.append("NA" if v == "NA" else v)
            r_lines.append(f"  {m} = c({', '.join(vals)}),")
        r_lines.append("  stringsAsFactors = FALSE")
        r_lines.append(")")
        r_path.write_text("\n".join(r_lines) + "\n")

        print(f"[SAVED] {csv_path}")
        print(f"[SAVED] {r_path}")


if __name__ == "__main__":
    main()
