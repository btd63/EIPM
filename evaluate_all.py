#!/usr/bin/env python3
"""
Run evaluation for EIPM v1~v5, equal-weight baseline, and Huling weights.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import re


def _maybe_add(args: List[str], flag: str, value) -> None:
    if value is None:
        return
    args.extend([flag, str(value)])


def _run_capture(cmd: List[str]) -> Tuple[int, str, str]:
    print("[RUN]", " ".join(cmd))
    proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    out = proc.stdout or ""
    err = proc.stderr or ""
    if out:
        print(out, end="")
    if err:
        print(err, end="", file=sys.stderr)
    return proc.returncode, out, err


def _parse_summary(text: str) -> Dict[str, Optional[float]]:
    # Matches: [SUMMARY] MSE_mean=... MAE_mean=... (optional BIAS_mean=...)
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="./datasets_huling")
    p.add_argument("--pattern", type=str, required=True, help="npz filename")
    p.add_argument("--huling_dir", type=str, default="./models/huling")
    p.add_argument("--equal_weight_ckpt_dir", type=str, default="./models/eipm_single_v1")
    p.add_argument("--equal_weight_out_dir", type=str, default="./models/equal_weight")
    p.add_argument("--max_reps", type=int, default=None)
    p.add_argument("--only_rep", type=int, default=None)
    p.add_argument("--top_k_errors", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    py = sys.executable
    pattern = args.pattern

    rows = []

    # Huling weights evaluation
    cmd = [py, "evaluate_huling.py", "--data_dir", args.data_dir, "--npz", pattern, "--huling_dir", args.huling_dir]
    _maybe_add(cmd, "--max_reps", args.max_reps)
    _maybe_add(cmd, "--only_rep", args.only_rep)
    rc, out, err = _run_capture(cmd)
    summ = _parse_summary(out)
    rows.append({"method": "huling", "status": "ok" if rc == 0 else "error", **summ})

    # Equal-weight baseline (uses EIPM evaluation pipeline)
    cmd = [
        py,
        "evaluate_equal_weight.py",
        "--data_dir",
        args.data_dir,
        "--ckpt_dir",
        args.equal_weight_ckpt_dir,
        "--out_dir",
        args.equal_weight_out_dir,
        "--pattern",
        pattern,
    ]
    _maybe_add(cmd, "--max_reps", args.max_reps)
    _maybe_add(cmd, "--only_rep", args.only_rep)
    _maybe_add(cmd, "--top_k_errors", args.top_k_errors)
    rc, out, err = _run_capture(cmd)
    summ = _parse_summary(out)
    rows.append({"method": "unweighted", "status": "ok" if rc == 0 else "error", **summ})

    # EIPM v1~v5
    for v in [1, 2, 3, 4, 5]:
        ckpt_dir = f"./models/eipm_single_v{v}"
        cmd = [
            py,
            f"evaluate_eipm_v{v}.py",
            "--data_dir",
            args.data_dir,
            "--ckpt_dir",
            ckpt_dir,
            "--pattern",
            pattern,
        ]
        _maybe_add(cmd, "--max_reps", args.max_reps)
        _maybe_add(cmd, "--only_rep", args.only_rep)
        _maybe_add(cmd, "--top_k_errors", args.top_k_errors)
        rc, out, err = _run_capture(cmd)
        summ = _parse_summary(out)
        rows.append({"method": f"eipm_v{v}", "status": "ok" if rc == 0 else "error", **summ})

    # Print as R data.frame
    methods = [r["method"] for r in rows]
    mse = [r["MSE_mean"] for r in rows]
    mae = [r["MAE_mean"] for r in rows]
    bias = [r["BIAS_mean"] for r in rows]
    status = [r["status"] for r in rows]

    def fmt_list(vals):
        out = []
        for v in vals:
            if v is None:
                out.append("NA")
            else:
                out.append(f"{v:.6g}")
        return ", ".join(out)

    print("\n[DATA.FRAME]")
    print("results <- data.frame(")
    print(f"  method = c({', '.join([repr(m) for m in methods])}),")
    print(f"  MSE_mean = c({fmt_list(mse)}),")
    print(f"  MAE_mean = c({fmt_list(mae)}),")
    print(f"  BIAS_mean = c({fmt_list(bias)}),")
    print(f"  status = c({', '.join([repr(s) for s in status])})")
    print(")")


if __name__ == "__main__":
    main()
