#!/usr/bin/env python3
"""
Verify that repXXX in an NPZ matches repXXX.csv in a given CSV folder.
Default: rep001, dataset sim_nonlinear_dx50_ntr1000_nev10000_rpt100_tk50_ok50_pi0.0_seed42.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="./datasets")
    p.add_argument(
        "--npz",
        type=str,
        default="sim_nonlinear_dx50_ntr1000_nev10000_rpt100_tk50_ok50_pi0.0_seed42.npz",
    )
    p.add_argument(
        "--csv_dir",
        type=str,
        default="./datasets/sim_nonlinear_dx50_ntr1000_nev10000_rpt100_tk50_ok50_pi0.0_seed42",
    )
    p.add_argument("--rep", type=int, default=1)
    p.add_argument("--atol", type=float, default=1e-8)
    p.add_argument("--rtol", type=float, default=1e-6)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    npz_path = Path(args.data_dir) / args.npz
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")
    csv_dir = Path(args.csv_dir)
    if not csv_dir.exists():
        raise FileNotFoundError(f"CSV dir not found: {csv_dir}")

    rep = int(args.rep)
    csv_path = csv_dir / f"rep{rep:03d}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    data = np.load(npz_path, allow_pickle=True)
    X_npz = np.array(data["X_train"][rep], dtype=np.float64)
    T_npz = np.array(data["T_train"][rep], dtype=np.float64).reshape(-1, 1)
    Y_npz = np.array(data["Y_train"][rep], dtype=np.float64).reshape(-1, 1)
    npz_rows = np.hstack([X_npz, T_npz, Y_npz])

    csv = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    csv = np.array(csv, dtype=np.float64)

    if npz_rows.shape != csv.shape:
        print(f"[FAIL] shape mismatch: npz {npz_rows.shape} vs csv {csv.shape}")
        return

    diff = np.abs(npz_rows - csv)
    max_abs = float(np.max(diff))
    ok = np.allclose(npz_rows, csv, rtol=float(args.rtol), atol=float(args.atol))

    print(f"[INFO] rep={rep:03d} rows={npz_rows.shape[0]} cols={npz_rows.shape[1]}")
    print(f"[INFO] max_abs_diff={max_abs:.6g}")
    print(f"[RESULT] match={ok} (rtol={args.rtol}, atol={args.atol})")

    if not ok:
        # show first few mismatches
        idx = np.argwhere(diff > (float(args.atol) + float(args.rtol) * np.abs(npz_rows)))
        idx = idx[:10]
        print("[MISMATCH] first few (row,col,npz,csv,diff):")
        for r, c in idx:
            print(
                f"{int(r)},{int(c)},"
                f"{npz_rows[int(r), int(c)]:.6g},"
                f"{csv[int(r), int(c)]:.6g},"
                f"{diff[int(r), int(c)]:.6g}"
            )


if __name__ == "__main__":
    main()
