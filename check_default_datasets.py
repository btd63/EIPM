from __future__ import annotations

from pathlib import Path
import glob


def main() -> None:
    data_dir = Path("./datasets")

    # evaluate_eipm.py default behavior
    pattern = "sim_*nonlinear*.npz"
    files = sorted(glob.glob(str(data_dir / pattern)))
    eipm = files[0] if files else None

    # evaluate_huling.py default behavior
    huling_default = data_dir / "sim_nonlinear_dx50_ntr1000_nev10000_rpt100_tk50_ok50_pi0.0_seed42.npz"
    if huling_default.exists():
        huling = str(huling_default)
    else:
        files2 = sorted(glob.glob(str(data_dir / "sim_*linear*.npz")))
        huling = files2[0] if files2 else None

    print("evaluate_eipm default:", eipm)
    print("evaluate_huling default:", huling)


if __name__ == "__main__":
    main()
