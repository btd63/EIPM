from pathlib import Path
import glob


def main() -> None:
    data_dir = Path(__file__).resolve().parent / "datasets"
    files = sorted(glob.glob(str(data_dir / "sim_*nonlinear*.npz")))
    if not files:
        print("No nonlinear dataset found.")
        return
    print(Path(files[0]).resolve())


if __name__ == "__main__":
    main()
