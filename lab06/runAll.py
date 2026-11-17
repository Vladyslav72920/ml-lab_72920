import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

quick = "--quick" in sys.argv
py = sys.executable
sub = ["--set", "data.subset=5000"] if quick else []

steps = [
    ("Task 1: main training + MLflow", [py, "train.py", "--name", "main"] + sub),
    ("Task 2: profiling", [py, "profile_epoch.py"] + sub),
    ("Tasks 3-7: studies", [py, "studies.py", "all"] + (["--quick"] if quick else [])),
]


def main():
    status = {}
    for title, cmd in steps:
        print(f"\n{'=' * 70}\n{title}\n{'=' * 70}", flush=True)
        status[title] = subprocess.call(cmd, cwd=HERE) == 0
    print("\n" + "=" * 70)
    for title, ok in status.items():
        print(("OK     " if ok else "FAILED ") + title)
    print("\nResults: results/   Checkpoints: checkpoints/")
    print("MLflow UI: mlflow ui --backend-store-uri sqlite:///mlflow.db  -> http://localhost:5000")


if __name__ == "__main__":
    main()