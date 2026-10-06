"""Print Python version and the installed version of every package in requirements.txt."""
import platform
import re
import sys
from importlib import metadata
from pathlib import Path

REQ = Path(__file__).resolve().parents[2] / "requirements.txt"


def main() -> None:
    print(f"Python {sys.version}")
    print(f"Platform: {platform.platform()}")
    print("-" * 40)
    for line in REQ.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = re.split(r"[=<>!~]", line)[0].strip()
        try:
            print(f"{name}=={metadata.version(name)}")
        except metadata.PackageNotFoundError:
            print(f"{name}: NOT INSTALLED")


if __name__ == "__main__":
    main()
