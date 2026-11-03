"""Shared helpers: config loading, paths, logging, seeding."""
from __future__ import annotations

import json
import logging
import os
import platform
import random
from pathlib import Path

import numpy as np
import yaml

SRC_DIR = Path(__file__).resolve().parent


def load_config() -> dict:
    with open(SRC_DIR / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def results_dir(cfg: dict) -> Path:
    d = (SRC_DIR / cfg["results_dir"]).resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_logger(cfg: dict) -> logging.Logger:
    log = logging.getLogger("lab05")
    if not log.handlers:
        log.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
        fh = logging.FileHandler(results_dir(cfg) / "run.log", mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        log.addHandler(fh)
        log.addHandler(sh)
    return log


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def save_json(obj, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def env_info() -> dict:
    import torch
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cpu_threads": torch.get_num_threads(),
    }
