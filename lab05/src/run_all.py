from __future__ import annotations

import time

import gradcheck
import torch_inspect
import train_moons
from common import env_info, get_logger, load_config, results_dir, save_json, seed_everything


def main():
    cfg = load_config()
    out = results_dir(cfg)
    (out / "run.log").write_text("", encoding="utf-8")  # fresh log per full run
    seed_everything(cfg["seed"])
    log = get_logger(cfg)
    log.info("Lab 5 run started | env %s", env_info())

    t0 = time.perf_counter()
    g = gradcheck.main()
    m = train_moons.main()
    torch_inspect.main()
    log.info("Lab 5 run finished in %.1fs | gradcheck %s", time.perf_counter() - t0,
             "PASS" if g["passed"] else "FAIL")
    save_json({"gradcheck_passed": g["passed"],
               "moons_speedup_torch64_vs_engine": m["speedup_torch64_vs_engine"]},
              out / "summary.json")


if __name__ == "__main__":
    main()
