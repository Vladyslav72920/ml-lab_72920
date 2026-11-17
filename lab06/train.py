import argparse

from pipeline import fit, load_config

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--name", default="main")
    p.add_argument("--resume", default=None)
    p.add_argument("--stop-after", type=int, default=None)
    a = p.parse_args()

    cfg = load_config(a.config, a.set)
    res = fit(cfg, run_name=a.name, resume=a.resume, stop_after=a.stop_after)
    if not res["interrupted"]:
        print(f"Done. best val acc={res['best_val_acc']:.4f}  test acc={res['test_acc']:.4f}")


if __name__ == "__main__":
    main()