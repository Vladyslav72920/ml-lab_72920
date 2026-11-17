import argparse
import copy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pipeline import fit, load_config, set_path

OUT = Path("results")
OUT.mkdir(exist_ok=True)


def run(base, name, over, epochs, **kw):
    cfg = copy.deepcopy(base)
    for k, v in over.items():
        set_path(cfg, k, v)
    cfg["epochs"] = epochs
    cfg["ckpt_dir"] = f"checkpoints/{name}"
    return fit(cfg, run_name=name, **kw), cfg


def save_table(df, name):
    df.to_csv(OUT / f"{name}.csv", index=False)
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.4g}" if isinstance(v, float) else str(v) for v in r) + " |")
    (OUT / f"{name}.md").write_text("\n".join(lines))
    print("\n" + "\n".join(lines) + "\n")


def curves(results, key, title, path):
    plt.figure(figsize=(6, 4))
    for name, r in results.items():
        h = pd.DataFrame(r["history"])
        plt.plot(h.epoch, h[key], marker="o", label=name)
    plt.xlabel("epoch"); plt.ylabel(key); plt.title(title); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


def study_workers(base):
    rows = []
    for nw in [0, 1, 2, 4, 8]:
        r, _ = run(base, f"workers_{nw}", {"data.num_workers": nw}, epochs=2)
        h = r["history"][-1]
        rows.append({"num_workers": nw, "samples_per_s": h["samples_per_s"],
                     "epoch_time_s": h["epoch_time_s"], "data_wait_pct": 100 * h["data_wait_frac"]})
    df = pd.DataFrame(rows)
    save_table(df, "workers_study")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(df.num_workers, df.samples_per_s, marker="o"); ax[0].set_title("Throughput (samples/s)")
    ax[1].plot(df.num_workers, df.data_wait_pct, marker="o", color="C3")
    ax[1].set_title("% of epoch spent waiting for data")
    for a in ax:
        a.set_xlabel("num_workers"); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "workers_study.png", dpi=120)
    best = df.loc[df.samples_per_s.idxmax()]
    verdict = ("data-loading bound" if df.data_wait_pct.iloc[0] > 30 else "compute bound")
    print(f"Bottleneck at num_workers=0: {verdict} (data wait {df.data_wait_pct.iloc[0]:.0f}%). "
          f"Best throughput at num_workers={int(best.num_workers)}.")


def study_batch(base):
    rows = []
    for bs in [32, 64, 128, 256, 512]:
        r, _ = run(base, f"batch_{bs}", {"data.batch_size": bs}, epochs=5)
        h = r["history"]
        rows.append({"batch_size": bs,
                     "samples_per_s": sum(x["samples_per_s"] for x in h) / len(h),
                     "peak_mem_mb": max(x["peak_mem_mb"] for x in h),
                     "final_val_acc": h[-1]["val_acc"], "test_acc": r["test_acc"]})
    df = pd.DataFrame(rows)
    save_table(df, "batch_study")
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for a, col in zip(ax, ["samples_per_s", "peak_mem_mb", "final_val_acc"]):
        a.plot(df.batch_size, df[col], marker="o"); a.set_xscale("log", base=2)
        a.set_title(col); a.set_xlabel("batch size"); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "batch_study.png", dpi=120)


def study_sched(base):
    specs = {
        "StepLR (SGD)": {"optim.name": "sgd", "optim.lr": 0.05, "sched.name": "step"},
        "Cosine (SGD)": {"optim.name": "sgd", "optim.lr": 0.05, "sched.name": "cosine"},
        "Plateau (SGD)": {"optim.name": "sgd", "optim.lr": 0.05, "sched.name": "plateau"},
        "Cosine (Adam)": {"optim.name": "adam", "optim.lr": 0.001, "sched.name": "cosine"},
    }
    res, rows = {}, []
    for label, over in specs.items():
        r, _ = run(base, "sched_" + label.split()[0].lower() + "_" + over["optim.name"], over, epochs=10)
        res[label] = r
        rows.append({"run": label, "final_val_acc": r["history"][-1]["val_acc"],
                     "best_val_acc": r["best_val_acc"], "test_acc": r["test_acc"],
                     "final_train_loss": r["history"][-1]["train_loss"]})
    save_table(pd.DataFrame(rows), "sched_optim_study")
    curves({k: v for k, v in res.items() if "SGD" in k}, "val_acc", "LR schedules (SGD+momentum)",
           OUT / "sched_study.png")
    curves({k: v for k, v in res.items() if "Cosine" in k}, "val_acc", "SGD+momentum vs Adam (cosine, 10 epochs)",
           OUT / "optim_study.png")


def study_accum(base):
    specs = {
        "bs256, accum 1": {"data.batch_size": 256, "accum_steps": 1},
        "bs64, accum 4": {"data.batch_size": 64, "accum_steps": 4},
        "bs64, accum 1 (reference)": {"data.batch_size": 64, "accum_steps": 1},
    }
    res, rows = {}, []
    for label, over in specs.items():
        r, _ = run(base, "accum_" + label.split(",")[0] + label.split(",")[1].split()[1], over, epochs=10)
        res[label] = r
        rows.append({"run": label, "final_train_loss": r["history"][-1]["train_loss"],
                     "final_val_acc": r["history"][-1]["val_acc"], "test_acc": r["test_acc"]})
    save_table(pd.DataFrame(rows), "accum_study")
    curves(res, "train_loss", "Gradient accumulation: train loss", OUT / "accum_train_loss.png")
    curves(res, "val_acc", "Gradient accumulation: val accuracy", OUT / "accum_val_acc.png")


def study_resume(base):
    full, cfg = run(base, "resume_baseline", {}, epochs=6)
    cfg = copy.deepcopy(base)
    cfg["epochs"], cfg["ckpt_dir"] = 6, "checkpoints/resume_interrupted"
    fit(cfg, run_name="resume_interrupted", stop_after=3)
    resumed = fit(cfg, run_name="resume_interrupted",
                  resume="checkpoints/resume_interrupted/last.pt")
    res = {"uninterrupted": full, "interrupted@3 + resumed": resumed}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, r in res.items():
        h = pd.DataFrame(r["history"])
        ax[0].plot(h.epoch, h.train_loss, marker="o", label=name)
        ax[1].plot(h.epoch, h.val_acc, marker="o", label=name)
    for a, t in zip(ax, ["train loss", "val acc"]):
        a.axvline(3, ls="--", color="gray"); a.set_title(t); a.set_xlabel("epoch"); a.legend()
    fig.tight_layout(); fig.savefig(OUT / "resume_curves.png", dpi=120)
    a_, b_ = [pd.DataFrame(r["history"]) for r in res.values()]
    print(f"Max |train_loss diff| after epoch 3: {(a_.train_loss - b_.train_loss).abs()[3:].max():.4f}")


STUDIES = {"workers": study_workers, "batch": study_batch, "sched": study_sched,
           "accum": study_accum, "resume": study_resume}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("which", choices=list(STUDIES) + ["all"])
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    base = load_config(a.config)
    if a.quick:
        base["data"]["subset"] = 5000
    for name in (STUDIES if a.which == "all" else [a.which]):
        STUDIES[name](base)