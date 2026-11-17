"""Task 2: profile one training epoch with torch.profiler and report time shares."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile, record_function

from pipeline import build_model, build_optimizer, get_device, load_config, make_loaders, set_seed

p = argparse.ArgumentParser()
p.add_argument("--config", default="config.yaml")
p.add_argument("--set", action="append", default=[])
a = p.parse_args()

cfg = load_config(a.config, a.set)
set_seed(cfg["seed"])
device = get_device(cfg)
cuda = device.type == "cuda"
train_loader, _, _, _ = make_loaders(cfg)
model = build_model(cfg).to(device)
opt = build_optimizer(model, cfg)
crit = nn.CrossEntropyLoss()
model.train()


def sync():
    if cuda:
        torch.cuda.synchronize()


acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])
with profile(activities=acts) as prof:
    it = iter(train_loader)
    while True:
        with record_function("data_loading"):
            try:
                x, y = next(it)
            except StopIteration:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            sync()
        with record_function("forward"):
            out = model(x)
            loss = crit(out, y)
            sync()
        with record_function("backward"):
            opt.zero_grad(set_to_none=True)
            loss.backward()
            sync()
        with record_function("optimizer_step"):
            opt.step()
            sync()

names = ["data_loading", "forward", "backward", "optimizer_step"]
times = {e.key: e.cpu_time_total / 1e6 for e in prof.key_averages() if e.key in names}  # seconds
total = sum(times.values())
df = pd.DataFrame({"stage": names,
                   "time_s": [times[n] for n in names],
                   "share_pct": [100 * times[n] / total for n in names]})
out = Path("results"); out.mkdir(exist_ok=True)
df.to_csv(out / "profile_breakdown.csv", index=False)
(out / "profile_top_ops.txt").write_text(
    prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
print(df.to_string(index=False))

plt.figure(figsize=(5, 4))
plt.bar(df.stage, df.share_pct)
plt.ylabel("% of epoch time"); plt.title("Profile of one epoch")
plt.xticks(rotation=20); plt.tight_layout(); plt.savefig(out / "profile_breakdown.png", dpi=120)
