import functools
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import yaml
from torch.utils.data import DataLoader, Subset

MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


def set_path(cfg, dotted_key, value):
    d = cfg
    parts = dotted_key.split(".")
    for p in parts[:-1]:
        d = d[p]
    d[parts[-1]] = value


def load_config(path, overrides=None):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        key, val = item.split("=", 1)
        set_path(cfg, key, yaml.safe_load(val))
    return cfg


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(cfg):
    d = cfg.get("device", "auto")
    if d == "auto":
        d = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(d)


def _winit(worker_id, seed):
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def make_loaders(cfg):
    d = cfg["data"]
    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                          T.ToTensor(), T.Normalize(MEAN, STD)])
    eval_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    ds_train = torchvision.datasets.CIFAR10(d["root"], train=True, download=True, transform=train_tf)
    ds_val = torchvision.datasets.CIFAR10(d["root"], train=True, download=True, transform=eval_tf)
    ds_test = torchvision.datasets.CIFAR10(d["root"], train=False, download=True, transform=eval_tf)

    perm = np.random.RandomState(cfg["seed"]).permutation(len(ds_train)).tolist()
    val_idx, train_idx = perm[: d["val_size"]], perm[d["val_size"]:]
    if d.get("subset"):
        train_idx = train_idx[: d["subset"]]

    gen = torch.Generator()
    gen.manual_seed(cfg["seed"])

    nw = d["num_workers"]
    train_kw = dict(num_workers=nw, pin_memory=torch.cuda.is_available(), worker_init_fn=functools.partial(_winit, seed=cfg["seed"]))
    if nw > 0 and d.get("persistent_workers", True):
        train_kw["persistent_workers"] = True
    eval_kw = dict(num_workers=min(nw, 2), pin_memory=torch.cuda.is_available())

    train_loader = DataLoader(Subset(ds_train, train_idx), batch_size=d["batch_size"],
                              shuffle=True, drop_last=True, generator=gen, **train_kw)
    val_loader = DataLoader(Subset(ds_val, val_idx), batch_size=512, shuffle=False, **eval_kw)
    test_loader = DataLoader(ds_test, batch_size=512, shuffle=False, **eval_kw)
    return train_loader, val_loader, test_loader, gen


class SmallCNN(nn.Module):
    def __init__(self, num_classes=10, width=32):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        w = width
        self.features = nn.Sequential(
            block(3, w), block(w, w), nn.MaxPool2d(2),
            block(w, 2 * w), block(2 * w, 2 * w), nn.MaxPool2d(2),
            block(2 * w, 4 * w), block(4 * w, 4 * w), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.fc = nn.Linear(4 * w, num_classes)

    def forward(self, x):
        return self.fc(self.features(x))


def build_model(cfg):
    m = cfg["model"]
    return SmallCNN(m["num_classes"], m["width"])


def build_optimizer(model, cfg):
    o = cfg["optim"]
    if o["name"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=o["lr"], momentum=o.get("momentum", 0.9),
                               weight_decay=o.get("weight_decay", 0.0))
    if o["name"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=o["lr"], weight_decay=o.get("weight_decay", 0.0))
    raise ValueError(o["name"])


def build_scheduler(opt, cfg):
    s = cfg["sched"]
    if s["name"] == "none":
        return None
    if s["name"] == "step":
        return torch.optim.lr_scheduler.StepLR(opt, step_size=s["step_size"], gamma=s["gamma"])
    if s["name"] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    if s["name"] == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=s["factor"],
                                                          patience=s["patience"])
    raise ValueError(s["name"])


def train_one_epoch(model, loader, opt, device, accum_steps=1):
    model.train()
    crit = nn.CrossEntropyLoss()
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), device=device)
    n, data_time, nb = 0, 0.0, len(loader)
    opt.zero_grad(set_to_none=True)
    t0 = t_end = time.perf_counter()
    for i, (x, y) in enumerate(loader):
        data_time += time.perf_counter() - t_end
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        loss = crit(out, y)
        (loss / accum_steps).backward()
        if (i + 1) % accum_steps == 0 or i + 1 == nb:
            opt.step()
            opt.zero_grad(set_to_none=True)
        loss_sum += loss.detach() * x.size(0)
        correct += (out.argmax(1) == y).sum()
        n += x.size(0)
        if cuda:
            torch.cuda.synchronize()
        t_end = time.perf_counter()
    elapsed = t_end - t0
    return {
        "train_loss": loss_sum.item() / n,
        "train_acc": correct.item() / n,
        "epoch_time_s": elapsed,
        "samples_per_s": n / elapsed,
        "data_wait_frac": data_time / elapsed,
        "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20 if cuda else float("nan"),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="sum")
    loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss += crit(out, y).item()
        correct += (out.argmax(1) == y).sum().item()
        n += y.size(0)
    return loss / n, correct / n


def save_curves(history, path):
    df = pd.DataFrame(history)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].plot(df.epoch, df.train_loss, label="train"); ax[0].plot(df.epoch, df.val_loss, label="val")
    ax[0].set_title("Loss"); ax[0].legend()
    ax[1].plot(df.epoch, df.train_acc, label="train"); ax[1].plot(df.epoch, df.val_acc, label="val")
    ax[1].set_title("Accuracy"); ax[1].legend()
    ax[2].plot(df.epoch, df.lr); ax[2].set_title("Learning rate")
    for a in ax:
        a.set_xlabel("epoch")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def save_checkpoint(path, **state):
    tmp = Path(str(path) + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def fit(cfg, run_name=None, resume=None, stop_after=None):
    set_seed(cfg["seed"])
    device = get_device(cfg)
    train_loader, val_loader, test_loader, gen = make_loaders(cfg)
    model = build_model(cfg).to(device)
    opt = build_optimizer(model, cfg)
    sched = build_scheduler(opt, cfg)
    ckpt_dir = Path(cfg["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    history, start_epoch, best_acc, run_id = [], 0, 0.0, None
    if resume:
        ck = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        if sched is not None and ck["scheduler"] is not None:
            sched.load_state_dict(ck["scheduler"])
        history, start_epoch = ck["history"], ck["epoch"]
        best_acc, run_id = ck["best_acc"], ck["run_id"]
        gen.set_state(ck["gen_state"])
        torch.set_rng_state(ck["torch_rng"])
        if torch.cuda.is_available() and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        print(f"Resumed from {resume} at epoch {start_epoch}")

    n_params = sum(p.numel() for p in model.parameters())
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    run = mlflow.start_run(run_id=run_id) if run_id else mlflow.start_run(run_name=run_name)
    run_id = run.info.run_id
    result = {"run_id": run_id, "interrupted": False}
    try:
        if resume:
            mlflow.set_tag("resumed_from_epoch", start_epoch)
        else:
            mlflow.log_params(flatten(cfg))
            mlflow.log_metrics({"n_params": n_params, "model_size_mb_fp32": size_mb})

        for epoch in range(start_epoch, cfg["epochs"]):
            tr = train_one_epoch(model, train_loader, opt, device, cfg.get("accum_steps", 1))
            vl, va = evaluate(model, val_loader, device)
            lr = opt.param_groups[0]["lr"]
            if sched is not None:
                sched.step(vl) if cfg["sched"]["name"] == "plateau" else sched.step()
            rec = {"epoch": epoch + 1, "lr": lr, "val_loss": vl, "val_acc": va, **tr}
            history.append(rec)
            mlflow.log_metrics({k: v for k, v in rec.items()
                                if k != "epoch" and not np.isnan(v)}, step=epoch + 1)
            if va > best_acc:
                best_acc = va
                torch.save(model.state_dict(), ckpt_dir / "best.pt")
            save_checkpoint(ckpt_dir / "last.pt", model=model.state_dict(), optimizer=opt.state_dict(),
                            scheduler=sched.state_dict() if sched else None, epoch=epoch + 1,
                            history=history, best_acc=best_acc, run_id=run_id,
                            gen_state=gen.get_state(), torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
            print(f"[{run_name}] epoch {epoch+1}/{cfg['epochs']} lr={lr:.4g} "
                  f"train_loss={tr['train_loss']:.4f} val_loss={vl:.4f} val_acc={va:.4f} "
                  f"{tr['samples_per_s']:.0f} samples/s data_wait={tr['data_wait_frac']:.0%}")
            if stop_after and epoch + 1 >= stop_after and epoch + 1 < cfg["epochs"]:
                result["interrupted"] = True
                mlflow.set_tag("interrupted_at_epoch", epoch + 1)
                break

        if not result["interrupted"]:
            model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=device))
            _, test_acc = evaluate(model, test_loader, device)
            mlflow.log_metrics({"test_acc": test_acc, "best_val_acc": best_acc})
            result.update(test_acc=test_acc, best_val_acc=best_acc)
            pd.DataFrame(history).to_csv(ckpt_dir / "history.csv", index=False)
            save_curves(history, ckpt_dir / "curves.png")
            with open(ckpt_dir / "config.yaml", "w") as f:
                yaml.safe_dump(cfg, f)
            for name in ("config.yaml", "history.csv", "curves.png", "best.pt"):
                mlflow.log_artifact(str(ckpt_dir / name))
    finally:
        mlflow.end_run()
    result["history"] = history
    return result