"""Task 5: train a 2 -> 8 -> 1 (ReLU hidden, sigmoid output) model with BCE on
make_moons using (a) our scalar Value engine and (b) PyTorch, starting from
identical weights, then compare wall-clock speed (median of 5 runs).
"""
from __future__ import annotations

import statistics
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_moons

from common import env_info, get_logger, load_config, results_dir, save_json, seed_everything
from engine import sum_values
from nn import MLP, bce

EPS = 1e-7


# ------------------------------------------------------------- our engine
def train_engine(X, y, cfg_m, seed):
    model = MLP(2, cfg_m["hidden"], seed=seed)
    params = model.parameters()
    n = len(X)
    losses = []
    for _ in range(cfg_m["epochs"]):
        model.zero_grad()
        loss = sum_values(bce(model(x), t, EPS) for x, t in zip(X, y)) * (1.0 / n)
        loss.backward()
        for p in params:
            p.data -= cfg_m["lr"] * p.grad
        losses.append(loss.data)
    return model, losses


# ---------------------------------------------------------------- PyTorch
class TorchMLP(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.l1 = torch.nn.Linear(2, hidden)
        self.l2 = torch.nn.Linear(hidden, 1)

    def forward(self, x):
        return torch.sigmoid(self.l2(torch.relu(self.l1(x))))


def torch_from_engine_weights(weights, hidden, dtype):
    W1, b1, W2, b2 = weights
    m = TorchMLP(hidden).to(dtype)
    with torch.no_grad():
        m.l1.weight.copy_(torch.tensor(W1, dtype=dtype))
        m.l1.bias.copy_(torch.tensor(b1, dtype=dtype))
        m.l2.weight.copy_(torch.tensor(W2, dtype=dtype))
        m.l2.bias.copy_(torch.tensor(b2, dtype=dtype))
    return m


def train_torch(X, y, cfg_m, init_weights, dtype=torch.float64):
    model = torch_from_engine_weights(init_weights, cfg_m["hidden"], dtype)
    Xt = torch.tensor(X, dtype=dtype)
    yt = torch.tensor(y, dtype=dtype).unsqueeze(1)
    opt = torch.optim.SGD(model.parameters(), lr=cfg_m["lr"])
    losses = []
    for _ in range(cfg_m["epochs"]):
        opt.zero_grad()
        p = model(Xt) * (1 - 2 * EPS) + EPS   # same clamp as the engine's bce()
        loss = -(yt * torch.log(p) + (1 - yt) * torch.log(1 - p)).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return model, losses


# ---------------------------------------------------------------- helpers
def engine_predict(model, X):
    W1, b1, W2, b2 = (np.array(w) for w in model.weights())
    h = np.maximum(X @ W1.T + b1, 0)
    return 1 / (1 + np.exp(-(h @ W2.T + b2))).ravel()


def time_runs(fn, runs):
    times = []
    out = None
    for _ in range(runs):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return times, out


def main():
    cfg = load_config()
    seed = cfg["seed"]
    seed_everything(seed)
    log = get_logger(cfg)
    out_dir = results_dir(cfg)
    m = cfg["moons"]

    X, y = make_moons(n_samples=m["n_samples"], noise=m["noise"], random_state=seed)
    X = X.astype(np.float64)
    Xl, yl = X.tolist(), y.astype(float).tolist()
    init = MLP(2, m["hidden"], seed=seed).weights()

    runs = m["timing_runs"]
    log.info("moons | training engine model, %d runs x %d epochs", runs, m["epochs"])
    t_eng, (eng_model, eng_losses) = time_runs(lambda: train_engine(Xl, yl, m, seed), runs)

    log.info("moons | training PyTorch model (float64 and float32)")
    torch.manual_seed(seed)
    train_torch(X, y, m, init)  # warm-up (allocator, kernels), not timed
    t_t64, (t_model, t_losses) = time_runs(lambda: train_torch(X, y, m, init, torch.float64), runs)
    t_t32, (_, t32_losses) = time_runs(lambda: train_torch(X, y, m, init, torch.float32), runs)

    eng_acc = float(((engine_predict(eng_model, X) > 0.5) == y).mean())
    with torch.no_grad():
        t_acc = float(((t_model(torch.tensor(X)).numpy().ravel() > 0.5) == y).mean())
    loss_gap = float(np.max(np.abs(np.array(eng_losses) - np.array(t_losses))))

    med_e, med_t64, med_t32 = (statistics.median(t) for t in (t_eng, t_t64, t_t32))
    result = {
        "data": {"n_samples": m["n_samples"], "noise": m["noise"], "seed": seed},
        "model": "2 -> 8 (ReLU) -> 1 (sigmoid), BCE, full-batch SGD",
        "lr": m["lr"], "epochs": m["epochs"], "runs": runs,
        "timings_s": {
            "engine": {"runs": t_eng, "median": med_e},
            "torch_float64": {"runs": t_t64, "median": med_t64},
            "torch_float32": {"runs": t_t32, "median": med_t32},
        },
        "speedup_torch64_vs_engine": med_e / med_t64,
        "speedup_torch32_vs_engine": med_e / med_t32,
        "per_epoch_ms": {"engine": 1000 * med_e / m["epochs"],
                         "torch_float64": 1000 * med_t64 / m["epochs"],
                         "torch_float32": 1000 * med_t32 / m["epochs"]},
        "final_loss": {"engine": eng_losses[-1], "torch_float64": t_losses[-1],
                       "torch_float32": t32_losses[-1]},
        "train_accuracy": {"engine": eng_acc, "torch_float64": t_acc},
        "max_abs_loss_curve_diff_engine_vs_torch64": loss_gap,
        "env": env_info(),
    }
    save_json(result, out_dir / "moons_timing.json")

    # ---- figure: loss curves + decision boundary of the engine model
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(eng_losses, label="Value engine", lw=2)
    ax[0].plot(t_losses, "--", label="PyTorch (float64)", lw=1.5)
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("BCE loss"); ax[0].set_title("Training loss (identical init)")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - .5, X[:, 0].max() + .5, 300),
                         np.linspace(X[:, 1].min() - .5, X[:, 1].max() + .5, 300))
    zz = engine_predict(eng_model, np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax[1].contourf(xx, yy, zz, levels=[0, .5, 1], alpha=0.25, colors=["#4C72B0", "#DD8452"])
    ax[1].scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=14, edgecolors="k", linewidths=0.3)
    ax[1].set_title(f"Engine decision boundary (train acc {eng_acc:.1%})")
    fig.tight_layout()
    fig.savefig(out_dir / "moons_training.png", dpi=130)
    plt.close(fig)

    log.info("moons | median engine %.3fs, torch64 %.3fs, torch32 %.3fs -> torch is %.1fx faster; "
             "acc engine %.3f torch %.3f; max loss-curve diff %.2e",
             med_e, med_t64, med_t32, med_e / med_t64, eng_acc, t_acc, loss_gap)
    return result


if __name__ == "__main__":
    main()
