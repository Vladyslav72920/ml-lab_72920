import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data import X_tr, Y_tr, X_te, y_te
from nn import init_params, forward, backward, loss_fn, accuracy

WIDTHS = [16, 64]
LRS = [0.001, 0.01, 0.1]


def train(W, lr, epochs=50, bs=32, seed=0):
    p = init_params(W, seed=seed)
    r = np.random.default_rng(seed)
    hist = []
    for ep in range(epochs):
        idx = r.permutation(len(X_tr))                 
        for s in range(0, len(idx), bs):
            b = idx[s:s + bs]
            _, c = forward(X_tr[b], p)
            g = backward(Y_tr[b], p, c)
            for k in p:
                p[k] -= lr * g[k]
        L = loss_fn(forward(X_tr, p)[0], Y_tr)
        hist.append(L)
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"W={W:2d} lr={lr:<5} epoch {ep + 1:2d}/{epochs} train loss={L:.4f}")
    return p, hist, accuracy(p, X_te, y_te)


if __name__ == "__main__":
    results = {}
    for W in WIDTHS:
        for lr in LRS:
            _, hist, acc = train(W, lr)
            results[(W, lr)] = (hist, acc)
            print(f">>> W={W} lr={lr} final test accuracy = {acc:.4f}\n")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, W in zip(axes, WIDTHS):
        for lr in LRS:
            h, a = results[(W, lr)]
            ax.plot(range(1, 51), h, label=f"lr={lr} (test acc={a:.3f})")
        ax.set_title(f"Hidden width W={W}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train loss")
        ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "loss_curves.png")
    plt.savefig(path, dpi=150)
    print("saved", path)

    print("\n accuracy table ")
    print(f"{'W':>4} | " + " | ".join(f"lr={lr:<6}" for lr in LRS))
    for W in WIDTHS:
        print(f"{W:>4} | " + " | ".join(f"{results[(W, lr)][1]:<9.4f}" for lr in LRS))