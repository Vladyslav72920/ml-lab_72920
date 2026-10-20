import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchinfo import summary
import thop

from dataset import get_dataloaders
from models import MLP, CNN, AttentionNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def measure_latency(model, batch_size, runs, warmup=5):
    model.eval()
    dummy = torch.randn(batch_size, 1, 28, 28).to(device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(dummy)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
    return float(np.median(times))

def train_model(model, train_loader, epochs=5, lr=1e-3):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
    return model

def evaluate_accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            preds = model(X).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(out_dir, exist_ok=True)

    print("Loading Fashion-MNIST dataset...")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=128)

    models_dict = {
        "MLP": MLP(),
        "CNN": CNN(),
        "Attention": AttentionNet()
    }

    metrics = []

    for name, model in models_dict.items():
        print(f"\nProfiling {name}...")
        model.to(device)

        info = summary(model, input_size=(1, 1, 28, 28), verbose=0)
        params_count = info.total_params

        dummy = torch.randn(1, 1, 28, 28).to(device)
        macs, _ = thop.profile(model, inputs=(dummy,), verbose=False)
        thop_flops = macs * 2

        lat_b1 = measure_latency(model, batch_size=1, runs=100)
        lat_b128 = measure_latency(model, batch_size=128, runs=20)

        print(f"Training {name} (5 epochs, Adam, lr=1e-3)...")
        train_model(model, train_loader, epochs=5, lr=1e-3)
        acc = evaluate_accuracy(model, test_loader)
        print(f"{name} test accuracy: {acc * 100:.2f}%")

        metrics.append({
            "model": name,
            "params": params_count,
            "thop_flops": thop_flops,
            "latency_b1_ms": round(lat_b1, 3),
            "latency_b128_ms": round(lat_b128, 3),
            "test_accuracy": round(acc, 4)
        })

    df = pd.DataFrame(metrics)
    csv_path = os.path.join(out_dir, "summary_metrics.csv")
    df.to_csv(csv_path, index=False)
    print("\nSummary metrics:")
    print(df.to_string(index=False))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for _, row in df.iterrows():
        ax1.scatter(row["params"], row["test_accuracy"] * 100, s=120, label=row["model"])
        ax1.annotate(row["model"], (row["params"], row["test_accuracy"] * 100 + 0.2))
    ax1.set_title("Accuracy vs Parameters")
    ax1.set_xlabel("Number of Parameters")
    ax1.set_ylabel("Test Accuracy (%)")
    ax1.grid(True, linestyle="--", alpha=0.6)

    for _, row in df.iterrows():
        ax2.scatter(row["latency_b128_ms"], row["test_accuracy"] * 100, s=120, label=row["model"])
        ax2.annotate(row["model"], (row["latency_b128_ms"] + 0.05, row["test_accuracy"] * 100))
    ax2.set_title("Accuracy vs Latency (Batch=128)")
    ax2.set_xlabel("Inference Latency (ms)")
    ax2.set_ylabel("Test Accuracy (%)")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plot_path = os.path.join(out_dir, "trade_off_plots.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    print(f"Artifacts saved to {out_dir}")