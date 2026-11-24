import os
import copy
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from thop import profile

from models import OversizedCNN, StudentCNN, build_structured_pruned_model

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_loaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    train_ds = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            preds = model(X).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

def measure_cpu_latency_ms(model, runs=300):
    cpu_model = copy.deepcopy(model).cpu().eval()
    dummy = torch.randn(1, 1, 28, 28)
    with torch.no_grad():
        for _ in range(50):
            _ = cpu_model(dummy)
        t0 = time.perf_counter()
        for _ in range(runs):
            _ = cpu_model(dummy)
        elapsed = time.perf_counter() - t0
    return (elapsed / runs) * 1000.0

def get_model_stats(model, path):
    torch.save(model.state_dict(), path)
    size_mb = os.path.getsize(path) / (1024 ** 2)
    params = sum(p.numel() for p in model.parameters())
    cpu_m = copy.deepcopy(model).cpu().eval()
    macs, _ = profile(cpu_m, inputs=(torch.randn(1, 1, 28, 28),), verbose=False)
    flops_m = (macs * 2) / 1e6
    lat_ms = measure_cpu_latency_ms(model)
    return params, size_mb, flops_m, lat_ms

def apply_unstructured_mask(model, sparsity, mode="global"):
    masks = {}
    target_modules = [m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    if sparsity == 0.0:
        for m in target_modules:
            masks[m] = torch.ones_like(m.weight.data)
        return masks

    if mode == "global":
        all_weights = torch.cat([m.weight.data.abs().view(-1) for m in target_modules])
        threshold = torch.quantile(all_weights, sparsity)
        for m in target_modules:
            mask = (m.weight.data.abs() >= threshold).float()
            m.weight.data.mul_(mask)
            masks[m] = mask
    elif mode == "layerwise":
        for m in target_modules:
            w_abs = m.weight.data.abs().view(-1)
            threshold = torch.quantile(w_abs, sparsity)
            mask = (m.weight.data.abs() >= threshold).float()
            m.weight.data.mul_(mask)
            masks[m] = mask
    return masks

def train_epochs(model, train_loader, device, epochs=3, lr=1e-3, masks=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            if masks is not None:
                for m, mask in masks.items():
                    if m.weight.grad is not None:
                        m.weight.grad.mul_(mask)
            optimizer.step()
            if masks is not None:
                with torch.no_grad():
                    for m, mask in masks.items():
                        m.weight.data.mul_(mask)

def train_distillation(student, teacher, train_loader, device, epochs=6, lr=1e-3, T=4.0, alpha=0.5):
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    teacher.eval()
    for _ in range(epochs):
        student.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            s_logits = student(X)
            with torch.no_grad():
                t_logits = teacher(X)
            loss_ce = F.cross_entropy(s_logits, y)
            loss_kl = F.kl_div(
                F.log_softmax(s_logits / T, dim=1),
                F.softmax(t_logits / T, dim=1),
                reduction="batchmean"
            ) * (T * T)
            loss = alpha * loss_ce + (1.0 - alpha) * loss_kl
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    set_seed(42)
    out_dir = "lab07/results"
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_loaders(batch_size=128)

    print("Training Oversized Baseline CNN")
    teacher = OversizedCNN().to(device)
    train_epochs(teacher, train_loader, device, epochs=6, lr=1e-3)
    t_acc = evaluate(teacher, test_loader, device)
    t_params, t_size, t_flops, t_lat = get_model_stats(teacher, os.path.join(out_dir, "teacher.pt"))
    print(f"Teacher | Params: {t_params:,} | Size: {t_size:.2f} MB | FLOPs: {t_flops:.2f}M | Latency: {t_lat:.2f} ms | Acc: {t_acc*100:.2f}%")

    print("\nUnstructured Magnitude Pruning")
    sparsities = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95]
    unstruct_records = []
    model_90_global = None

    for sp in sparsities:
        for mode in ["global", "layerwise"]:
            m = copy.deepcopy(teacher).to(device)
            masks = apply_unstructured_mask(m, sp, mode=mode)
            if sp > 0.0:
                train_epochs(m, train_loader, device, epochs=3, lr=3e-4, masks=masks)
            acc = evaluate(m, test_loader, device)
            lat = measure_cpu_latency_ms(m)
            unstruct_records.append({
                "mode": mode,
                "sparsity": sp,
                "accuracy": round(acc, 4),
                "latency_ms": round(lat, 2)
            })
            print(f"Unstructured ({mode}, sp={sp:.2f}) | Acc: {acc*100:.2f}% | Latency: {lat:.2f} ms")
            if mode == "global" and sp == 0.9:
                model_90_global = copy.deepcopy(m).cpu()

    df_unstruct = pd.DataFrame(unstruct_records)
    df_unstruct.to_csv(os.path.join(out_dir, "unstructured_pruning.csv"), index=False)

    print("\n Dense vs Sparse Disk Storage (90% Global sparsity)")
    dense_path = os.path.join(out_dir, "pruned90_dense.pt")
    sparse_path = os.path.join(out_dir, "pruned90_sparse.pt")
    torch.save(model_90_global.state_dict(), dense_path)
    sparse_sd = {
        k: (v.to_sparse_coo() if v.ndim >= 2 else v)
        for k, v in model_90_global.state_dict().items()
    }
    torch.save(sparse_sd, sparse_path)
    dense_mb = os.path.getsize(dense_path) / (1024 ** 2)
    sparse_mb = os.path.getsize(sparse_path) / (1024 ** 2)
    print(f"90% Pruned Storage | Dense: {dense_mb:.2f} MB vs Sparse COO: {sparse_mb:.2f} MB")

    print("\n\Structured Filter Pruning (L2 Norm: 10%, 25%, 50%)")
    struct_records = [{"prune_ratio": 0.0, "params": t_params, "size_mb": round(t_size, 2), "flops_m": round(t_flops, 2), "latency_ms": round(t_lat, 2), "accuracy": round(t_acc, 4)}]
    model_struct_50 = None
    for pr in [0.10, 0.25, 0.50]:
        sm = build_structured_pruned_model(teacher.cpu(), keep_ratio=1.0 - pr).to(device)
        train_epochs(sm, train_loader, device, epochs=3, lr=3e-4)
        s_acc = evaluate(sm, test_loader, device)
        s_params, s_size, s_flops, s_lat = get_model_stats(sm, os.path.join(out_dir, f"struct_{int(pr*100)}.pt"))
        struct_records.append({
            "prune_ratio": pr,
            "params": s_params,
            "size_mb": round(s_size, 2),
            "flops_m": round(s_flops, 2),
            "latency_ms": round(s_lat, 2),
            "accuracy": round(s_acc, 4)
        })
        print(f"Structured (prune={int(pr*100)}%) | Params: {s_params:,} | FLOPs: {s_flops:.2f}M | Latency: {s_lat:.2f} ms | Acc: {s_acc*100:.2f}%")
        if pr == 0.50:
            model_struct_50 = copy.deepcopy(sm)

    df_struct = pd.DataFrame(struct_records)
    df_struct.to_csv(os.path.join(out_dir, "structured_pruning.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for mode in ["global", "layerwise"]:
        sub = df_unstruct[df_unstruct["mode"] == mode]
        axes[0].plot(sub["sparsity"] * 100, sub["accuracy"] * 100, marker="o", label=mode)
    axes[0].set_title("Unstructured Magnitude Pruning")
    axes[0].set_xlabel("Sparsity (%)")
    axes[0].set_ylabel("Test Accuracy (%)")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(df_struct["prune_ratio"] * 100, df_struct["accuracy"] * 100, marker="s", color="crimson", label="L2 Filter Pruning")
    axes[1].set_title("Structured Filter Pruning")
    axes[1].set_xlabel("Pruned Filters (%)")
    axes[1].set_ylabel("Test Accuracy (%)")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pruning_curves.png"), dpi=150)

    print("\nKnowledge Distillation vs Plain Student (~100k params)")
    set_seed(42)
    plain_student = StudentCNN().to(device)
    train_epochs(plain_student, train_loader, device, epochs=6, lr=1e-3)
    ps_acc = evaluate(plain_student, test_loader, device)
    ps_params, ps_size, ps_flops, ps_lat = get_model_stats(plain_student, os.path.join(out_dir, "plain_student.pt"))
    print(f"Plain Student | Params: {ps_params:,} | Size: {ps_size:.2f} MB | FLOPs: {ps_flops:.2f}M | Latency: {ps_lat:.2f} ms | Acc: {ps_acc*100:.2f}%")

    set_seed(42)
    distilled_student = StudentCNN().to(device)
    train_distillation(distilled_student, teacher.to(device), train_loader, device, epochs=6, lr=1e-3, T=4.0, alpha=0.5)
    ds_acc = evaluate(distilled_student, test_loader, device)
    ds_params, ds_size, ds_flops, ds_lat = get_model_stats(distilled_student, os.path.join(out_dir, "distilled_student.pt"))
    print(f"Distilled Student | Params: {ds_params:,} | Size: {ds_size:.2f} MB | FLOPs: {ds_flops:.2f}M | Latency: {ds_lat:.2f} ms | Acc: {ds_acc*100:.2f}%")

    summary = pd.DataFrame([
        {"Model": "Oversized Baseline", "Params": t_params, "Dense_MB": round(t_size, 2), "Sparse_MB": round(t_size, 2), "FLOPs_M": round(t_flops, 2), "Latency_ms": round(t_lat, 2), "Accuracy": round(t_acc, 4)},
        {"Model": "Unstructured Pruned (90%)", "Params": t_params, "Dense_MB": round(dense_mb, 2), "Sparse_MB": round(sparse_mb, 2), "FLOPs_M": round(t_flops, 2), "Latency_ms": round(measure_cpu_latency_ms(model_90_global), 2), "Accuracy": df_unstruct[(df_unstruct["mode"]=="global") & (df_unstruct["sparsity"]==0.9)]["accuracy"].values[0]},
        {"Model": "Structured Pruned (50%)", "Params": struct_records[-1]["params"], "Dense_MB": struct_records[-1]["size_mb"], "Sparse_MB": struct_records[-1]["size_mb"], "FLOPs_M": struct_records[-1]["flops_m"], "Latency_ms": struct_records[-1]["latency_ms"], "Accuracy": struct_records[-1]["accuracy"]},
        {"Model": "Plain Student (~100k)", "Params": ps_params, "Dense_MB": round(ps_size, 2), "Sparse_MB": round(ps_size, 2), "FLOPs_M": round(ps_flops, 2), "Latency_ms": round(ps_lat, 2), "Accuracy": round(ps_acc, 4)},
        {"Model": "Distilled Student (T=4)", "Params": ds_params, "Dense_MB": round(ds_size, 2), "Sparse_MB": round(ds_size, 2), "FLOPs_M": round(ds_flops, 2), "Latency_ms": round(ds_lat, 2), "Accuracy": round(ds_acc, 4)},
    ])
    summary.to_csv(os.path.join(out_dir, "ch9_summary.csv"), index=False)
    print("\nFinal Summary Saved to lab07/results/ch9_summary.csv")
