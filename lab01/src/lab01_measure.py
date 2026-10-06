import os
import random
import statistics
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
import torch
from memory_profiler import memory_usage
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

N_TRAIN_RUNS = 5
N_INFER_RUNS = 100

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
sample = X_test[:1]

MODELS = {
    "LogisticRegression": lambda: LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
}


def peak_mem_mb(fn) -> float:
    peak = memory_usage((fn, (), {}), interval=0.005, max_usage=True)
    if isinstance(peak, (list, tuple)):
        peak = max(peak)
    return float(peak)


def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


if __name__ == '__main__':
    acc_rows, meas_rows = [], []
    for name, make in MODELS.items():
        print(f"== {name} ==")

        model = make().fit(X_train, y_train)
        acc = round(float(model.score(X_test, y_test)), 4)
        acc_rows.append({"model": name, "test_accuracy": f"{acc:.4f}"})

        make().fit(X_train, y_train)
        times = []
        for _ in range(N_TRAIN_RUNS):
            m = make()
            t0 = time.perf_counter()
            m.fit(X_train, y_train)
            times.append(time.perf_counter() - t0)
        train_ms = statistics.median(times) * 1000

        base_train = rss_mb()
        peak_train = peak_mem_mb(lambda: make().fit(X_train, y_train))

        model.predict(sample)
        lat = []
        for _ in range(N_INFER_RUNS):
            t0 = time.perf_counter()
            model.predict(sample)
            lat.append(time.perf_counter() - t0)
        infer_ms = statistics.median(lat) * 1000

        base_inf = rss_mb()
        peak_inf = peak_mem_mb(lambda: [model.predict(sample) for _ in range(N_INFER_RUNS)])

        path = RESULTS / f"model_{name}.joblib"
        joblib.dump(model, path)
        size_bytes = path.stat().st_size

        meas_rows.append({
            "model": name,
            "test_accuracy": f"{acc:.4f}",
            "train_time_ms_median": round(train_ms, 3),
            "infer_latency_ms_median": round(infer_ms, 4),
            "model_size_bytes": size_bytes,
            "model_size_KB": round(size_bytes / 1024, 2),
            "peak_rss_train_MB": round(peak_train, 1),
            "peak_rss_infer_MB": round(peak_inf, 1),
            "rss_before_train_MB": round(base_train, 1),
            "rss_before_infer_MB": round(base_inf, 1),
        })
        print(meas_rows[-1])

    pd.DataFrame(acc_rows).to_csv(RESULTS / "baseline_accuracy.csv", index=False)
    meas = pd.DataFrame(meas_rows)
    meas.to_csv(RESULTS / "measurements.csv", index=False)

    BUDGETS = {
        "Cloud":  {"mem_max_MB": float("inf"), "lat_ms": 100, "size_KB": 500 * 1024},
        "Edge":   {"mem_max_MB": 1024,         "lat_ms": 50,  "size_KB": 50 * 1024},
        "Mobile": {"mem_max_MB": 256,          "lat_ms": 20,  "size_KB": 10 * 1024},
        "TinyML": {"mem_max_MB": 256 / 1024,   "lat_ms": 10,  "size_KB": 100},
    }

    rows = []
    for _, r in meas.iterrows():
        peak_mem = max(r["peak_rss_train_MB"], r["peak_rss_infer_MB"])
        for tier, b in BUDGETS.items():
            ok_mem = peak_mem <= b["mem_max_MB"]
            ok_lat = r["infer_latency_ms_median"] <= b["lat_ms"]
            ok_size = r["model_size_KB"] <= b["size_KB"]
            rows.append({
                "model": r["model"], "tier": tier,
                "memory_ok": ok_mem, "latency_ok": ok_lat, "size_ok": ok_size,
                "fits": ok_mem and ok_lat and ok_size,
            })
    fit = pd.DataFrame(rows)
    fit.to_csv(RESULTS / "budget_fit.csv", index=False)
    print(fit.to_string(index=False))