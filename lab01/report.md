# Lab 1: Environment and first system measurements

## 1. Goal

The main objective of this lab is to establish a fully reproducible Python environment with pinned dependencies and train baseline classification models (Logistic Regression and Random Forest) on the Breast Cancer Wisconsin dataset[cite: 5, 6]. Following rigorous measurement protocols, empirical benchmarks are collected for training duration, single-sample inference latency, storage footprint, and peak RAM consumption[cite: 4, 6]. Finally, these measured system costs are evaluated against hardware budget profiles to determine the deployment feasibility of each model across Cloud, Edge, Mobile, and TinyML tiers.

## 2. Method
- Environment: Python 3.11.8, venv, pinned `requirements.txt` (versions in `results/versions.txt`).
- Data: Breast Cancer Wisconsin, 70/30 stratified split, `random_state=42`.
- Models: LogisticRegression(max_iter=1000), RandomForest(n_estimators=100), seed 42.
- Training time: 1 warm-up + median of 5 runs.
- Inference: single sample, 1 warm-up + 100 repeats, median.
- Model size: `joblib.dump`, bytes and KB.
- Peak memory: `memory_profiler` (peak RSS) during training and during inference.
- Budgets: Cloud / Edge / Mobile / TinyML as given in the manual.

## 3. Results

### Accuracy and system cost
| Model | Test acc. | Train time (ms, median) | Inference (ms, median) | Size (bytes) | Size (KB) | Peak RSS train (MB) | Peak RSS infer (MB) |
|---|---|---|---|---|---|---|---|
| LogisticRegression | 0.9415 | 133.362 | 0.0469 | 1103 | 1.08 | 312.9 | 312.9 |
| RandomForest | 0.9357 | 128.104 | 5.8295 | 290873 | 284.06 | 314.1 | 314.6 |

### Deployment fit
| Model | Cloud | Edge | Mobile | TinyML |
|---|---|---|---|---|
| LogisticRegression | PASS | PASS | FAIL | FAIL |
| RandomForest | PASS | PASS | FAIL | FAIL |

Short justification per model (use your numbers; note that process RSS includes the Python + sklearn baseline, so discuss memory carefully):
- **LogisticRegression:** Fits Cloud and Edge tiers without any violations[cite: 6]. It satisfies latency (0.0469 ms vs. 20 ms max) and storage (1.08 KB vs. 10 MB max) for Mobile, but fails the tier because overall process RSS (~313 MB) exceeds the 256 MB ceiling due to baseline Python and scikit-learn runtime overhead[cite: 6]. It fails TinyML because memory consumption (~313 MB) vastly exceeds the strict 256 KB constraint[cite: 6].
- **RandomForest:** Fits Cloud and Edge tiers[cite: 6]. It fails Mobile because peak process RSS (~315 MB) exceeds 256 MB[cite: 6]. It fails TinyML due to both an extreme memory budget violation (~315 MB vs. 256 KB max) and model artifact size overflow (284.06 KB vs. 100 KB max)[cite: 6].

## 4. Conclusions (own words)
1. Both models deliver comparable test accuracy (~94%), but Logistic Regression achieves an inference latency over 120 times lower (0.047 ms vs. 5.83 ms) and an artifact size roughly 260 times smaller than Random Forest.
2. In a standard CPython/scikit-learn runtime, the base process memory footprint (~313–315 MB) dominates resource usage, which causes both models to fail deployment budgets for Mobile (≤256 MB) and TinyML (≤256 KB) despite meeting latency targets[cite: 6].
3. For resource- and latency-sensitive deployments, Logistic Regression is clearly preferable, whereas Random Forest is practical only in Cloud or Edge environments where memory and storage limits are sufficiently relaxed[cite: 6].