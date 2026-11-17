# Individual Project Proposal

**Student:** Vladyslav Ryzhyk **ID:** 72920 **Course:** ML Systems, Lab 6

## Problem
Compact on-device image classifier: recognise 10 everyday object classes from a 32×32 colour image on a
CPU-only edge device, where a cloud round-trip is undesirable (privacy, offline use, latency).

## Dataset
CIFAR-10 (60 000 images, 10 classes; 50 000 train / 10 000 test). A fixed 5 000-image validation split is
taken from the training set (seeded), so the test set is touched only for the final report.

## Baseline
Small 6-conv-layer CNN with BatchNorm (≈0.3 M parameters, ≈1.2 MB in FP32), trained with the YAML/MLflow
pipeline from Lab 6 (SGD+momentum, cosine schedule, random-crop + flip augmentation).

## Success metric
- Top-1 test accuracy ≥ 88 % for the final optimised model, and ≤ 1 percentage point lower than the FP32 baseline.
- Reported as mean ± std over 3 seeds.

## Deployment constraints
- **Latency ≤ 50 ms** per image, batch size 1, CPU only; median of 200 timed runs after warm-up.
- **Model size ≤ 5 MB** on disk (serialised weights).

## Planned optimisation
Post-training static **INT8 quantization** (with a small calibration set), followed by quantization-aware
training if the accuracy drop exceeds 1 pp. If the latency budget is still tight, add structured
**pruning** of the widest conv layers and fine-tune. Each variant is logged in MLflow with accuracy,
size (MB) and latency (ms), and compared in one trade-off table.

## Risks
Accuracy loss from INT8 on a small network; latency measured on a laptop CPU may not match the target device.
Mitigation: report both FP32 and INT8, and keep the width multiplier configurable.
