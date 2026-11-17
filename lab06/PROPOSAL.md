# Individual Project Proposal

**Student:** Vladyslav Ryzhyk **ID:** 72920 **Course:** ML Systems, Lab 6

## Title

Fitting a 100-class image classifier into a CPU-only edge budget: knowledge distillation, structured pruning and INT8 quantization, alone and combined

## 1. Problem

A strong CIFAR-100 model (ResNet-18, ~11 M parameters, ~45 MB) is far too large for an offline, CPU-only device. Shrinking the network naively loses a lot of accuracy. The project asks: **which combination and order of distillation, pruning and quantization gives the best accuracy under a hard 50 ms / 5 MB budget, and how much does each technique contribute?**

## 2. Dataset

**CIFAR-100**: 50,000 training and 10,000 test images (32×32, 100 fine-grained classes). A fixed 5,000-image validation split is taken from the training set (seeded) and used for all model selection. The test set is used only for final models. It is much harder than CIFAR-10, so the cost of compression is clearly visible.

## 3. Models

- **Teacher:** ResNet-18 (CIFAR variant), trained from scratch; accuracy reference, never deployed.
- **Student:** the same architecture at 0.5× width (~2.8 M parameters, ~11 MB in FP32), so it does **not** meet the size limit without compression. Standard convolutions are used (no depthwise), because they quantize reliably.

## 4. Success metrics and deployment constraints

| Metric                            | Target                                                                                                                     |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Top-1 test accuracy (final model) | **≥ 60 %** (stretch goal 65 %), mean ± std over 3 seeds                                                                    |
| Latency                           | **≤ 50 ms** per image, batch 1, CPU, 1 thread, median of 200 runs after warm-up (p95 also reported); internal goal ≤ 10 ms |
| Model size                        | **≤ 5 MB** serialized (internal goal ≤ 2 MB)                                                                               |

## 5. Experiment plan

| ID  | Model    | What is added                                                                                                |
| --- | -------- | ------------------------------------------------------------------------------------------------------------ |
| A   | Teacher  | reference accuracy, size, latency                                                                            |
| B   | Student  | trained from scratch (baseline)                                                                              |
| C   | Student  | **distillation** from teacher (temperature T ∈ {2, 4}, loss weight α ∈ {0.5, 0.9})                           |
| D   | C        | **structured channel pruning** (30 / 50 / 70 %) + fine-tuning with distillation                              |
| E   | D        | **INT8 quantization**: post-training with calibration set, quantization-aware training if the drop is > 1 pp |
| F   | Ablation | each technique alone (KD only, pruning only, INT8 only) versus all three combined                            |

All runs use the YAML + MLflow pipeline from this lab (seeded, checkpointed, every run logged with parameters, accuracy, size, latency, parameter count and FLOPs). Latency is measured in PyTorch and again after ONNX export to check that conclusions do not depend on the runtime.

## 6. Deliverables

Accuracy–latency and accuracy–size Pareto plots, the ablation table, the final model that meets all constraints, and a short report stating which technique mattered most and why.

## 7. Risks and mitigation

- **Teacher training takes too long:** cap at ~40 epochs with mixed precision; fall back to a narrower teacher.
- **INT8 loses too much accuracy:** use quantization-aware training and per-channel weight quantization.
- **Over-aggressive pruning:** choose sparsity on the validation set from the 30 / 50 / 70 % sweep.
- **Laptop latency differs from the target device:** report hardware and thread count, and also report speed-up relative to the FP32 student.
