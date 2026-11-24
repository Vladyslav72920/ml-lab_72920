# Individual Project Proposal: Real-Time Edge Driver Microsleep & Drowsiness Detection via Spatio-Temporal Knowledge Distillation

- **Student Name:** Vladyslav Ryzhyk
- **Student ID:** 72920

---

## 1. Problem Statement
Driver fatigue and microsleep episodes account for a substantial share of severe road accidents. Cloud-based video analytics are unviable in vehicles due to intermittent cellular connectivity, high round-trip latency, and strict biometric privacy regulations. The objective of this project is to design and compress an on-device spatio-temporal neural network capable of detecting early signs of driver drowsiness (eye closure dynamics, yawning, and head nodding) directly from low-power cabin infrared/RGB camera streams under strict real-time edge hardware constraints.

## 2. Dataset
- **Primary Dataset:** **NTHU-DDD (National Tsing Hua University Driver Drowsiness Detection)** and **UTA-RLDD (Real-Life Drowsiness Dataset)** — ~9.5 hours of annotated driving videos across 36 subjects under varied illumination (daylight, night IR, with/without glasses).
- **Preprocessing & Input Representation:** Face and eye/mouth Region-of-Interest (ROI) alignment via lightweight MediaPipe landmarks, sampled into short temporal clips of $T = 16$ frames at spatial resolution $112 \times 112 \times 3$ (2.5-second sliding window at ~6.4 FPS). Subject-independent cross-validation splits (train/val/test by distinct driver IDs) will be strictly enforced to prevent identity leakage.

## 3. Success Metric
- **Primary Metric:** **Subject-Independent Macro F1-Score $\ge 0.88$** and **Drowsy-Class Recall (Sensitivity) $\ge 0.92$** on unseen test subjects.
- **Secondary Metric:** **Area Under the Precision-Recall Curve (PR-AUC) $\ge 0.93$**, ensuring minimal false alarms during normal driving while preventing missed microsleep events.

## 4. Deployment Constraints
To operate continuously on an embedded automotive ARM Cortex-A76 CPU (single-core inference, batch size $= 1$, no discrete GPU):
- **End-to-End Inference Latency:** $\le 50\text{ ms}$ per 16-frame temporal clip ($\ge 20\text{ FPS}$ throughput).
- **Model Storage Footprint:** $\le 5\text{ MB}$ serialized artifact size on disk (including all weights and buffers).

## 5. Planned Optimization Pipeline
A raw spatio-temporal teacher model exceeds both the $5\text{ MB}$ storage limit and the $50\text{ ms}$ CPU latency budget. To meet all deployment constraints while preserving accuracy, a three-stage compression pipeline will be executed:
1. **Teacher-Student Knowledge Distillation:** Train a high-capacity **Video Swin-Tiny / ResNet-3D-18** teacher ($\approx 45\text{ MB}$) on the 16-frame clips, then distill both soft output logits and intermediate feature attention maps into a compact **MobileNetV3-Small + Temporal Shift Module (TSM)** student architecture ($\approx 6.2\text{ MB}$ in FP32).
2. **Structured Channel Pruning:** Apply $L_1$-norm structured filter pruning (30% sparsity target) across depthwise-separable convolutional blocks, followed by fine-tuning to recover temporal feature sensitivity and reduce FLOPs by ~35%.
3. **Quantization-Aware Training (QAT) to INT8:** Convert weights and activations from FP32 to **INT8** using PyTorch QAT with ONNX Runtime / XNNPACK backend execution, compressing the final model size to **$\approx 1.8\text{ MB}$** ($\le 5\text{ MB}$) and reducing single-thread CPU latency from ~115 ms down to **$\approx 28\text{ ms}$** ($\le 50\text{ ms}$).