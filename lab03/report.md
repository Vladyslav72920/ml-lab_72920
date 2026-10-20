# Lab 3: Architectures and their costs

## 1. Goal
The objective of this lab is to evaluate and compare three distinct deep learning architectures (MLP, CNN, and Attention/Transformer) of comparable parameter scale on the Fashion-MNIST dataset[cite: 23]. The assignment focuses on calculating parameter counts and FLOPs by hand, profiling representative execution costs via automated tools, measuring inference latency across different batch sizes, and explaining architectural trade-offs using memory access patterns[cite: 23].

## 2. Method
- **Dataset:** Fashion-MNIST loaded via `torchvision.datasets.FashionMNIST` with normalization `Normalize((0.2860,), (0.3530,))`[cite: 23]. The official 60k training dataset is partitioned into 55,000 training and 5,000 validation samples; the official 10,000 sample test set is used for final evaluation[cite: 23].
- **Architectures:**
  - **MLP:** Flatten 784 -> Linear(784, 150) -> ReLU -> Linear(150, 100) -> ReLU -> Linear(100, 10).
  - **CNN:** Conv2d(1, 32, 3, pad=1) -> ReLU -> MaxPool2d(2) -> Conv2d(32, 64, 3, pad=1) -> ReLU -> MaxPool2d(2) -> Flatten -> Linear(3136, 32) -> ReLU -> Linear(32, 10).
  - **Attention:** Image partitioned into 16 non-overlapping patches of 7x7, projected to dimension 64, processed by two Transformer encoder layers (d_model=64, nhead=4, dim_feedforward=128), followed by sequence mean pooling and a Linear(64, 10) classifier[cite: 23].
- **Cost Metrics & Profiling:** Hand calculation of parameters and FLOPs based on Chapter 4 formulas, automated verification via `torchinfo` and `thop.profile`, and latency benchmarking for batch sizes 1 (100 runs) and 128 (20 runs)[cite: 23].
- **Training:** Each model was trained for 5 epochs using the Adam optimizer (lr=1e-3, batch size 128)[cite: 23].

## 3. Results

### Hand Calculations vs Profiler

#### Representative Layer Calculations:
1. **Linear Layer (`Linear(784, 150)`):**
   - $\text{Params} = (784 \times 150) + 150 = 117,600 + 150 = 117,750$
   - $\text{FLOPs} = 2 \times 784 \times 150 = 235,200 \text{ FLOPs}$[cite: 23]
2. **Convolutional Layer (`Conv2d(1, 32, 3, padding=1)` on $28 \times 28$ input):**
   - $\text{Params} = (3 \times 3 \times 1 \times 32) + 32 = 288 + 32 = 320$
   - $\text{FLOPs} = 2 \times K \times K \times C_{in} \times C_{out} \times H_{out} \times W_{out} = 2 \times 3 \times 3 \times 1 \times 32 \times 28 \times 28 = 451,584 \text{ FLOPs}$[cite: 23]
3. **Attention Layer (Single Multi-Head Attention block, $seq=16, d=64$):**
   - $\text{QKV projections:} \ 6 \times seq \times d^2 = 6 \times 16 \times 4096 = 393,216 \text{ FLOPs}$[cite: 23]
   - $\text{Attention scores + context:} \ 4 \times seq^2 \times d = 4 \times 256 \times 64 = 65,536 \text{ FLOPs}$[cite: 23]
   - $\text{Out projection:} \ 2 \times seq \times d^2 = 2 \times 16 \times 4096 = 131,072 \text{ FLOPs}$
   - $\text{Total Attention FLOPs} = 589,824 \text{ FLOPs}$

#### Metric Summary Table:
| Model | Params (`torchinfo`) | FLOPs (`thop`) | Latency B=1 (ms) | Latency B=128 (ms) | Test Accuracy (%) |
|---|---|---|---|---|---|
| **MLP** | 133,860 | 267,200 | 0.053 | 0.196 | 87.95% |
| **CNN** | 119,530 | 7,878,272 | 0.361 | 27.143 | 89.17% |
| **Attention** | 70,794 | 1,182,976 | 0.673 | 5.464 | 83.39% |

*Gap Explanation:* `thop` measures Multiply-Accumulate operations (MACs), which are doubled to represent standard FLOPs ($1 \text{ MAC} = 2 \text{ FLOPs}$)[cite: 23]. Minor discrepancies between theoretical formulas and profiler totals stem from non-GEMM operations (layer normalization, Softmax, activation functions) that profilers omit or compute differently[cite: 23].

## 4. Conclusions
1. **Architectural Trade-offs & Accuracy:** The CNN achieved the highest classification accuracy (89.17%) while maintaining a competitive parameter footprint (119.5k), demonstrating the superior inductive bias of translation invariance and spatial weight sharing for image data compared to MLP (87.95%) and Attention (83.39%).
2. **Memory Access Patterns & Latency:** While MLP exhibits minimal latency due to contiguous memory GEMM executions, CNN and Attention demand significantly more computational cycles[cite: 28]. In particular, CNN requires intensive spatial slide operations over activation maps, resulting in high FLOP counts (7.88M) and higher batched latency on CPU[cite: 28]. Attention suffers from lower arithmetic intensity and non-sequential memory layouts (patch flattening, matrix transpositions for QK attention maps), yielding higher latency at batch size 1 despite lower parameter count[cite: 28].
3. **Cost-Effectiveness:** When balancing parameter efficiency, hardware utilization, and predictive capability, CNN represents the most cost-effective architecture for vision tasks at this scale, whereas the MLP serves as an ultra-low-latency alternative where inference speed is strictly constrained[cite: 28].