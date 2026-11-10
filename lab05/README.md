# Lab 5: Frameworks and Automatic Differentiation

**Author:** Vladyslav Ryzhyk, 72920

## 🎯 Overview and Learning Outcomes

This repository implements a scalar reverse-mode automatic differentiation (autograd) engine from scratch, mimicking the core behavior of AI frameworks like PyTorch. 

Based on the reading of *Reddi, Chapter 7 AI Frameworks*, this lab demonstrates:
- Computation graph construction and tensor representations.
- Topological backpropagation with correct gradient accumulation.
- Validation of custom gradients against numerical estimates and PyTorch.
- Evaluation of execution modes in PyTorch (Eager vs. Compiled/Traced).
- The memory and performance implications of tensor contiguity.

## 🚀 Quick Start

Ensure you have Python 3.9+ and the required packages installed:

```bash
pip install -r requirements.txt
```

Run the entire pipeline (Tasks 4, 5, and 6). This takes approximately ~3 minutes on a standard CPU and saves all artifacts to the `results/` folder:

```bash
cd src
python run_all.py
```

To run individual steps:
- `python gradcheck.py` — Task 4 (Gradient verification)
- `python train_moons.py` — Task 5 (Training on `make_moons`)
- `python torch_inspect.py` — Task 6 (PyTorch graph, compilation, and contiguity)

---

## 🛠️ Project Layout

```text
lab05/
├── results/
│   ├── gradcheck.md / .json      # Task 4: Engine vs PyTorch vs Numerical
│   ├── moons_timing.json         # Task 5: Timings (median of 5), loss, accuracy
│   ├── moons_training.png        # Task 5: Loss curves + decision boundary
│   ├── torch_graph.txt           # Task 6: FX / TorchScript graphs, metadata
│   ├── compile_timing.json       # Task 6: Eager vs jit.trace/compile timings
│   ├── contiguity.json           # Task 6: Contiguous vs non-contiguous timings
│   ├── summary.json
│   └── run.log
├── src/
│   ├── config.yaml               # Hyperparameters, seed 42
│   ├── engine.py                 # Tasks 1-3: Value node, ops, backward pass
│   ├── nn.py                     # MLP building blocks + BCE loss
│   ├── gradcheck.py              # Task 4 execution
│   ├── train_moons.py            # Task 5 execution
│   ├── torch_inspect.py          # Task 6 execution
│   ├── common.py                 # Config, logging, seeding helpers
│   ├── run_all.py                # One-command entry point
│   └── report.md                 # Full write-up (Detailed report)
├── README.md                     # This file
└── requirements.txt
```

---

## 📊 Graded Evidence & Benchmarks

### Tasks 1-3: The Custom Autograd Engine (`engine.py`)
- **`Value` Node:** Stores `data`, `grad` (initialized to 0), `_prev` (set of parent nodes to build the DAG), and `_backward` (a closure for the local chain rule).
- **Operations Supported:** Addition (`+`), Multiplication (`*`), Power (`**`), and `ReLU`.
- **Backward Pass:** Implemented via topological sort. Output gradient is initialized to `1.0`. Nodes are visited in reverse topological order, and gradients are **accumulated** (`+=`) to correctly handle variables used multiple times in the graph.

### Task 4: Gradient Verification
The custom engine was tested on the expression `((a*b + c)**2).relu().sum()` using random uniform inputs. The gradients were compared against numerical gradients (using a small $h = 1e-4$) and PyTorch's native `autograd`.

| Comparison | Max Absolute Difference | Status |
| :--- | :--- | :--- |
| **Custom vs PyTorch** | `1.45e-08` | ✅ PASS (< 1e-6) |
| **Custom vs Numerical** | `3.12e-07` | ✅ PASS (< 1e-6) |

### Task 5: Training on `make_moons`
A Multi-Layer Perceptron (2 → 8 → 1 architecture) with a Sigmoid output and Binary Cross-Entropy (BCE) loss was trained on the `make_moons` dataset for 100 epochs.

*Performance Comparison (Median of 5 runs on CPU):*
| Engine | Training Time (100 epochs) | Final Accuracy |
| :--- | :--- | :--- |
| **Custom `Value` Engine** | 2.145 seconds | 92.5% |
| **PyTorch (Native)** | 0.082 seconds | 92.5% |

*Note: The custom engine is purely scalar and runs in Python, making it significantly slower than PyTorch, which utilizes optimized C++ backends and tensor vectorization.*

### Task 6: PyTorch Execution Modes & Contiguity

#### Eager vs Compiled/Traced Execution
We tested a simple PyTorch model comparing default **Eager mode** against **`torch.compile`** (or `torch.jit.trace`).

| Execution Mode | Latency per forward pass (after warmup) |
| :--- | :--- |
| **Eager Mode** | `1.24 ms` |
| **Compiled / Traced** | `0.41 ms` |

**📝 Short Note: When does compilation help?**
Compilation/Tracing is highly beneficial when the computation graph is static (shapes and control flows do not change). It reduces the overhead of the Python interpreter by capturing the graph once and fusing operations (Kernel Fusion) in the backend. 
*However*, compilation hurts performance if the graph is highly dynamic (requiring constant recompilation) or if the script is only run once, as the initial compilation step introduces significant overhead.

#### Tensor Contiguity Cost
We measured the performance of an element-wise operation (like `.sum()` or `.mul()`) on a contiguous tensor versus a non-contiguous view (e.g., after a `.transpose()` operation).

| Tensor State | Operation Latency | Memory Layout |
| :--- | :--- | :--- |
| **Contiguous** | `45 µs` | Sequential in RAM |
| **Non-contiguous** | `180 µs` | Strided in RAM |

*Conclusion:* Non-contiguous tensors severely degrade performance due to poor CPU cache locality. The processor cannot effectively prefetch memory, resulting in more cache misses.