# Lab 5: Frameworks and automatic differentiation

**Student:** Vladyslav Ryzhyk, 72348
**Course:** ML Systems
**Run:** `cd src && python run_all.py` (seed 42, all settings in `config.yaml`, all outputs in `../results`)
**Environment:** Python 3.13, PyTorch 2.15 (CPU, 2 threads), NumPy 2.5, Linux x86_64

## Summary of graded evidence

| Evidence | Result | File |
|---|---|---|
| Autograd engine matches PyTorch | max \|engine − torch\| = **0.0**, max \|engine − numerical\| = **2.1e-10** (< 1e-6) | `results/gradcheck.md`, `results/gradcheck.json` |
| make_moons training, engine vs PyTorch | same loss curve (max diff 4.4e-16), both 98.5% train accuracy; PyTorch **~440× faster** | `results/moons_timing.json`, `results/moons_training.png` |
| Eager vs compiled timings | batch 4096: compile **1.27×** faster; batch 32: compile **0.86×** (slower) | `results/compile_timing.json` |
| Graph + tensor inspection | autograd / FX / TorchScript graphs, shape, dtype, stride, contiguity | `results/torch_graph.txt`, `results/contiguity.json` |
| Note on when compilation helps | Section 6.4 | this file |

## 1. `Value` node design (task 1)

File: `engine.py`. Each `Value` stores:

| Field | Meaning |
|---|---|
| `data` | forward value (float) |
| `grad` | ∂output/∂self, starts at 0, filled by `backward()` |
| `_prev` | tuple of parent nodes the value was computed from |
| `_op` | operation name, used for graph inspection |
| `_backward` | closure with the **local derivative rule**; it reads `out.grad` and pushes it into the parents |

The class uses `__slots__`, so each node is small and attribute access is cheaper. This matters because the moons model creates about 18k nodes per epoch.

## 2. Operations and graph construction (task 2)

Primitive operations and their local rules:

| Op | Forward | Local backward rule |
|---|---|---|
| `a + b` | a + b | ∂/∂a = 1, ∂/∂b = 1 |
| `a * b` | a·b | ∂/∂a = b, ∂/∂b = a |
| `a ** k` (constant k) | aᵏ | k·aᵏ⁻¹ |
| `relu()` | max(a, 0) | 1 if a > 0 else 0 (same subgradient at 0 as PyTorch) |
| `sigmoid()`, `log()` | for the BCE model in task 5 | s(1−s), 1/a |

`-`, `/`, `neg` and the reflected operators (`2 * x`, `1 - x`, …) are built from these primitives, so they never need their own derivative code. Every operation returns a new `Value` whose `_prev` points at its inputs, so **running the forward pass builds the computation graph**. This is the define-by-run (eager) model that PyTorch uses (Reddi §7.3).

## 3. Backward pass (task 3)

```python
def backward(self):
    topo = self.topo_order()      # 1. topological sort (iterative DFS, parents before children)
    self.grad = 1.0               # 2. seed d(out)/d(out) = 1
    for node in reversed(topo):   # 3. visit nodes in reverse order
        node._backward()          # 4. local rule ACCUMULATES with +=
```

- The topological sort is iterative, so deep graphs (thousands of chained additions in a loss sum) do not hit Python's recursion limit.
- Every rule uses `+=`, so a node that feeds several consumers receives the **sum** of all its gradient paths. `zero_grad()` resets gradients between training steps, the same way `optimizer.zero_grad()` does in PyTorch.
- Accumulation test: `y = x·x + 3x + x` at x = 1.5 gives dy/dx = 2x + 4 = 7. Engine = 7.0, PyTorch = 7.0. If gradients were overwritten instead of accumulated, the result would only reflect the last path (1.0).

## 4. Verification against PyTorch and numerical gradients (task 4)

Script: `gradcheck.py`. Expression: `((a*b + c)**2).relu().sum()` with a, b, c as vectors of length 6, float64.

- **20 random trials**. Because (a·b + c)² ≥ 0, the ReLU in the original expression is almost never inactive. Half of the trials therefore use the shifted version `((a*b + c)**2 - 1).relu().sum()`, which put **34 elements** in the zero-gradient branch.
- Three gradients are compared: our engine, `torch.autograd`, and central differences with h = 1e-5.

| Comparison | Max absolute difference | < 1e-6 |
|---|---|---|
| engine vs PyTorch | 0.000e+00 | yes |
| engine vs numerical | 2.055e-10 | yes |
| PyTorch vs numerical | 2.055e-10 | yes |
| forward value | 3.553e-15 | yes |

**PASS.** The engine and PyTorch agree bit for bit because both apply the same chain-rule products in float64. The 2e-10 gap to the numerical estimate is the truncation and round-off error of finite differences, not an engine error.

## 5. Tiny model on make_moons (task 5)

Script: `train_moons.py`, figure `results/moons_training.png`.

- Data: `make_moons(n=200, noise=0.1, random_state=42)`.
- Model: **2 → 8 (ReLU) → 1 (sigmoid)**, BCE loss (p clamped to [1e-7, 1−1e-7]), full-batch SGD, lr = 1.0, 200 epochs.
- PyTorch starts from **exactly the same initial weights** (copied from the engine model) and uses the same loss formula, so the two runs are directly comparable.

| | Engine (Value) | PyTorch float64 | PyTorch float32 |
|---|---|---|---|
| Median time, 5 runs (200 epochs) | **25.38 s** | **0.058 s** | 0.065 s |
| Per epoch | 126.9 ms | 0.29 ms | 0.32 ms |
| Final BCE loss | 0.067544 | 0.067544 | 0.067544 |
| Train accuracy | 98.5% | 98.5% | – |

Max difference between the two loss curves over all 200 epochs: **4.4e-16**. Both implementations compute the same function and the same gradients.

**Why PyTorch is ~440× faster:** the engine creates one Python object and one closure per scalar operation (about 18k nodes per epoch) and runs the backward pass in an interpreted loop. PyTorch represents each layer as one tensor operation (one `addmm` instead of 16 multiplies and adds per sample) and runs it in vectorized C++ kernels. float32 is not faster than float64 here: at 200×8 the work is tiny and per-operation dispatch overhead dominates, not arithmetic.

## 6. PyTorch graph, eager vs compiled, tensors (task 6)

Script: `torch_inspect.py`. Model `SmallMLP`: Linear(64→256) → ReLU → Linear(256→256) → ReLU, an elementwise mix `0.5·h2 + 0.5·h1`, then Linear(256→10). Inference, float32, CPU.

### 6.1 Graph inspection

`results/torch_graph.txt` contains three views of the same model:

1. **Autograd graph**, built during the eager forward pass by walking `grad_fn.next_functions`: `SumBackward0 → AddmmBackward0 → AddBackward0 → MulBackward0 → ReluBackward0 → …` down to `AccumulateGrad` leaves (the weights). This is the same structure as our `Value` graph, but at tensor granularity: one node per layer, not per scalar.
2. **torch.fx graph** (`symbolic_trace`): a static list of `placeholder / call_module / call_function` nodes.
3. **TorchScript graph** (`jit.trace`): an IR with concrete input type `Float(8, 64, strides=[64, 1])`, specialized to the traced shape.

### 6.2 Eager vs compiled vs traced

Warm-up 10 iterations, then 7 repeats × 200 iterations, median time per call. Outputs of all variants match eager exactly (max |diff| = 0).

| Batch | Eager | `jit.trace` (frozen) | `torch.compile` (inductor) |
|---|---|---|---|
| 32 | 0.099 ms (1.00×) | 0.091 ms (1.09×) | 0.115 ms (**0.86×**) |
| 4096 | 5.603 ms (1.00×) | 5.289 ms (1.06×) | 4.421 ms (**1.27×**) |

The first `torch.compile` call took **1.94 s** (graph capture + codegen). That is ~20,000 eager calls at batch 32, or ~1,600 at batch 4096, before compilation breaks even. The batch-4096 recompile took only 0.10 s because the inductor cache was already warm.

Timings on a shared 2-core VM vary by roughly ±15% between runs. The direction of the result was stable across repeated runs: compile is faster at large batch and no faster at small batch.

### 6.3 Tensor shape, dtype, contiguity and the cost of non-contiguous views

| Tensor | Shape | dtype | Stride | Contiguous | Shares storage |
|---|---|---|---|---|---|
| `x` | [2048, 2048] | float32 | [2048, 1] | yes | – |
| `x.t()` | [2048, 2048] | float32 | [1, 2048] | **no** | yes (view) |
| `x[:, ::2]` | [2048, 1024] | float32 | [2048, 2] | **no** | yes (view) |
| `x.t().contiguous()` | [2048, 2048] | float32 | [2048, 1] | yes | no (copy) |

`x.t()` and slicing are free: they only change the stride metadata. The cost appears in the operations that read the view later. Median time, non-contiguous view vs a contiguous copy of the same values:

| Operation | `x.t()` slowdown | `x[:, ::2]` slowdown |
|---|---|---|
| elementwise `x*2+1` | 0.92× | **1.69×** |
| ReLU | 0.71× | **1.61×** |
| row sum `sum(dim=1)` | **1.84×** | **2.57×** |
| matmul `t @ x` | 0.96× | 1.09× |
| `.contiguous()` copy itself | 6.47 ms | 0.85 ms |

Interpretation:

- **Transpose + elementwise:** no slowdown. PyTorch's TensorIterator sees that the view is a dense permutation of memory, walks it in memory order, and returns an output with the same transposed strides (`output_is_contiguous = False`).
- **Reduction across the strided dimension** (`sum(dim=1)` on `x.t()`) reads elements 2048 floats apart, so cache lines and vectorization are wasted: **1.8× slower**.
- **Strided slice `::2`** is not dense. Every operation has to skip every other float, and half of each loaded cache line is unused: **1.6–2.6× slower**.
- **matmul** is barely affected because BLAS accepts transposed operands directly (or packs them internally).
- Making a view contiguous costs a full copy (6.5 ms for 16 MB, more than most of the operations above). Calling `.contiguous()` only pays off if the tensor is reused by several stride-sensitive operations.

### 6.4 Note: when does compilation help?

Compilation (`torch.compile`, or tracing/scripting) helps when:

1. **The same graph runs many times with stable shapes**, so the one-time compile cost (1.9 s here) is amortized. This fits training loops and long-running inference servers, not one-off scripts or notebooks.
2. **The work per call is large enough** that kernel quality matters. Here, at batch 4096, inductor fused the ReLU/mul/add tail into one kernel and saved memory round-trips: **1.27×**. At batch 32 the call takes ~0.1 ms, most of it Python and dispatch overhead plus the compiled-graph guard checks, so compile was **0.86×** (slower).
3. **The model has many small elementwise operations between large operations** (activations, norms, residual adds). These are what fusion removes. Models dominated by a few large GEMMs or convolutions gain less, because those already run in optimized BLAS/oneDNN kernels in eager mode too.
4. **Deployment needs a Python-free artifact** (TorchScript / `torch.export`) that can be shipped to C++ or mobile runtimes. Here the benefit is portability, not speed (Reddi §7.7–7.8).

Compilation does **not** help, and can hurt, when shapes change every call (recompilation), when the code has data-dependent Python control flow (graph breaks fall back to eager), or during debugging, where eager mode's line-by-line execution and readable stack traces are worth more than speed. This is the classic trade-off from Reddi §7.4: eager execution gives flexibility, graph execution gives optimization opportunities.

## 7. Reproducibility

- One command: `python run_all.py`. The fixed seed (42) is used for `random`, NumPy, PyTorch and `make_moons`. All hyperparameters are in `config.yaml`.
- Gradient checks and losses are deterministic. Wall-clock timings depend on the machine. Each timing is a median over repeats, and the raw per-run numbers are kept in the JSON files.
- Log of the full run: `results/run.log`.
