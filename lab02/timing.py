import time
import numpy as np
from data import X_tr, Y_tr
from nn import init_params, forward, backward

WIDTHS = [16, 64]


def time_passes(W, batch=32, runs=100):
    p = init_params(W)
    Xb, Yb = X_tr[:batch], Y_tr[:batch]
    forward(Xb, p) 
    tf, tb = [], []
    for _ in range(runs):
        t0 = time.perf_counter()
        _, c = forward(Xb, p)
        t1 = time.perf_counter()
        backward(Yb, p, c)
        t2 = time.perf_counter()
        tf.append(t1 - t0)
        tb.append(t2 - t1)
    return np.median(tf), np.median(tb)


if __name__ == "__main__":
    print("--- Task 7: timing (median of 100 runs, batch=32) ---")
    for W in WIDTHS:
        f, b = time_passes(W)
        print(f"W={W:2d}: forward={f * 1e6:7.1f} us | backward={b * 1e6:7.1f} us | backward/forward={b / f:.2f}")

    print("""
Timing explanation
The backward pass dominates: it is slower than the forward pass for both widths,
and the gap grows with the hidden width W (see backward/forward ratios above).

Reason: matrix multiplication. Forward needs one matmul per layer
(A_prev @ W), i.e. 3 matmuls in total. Backward needs two matmuls per layer:
A_prev.T @ dZ (gradient of the weights) and dZ @ W.T (gradient passed to the
previous layer), i.e. 5 matmuls in total (the input layer needs no dA).
A matmul costs ~ N * n_in * n_out multiply-adds, while ReLU, softmax and bias
sums cost only ~ N * n_out, so the matmuls determine the runtime.

The ratio is below the theoretical ~5/3 for small W because with batch size 32
and tiny matrices a large part of the time is fixed Python/NumPy call overhead.
For larger W the matmul cost grows (~W^2), overhead matters less, and the
backward/forward ratio increases.
""")