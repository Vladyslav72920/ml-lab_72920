import numpy as np
from data import X_tr, Y_tr
from nn import init_params, forward, loss_fn

p = init_params(16)
P, _ = forward(X_tr[:32], p)

print("task3")
print("Output shape:", P.shape, "(expected (32, 10))")
print("Rows sum to 1 (within 1e-6):", bool(np.abs(P.sum(axis=1) - 1).max() < 1e-6))

print("task4")
print(f"Untrained loss = {loss_fn(P, Y_tr[:32]):.4f}, ln(10) = {np.log(10):.4f}")