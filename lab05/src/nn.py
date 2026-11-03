"""Tiny MLP built on the scalar Value engine (Lab 5, task 5).

Architecture: 2 -> 8 (ReLU) -> 1 (sigmoid), trained with binary cross-entropy.
"""
from __future__ import annotations

import random

from engine import Value, sum_values


class Neuron:
    def __init__(self, n_in: int, rng: random.Random, act: str):
        bound = 1.0 / n_in ** 0.5  # same scale as torch.nn.Linear default init
        self.w = [Value(rng.uniform(-bound, bound)) for _ in range(n_in)]
        self.b = Value(rng.uniform(-bound, bound))
        self.act = act

    def __call__(self, x):
        z = sum_values(wi * xi for wi, xi in zip(self.w, x)) + self.b
        if self.act == "relu":
            return z.relu()
        if self.act == "sigmoid":
            return z.sigmoid()
        return z

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, n_in: int, n_out: int, rng: random.Random, act: str):
        self.neurons = [Neuron(n_in, rng, act) for _ in range(n_out)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    """2 -> hidden (ReLU) -> 1 (sigmoid)."""

    def __init__(self, n_in: int = 2, hidden: int = 8, seed: int = 42):
        rng = random.Random(seed)
        self.l1 = Layer(n_in, hidden, rng, "relu")
        self.l2 = Layer(hidden, 1, rng, "sigmoid")

    def __call__(self, x):
        return self.l2(self.l1(x))[0]

    def parameters(self):
        return self.l1.parameters() + self.l2.parameters()

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    # weight export so PyTorch can start from identical parameters
    def weights(self):
        W1 = [[w.data for w in n.w] for n in self.l1.neurons]
        b1 = [n.b.data for n in self.l1.neurons]
        W2 = [[w.data for w in n.w] for n in self.l2.neurons]
        b2 = [n.b.data for n in self.l2.neurons]
        return W1, b1, W2, b2


def bce(p: Value, y: float, eps: float = 1e-7) -> Value:
    """Binary cross-entropy for one sample; p is clamped away from 0/1."""
    p = p * (1 - 2 * eps) + eps
    return -(p.log() * y + (1 - p).log() * (1 - y))
