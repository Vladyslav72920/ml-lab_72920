"""Scalar reverse-mode automatic differentiation engine (Lab 5, tasks 1-3).

Every arithmetic operation on a Value creates a new Value that remembers
its parents and a closure (`_backward`) holding the local derivative rule.
The forward pass therefore builds the computation graph as a side effect,
and `backward()` walks that graph in reverse topological order.
"""
from __future__ import annotations

import math
from typing import Callable, Iterable, Tuple, Union

Number = Union[int, float]


class Value:
    """A node in the computation graph.

    Attributes
    ----------
    data      : float  - forward value
    grad      : float  - d(output)/d(self), filled in by backward()
    _prev     : tuple  - parent nodes this value was computed from
    _op       : str    - name of the operation (for graph inspection)
    _backward : callable - local derivative rule; pushes self.grad into parents
    """

    __slots__ = ("data", "grad", "_prev", "_op", "_backward", "label")

    def __init__(self, data: Number, _prev: Iterable["Value"] = (), _op: str = "", label: str = ""):
        self.data = float(data)
        self.grad = 0.0
        self._prev: Tuple[Value, ...] = tuple(_prev)
        self._op = _op
        self._backward: Callable[[], None] = lambda: None
        self.label = label

    # ------------------------------------------------------------------ ops
    @staticmethod
    def _wrap(x) -> "Value":
        return x if isinstance(x, Value) else Value(x)

    def __add__(self, other):
        other = self._wrap(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # d(a+b)/da = 1, d(a+b)/db = 1  -> accumulate with +=
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = self._wrap(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # d(a*b)/da = b, d(a*b)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, k: Number):
        assert isinstance(k, (int, float)), "only constant exponents are supported"
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():
            # d(x^k)/dx = k * x^(k-1)
            self.grad += k * (self.data ** (k - 1)) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward():
            # subgradient 0 at x == 0, same convention as PyTorch
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    # extra ops needed for the make_moons model (sigmoid + BCE)
    def sigmoid(self):
        x = self.data
        s = 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))
        out = Value(s, (self,), "sigmoid")

        def _backward():
            self.grad += s * (1.0 - s) * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    # derived ops (built from the primitives above -> graph stays consistent)
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-self._wrap(other))

    def __truediv__(self, other):
        return self * self._wrap(other) ** -1

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return self._wrap(other) + (-self)

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return self._wrap(other) * self ** -1

    def sum(self):
        """Scalar engine: sum of a single scalar is the identity (keeps the
        task-4 expression ((a*b + c)**2).relu().sum() valid)."""
        return self

    # ------------------------------------------------------------ backward
    def topo_order(self):
        """Iterative DFS topological sort (no recursion limit on deep graphs)."""
        order, visited = [], set()
        stack = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for p in node._prev:
                if id(p) not in visited:
                    stack.append((p, False))
        return order  # parents always before children

    def backward(self):
        topo = self.topo_order()   # 1. topologically sort the graph
        self.grad = 1.0            # 2. seed output gradient d(out)/d(out) = 1
        for node in reversed(topo):  # 3. visit nodes in reverse order
            node._backward()       # 4. each rule ACCUMULATES (+=) into parents

    def zero_grad(self):
        for node in self.topo_order():
            node.grad = 0.0

    def __repr__(self):
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g}, op='{self._op}')"


def sum_values(values):
    """Balanced-free left fold sum; avoids Python's sum() starting from int 0."""
    it = iter(values)
    total = next(it)
    for v in it:
        total = total + v
    return total
