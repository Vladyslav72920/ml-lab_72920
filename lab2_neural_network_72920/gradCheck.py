import numpy as np
from data import X_tr, Y_tr
from nn import init_params, forward, backward, loss_fn


def grad_check(p, X, Y, n=10, eps=1e-5, seed=1):
    r = np.random.default_rng(seed)
    _, c = forward(X, p)
    g = backward(Y, p, c)
    passed = 0
    for _ in range(n):
        k = str(r.choice(["W1", "W2", "W3"]))
        idx = tuple(int(r.integers(0, s)) for s in p[k].shape)
        old = p[k][idx]
        p[k][idx] = old + eps
        Lp = loss_fn(forward(X, p)[0], Y)
        p[k][idx] = old - eps
        Lm = loss_fn(forward(X, p)[0], Y)
        p[k][idx] = old
        num, ana = (Lp - Lm) / (2 * eps), g[k][idx]
        rel = abs(ana - num) / max(abs(ana) + abs(num), 1e-12)
        ok = rel < 1e-6
        passed += ok
        print(f"{k}{idx}: analytic={ana:.4e} numeric={num:.4e} rel={rel:.2e} {'OK' if ok else 'FAIL'}")
    print(f"PASSED {passed}/{n} (need >= 9/10)")
    return passed


if __name__ == "__main__":
    print("task5")
    grad_check(init_params(16), X_tr[:32], Y_tr[:32])