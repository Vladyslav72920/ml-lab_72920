"""Task 4: verify the Value engine against PyTorch autograd and numerical
gradients on f = ((a*b + c)**2).relu().sum().

a, b, c are vectors; our engine works on scalars, so each element is its own
Value and .sum() is a chain of Value additions. Because (a*b + c)**2 >= 0,
the ReLU in the original expression is almost never inactive, so half of the
trials use the shifted variant ((a*b + c)**2 - 1).relu().sum() to also test the
zero-gradient branch of ReLU.
"""
from __future__ import annotations

import numpy as np
import torch

from common import get_logger, load_config, results_dir, save_json, seed_everything
from engine import Value, sum_values


# ----------------------------------------------------------- expressions
def f_engine(a, b, c, shift=0.0):
    terms = [(((ai * bi + ci) ** 2) - shift).relu() for ai, bi, ci in zip(a, b, c)]
    return sum_values(terms).sum()


def f_torch(a, b, c, shift=0.0):
    return (((a * b + c) ** 2) - shift).relu().sum()


def f_numpy(a, b, c, shift=0.0):
    return np.maximum(((a * b + c) ** 2) - shift, 0.0).sum()


# --------------------------------------------------------------- gradients
def grads_engine(a, b, c, shift):
    A = [Value(x) for x in a]
    B = [Value(x) for x in b]
    C = [Value(x) for x in c]
    out = f_engine(A, B, C, shift)
    out.backward()
    return out.data, [np.array([v.grad for v in V]) for V in (A, B, C)]


def grads_torch(a, b, c, shift):
    A, B, C = (torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in (a, b, c))
    out = f_torch(A, B, C, shift)
    out.backward()
    return out.item(), [t.grad.numpy().copy() for t in (A, B, C)]


def grads_numerical(a, b, c, shift, h):
    inputs = [a.copy(), b.copy(), c.copy()]
    grads = []
    for k in range(3):
        g = np.zeros_like(inputs[k])
        for i in range(len(g)):
            orig = inputs[k][i]
            inputs[k][i] = orig + h
            fp = f_numpy(*inputs, shift)
            inputs[k][i] = orig - h
            fm = f_numpy(*inputs, shift)
            inputs[k][i] = orig
            g[i] = (fp - fm) / (2 * h)
        grads.append(g)
    return grads


def accumulation_test():
    """A node used several times must receive the SUM of its gradient paths.
    y = x*x + x*3 + x  ->  dy/dx = 2x + 4. If backward overwrote instead of
    accumulating, the result would be wrong."""
    x = Value(1.5)
    y = x * x + x * 3 + x
    y.backward()
    expected = 2 * 1.5 + 4
    xt = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
    (xt * xt + xt * 3 + xt).backward()
    return {"engine": x.grad, "torch": xt.grad.item(), "analytic": expected,
            "passed": abs(x.grad - expected) < 1e-12 and abs(x.grad - xt.grad.item()) < 1e-12}


def main():
    cfg = load_config()
    seed_everything(cfg["seed"])
    log = get_logger(cfg)
    out_dir = results_dir(cfg)
    gc = cfg["gradcheck"]
    rng = np.random.default_rng(cfg["seed"])

    rows = []
    worst = {"engine_vs_torch": 0.0, "engine_vs_numerical": 0.0, "torch_vs_numerical": 0.0, "forward": 0.0}
    inactive_elems = 0
    for trial in range(gc["n_trials"]):
        shift = 0.0 if trial % 2 == 0 else 1.0
        a, b, c = (rng.normal(size=gc["vector_size"]) for _ in range(3))
        inactive_elems += int(((a * b + c) ** 2 - shift <= 0).sum())
        fe, ge = grads_engine(a, b, c, shift)
        ft, gt = grads_torch(a, b, c, shift)
        gn = grads_numerical(a, b, c, shift, gc["numerical_h"])
        d_et = max(np.abs(x - y).max() for x, y in zip(ge, gt))
        d_en = max(np.abs(x - y).max() for x, y in zip(ge, gn))
        d_tn = max(np.abs(x - y).max() for x, y in zip(gt, gn))
        worst["engine_vs_torch"] = max(worst["engine_vs_torch"], d_et)
        worst["engine_vs_numerical"] = max(worst["engine_vs_numerical"], d_en)
        worst["torch_vs_numerical"] = max(worst["torch_vs_numerical"], d_tn)
        worst["forward"] = max(worst["forward"], abs(fe - ft))
        rows.append({"trial": trial, "shift": shift, "f": fe,
                     "max_abs_engine_vs_torch": d_et,
                     "max_abs_engine_vs_numerical": d_en,
                     "max_abs_torch_vs_numerical": d_tn})

    # one fully printed example for the report
    a, b, c = (np.array([0.5, -1.2, 2.0]), np.array([1.5, 0.3, -0.7]), np.array([-0.2, 0.9, 0.4]))
    _, ge = grads_engine(a, b, c, 0.0)
    _, gt = grads_torch(a, b, c, 0.0)
    example = {"a": a.tolist(), "b": b.tolist(), "c": c.tolist(),
               "engine_grad": {k: g.tolist() for k, g in zip("abc", ge)},
               "torch_grad": {k: g.tolist() for k, g in zip("abc", gt)}}

    acc = accumulation_test()
    tol = gc["tolerance"]
    passed = worst["engine_vs_torch"] < tol and worst["engine_vs_numerical"] < tol and acc["passed"]
    result = {
        "expression": "((a*b + c)**2).relu().sum()  (+ shifted variant ((a*b+c)**2 - 1).relu().sum())",
        "dtype": "float64",
        "n_trials": gc["n_trials"], "vector_size": gc["vector_size"],
        "relu_inactive_elements_tested": inactive_elems,
        "numerical_h": gc["numerical_h"], "tolerance": tol,
        "max_abs_diff": worst, "accumulation_test": acc,
        "passed": bool(passed), "example": example, "trials": rows,
    }
    save_json(result, out_dir / "gradcheck.json")

    md = ["# Gradient check (task 4)", "",
          f"Expression: `((a*b + c)**2).relu().sum()`, vectors of size {gc['vector_size']}, "
          f"{gc['n_trials']} random trials (half with a `-1` shift so ReLU is inactive for "
          f"{inactive_elems} elements), float64.", "",
          "| Comparison | Max absolute difference | < 1e-6 |", "|---|---|---|"]
    for k in ("engine_vs_torch", "engine_vs_numerical", "torch_vs_numerical", "forward"):
        md.append(f"| {k.replace('_', ' ')} | {worst[k]:.3e} | {'yes' if worst[k] < tol else 'NO'} |")
    md += ["", f"Accumulation test `y = x*x + 3x + x` at x=1.5: engine {acc['engine']}, "
               f"torch {acc['torch']}, analytic {acc['analytic']} -> {'pass' if acc['passed'] else 'FAIL'}.",
           "", f"**Overall: {'PASS' if passed else 'FAIL'}**", ""]
    (out_dir / "gradcheck.md").write_text("\n".join(md), encoding="utf-8")

    log.info("gradcheck | max|engine-torch|=%.3e max|engine-num|=%.3e accumulation=%s -> %s",
             worst["engine_vs_torch"], worst["engine_vs_numerical"], acc["passed"],
             "PASS" if passed else "FAIL")
    return result


if __name__ == "__main__":
    main()
