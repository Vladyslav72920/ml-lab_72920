"""Task 6: inspect PyTorch graphs, compare eager vs compiled/traced execution
after warm-up, inspect tensor metadata and measure the cost of non-contiguous
views.
"""
from __future__ import annotations

import statistics
import time
import traceback
import warnings

import torch
import torch.fx

warnings.filterwarnings("ignore", category=FutureWarning)  # jit.trace deprecation notice

from common import env_info, get_logger, load_config, results_dir, save_json, seed_everything


class SmallMLP(torch.nn.Module):
    def __init__(self, d_in=64, hidden=256, d_out=10):
        super().__init__()
        self.fc1 = torch.nn.Linear(d_in, hidden)
        self.fc2 = torch.nn.Linear(hidden, hidden)
        self.fc3 = torch.nn.Linear(hidden, d_out)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h)) * 0.5 + h * 0.5   # small fusable elementwise tail
        return self.fc3(h)


# ------------------------------------------------------------ graph dumps
def autograd_graph(t: torch.Tensor) -> list[str]:
    """Walk grad_fn.next_functions (the dynamic graph PyTorch recorded)."""
    lines, seen = [], set()

    def walk(fn, depth):
        if fn is None:
            return
        name = type(fn).__name__
        extra = ""
        if hasattr(fn, "variable"):
            v = fn.variable
            extra = f"  <- leaf {tuple(v.shape)} {str(v.dtype).replace('torch.', '')}"
        if id(fn) in seen:
            lines.append("  " * depth + name + extra + "  (shared node, expanded above)")
            return
        lines.append("  " * depth + name + extra)
        seen.add(id(fn))
        for nxt, _ in fn.next_functions:
            walk(nxt, depth + 1)

    walk(t.grad_fn, 0)
    return lines


# ---------------------------------------------------------------- timing
def bench(fn, x, warmup, iters, repeats):
    with torch.no_grad():
        for _ in range(warmup):
            fn(x)
        per_iter = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            for _ in range(iters):
                fn(x)
            per_iter.append((time.perf_counter() - t0) / iters)
    return statistics.median(per_iter), per_iter


def try_compile(model, x, log):
    """torch.compile first; fall back to torch.jit.trace if unsupported."""
    try:
        t0 = time.perf_counter()
        cm = torch.compile(model)
        with torch.no_grad():
            cm(x)
        return cm, "torch.compile (inductor)", time.perf_counter() - t0, None
    except Exception as e:  # pragma: no cover - environment dependent
        log.warning("torch.compile unavailable: %s", e)
        err = "".join(traceback.format_exception_only(type(e), e)).strip()
        return None, None, None, err


def main():
    cfg = load_config()
    seed_everything(cfg["seed"])
    log = get_logger(cfg)
    out_dir = results_dir(cfg)
    c = cfg["torch_inspect"]

    model = SmallMLP(hidden=c["hidden"]).eval()
    report_lines = []

    # ---- 1. graph inspection -------------------------------------------
    x_small = torch.randn(8, 64, requires_grad=False)
    out = model(x_small).sum()
    report_lines += ["=== Autograd graph (eager, recorded during forward of model(x).sum()) ==="]
    report_lines += autograd_graph(out)

    fx = torch.fx.symbolic_trace(model)
    report_lines += ["", "=== torch.fx symbolic_trace graph ===", str(fx.graph)]

    with torch.no_grad():
        traced = torch.jit.trace(model, x_small)
    report_lines += ["", "=== torch.jit.trace graph ===", str(traced.graph)]

    # ---- 2. eager vs compiled vs traced --------------------------------
    timings = {}
    compile_info = {}
    for batch in (32, c["batch"]):
        x = torch.randn(batch, 64)
        torch._dynamo.reset() if hasattr(torch, "_dynamo") else None
        compiled, cname, ctime, cerr = try_compile(model, x, log)
        with torch.no_grad():
            tr = torch.jit.trace(model, x)
            tr = torch.jit.freeze(tr)
        variants = {"eager": model, "jit.trace (frozen)": tr}
        if compiled is not None:
            variants[cname] = compiled
        compile_info[batch] = {"compile_backend": cname, "first_call_compile_s": ctime, "error": cerr}

        with torch.no_grad():
            ref = model(x)
            max_diff = {k: float((v(x) - ref).abs().max()) for k, v in variants.items()}

        res = {}
        for name, fn in variants.items():
            med, runs = bench(fn, x, c["warmup_iters"], c["timed_iters"], c["repeats"])
            res[name] = {"median_ms": med * 1e3, "runs_ms": [r * 1e3 for r in runs],
                         "max_abs_diff_vs_eager": max_diff[name]}
        for name in res:
            res[name]["speedup_vs_eager"] = res["eager"]["median_ms"] / res[name]["median_ms"]
        timings[f"batch_{batch}"] = res
        log.info("compile | batch %d | %s", batch,
                 ", ".join(f"{k}: {v['median_ms']:.3f} ms ({v['speedup_vs_eager']:.2f}x)" for k, v in res.items()))

    # ---- 3. tensor metadata ---------------------------------------------
    n = c["contig_size"]
    base = torch.randn(n, n)
    views = {
        "x": base,
        "x.t()": base.t(),
        "x[:, ::2]": base[:, ::2],
        "x.t().contiguous()": base.t().contiguous(),
    }
    meta = {k: {"shape": list(v.shape), "dtype": str(v.dtype), "stride": list(v.stride()),
                "is_contiguous": v.is_contiguous(),
                "shares_storage_with_x": v.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()}
            for k, v in views.items()}
    report_lines += ["", "=== Tensor metadata ==="]
    for k, v in meta.items():
        report_lines.append(f"{k:22s} shape={v['shape']} dtype={v['dtype']} stride={v['stride']} "
                            f"contiguous={v['is_contiguous']} shares_storage={v['shares_storage_with_x']}")

    # ---- 4. contiguous vs non-contiguous op speed ------------------------
    pairs = {
        "transpose x.t()": (base.t(), base.t().contiguous()),
        "strided slice x[:, ::2]": (base[:, ::2], base[:, ::2].contiguous()),
    }
    ops = {
        "elementwise x*2+1": lambda t: t * 2 + 1,
        "row sum (dim=1)": lambda t: t.sum(dim=1),
        "relu": torch.relu,
        "matmul t @ x": lambda t: t @ base,
    }
    contig = {}
    for vname, (view, dense) in pairs.items():
        rows = {}
        for name, op in ops.items():
            if name.startswith("matmul") and view.shape[1] != base.shape[0]:
                op = lambda t: t @ base[: t.shape[1]]
            mc, _ = bench(op, dense, 3, 10, c["repeats"])
            mv, _ = bench(op, view, 3, 10, c["repeats"])
            out_contig = op(view).is_contiguous()
            rows[name] = {"contiguous_ms": mc * 1e3, "non_contiguous_ms": mv * 1e3,
                          "slowdown": mv / mc, "output_is_contiguous": out_contig}
        mcopy, _ = bench(lambda t: t.contiguous(), view, 3, 10, c["repeats"])
        rows[".contiguous() copy"] = {"ms": mcopy * 1e3}
        contig[vname] = rows
        log.info("contiguity | %s | %s", vname, ", ".join(
            f"{k}: {v['slowdown']:.2f}x" for k, v in rows.items() if "slowdown" in v))

    (out_dir / "torch_graph.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    save_json({"model": "SmallMLP 64 -> 256 -> 256 -> 10, inference (no_grad), float32, CPU",
               "warmup_iters": c["warmup_iters"], "timed_iters": c["timed_iters"], "repeats": c["repeats"],
               "compile_info": compile_info, "timings": timings, "env": env_info()},
              out_dir / "compile_timing.json")
    save_json({"matrix": f"{n}x{n} float32", "metadata": meta, "timings": contig},
              out_dir / "contiguity.json")
    return timings, contig


if __name__ == "__main__":
    main()
