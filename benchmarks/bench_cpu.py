"""
vectorphile CPU benchmark suite — measures throughput vs PyTorch on key ops.

Run with:
    python benchmarks/bench_cpu.py

Output columns: op | shape | vp_ms | torch_ms | speedup
"""
import time
import gc
import math

WARMUP = 2
ITERS  = 10

def _timeit(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    gc.collect()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1000  # ms


def _header():
    print(f"{'Op':<30} {'Shape':<20} {'VP (ms)':>10} {'PT (ms)':>10} {'Speedup':>9}")
    print("-" * 82)


def _row(name, shape, vp_ms, pt_ms):
    sp = pt_ms / vp_ms if vp_ms > 0 else float('inf')
    flag = " ✓" if sp >= 1.0 else " ✗"
    print(f"{name:<30} {str(shape):<20} {vp_ms:>10.3f} {pt_ms:>10.3f} {sp:>8.2f}x{flag}")


# ─── import vectorphile ───────────────────────────────────────────────────────
try:
    import vectorphile as vp
except OSError as e:
    print(f"Cannot run benchmarks: {e}")
    print("Tip: increase Windows paging file size (System > Advanced > Performance > Virtual Memory)")
    raise SystemExit(0)

# ─── try to import torch (optional — skip if memory constrained) ─────────────
HAS_TORCH = False
if '--with-torch' in __import__('sys').argv:
    try:
        import torch
        HAS_TORCH = True
    except Exception:
        print("NOTE: torch import failed — showing VP timings only\n")
else:
    print("NOTE: running VP-only mode. Pass --with-torch to compare against PyTorch.\n")


def bench_matmul():
    for N in [128, 256, 512]:
        shape = (N, N)
        a_vp = vp.randn(N, N)
        b_vp = vp.randn(N, N)
        vp_ms = _timeit(lambda: a_vp @ b_vp)
        if HAS_TORCH:
            a_pt = torch.randn(N, N)
            b_pt = torch.randn(N, N)
            pt_ms = _timeit(lambda: a_pt @ b_pt)
        else:
            pt_ms = float('nan')
        _row("matmul", shape, vp_ms, pt_ms)


def bench_linear_forward():
    for (B, D) in [(32, 256), (64, 512)]:
        m_vp = vp.nn.Linear(D, D)
        x_vp = vp.randn(B, D)
        vp_ms = _timeit(lambda: m_vp(x_vp))
        if HAS_TORCH:
            m_pt = torch.nn.Linear(D, D)
            x_pt = torch.randn(B, D)
            pt_ms = _timeit(lambda: m_pt(x_pt))
        else:
            pt_ms = float('nan')
        _row("nn.Linear forward", (B, D), vp_ms, pt_ms)


def bench_relu():
    for N in [10000, 100000]:
        x_vp = vp.randn(N)
        vp_ms = _timeit(lambda: vp.relu(x_vp))
        if HAS_TORCH:
            x_pt = torch.randn(N)
            pt_ms = _timeit(lambda: torch.relu(x_pt))
        else:
            pt_ms = float('nan')
        _row("relu", (N,), vp_ms, pt_ms)


def bench_softmax():
    for (B, D) in [(64, 512), (256, 1024)]:
        x_vp = vp.randn(B, D)
        vp_ms = _timeit(lambda: vp.softmax(x_vp, -1))
        if HAS_TORCH:
            x_pt = torch.randn(B, D)
            pt_ms = _timeit(lambda: torch.softmax(x_pt, dim=-1))
        else:
            pt_ms = float('nan')
        _row("softmax", (B, D), vp_ms, pt_ms)


def bench_batchnorm():
    for (B, C, H, W) in [(16, 32, 4, 4), (32, 64, 4, 4)]:
        m_vp = vp.nn.BatchNorm2d(C)
        m_vp.training = False
        x_vp = vp.randn(B, C, H, W)
        vp_ms = _timeit(lambda: m_vp(x_vp))
        if HAS_TORCH:
            m_pt = torch.nn.BatchNorm2d(C)
            m_pt.eval()
            x_pt = torch.randn(B, C, H, W)
            pt_ms = _timeit(lambda: m_pt(x_pt))
        else:
            pt_ms = float('nan')
        _row("BatchNorm2d (eval)", (B, C, H, W), vp_ms, pt_ms)


def bench_embedding():
    for (V, D, N) in [(5000, 128, 32), (10000, 256, 64)]:
        m_vp = vp.nn.Embedding(V, D)
        idx_vp = vp.randint(0, V, (N,))
        vp_ms = _timeit(lambda: m_vp(idx_vp))
        if HAS_TORCH:
            m_pt = torch.nn.Embedding(V, D)
            idx_pt = torch.randint(0, V, (N,))
            pt_ms = _timeit(lambda: m_pt(idx_pt))
        else:
            pt_ms = float('nan')
        _row("Embedding lookup", (N, D), vp_ms, pt_ms)


def bench_layernorm():
    for (B, T, D) in [(8, 32, 256), (16, 64, 512)]:
        m_vp = vp.nn.LayerNorm(D)
        x_vp = vp.randn(B, T, D)
        vp_ms = _timeit(lambda: m_vp(x_vp))
        if HAS_TORCH:
            m_pt = torch.nn.LayerNorm(D)
            x_pt = torch.randn(B, T, D)
            pt_ms = _timeit(lambda: m_pt(x_pt))
        else:
            pt_ms = float('nan')
        _row("LayerNorm", (B, T, D), vp_ms, pt_ms)


def bench_adam_step():
    for D in [256, 512]:
        m_vp = vp.nn.Linear(D, D)
        opt_vp = vp.optim.Adam(m_vp.parameters(), lr=1e-3)
        x_vp = vp.randn(16, D)
        def vp_step():
            y = m_vp(x_vp)
            y.mean().backward()
            opt_vp.step()
            opt_vp.zero_grad()
        vp_ms = _timeit(vp_step)
        if HAS_TORCH:
            m_pt = torch.nn.Linear(D, D)
            opt_pt = torch.optim.Adam(m_pt.parameters(), lr=1e-3)
            x_pt = torch.randn(16, D)
            def pt_step():
                y = m_pt(x_pt)
                y.mean().backward()
                opt_pt.step()
                opt_pt.zero_grad()
            pt_ms = _timeit(pt_step)
        else:
            pt_ms = float('nan')
        _row("Adam step (fwd+bwd+step)", (16, D), vp_ms, pt_ms)


def bench_import():
    """Cold import time (approximate via reimport)."""
    import importlib, sys
    # Remove from cache
    mods = [k for k in sys.modules if k.startswith('vectorphile')]
    for m in mods:
        del sys.modules[m]
    t0 = time.perf_counter()
    import vectorphile as _vp2
    vp_ms = (time.perf_counter() - t0) * 1000

    if HAS_TORCH:
        torch_mods = [k for k in sys.modules if k.startswith('torch')]
        for m in torch_mods:
            del sys.modules[m]
        t0 = time.perf_counter()
        import torch as _t2
        pt_ms = (time.perf_counter() - t0) * 1000
    else:
        pt_ms = float('nan')
    _row("cold import", "(library)", vp_ms, pt_ms)


if __name__ == '__main__':
    # Note: requires sufficient virtual memory. On Windows, increase paging file
    # (System > Advanced > Performance > Virtual Memory) if killed with exit code 29.
    print("\nvectorphile CPU Benchmark Suite")
    print(f"Python {__import__('sys').version.split()[0]}  |  VP v0.5.0")
    if HAS_TORCH:
        print(f"PyTorch {torch.__version__}  |  CPU-only")
    print()
    _header()
    print()
    bench_matmul()
    print()
    bench_linear_forward()
    print()
    bench_relu()
    print()
    bench_softmax()
    print()
    bench_batchnorm()
    print()
    bench_embedding()
    print()
    bench_layernorm()
    print()
    # bench_adam_step() is expensive (fwd+bwd+opt) — run with --full flag
    if '--full' in __import__('sys').argv:
        bench_adam_step()
        print()
    bench_import()
    print()
    print("Done. ✓ = VP faster than PyTorch  ✗ = PyTorch faster")
