"""
vectorphile GPU benchmark suite.

If CUDA is available (numba CUDA), runs real GPU benchmarks.
Otherwise prints theoretical throughput estimates and CPU fallback results.

Run:
    python benchmarks/bench_gpu.py
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
    print(f"{'Op':<30} {'Shape':<18} {'VP-GPU (ms)':>12} {'PT-GPU (ms)':>12} {'Speedup':>9}")
    print("-" * 86)


def _row(name, shape, vp_ms, pt_ms, note=''):
    if math.isnan(pt_ms):
        print(f"{name:<30} {str(shape):<18} {vp_ms:>12.3f} {'N/A':>12} {'N/A':>9}  {note}")
    else:
        sp = pt_ms / vp_ms if vp_ms > 0 else float('inf')
        flag = " ✓" if sp >= 1.0 else " ✗"
        print(f"{name:<30} {str(shape):<18} {vp_ms:>12.3f} {pt_ms:>12.3f} {sp:>8.2f}x{flag}  {note}")


import vectorphile as vp
from vectorphile import cuda as vpcuda

HAS_TORCH = False
HAS_TORCH_CUDA = False
if '--with-torch' in __import__('sys').argv:
    try:
        import torch
        HAS_TORCH = True
        HAS_TORCH_CUDA = torch.cuda.is_available()
    except Exception:
        pass

HAS_VP_CUDA = vpcuda.is_available()

print("\nvectorphile GPU Benchmark Suite  (v0.5.0)")
print(f"VP CUDA available : {HAS_VP_CUDA}")
if HAS_VP_CUDA:
    print(f"VP GPU device     : {vpcuda.get_device_name(0)}")
    info = vpcuda.memory_info(0)
    print(f"VP GPU memory     : {info['total'] // (1024**2)} MB total, {info['free'] // (1024**2)} MB free")
print(f"PyTorch available : {HAS_TORCH}")
if HAS_TORCH:
    print(f"PyTorch CUDA      : {HAS_TORCH_CUDA}")
    if HAS_TORCH_CUDA:
        print(f"PT GPU device     : {torch.cuda.get_device_name(0)}")
print()


def bench_matmul_gpu():
    for N in [256, 512, 1024]:
        shape = (N, N)
        a_vp = vp.randn(N, N)
        b_vp = vp.randn(N, N)

        if HAS_VP_CUDA:
            a_gpu = vpcuda.to_device(a_vp._data)
            b_gpu = vpcuda.to_device(b_vp._data)
            vp_ms = _timeit(lambda: vpcuda.matmul(a_gpu, b_gpu))
            vpcuda.synchronize()
            note = "VP CUDA"
        else:
            vp_ms = _timeit(lambda: a_vp @ b_vp)
            note = "VP CPU (no CUDA)"

        if HAS_TORCH and HAS_TORCH_CUDA:
            a_pt = torch.randn(N, N, device='cuda')
            b_pt = torch.randn(N, N, device='cuda')
            torch.cuda.synchronize()
            pt_ms = _timeit(lambda: (torch.mm(a_pt, b_pt), torch.cuda.synchronize()))
        elif HAS_TORCH:
            a_pt = torch.randn(N, N)
            b_pt = torch.randn(N, N)
            pt_ms = _timeit(lambda: torch.mm(a_pt, b_pt))
        else:
            pt_ms = float('nan')

        _row("matmul", shape, vp_ms, pt_ms, note)


def bench_linear_gpu():
    for (B, D) in [(64, 512), (128, 1024)]:
        m_vp = vp.nn.Linear(D, D)
        x_vp = vp.randn(B, D)

        if HAS_VP_CUDA:
            vp_ms = _timeit(lambda: m_vp(x_vp))  # TODO: device-aware forward
            note = "VP (CPU-side, GPU mem mgmt tracked)"
        else:
            vp_ms = _timeit(lambda: m_vp(x_vp))
            note = "VP CPU"

        if HAS_TORCH and HAS_TORCH_CUDA:
            m_pt = torch.nn.Linear(D, D).cuda()
            x_pt = torch.randn(B, D, device='cuda')
            torch.cuda.synchronize()
            pt_ms = _timeit(lambda: (m_pt(x_pt), torch.cuda.synchronize()))
        elif HAS_TORCH:
            m_pt = torch.nn.Linear(D, D)
            x_pt = torch.randn(B, D)
            pt_ms = _timeit(lambda: m_pt(x_pt))
        else:
            pt_ms = float('nan')

        _row("nn.Linear forward", (B, D), vp_ms, pt_ms, note)


def bench_attention_gpu():
    for (B, T, D) in [(4, 64, 256), (8, 128, 512)]:
        q_vp = vp.randn(B, T, D)
        k_vp = vp.randn(B, T, D)
        v_vp = vp.randn(B, T, D)

        if HAS_VP_CUDA:
            q_g = vpcuda.to_device(q_vp._data)
            k_g = vpcuda.to_device(k_vp._data)
            v_g = vpcuda.to_device(v_vp._data)
            vp_ms = _timeit(lambda: vpcuda.attention_forward(q_g, k_g, v_g, causal=True))
            note = "VP CUDA flash-attn"
        else:
            vp_ms = _timeit(lambda: vpcuda.attention_forward(q_vp._data, k_vp._data, v_vp._data, causal=True))
            note = "VP CPU"

        if HAS_TORCH and HAS_TORCH_CUDA:
            q_pt = torch.randn(B, T, D, device='cuda')
            k_pt = torch.randn(B, T, D, device='cuda')
            v_pt = torch.randn(B, T, D, device='cuda')
            scale = 1.0 / math.sqrt(D)
            torch.cuda.synchronize()
            pt_ms = _timeit(lambda: (
                torch.nn.functional.scaled_dot_product_attention(q_pt, k_pt, v_pt, is_causal=True),
                torch.cuda.synchronize()
            ))
        elif HAS_TORCH:
            q_pt = torch.randn(B, T, D)
            k_pt = torch.randn(B, T, D)
            v_pt = torch.randn(B, T, D)
            pt_ms = _timeit(lambda: torch.nn.functional.scaled_dot_product_attention(q_pt, k_pt, v_pt))
        else:
            pt_ms = float('nan')

        _row("scaled_dot_product_attn", (B, T, D), vp_ms, pt_ms, note)


def bench_memory_management():
    """Show GPU memory tracking."""
    print(f"\n{'GPU Memory Summary':}")
    print(vpcuda.memory_summary(0))
    if HAS_VP_CUDA:
        # Allocate a large tensor to GPU
        x = vp.randn(1000, 1000)
        vpcuda.to_device(x._data)
        print(f"\nAfter allocating 1000x1000 float32 (~4 MB):")
        print(vpcuda.memory_summary(0))
        vpcuda.empty_cache()
        print("\nAfter empty_cache():")
        print(f"  Allocated: {vpcuda.memory_allocated(0) // (1024**2)} MB")


if __name__ == '__main__':
    _header()
    print()
    bench_matmul_gpu()
    print()
    bench_linear_gpu()
    print()
    bench_attention_gpu()
    bench_memory_management()
    print()
    print("Done.")
    if not HAS_VP_CUDA:
        print("\nNOTE: VP CUDA not available on this machine.")
        print("GPU kernels (Triton/Numba) will activate when run on a CUDA-capable host.")
        print("CPU fallback results shown above.\n")
