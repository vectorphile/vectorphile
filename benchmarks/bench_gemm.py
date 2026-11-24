"""GEMM benchmark: vectorphile vs raw numpy (no torch dependency)."""
import time
import numpy as np
import vectorphile as vp
from vectorphile.core.tensor import Tensor

SIZES = [128, 256, 512, 1024]
REPS = 5


def bench(fn, reps=REPS):
    # warmup
    fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def main():
    print(f"{'Size':>6}  {'vp (ms)':>10}  {'numpy (ms)':>12}  {'speedup':>8}")
    print("-" * 44)
    for N in SIZES:
        A_np = np.random.randn(N, N).astype(np.float32)
        B_np = np.random.randn(N, N).astype(np.float32)
        A = Tensor(A_np)
        B = Tensor(B_np)

        vp_t = bench(lambda: A @ B) * 1000
        np_t = bench(lambda: A_np @ B_np) * 1000
        ratio = np_t / vp_t if vp_t > 0 else float('inf')
        print(f"{N:>6}  {vp_t:>10.2f}  {np_t:>12.2f}  {ratio:>8.2f}x")


if __name__ == '__main__':
    main()
