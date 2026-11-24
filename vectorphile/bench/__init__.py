"""vectorphile.bench — Performance benchmarking utilities — pure Python."""
from __future__ import annotations
import time
import math
import gc
from typing import Any, Callable, List, Optional, Tuple


class BenchmarkResult:
    """Container for benchmark results."""

    def __init__(self, name: str, times_ms: List[float],
                 batch_size: int = 1, n_params: int = 0):
        self.name = name
        self.times_ms = sorted(times_ms)
        self.batch_size = batch_size
        self.n_params = n_params

    @property
    def n_runs(self) -> int:
        return len(self.times_ms)

    @property
    def mean_ms(self) -> float:
        return sum(self.times_ms) / max(len(self.times_ms), 1)

    @property
    def std_ms(self) -> float:
        mu = self.mean_ms
        var = sum((t - mu)**2 for t in self.times_ms) / max(len(self.times_ms), 1)
        return math.sqrt(var)

    @property
    def min_ms(self) -> float:
        return self.times_ms[0]

    @property
    def max_ms(self) -> float:
        return self.times_ms[-1]

    def percentile(self, p: float) -> float:
        idx = int(len(self.times_ms) * p / 100)
        return self.times_ms[min(idx, len(self.times_ms) - 1)]

    @property
    def throughput(self) -> float:
        mean_s = self.mean_ms / 1000.0
        return self.batch_size / mean_s if mean_s > 0 else 0.0

    def print(self) -> None:
        print(f'\n=== Benchmark: {self.name} ===')
        print(f'  Runs:        {self.n_runs}')
        print(f'  Batch size:  {self.batch_size}')
        if self.n_params:
            print(f'  Parameters:  {self.n_params:,}')
        print(f'  Mean:        {self.mean_ms:.3f} ms ± {self.std_ms:.3f} ms')
        print(f'  Min:         {self.min_ms:.3f} ms')
        print(f'  Median (p50):{self.percentile(50):.3f} ms')
        print(f'  p95:         {self.percentile(95):.3f} ms')
        print(f'  p99:         {self.percentile(99):.3f} ms')
        print(f'  Max:         {self.max_ms:.3f} ms')
        print(f'  Throughput:  {self.throughput:.1f} samples/s')

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'n_runs': self.n_runs,
            'batch_size': self.batch_size,
            'mean_ms': self.mean_ms,
            'std_ms': self.std_ms,
            'p50_ms': self.percentile(50),
            'p95_ms': self.percentile(95),
            'p99_ms': self.percentile(99),
            'throughput': self.throughput,
        }


def benchmark(fn: Callable, args: tuple = (), kwargs: dict = None,
              n_warmup: int = 5, n_runs: int = 50,
              batch_size: int = 1, name: str = 'fn') -> BenchmarkResult:
    """Benchmark a callable with timing statistics."""
    kwargs = kwargs or {}
    for _ in range(n_warmup):
        fn(*args, **kwargs)
    gc.collect()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append((time.perf_counter() - t0) * 1000)
    return BenchmarkResult(name=name, times_ms=times, batch_size=batch_size)


def benchmark_model(model, input_shape: Tuple[int, ...],
                    n_warmup: int = 5, n_runs: int = 50,
                    name: Optional[str] = None) -> BenchmarkResult:
    """Benchmark a vectorphile nn.Module forward pass."""
    import random as _rand
    from vectorphile.core.tensor import Tensor
    from vectorphile._core.ndarray import array as _nd
    from vectorphile.profiler import count_parameters
    import vectorphile as vp

    name = name or type(model).__name__
    batch_size = input_shape[0] if input_shape else 1
    total = 1
    for s in input_shape:
        total *= s
    flat = [_rand.gauss(0, 1) for _ in range(total)]
    x = Tensor(_nd(flat))

    def _run():
        with vp.autograd.no_grad():
            return model(x)

    n_params = count_parameters(model)
    result = benchmark(_run, n_warmup=n_warmup, n_runs=n_runs,
                       batch_size=batch_size, name=name)
    result.n_params = n_params
    return result


def compare(*results: BenchmarkResult, metric: str = 'mean_ms') -> None:
    print(f'\n{"Model":<30} {"Mean(ms)":>10} {"p50(ms)":>10} {"p99(ms)":>10} {"Throughput":>14}')
    print('-' * 80)
    for r in results:
        print(f'{r.name:<30} {r.mean_ms:>10.3f} {r.percentile(50):>10.3f} '
              f'{r.percentile(99):>10.3f} {r.throughput:>13.1f}/s')


__all__ = ['BenchmarkResult', 'benchmark', 'benchmark_model', 'compare']
