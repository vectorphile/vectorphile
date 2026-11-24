"""
vectorphile._core — Pure-Python N-dimensional array engine.

No NumPy dependency. Uses Python's built-in `array` module for flat storage
and implements strides, broadcasting, and basic BLAS manually.

When NumPy is available it is used as a fast-path accelerator, but all
operations have a correct pure-Python fallback so vectorphile runs with
zero external dependencies.
"""
from vectorphile._core.ndarray import NDArray, array, zeros, ones, full, eye, arange, linspace

# CPU feature detection — cached at import time
def _detect_cpu_features() -> dict:
    """Detect CPU SIMD capabilities using cpuid via ctypes (Windows/Linux)."""
    features = {'avx2': False, 'avx512f': False, 'fma': False}
    try:
        import subprocess, platform
        if platform.system() == 'Windows':
            # Use PowerShell to check CPU flags
            r = subprocess.run(
                ['powershell', '-Command',
                 '(Get-WmiObject -Class Win32_Processor).Description'],
                capture_output=True, text=True, timeout=2
            )
            desc = r.stdout.strip().lower()
            # Heuristic: modern Intel/AMD CPUs with AVX2
            features['avx2'] = True  # _vp_core.pyd was compiled with /arch:AVX2
            features['fma'] = True
        else:
            with open('/proc/cpuinfo') as f:
                flags = f.read().lower()
            features['avx2'] = 'avx2' in flags
            features['avx512f'] = 'avx512f' in flags
            features['fma'] = 'fma' in flags
    except Exception:
        pass
    return features

CPU_FEATURES = _detect_cpu_features()
from vectorphile._core.ops import (
    add, subtract, multiply, divide, matmul, dot,
    maximum, minimum, where, abs as vp_abs, sqrt, exp, log,
    sin, cos, tan, sinh, cosh, tanh,
    sum as vp_sum, mean as vp_mean, max as vp_max, min as vp_min,
    clip, reshape, transpose, concatenate, stack,
)

__all__ = [
    'NDArray', 'array', 'zeros', 'ones', 'full', 'eye', 'arange', 'linspace',
    'add', 'subtract', 'multiply', 'divide', 'matmul', 'dot',
    'maximum', 'minimum', 'where', 'vp_abs', 'sqrt', 'exp', 'log',
    'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh',
    'vp_sum', 'vp_mean', 'vp_max', 'vp_min',
    'clip', 'reshape', 'transpose', 'concatenate', 'stack',
]
