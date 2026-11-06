"""Automatic Mixed Precision (AMP) — Phase 2 will add real CUDA fp16 support."""
from contextlib import contextmanager
from typing import Optional
import numpy as np
from vectorphile.core.tensor import Tensor


class GradScaler:
    """Gradient scaler stub — no-op in Phase 1 (CPU only)."""
    def __init__(self, init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5,
                 growth_interval=2000, enabled=True):
        self.enabled = enabled
        self._scale = init_scale

    @property
    def scale(self):
        return self._scale

    def scale(self, loss):
        return loss

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        pass

    def unscale_(self, optimizer):
        pass

    def get_scale(self) -> float:
        return self._scale

    def is_enabled(self) -> bool:
        return self.enabled


@contextmanager
def autocast(device_type='cpu', dtype=None, enabled=True, cache_enabled=None):
    """Context manager for automatic mixed precision.

    In Phase 1 (CPU), this is a no-op. Phase 2 will add real fp16/bf16 on CUDA.
    """
    yield


def is_autocast_enabled() -> bool:
    return False


def get_autocast_dtype() -> str:
    return 'float32'
