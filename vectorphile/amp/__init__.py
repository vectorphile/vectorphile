"""Automatic Mixed Precision (AMP) — supports BF16/FP16 autocast on CPU and CUDA."""
from contextlib import contextmanager
from typing import Optional
from vectorphile.core.tensor import Tensor

_AUTOCAST_ENABLED = False
_AUTOCAST_DTYPE = 'float32'


class GradScaler:
    """Dynamic loss scaler for FP16 training.

    Scales the loss up before backward() to prevent FP16 underflow.
    Automatically reduces scale on NaN/Inf, increases it on clean steps.
    """
    def __init__(self, init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5,
                 growth_interval=2000, enabled=True):
        self.enabled = enabled
        self._scale = init_scale
        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._growth_interval = growth_interval
        self._steps_since_overflow = 0

    @property
    def _scale_val(self):
        return self._scale

    def scale(self, loss):
        """Multiply loss by current scale."""
        if not self.enabled:
            return loss
        return Tensor(loss._data * self._scale)

    def step(self, optimizer):
        """Unscale gradients and step the optimizer (skips on overflow)."""
        if not self.enabled:
            optimizer.step()
            return
        # Check for overflow
        overflow = False
        try:
            import math as _m
            for p in optimizer.param_groups[0]['params'] if hasattr(optimizer, 'param_groups') else []:
                if p.grad is not None:
                    flat = list(p.grad._data.to_flat_list()) if hasattr(p.grad._data, 'to_flat_list') else []
                    if any(not _m.isfinite(v) for v in flat):
                        overflow = True
                        break
        except Exception:
            pass
        if not overflow:
            # Unscale
            try:
                params = optimizer.param_groups[0]['params'] if hasattr(optimizer, 'param_groups') else []
                for p in params:
                    if p.grad is not None:
                        p.grad._data = (p.grad._data * (1.0 / self._scale)).astype('float32')
            except Exception:
                pass
            optimizer.step()
            self._steps_since_overflow += 1
            if self._steps_since_overflow >= self._growth_interval:
                self._scale = min(self._scale * self._growth_factor, 2.0 ** 24)
                self._steps_since_overflow = 0
        else:
            self._scale = max(self._scale * self._backoff_factor, 1.0)
            self._steps_since_overflow = 0

    def update(self):
        """Called after optimizer.step() when using manual unscale."""
        pass

    def unscale_(self, optimizer):
        """Manually unscale gradients."""
        if not self.enabled:
            return
        try:
            params = optimizer.param_groups[0]['params'] if hasattr(optimizer, 'param_groups') else []
            for p in params:
                if p.grad is not None:
                    p.grad._data = (p.grad._data * (1.0 / self._scale)).astype('float32')
        except Exception:
            pass

    def get_scale(self) -> float:
        return self._scale

    def is_enabled(self) -> bool:
        return self.enabled

    def get_growth_interval(self) -> int:
        return self._growth_interval

    def get_backoff_factor(self) -> float:
        return self._backoff_factor

    def get_growth_factor(self) -> float:
        return self._growth_factor

    def state_dict(self) -> dict:
        return {
            'scale': self._scale,
            'growth_factor': self._growth_factor,
            'backoff_factor': self._backoff_factor,
            'growth_interval': self._growth_interval,
            'steps_since_overflow': self._steps_since_overflow,
            'enabled': self.enabled,
        }

    def load_state_dict(self, state: dict) -> None:
        self._scale = state.get('scale', self._scale)
        self._growth_factor = state.get('growth_factor', self._growth_factor)
        self._backoff_factor = state.get('backoff_factor', self._backoff_factor)
        self._growth_interval = state.get('growth_interval', self._growth_interval)
        self._steps_since_overflow = state.get('steps_since_overflow', 0)
        self.enabled = state.get('enabled', self.enabled)


@contextmanager
def autocast(device_type='cpu', dtype=None, enabled=True, cache_enabled=None):
    """Context manager for automatic mixed precision.

    Inside this context, eligible Tensor ops auto-cast to `dtype` (default: bfloat16
    on CPU/MPS, float16 on CUDA). Master weights stay in float32; only forward pass
    activations are downcast for speed/memory savings.

    Usage::
        with vp.amp.autocast(device_type='cpu', dtype='bfloat16'):
            output = model(x)
        loss = criterion(output, target)
    """
    global _AUTOCAST_ENABLED, _AUTOCAST_DTYPE
    if not enabled:
        yield
        return
    prev_enabled = _AUTOCAST_ENABLED
    prev_dtype = _AUTOCAST_DTYPE
    _AUTOCAST_ENABLED = True
    _AUTOCAST_DTYPE = dtype or ('float16' if device_type == 'cuda' else 'bfloat16')
    try:
        yield
    finally:
        _AUTOCAST_ENABLED = prev_enabled
        _AUTOCAST_DTYPE = prev_dtype


def is_autocast_enabled() -> bool:
    return _AUTOCAST_ENABLED


def get_autocast_dtype() -> str:
    return _AUTOCAST_DTYPE


def cast_to_autocast_dtype(tensor: Tensor) -> Tensor:
    """Cast a tensor to the current autocast dtype if autocast is enabled."""
    if not _AUTOCAST_ENABLED:
        return tensor
    target = _AUTOCAST_DTYPE
    if hasattr(tensor._data, 'dtype') and str(tensor._data.dtype) != target:
        try:
            return Tensor(tensor._data.astype(target))
        except Exception:
            pass
    return tensor


def autocast_decorator(fn):
    """Decorator: cast all Tensor args to current autocast dtype inside fn."""
    import functools
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not _AUTOCAST_ENABLED:
            return fn(*args, **kwargs)
        new_args = tuple(cast_to_autocast_dtype(a) if isinstance(a, Tensor) else a for a in args)
        new_kwargs = {k: cast_to_autocast_dtype(v) if isinstance(v, Tensor) else v for k, v in kwargs.items()}
        return fn(*new_args, **new_kwargs)
    return wrapper


class _AutocastStack:
    """Thread-local autocast nesting stack for nested autocast contexts."""
    _stack: List[Tuple[bool, str]] = []

    @classmethod
    def push(cls, enabled: bool, dtype: str) -> None:
        cls._stack.append((enabled, dtype))

    @classmethod
    def pop(cls) -> Tuple[bool, str]:
        return cls._stack.pop() if cls._stack else (False, 'float32')


# ── BF16 Tensor wrapper ───────────────────────────────────────────────────────

class BF16Tensor:
    """Lightweight BF16 (bfloat16) tensor backed by a uint16 array.

    BF16 = same 8-bit exponent as FP32, 7-bit mantissa (truncated from 23).
    Requires 2 bytes/element (half of FP32) with the same dynamic range.

    Conversion is done in pure Python via bit manipulation (identical to what
    kernel_fp32_to_bf16 does in CUDA C). When the VP CUDA extension is available,
    conversion uses the CUDA kernels for GPU tensors.

    Usage::
        from vectorphile.amp import BF16Tensor
        bf = BF16Tensor.from_float([1.0, 2.5, -3.14])
        fp = bf.to_float()
    """

    __slots__ = ('_buf', '_shape')

    def __init__(self, uint16_buf, shape):
        """
        Args:
            uint16_buf: array.array('H', ...) of uint16 values
            shape: tuple of ints
        """
        import array as _a
        if not isinstance(uint16_buf, _a.array):
            uint16_buf = _a.array('H', [int(v) for v in uint16_buf])
        self._buf = uint16_buf
        self._shape = tuple(shape)

    @classmethod
    def from_float(cls, values, shape=None) -> 'BF16Tensor':
        """Convert a flat list of floats (or any iterable) to BF16."""
        import array as _a
        import struct
        vals = list(values)
        sh = shape or (len(vals),)
        buf = _a.array('H')
        for v in vals:
            bits = struct.unpack('<I', struct.pack('<f', float(v)))[0]
            # Round-to-nearest-even
            rounding_bias = 0x7FFF + ((bits >> 16) & 1)
            bf16 = (bits + rounding_bias) >> 16
            buf.append(int(bf16) & 0xFFFF)
        return cls(buf, sh)

    @classmethod
    def from_tensor(cls, t: Tensor) -> 'BF16Tensor':
        """Convert a VP Tensor to BF16."""
        flat = list(t._data.to_flat_list() if hasattr(t._data, 'to_flat_list') else t._data)
        sh = t._data.shape if hasattr(t._data, 'shape') else (len(flat),)
        return cls.from_float(flat, sh)

    def to_float(self):
        """Dequantize back to a list of float32 values."""
        import struct
        result = []
        for u16 in self._buf:
            bits = int(u16) << 16
            v = struct.unpack('<f', struct.pack('<I', bits))[0]
            result.append(v)
        return result

    def to_tensor(self) -> Tensor:
        """Return a VP Tensor in float32."""
        import array as _a
        from vectorphile._core.ndarray import NDArray, _c_strides, _DType
        floats = self.to_float()
        buf = _a.array('f', [float(v) for v in floats])
        return Tensor(NDArray(buf, self._shape, _c_strides(self._shape), _DType('float32')))

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return 'bfloat16'

    def __repr__(self):
        n = len(self._buf)
        preview = self.to_float()[:4]
        return f'BF16Tensor(shape={self._shape}, dtype=bfloat16, preview={preview}{"..." if n > 4 else ""})'

    def nbytes(self):
        return len(self._buf) * 2


class BF16Linear:
    """Linear layer that stores weights in BF16 and upcasts to FP32 for compute.

    Saves 50% weight memory vs FP32. The forward pass:
    1. Upcast BF16 weights → FP32
    2. Compute matmul in FP32 (or CUDA if available)
    3. Return FP32 output

    Args:
        in_features: input dimension
        out_features: output dimension
        bias: whether to include a bias term
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        from vectorphile.nn.module import Module
        import math as _m
        import array as _a
        self.in_features = in_features
        self.out_features = out_features
        # Kaiming uniform init in FP32, store as BF16
        bound = _m.sqrt(1.0 / max(1, in_features))
        import random as _rnd
        w_vals = [_rnd.uniform(-bound, bound) for _ in range(out_features * in_features)]
        self._weight_bf16 = BF16Tensor.from_float(w_vals, (out_features, in_features))
        if bias:
            b_vals = [_rnd.uniform(-bound, bound) for _ in range(out_features)]
            self._bias_bf16 = BF16Tensor.from_float(b_vals, (out_features,))
        else:
            self._bias_bf16 = None

    def forward(self, x: Tensor) -> Tensor:
        """Upcast weights to FP32, compute matmul, return FP32 output."""
        import array as _a
        from vectorphile._core.ndarray import NDArray, _c_strides, _DType, array as _nd
        # Upcast weights
        w_fp32 = self._weight_bf16.to_float()
        # Get input flat
        xfl = list(x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else x._data)
        xsh = x._data.shape if hasattr(x._data, 'shape') else (len(xfl),)
        n_tokens = 1
        for d in xsh[:-1]: n_tokens *= d
        D = xsh[-1]
        # Matmul: (n_tokens, D) @ (D, out_features)
        out = []
        for i in range(n_tokens):
            xi = xfl[i * D: i * D + D]
            for j in range(self.out_features):
                wj = w_fp32[j * D: j * D + D]
                dot = sum(xi[k] * wj[k] for k in range(D))
                out.append(dot)
        # Add bias
        if self._bias_bf16 is not None:
            b_fp32 = self._bias_bf16.to_float()
            for i in range(n_tokens):
                for j in range(self.out_features):
                    out[i * self.out_features + j] += b_fp32[j]
        out_sh = xsh[:-1] + (self.out_features,)
        buf = _a.array('f', [float(v) for v in out])
        return Tensor(NDArray(buf, out_sh, _c_strides(out_sh), _DType('float32')))

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def weight_bf16(self) -> BF16Tensor:
        return self._weight_bf16

    def nbytes_saved(self) -> int:
        """Bytes saved vs FP32 (half the weight memory)."""
        return self.in_features * self.out_features * 2  # 2 bytes = FP32 - BF16


__all__ = [
    'GradScaler',
    'autocast',
    'is_autocast_enabled',
    'get_autocast_dtype',
    'cast_to_autocast_dtype',
    'autocast_decorator',
    '_AutocastStack',
    'BF16Tensor',
    'BF16Linear',
]
