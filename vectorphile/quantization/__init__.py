"""Model quantization — INT8, INT4, FP16 for inference efficiency."""
from __future__ import annotations
import numpy as np
from typing import Any, Dict, Optional, Union


# ── Quantization schemes ──────────────────────────────────────────────────────

def quantize_int8(weights: np.ndarray, symmetric: bool = True) -> tuple:
    """Quantize float32 weights to INT8.

    Returns (quantized_weights, scale, zero_point).
    """
    weights = np.array(weights, dtype=np.float32)
    if symmetric:
        scale = np.max(np.abs(weights)) / 127.0
        scale = max(scale, 1e-8)
        q = np.clip(np.round(weights / scale), -127, 127).astype(np.int8)
        return q, scale, 0
    else:
        w_min, w_max = weights.min(), weights.max()
        scale = (w_max - w_min) / 255.0
        scale = max(scale, 1e-8)
        zero_point = int(np.round(-w_min / scale))
        q = np.clip(np.round(weights / scale) + zero_point, 0, 255).astype(np.uint8)
        return q, scale, zero_point


def dequantize_int8(q: np.ndarray, scale: float, zero_point: int = 0) -> np.ndarray:
    """Dequantize INT8 back to float32."""
    return (q.astype(np.float32) - zero_point) * scale


def quantize_int4(weights: np.ndarray, group_size: int = 128) -> tuple:
    """Quantize float32 weights to INT4 (packed as uint8).

    Uses group quantization for better accuracy.
    Returns (packed_weights, scales, zero_points, original_shape).
    """
    weights = np.array(weights, dtype=np.float32)
    orig_shape = weights.shape
    flat = weights.flatten()
    n = len(flat)
    # Pad to multiple of group_size
    pad = (-n) % group_size
    if pad:
        flat = np.concatenate([flat, np.zeros(pad)])
    groups = flat.reshape(-1, group_size)
    scales = []
    zero_points = []
    quants = []
    for g in groups:
        g_min, g_max = g.min(), g.max()
        scale = (g_max - g_min) / 15.0
        scale = max(scale, 1e-8)
        zp = int(np.round(-g_min / scale))
        q = np.clip(np.round(g / scale) + zp, 0, 15).astype(np.uint8)
        scales.append(scale)
        zero_points.append(zp)
        quants.append(q)
    quants = np.array(quants)  # (n_groups, group_size)
    # Pack two 4-bit values per byte
    packed = (quants[:, 0::2] | (quants[:, 1::2] << 4))
    return packed, np.array(scales), np.array(zero_points), orig_shape, group_size, n


def dequantize_int4(packed, scales, zero_points, orig_shape, group_size, orig_n) -> np.ndarray:
    """Dequantize INT4 (packed) back to float32."""
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    quants = np.empty((packed.shape[0], group_size), dtype=np.uint8)
    quants[:, 0::2] = lo
    quants[:, 1::2] = hi
    result = np.empty_like(quants, dtype=np.float32)
    for i in range(len(scales)):
        result[i] = (quants[i].astype(np.float32) - zero_points[i]) * scales[i]
    return result.flatten()[:orig_n].reshape(orig_shape)


def quantize_fp16(weights: np.ndarray) -> np.ndarray:
    """Cast weights to FP16."""
    return np.array(weights, dtype=np.float16)


def dequantize_fp16(weights: np.ndarray) -> np.ndarray:
    """Cast FP16 back to float32."""
    return weights.astype(np.float32)


# ── Calibration ───────────────────────────────────────────────────────────────

class MinMaxCalibrator:
    """Collect activation statistics for post-training quantization."""

    def __init__(self):
        self._min = {}
        self._max = {}
        self._count = {}

    def update(self, name: str, tensor: np.ndarray) -> None:
        t = np.array(tensor, dtype=np.float32)
        if name not in self._min:
            self._min[name] = t.min()
            self._max[name] = t.max()
            self._count[name] = 1
        else:
            self._min[name] = min(self._min[name], t.min())
            self._max[name] = max(self._max[name], t.max())
            self._count[name] += 1

    def scale_zp(self, name: str, bits: int = 8, symmetric: bool = True):
        w_min = self._min.get(name, -1.0)
        w_max = self._max.get(name, 1.0)
        qmax = (1 << (bits - 1)) - 1 if symmetric else (1 << bits) - 1
        if symmetric:
            scale = max(abs(w_min), abs(w_max)) / qmax
            return max(scale, 1e-8), 0
        else:
            scale = (w_max - w_min) / qmax
            scale = max(scale, 1e-8)
            zp = int(round(-w_min / scale))
            return scale, zp

    def stats(self) -> Dict:
        return {name: {'min': self._min[name], 'max': self._max[name],
                       'count': self._count[name]}
                for name in self._min}


class HistogramCalibrator:
    """Histogram-based calibration with percentile clipping."""

    def __init__(self, percentile: float = 99.99, num_bins: int = 2048):
        self._histograms = {}
        self._bin_edges = {}
        self.percentile = percentile
        self.num_bins = num_bins

    def update(self, name: str, tensor: np.ndarray) -> None:
        t = np.abs(np.array(tensor, dtype=np.float32)).flatten()
        counts, edges = np.histogram(t, bins=self.num_bins)
        if name not in self._histograms:
            self._histograms[name] = counts
            self._bin_edges[name] = edges
        else:
            self._histograms[name] += counts

    def scale(self, name: str) -> float:
        counts = self._histograms[name]
        edges = self._bin_edges[name]
        total = counts.sum()
        threshold_count = total * self.percentile / 100.0
        cumsum = np.cumsum(counts)
        idx = np.searchsorted(cumsum, threshold_count)
        threshold = edges[min(idx + 1, len(edges) - 1)]
        return max(threshold / 127.0, 1e-8)


# ── Model quantizer ───────────────────────────────────────────────────────────

class QuantizedModel:
    """Wrapper that holds quantized weights for a vectorphile nn.Module."""

    def __init__(self, model: Any, scheme: str = 'int8', symmetric: bool = True,
                 group_size: int = 128):
        self.scheme = scheme
        self.symmetric = symmetric
        self.group_size = group_size
        self._orig_class = type(model).__name__
        self._quant_data: Dict = {}
        self._meta: Dict = {}
        self._quantize(model)

    def _quantize(self, model) -> None:
        if not hasattr(model, 'named_parameters'):
            raise TypeError("Model must have named_parameters() (a vectorphile nn.Module)")
        for name, param in model.named_parameters():
            data = param._data if hasattr(param, '_data') else np.array(param)
            if self.scheme == 'int8':
                q, scale, zp = quantize_int8(data, self.symmetric)
                self._quant_data[name] = q
                self._meta[name] = {'scale': scale, 'zp': zp, 'scheme': 'int8',
                                    'shape': data.shape, 'dtype': str(data.dtype)}
            elif self.scheme == 'int4':
                result = quantize_int4(data, self.group_size)
                self._quant_data[name] = result
                self._meta[name] = {'scheme': 'int4', 'shape': data.shape}
            elif self.scheme == 'fp16':
                self._quant_data[name] = quantize_fp16(data)
                self._meta[name] = {'scheme': 'fp16', 'shape': data.shape}

    def dequantize(self, name: str) -> np.ndarray:
        meta = self._meta[name]
        q = self._quant_data[name]
        scheme = meta['scheme']
        if scheme == 'int8':
            return dequantize_int8(q, meta['scale'], meta['zp'])
        elif scheme == 'int4':
            return dequantize_int4(*q)
        elif scheme == 'fp16':
            return dequantize_fp16(q)
        raise ValueError(f"Unknown scheme: {scheme}")

    def memory_mb(self) -> float:
        total = 0
        for name, q in self._quant_data.items():
            if isinstance(q, tuple):
                total += sum(np.array(x).nbytes for x in q)
            else:
                total += np.array(q).nbytes
        return total / (1024 ** 2)

    def compression_ratio(self, model: Any) -> float:
        orig_bytes = sum(
            (p._data if hasattr(p, '_data') else np.array(p)).nbytes
            for p in model.parameters()
        )
        return orig_bytes / (self.memory_mb() * 1024 ** 2 + 1e-9)

    def __repr__(self):
        return (f"QuantizedModel(original={self._orig_class}, scheme={self.scheme}, "
                f"layers={len(self._quant_data)}, size={self.memory_mb():.2f}MB)")


def quantize(model: Any, scheme: str = 'int8', symmetric: bool = True,
             group_size: int = 128) -> QuantizedModel:
    """Quantize a vectorphile model to a reduced-precision format.

    Args:
        model: A vectorphile nn.Module with named_parameters().
        scheme: 'int8', 'int4', or 'fp16'.
        symmetric: Use symmetric quantization (INT8 only).
        group_size: Group size for INT4 group quantization.

    Returns:
        QuantizedModel with quantized weights.
    """
    return QuantizedModel(model, scheme=scheme, symmetric=symmetric, group_size=group_size)


# ── Pruning (bonus: related to compression) ───────────────────────────────────

def magnitude_prune(model: Any, sparsity: float = 0.5) -> Dict[str, np.ndarray]:
    """Create magnitude-based pruning masks (unstructured).

    Args:
        model: vectorphile nn.Module.
        sparsity: Fraction of weights to zero out (0.0 – 1.0).

    Returns:
        Dict mapping parameter names to binary masks (1=keep, 0=prune).
    """
    masks = {}
    for name, param in model.named_parameters():
        data = param._data if hasattr(param, '_data') else np.array(param)
        threshold = np.percentile(np.abs(data), sparsity * 100)
        mask = (np.abs(data) >= threshold).astype(np.float32)
        masks[name] = mask
    return masks


def apply_masks(model: Any, masks: Dict[str, np.ndarray]) -> None:
    """Apply pruning masks to model weights in-place."""
    for name, param in model.named_parameters():
        if name in masks:
            if hasattr(param, '_data'):
                param._data *= masks[name]
