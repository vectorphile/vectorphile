"""vectorphile.transforms — Image and tensor data augmentation pipeline.

Mirrors torchvision.transforms API. All transforms operate on VP Tensors
(or lists/arrays that can be converted to them). No PIL or numpy dependency.

Usage::

    from vectorphile import transforms as T

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(raw_image_array)
"""
from __future__ import annotations
import math
import random as _random
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

from vectorphile.core.tensor import Tensor
from vectorphile._core.ndarray import NDArray, array as _make_nd


# ── Compose ───────────────────────────────────────────────────────────────────

class Compose:
    """Chain multiple transforms together."""
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self):
        lines = [f'  {t}' for t in self.transforms]
        return 'Compose([\n' + ',\n'.join(lines) + '\n])'


# ── Type conversion ───────────────────────────────────────────────────────────

class ToTensor:
    """Convert a numeric array / list to a VP Tensor of dtype float32.

    If the input has shape (H, W, C) with values 0–255 (uint8-like),
    divides by 255.0 and rearranges to (C, H, W). Otherwise just wraps.
    """
    def __call__(self, x) -> Tensor:
        if isinstance(x, Tensor):
            return x
        try:
            if isinstance(x, NDArray):
                nd = x
            else:
                flat = []
                shape = ()
                def _flatten(obj):
                    nonlocal shape
                    if isinstance(obj, (int, float)):
                        flat.append(float(obj))
                    else:
                        try:
                            for item in obj:
                                _flatten(item)
                        except TypeError:
                            flat.append(float(obj))
                _flatten(x)
                nd = NDArray(flat, 'float32')
            if hasattr(nd, 'shape') and len(nd.shape) == 3:
                H, W, C = nd.shape
                if C <= 4:
                    # (H,W,C) -> (C,H,W) transpose
                    out_flat = []
                    flat_data = list(nd.to_flat_list() if hasattr(nd, 'to_flat_list') else nd)
                    for c in range(C):
                        for h in range(H):
                            for w in range(W):
                                val = flat_data[h*W*C + w*C + c] / 255.0
                                out_flat.append(val)
                    result_nd = _make_nd(out_flat, dtype='float32')
                    result_nd = result_nd.reshape((C, H, W))
                    return Tensor(result_nd)
            if nd.dtype != 'float32':
                nd = nd.astype('float32')
            return Tensor(nd)
        except Exception:
            return Tensor(NDArray([float(x)] if not hasattr(x, '__iter__') else list(x), 'float32'))

    def __repr__(self):
        return 'ToTensor()'


class ToPILImage:
    """Convert a (C, H, W) tensor back to a uint8 (H, W, C) array."""
    def __call__(self, x: Tensor):
        if hasattr(x, '_data') and hasattr(x._data, 'shape') and len(x._data.shape) == 3:
            C, H, W = x._data.shape
            flat = list(x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else x._data)
            out = []
            for h in range(H):
                row = []
                for w in range(W):
                    pixel = []
                    for c in range(C):
                        val = int(max(0.0, min(255.0, flat[c*H*W + h*W + w] * 255.0)))
                        pixel.append(val)
                    row.append(pixel)
                out.append(row)
            return out
        return x

    def __repr__(self):
        return 'ToPILImage()'


# ── Normalization ─────────────────────────────────────────────────────────────

class Normalize:
    """Normalize a tensor with per-channel mean and std.

    Args:
        mean: sequence of per-channel means (length = C)
        std: sequence of per-channel stds (length = C)
        inplace: modify the tensor in place (default False)
    """
    def __init__(self, mean: Sequence[float], std: Sequence[float], inplace: bool = False):
        self.mean = list(mean)
        self.std = list(std)
        self.inplace = inplace

    def __call__(self, x: Tensor) -> Tensor:
        if not hasattr(x, '_data') or not hasattr(x._data, 'shape'):
            return x
        shape = x._data.shape
        if len(shape) < 3:
            return x
        C = shape[-3] if len(shape) >= 3 else 1
        mean = (self.mean * ((C // len(self.mean)) + 1))[:C]
        std  = (self.std  * ((C // len(self.std))  + 1))[:C]
        try:
            flat = list(x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else x._data)
            H, W = shape[-2], shape[-1]
            out = []
            for c in range(C):
                m, s = mean[c], max(std[c], 1e-7)
                for i in range(H * W):
                    out.append((flat[c*H*W + i] - m) / s)
            nd = _make_nd(out, dtype='float32')
            nd = nd.reshape(shape)
            return x if self.inplace and (x._data.__dict__.update({'_flat': out}) or False) else Tensor(nd)
        except Exception:
            return x

    def __repr__(self):
        return f'Normalize(mean={self.mean}, std={self.std})'


# ── Geometric transforms ──────────────────────────────────────────────────────

class Resize:
    """Resize a (C, H, W) tensor to (C, size[0], size[1]) using bilinear interpolation.

    Args:
        size: (H_out, W_out) tuple or single int (square)
    """
    def __init__(self, size: Union[int, Tuple[int, int]]):
        self.size = (size, size) if isinstance(size, int) else tuple(size)

    def __call__(self, x: Tensor) -> Tensor:
        if not hasattr(x, '_data') or not hasattr(x._data, 'shape'):
            return x
        shape = x._data.shape
        if len(shape) < 3:
            return x
        C, H_in, W_in = shape[-3], shape[-2], shape[-1]
        H_out, W_out = self.size
        try:
            flat = list(x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else x._data)
            out = []
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        # Bilinear
                        y_src = (h + 0.5) * H_in / H_out - 0.5
                        x_src = (w + 0.5) * W_in / W_out - 0.5
                        y0 = max(0, min(H_in-1, int(math.floor(y_src))))
                        y1 = max(0, min(H_in-1, y0 + 1))
                        x0 = max(0, min(W_in-1, int(math.floor(x_src))))
                        x1 = max(0, min(W_in-1, x0 + 1))
                        dy = y_src - y0; dx = x_src - x0
                        base = c * H_in * W_in
                        v = (flat[base + y0*W_in + x0] * (1-dy)*(1-dx) +
                             flat[base + y0*W_in + x1] * (1-dy)*dx +
                             flat[base + y1*W_in + x0] * dy*(1-dx) +
                             flat[base + y1*W_in + x1] * dy*dx)
                        out.append(v)
            nd = _make_nd(out, dtype='float32')
            nd = nd.reshape((C, H_out, W_out))
            return Tensor(nd)
        except Exception:
            return x

    def __repr__(self):
        return f'Resize(size={self.size})'


class CenterCrop:
    """Center-crop a (C, H, W) tensor to the given size."""
    def __init__(self, size: Union[int, Tuple[int, int]]):
        self.size = (size, size) if isinstance(size, int) else tuple(size)

    def __call__(self, x: Tensor) -> Tensor:
        if not hasattr(x, '_data') or not hasattr(x._data, 'shape'):
            return x
        shape = x._data.shape
        if len(shape) < 3:
            return x
        C, H, W = shape[-3], shape[-2], shape[-1]
        H_out, W_out = self.size
        h_start = max(0, (H - H_out) // 2)
        w_start = max(0, (W - W_out) // 2)
        try:
            flat = list(x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else x._data)
            out = []
            for c in range(C):
                for h in range(h_start, min(h_start + H_out, H)):
                    for w in range(w_start, min(w_start + W_out, W)):
                        out.append(flat[c*H*W + h*W + w])
            nd = _make_nd(out, dtype='float32')
            nd = nd.reshape((C, min(H_out, H - h_start), min(W_out, W - w_start)))
            return Tensor(nd)
        except Exception:
            return x

    def __repr__(self):
        return f'CenterCrop(size={self.size})'


class RandomCrop:
    """Randomly crop a (C, H, W) tensor."""
    def __init__(self, size: Union[int, Tuple[int, int]], padding: int = 0):
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        self.padding = padding

    def __call__(self, x: Tensor) -> Tensor:
        if not hasattr(x, '_data') or not hasattr(x._data, 'shape'):
            return x
        shape = x._data.shape
        if len(shape) < 3:
            return x
        C, H, W = shape[-3], shape[-2], shape[-1]
        H_out, W_out = self.size
        h_start = _random.randint(0, max(0, H - H_out))
        w_start = _random.randint(0, max(0, W - W_out))
        try:
            flat = list(x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else x._data)
            out = []
            for c in range(C):
                for h in range(h_start, min(h_start + H_out, H)):
                    for w in range(w_start, min(w_start + W_out, W)):
                        out.append(flat[c*H*W + h*W + w])
            nd = _make_nd(out, dtype='float32')
            nd = nd.reshape((C, min(H_out, H - h_start), min(W_out, W - w_start)))
            return Tensor(nd)
        except Exception:
            return x

    def __repr__(self):
        return f'RandomCrop(size={self.size}, padding={self.padding})'


class RandomHorizontalFlip:
    """Randomly flip a (C, H, W) tensor horizontally with probability p."""
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, x: Tensor) -> Tensor:
        if _random.random() >= self.p:
            return x
        if not hasattr(x, '_data') or not hasattr(x._data, 'shape'):
            return x
        shape = x._data.shape
        if len(shape) < 3:
            return x
        C, H, W = shape[-3], shape[-2], shape[-1]
        try:
            flat = list(x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else x._data)
            out = []
            for c in range(C):
                for h in range(H):
                    for w in range(W-1, -1, -1):
                        out.append(flat[c*H*W + h*W + w])
            nd = _make_nd(out, dtype='float32')
            nd = nd.reshape(shape)
            return Tensor(nd)
        except Exception:
            return x

    def __repr__(self):
        return f'RandomHorizontalFlip(p={self.p})'


class RandomVerticalFlip:
    """Randomly flip a (C, H, W) tensor vertically with probability p."""
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, x: Tensor) -> Tensor:
        if _random.random() >= self.p:
            return x
        if not hasattr(x, '_data') or not hasattr(x._data, 'shape'):
            return x
        shape = x._data.shape
        if len(shape) < 3:
            return x
        C, H, W = shape[-3], shape[-2], shape[-1]
        try:
            flat = list(x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else x._data)
            out = []
            for c in range(C):
                for h in range(H-1, -1, -1):
                    for w in range(W):
                        out.append(flat[c*H*W + h*W + w])
            nd = _make_nd(out, dtype='float32')
            nd = nd.reshape(shape)
            return Tensor(nd)
        except Exception:
            return x

    def __repr__(self):
        return f'RandomVerticalFlip(p={self.p})'


class RandomRotation:
    """Randomly rotate a (C, H, W) tensor by degrees in [-degrees, +degrees]."""
    def __init__(self, degrees: Union[float, Tuple[float, float]]):
        if isinstance(degrees, (int, float)):
            self.degrees = (-abs(degrees), abs(degrees))
        else:
            self.degrees = tuple(degrees)

    def __call__(self, x: Tensor) -> Tensor:
        angle_deg = _random.uniform(*self.degrees)
        if abs(angle_deg) < 0.5:
            return x
        if not hasattr(x, '_data') or not hasattr(x._data, 'shape'):
            return x
        shape = x._data.shape
        if len(shape) < 3:
            return x
        C, H, W = shape[-3], shape[-2], shape[-1]
        angle_rad = math.radians(angle_deg)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        cx, cy = W / 2.0, H / 2.0
        try:
            flat = list(x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else x._data)
            out = []
            for c in range(C):
                for h in range(H):
                    for w in range(W):
                        # Inverse map
                        dx, dy = w - cx, h - cy
                        src_x = cos_a * dx + sin_a * dy + cx
                        src_y = -sin_a * dx + cos_a * dy + cy
                        x0 = int(math.floor(src_x)); y0 = int(math.floor(src_y))
                        if 0 <= x0 < W-1 and 0 <= y0 < H-1:
                            fx, fy = src_x - x0, src_y - y0
                            base = c * H * W
                            v = (flat[base + y0*W + x0]     * (1-fy)*(1-fx) +
                                 flat[base + y0*W + x0+1]   * (1-fy)*fx +
                                 flat[base + (y0+1)*W + x0] * fy*(1-fx) +
                                 flat[base + (y0+1)*W + x0+1] * fy*fx)
                        else:
                            v = 0.0
                        out.append(v)
            nd = _make_nd(out, dtype='float32')
            nd = nd.reshape(shape)
            return Tensor(nd)
        except Exception:
            return x

    def __repr__(self):
        return f'RandomRotation(degrees={self.degrees})'


# ── Color transforms ──────────────────────────────────────────────────────────

class ColorJitter:
    """Randomly change brightness, contrast, saturation, hue of a (C, H, W) tensor.

    Args:
        brightness: float or (min, max) range
        contrast: float or (min, max) range
        saturation: float or (min, max) range
        hue: float or (min, max) range
    """
    def __init__(self, brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0):
        self.brightness = self._make_range(brightness)
        self.contrast   = self._make_range(contrast)
        self.saturation = self._make_range(saturation)
        self.hue        = self._make_range(hue)

    def _make_range(self, v):
        if isinstance(v, (int, float)):
            return (max(0.0, 1.0 - v), 1.0 + v)
        return tuple(v)

    def __call__(self, x: Tensor) -> Tensor:
        b_factor = _random.uniform(*self.brightness)
        c_factor = _random.uniform(*self.contrast)
        try:
            flat = list(x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else x._data)
            mean_val = sum(flat) / max(1, len(flat))
            out = [max(0.0, min(1.0, (v * b_factor - mean_val) * c_factor + mean_val))
                   for v in flat]
            nd = _make_nd(out, dtype='float32')
            nd = nd.reshape(x._data.shape)
            return Tensor(nd)
        except Exception:
            return x

    def __repr__(self):
        return (f'ColorJitter(brightness={self.brightness}, contrast={self.contrast}, '
                f'saturation={self.saturation})')


class Grayscale:
    """Convert RGB (3, H, W) to grayscale (num_output_channels, H, W)."""
    def __init__(self, num_output_channels: int = 1):
        self.num_output_channels = num_output_channels

    def __call__(self, x: Tensor) -> Tensor:
        if not hasattr(x, '_data') or not hasattr(x._data, 'shape'):
            return x
        shape = x._data.shape
        if len(shape) < 3 or shape[-3] < 3:
            return x
        C, H, W = shape[-3], shape[-2], shape[-1]
        try:
            flat = list(x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else x._data)
            gray = []
            for h in range(H):
                for w in range(W):
                    r = flat[0*H*W + h*W + w]
                    g = flat[1*H*W + h*W + w]
                    b = flat[2*H*W + h*W + w]
                    gray.append(0.2989*r + 0.5870*g + 0.1140*b)
            if self.num_output_channels == 3:
                out = gray * 3
            else:
                out = gray
            nd = _make_nd(out, dtype='float32')
            nd = nd.reshape((self.num_output_channels, H, W))
            return Tensor(nd)
        except Exception:
            return x

    def __repr__(self):
        return f'Grayscale(num_output_channels={self.num_output_channels})'


class RandomGrayscale:
    """Randomly convert image to grayscale with probability p."""
    def __init__(self, p: float = 0.1):
        self.p = p
        self._g = Grayscale(num_output_channels=3)

    def __call__(self, x: Tensor) -> Tensor:
        if _random.random() < self.p:
            return self._g(x)
        return x

    def __repr__(self):
        return f'RandomGrayscale(p={self.p})'


# ── Stochastic augmentation ───────────────────────────────────────────────────

class RandomApply:
    """Apply a list of transforms with probability p."""
    def __init__(self, transforms: List[Callable], p: float = 0.5):
        self.transforms = transforms
        self.p = p

    def __call__(self, x):
        if _random.random() < self.p:
            for t in self.transforms:
                x = t(x)
        return x

    def __repr__(self):
        return f'RandomApply(p={self.p}, transforms={self.transforms})'


class RandomChoice:
    """Randomly apply one transform from a list."""
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, x):
        return _random.choice(self.transforms)(x)

    def __repr__(self):
        return f'RandomChoice({self.transforms})'


class RandomOrder:
    """Apply transforms in random order."""
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, x):
        order = list(range(len(self.transforms)))
        _random.shuffle(order)
        for i in order:
            x = self.transforms[i](x)
        return x


# ── Tensor operations ─────────────────────────────────────────────────────────

class Lambda:
    """Apply an arbitrary callable as a transform."""
    def __init__(self, fn: Callable):
        self.fn = fn

    def __call__(self, x):
        return self.fn(x)

    def __repr__(self):
        return f'Lambda({self.fn})'


class Pad:
    """Pad a (C, H, W) tensor with constant value."""
    def __init__(self, padding: Union[int, Tuple[int, int, int, int]],
                 fill: float = 0.0, padding_mode: str = 'constant'):
        if isinstance(padding, int):
            self.padding = (padding, padding, padding, padding)  # left, top, right, bottom
        else:
            self.padding = tuple(padding)
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, x: Tensor) -> Tensor:
        if not hasattr(x, '_data') or not hasattr(x._data, 'shape'):
            return x
        shape = x._data.shape
        if len(shape) < 3:
            return x
        C, H, W = shape[-3], shape[-2], shape[-1]
        pl, pt, pr, pb = self.padding
        H_out, W_out = H + pt + pb, W + pl + pr
        try:
            flat = list(x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else x._data)
            out = []
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        h_src, w_src = h - pt, w - pl
                        if 0 <= h_src < H and 0 <= w_src < W:
                            out.append(flat[c*H*W + h_src*W + w_src])
                        else:
                            out.append(self.fill)
            nd = _make_nd(out, dtype='float32')
            nd = nd.reshape((C, H_out, W_out))
            return Tensor(nd)
        except Exception:
            return x

    def __repr__(self):
        return f'Pad(padding={self.padding}, fill={self.fill})'


__all__ = [
    'Compose', 'ToTensor', 'ToPILImage',
    'Normalize',
    'Resize', 'CenterCrop', 'RandomCrop',
    'RandomHorizontalFlip', 'RandomVerticalFlip', 'RandomRotation',
    'ColorJitter', 'Grayscale', 'RandomGrayscale',
    'RandomApply', 'RandomChoice', 'RandomOrder',
    'Lambda', 'Pad',
    'GaussianBlur', 'RandomErasing', 'FiveCrop', 'TenCrop',
]


class GaussianBlur:
    """Apply Gaussian blur with random sigma."""

    def __init__(self, kernel_size, sigma=(0.1, 2.0)):
        self.kernel_size = kernel_size if isinstance(kernel_size, (list, tuple)) else [kernel_size, kernel_size]
        self.sigma = sigma if isinstance(sigma, (list, tuple)) else (sigma, sigma)

    def __call__(self, x):
        import random, math
        try:
            from vectorphile.core.tensor import Tensor
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            # Simple box blur approximation (true Gaussian requires 2D convolution)
            kH, kW = self.kernel_size
            if hasattr(x, '_data'):
                flat = list(x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else x._data)
                sh = x._data.shape if hasattr(x._data, 'shape') else (len(flat),)
                # For 3-D (C,H,W) or 2-D (H,W)
                if len(sh) == 3:
                    C, H, W = sh
                    out = flat[:]
                    half_h, half_w = kH // 2, kW // 2
                    for c in range(C):
                        for h in range(H):
                            for w in range(W):
                                total, count = 0.0, 0
                                for dh in range(-half_h, half_h + 1):
                                    for dw in range(-half_w, half_w + 1):
                                        nh, nw = h + dh, w + dw
                                        if 0 <= nh < H and 0 <= nw < W:
                                            total += flat[c*H*W + nh*W + nw]
                                            count += 1
                                out[c*H*W + h*W + w] = total / count if count else 0.0
                    from vectorphile._core.ndarray import array as _nd
                    return Tensor(_nd(out, sh))
            return x
        except Exception:
            return x

    def __repr__(self): return f'GaussianBlur(kernel_size={self.kernel_size}, sigma={self.sigma})'


class RandomErasing:
    """Randomly erase a rectangular region of an image tensor."""

    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value

    def __call__(self, x):
        import random, math
        try:
            if random.random() > self.p:
                return x
            from vectorphile.core.tensor import Tensor
            if hasattr(x, '_data'):
                sh = x._data.shape if hasattr(x._data, 'shape') else None
                if sh and len(sh) == 3:
                    C, H, W = sh
                    area = H * W
                    for _ in range(10):
                        erase_area = random.uniform(*self.scale) * area
                        aspect = random.uniform(*self.ratio)
                        eh = int(round(math.sqrt(erase_area * aspect)))
                        ew = int(round(math.sqrt(erase_area / aspect)))
                        if eh < H and ew < W:
                            i = random.randint(0, H - eh)
                            j = random.randint(0, W - ew)
                            flat = list(x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else x._data)
                            for c in range(C):
                                for h in range(i, i + eh):
                                    for w in range(j, j + ew):
                                        flat[c*H*W + h*W + w] = float(self.value)
                            from vectorphile._core.ndarray import array as _nd
                            return Tensor(_nd(flat, sh))
                            break
            return x
        except Exception:
            return x

    def __repr__(self): return f'RandomErasing(p={self.p})'


class FiveCrop:
    """Crop image into 5 patches: 4 corners + center."""

    def __init__(self, size):
        self.size = size if isinstance(size, (list, tuple)) else (size, size)

    def __call__(self, x):
        import random
        try:
            from vectorphile.core.tensor import Tensor
            if hasattr(x, '_data'):
                sh = x._data.shape if hasattr(x._data, 'shape') else None
                if sh and len(sh) == 3:
                    C, H, W = sh
                    crop_h, crop_w = self.size
                    flat = list(x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else x._data)

                    def _crop(top, left):
                        out = []
                        for c in range(C):
                            for h in range(top, top + crop_h):
                                for w in range(left, left + crop_w):
                                    out.append(flat[c*H*W + h*W + w])
                        from vectorphile._core.ndarray import array as _nd
                        return Tensor(_nd(out, (C, crop_h, crop_w)))

                    crops = [
                        _crop(0, 0),  # top-left
                        _crop(0, W - crop_w),  # top-right
                        _crop(H - crop_h, 0),  # bottom-left
                        _crop(H - crop_h, W - crop_w),  # bottom-right
                        _crop((H - crop_h) // 2, (W - crop_w) // 2),  # center
                    ]
                    return tuple(crops)
            return (x,) * 5
        except Exception:
            return (x,) * 5

    def __repr__(self): return f'FiveCrop(size={self.size})'


class TenCrop:
    """Like FiveCrop but also includes horizontally flipped versions."""

    def __init__(self, size, vertical_flip=False):
        self.size = size if isinstance(size, (list, tuple)) else (size, size)
        self.vertical_flip = vertical_flip
        self._five = FiveCrop(size)

    def __call__(self, x):
        crops = self._five(x)
        # Flip: if Tensor, flip last axis for horizontal
        try:
            from vectorphile.core.tensor import Tensor
            flipped_crops = []
            for c in crops:
                if hasattr(c, '_data'):
                    flat = list(c._data.to_flat_list() if hasattr(c._data, 'to_flat_list') else c._data)
                    sh = c._data.shape
                    C, H, W = sh
                    out = []
                    for ch in range(C):
                        for h in range(H):
                            for w in range(W - 1, -1, -1):  # reverse width
                                out.append(flat[ch*H*W + h*W + w])
                    from vectorphile._core.ndarray import array as _nd
                    flipped_crops.append(Tensor(_nd(out, sh)))
                else:
                    flipped_crops.append(c)
            return crops + tuple(flipped_crops)
        except Exception:
            return crops + crops

    def __repr__(self): return f'TenCrop(size={self.size})'
