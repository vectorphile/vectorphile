"""N-dimensional image processing — pure vectorphile, no scipy.

All functions operate on vectorphile Tensors and use only VP's own NDArray.
"""
from __future__ import annotations
import math
from typing import Union, Tuple, Optional
from vectorphile.core.tensor import Tensor
from vectorphile._core.ndarray import array as _nd


def _flat(t):
    if isinstance(t, Tensor):
        d = t._data
    else:
        d = t
    return list(d.to_flat_list() if hasattr(d, 'to_flat_list') else d)


def _make(vals, shape):
    nd = _nd(vals, dtype='float32')
    nd = nd.reshape(shape)
    return Tensor(nd)


def _shape(t):
    if isinstance(t, Tensor):
        return t._data.shape
    return t.shape


# ── Gaussian kernel ───────────────────────────────────────────────────────────

def _gaussian_kernel_1d(sigma: float, radius: int) -> list:
    """Return 1D Gaussian kernel values (unnormalized sum to 1)."""
    k = [math.exp(-0.5 * (i ** 2) / (sigma ** 2)) for i in range(-radius, radius + 1)]
    s = sum(k)
    return [v / s for v in k]


def gaussian_filter(input: Tensor, sigma: float, order: int = 0,
                    mode: str = 'reflect', cval: float = 0.0, truncate: float = 4.0) -> Tensor:
    """Apply Gaussian filter to an N-D Tensor (pure VP)."""
    shape = _shape(input)
    ndim = len(shape)
    radius = int(truncate * sigma + 0.5)
    kernel = _gaussian_kernel_1d(sigma, radius)
    result = input
    for axis in range(ndim):
        result = _convolve_axis(result, kernel, axis, mode, cval)
    return result


def gaussian_filter1d(input: Tensor, sigma: float, axis: int = -1,
                      order: int = 0, mode: str = 'reflect',
                      cval: float = 0.0, truncate: float = 4.0) -> Tensor:
    """Apply 1D Gaussian filter along one axis."""
    shape = _shape(input)
    ndim = len(shape)
    if axis < 0:
        axis = ndim + axis
    radius = int(truncate * sigma + 0.5)
    kernel = _gaussian_kernel_1d(sigma, radius)
    return _convolve_axis(input, kernel, axis, mode, cval)


def _convolve_axis(t: Tensor, kernel: list, axis: int, mode: str, cval: float) -> Tensor:
    """Convolve tensor along a single axis with a 1D kernel."""
    shape = _shape(t)
    flat = _flat(t)
    ndim = len(shape)
    radius = len(kernel) // 2
    n = shape[axis]

    # strides in flat array
    strides = [1] * ndim
    for i in range(ndim - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]

    out = []
    total = 1
    for s in shape:
        total *= s

    def get_val(indices):
        idx = indices[axis]
        if idx < 0 or idx >= n:
            if mode == 'constant':
                return cval
            elif mode == 'reflect':
                if idx < 0:
                    idx = -idx
                else:
                    idx = 2 * n - idx - 2
                idx = max(0, min(n - 1, idx))
            elif mode == 'nearest':
                idx = max(0, min(n - 1, idx))
            elif mode == 'wrap':
                idx = idx % n
            else:
                idx = max(0, min(n - 1, idx))
        new_indices = list(indices)
        new_indices[axis] = idx
        pos = sum(new_indices[i] * strides[i] for i in range(ndim))
        return flat[pos]

    for linear in range(total):
        rem = linear
        indices = []
        for s in reversed(shape):
            indices.append(rem % s)
            rem //= s
        indices.reverse()
        val = sum(kernel[k] * get_val([*indices[:axis], indices[axis] + (k - radius), *indices[axis+1:]]) for k in range(len(kernel)))
        out.append(val)

    return _make(out, shape)


def uniform_filter(input: Tensor, size: int = 3, mode: str = 'reflect', cval: float = 0.0) -> Tensor:
    """Uniform (mean) filter over all axes."""
    shape = _shape(input)
    ndim = len(shape)
    kernel = [1.0 / size] * size
    result = input
    for axis in range(ndim):
        result = _convolve_axis(result, kernel, axis, mode, cval)
    return result


def uniform_filter1d(input: Tensor, size: int, axis: int = -1,
                     mode: str = 'reflect', cval: float = 0.0) -> Tensor:
    """Uniform filter along one axis."""
    shape = _shape(input)
    ndim = len(shape)
    if axis < 0:
        axis = ndim + axis
    kernel = [1.0 / size] * size
    return _convolve_axis(input, kernel, axis, mode, cval)


def median_filter(input: Tensor, size: int = 3, mode: str = 'reflect', cval: float = 0.0) -> Tensor:
    """Median filter — 1D only along each axis sequentially."""
    shape = _shape(input)
    ndim = len(shape)
    flat = _flat(input)
    radius = size // 2

    strides = [1] * ndim
    for i in range(ndim - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]

    total = 1
    for s in shape:
        total *= s

    out = []
    for linear in range(total):
        rem = linear
        indices = []
        for s in reversed(shape):
            indices.append(rem % s)
            rem //= s
        indices.reverse()
        window = []
        # gather window for each axis
        for axis in range(ndim):
            n = shape[axis]
        # simplified: gather a 1D window along last axis only
        axis = ndim - 1
        n = shape[axis]
        for k in range(-radius, radius + 1):
            idx = indices[axis] + k
            if idx < 0 or idx >= n:
                window.append(cval if mode == 'constant' else flat[linear])
            else:
                new_indices = list(indices)
                new_indices[axis] = idx
                pos = sum(new_indices[i] * strides[i] for i in range(ndim))
                window.append(flat[pos])
        window.sort()
        out.append(window[len(window) // 2])

    return _make(out, shape)


def maximum_filter(input: Tensor, size: int = 3, mode: str = 'reflect', cval: float = 0.0) -> Tensor:
    """Maximum filter over a sliding window."""
    shape = _shape(input)
    ndim = len(shape)
    flat = _flat(input)
    radius = size // 2

    strides = [1] * ndim
    for i in range(ndim - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]

    total = 1
    for s in shape:
        total *= s

    out = []
    for linear in range(total):
        rem = linear
        indices = []
        for s in reversed(shape):
            indices.append(rem % s)
            rem //= s
        indices.reverse()
        best = None
        for axis in range(ndim):
            n = shape[axis]
            for k in range(-radius, radius + 1):
                idx = indices[axis] + k
                if idx < 0 or idx >= n:
                    v = cval if mode == 'constant' else flat[linear]
                else:
                    new_indices = list(indices)
                    new_indices[axis] = idx
                    pos = sum(new_indices[i] * strides[i] for i in range(ndim))
                    v = flat[pos]
                if best is None or v > best:
                    best = v
        out.append(best if best is not None else 0.0)

    return _make(out, shape)


def minimum_filter(input: Tensor, size: int = 3, mode: str = 'reflect', cval: float = 0.0) -> Tensor:
    """Minimum filter over a sliding window."""
    shape = _shape(input)
    ndim = len(shape)
    flat = _flat(input)
    radius = size // 2

    strides = [1] * ndim
    for i in range(ndim - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]

    total = 1
    for s in shape:
        total *= s

    out = []
    for linear in range(total):
        rem = linear
        indices = []
        for s in reversed(shape):
            indices.append(rem % s)
            rem //= s
        indices.reverse()
        best = None
        for axis in range(ndim):
            n = shape[axis]
            for k in range(-radius, radius + 1):
                idx = indices[axis] + k
                if idx < 0 or idx >= n:
                    v = cval if mode == 'constant' else flat[linear]
                else:
                    new_indices = list(indices)
                    new_indices[axis] = idx
                    pos = sum(new_indices[i] * strides[i] for i in range(ndim))
                    v = flat[pos]
                if best is None or v < best:
                    best = v
        out.append(best if best is not None else 0.0)

    return _make(out, shape)


def laplace(input: Tensor, mode: str = 'reflect', cval: float = 0.0) -> Tensor:
    """Laplace filter (sum of second derivatives along each axis)."""
    shape = _shape(input)
    ndim = len(shape)
    kernel = [-1.0, 2.0, -1.0]
    result = None
    for axis in range(ndim):
        contrib = _convolve_axis(input, kernel, axis, mode, cval)
        flat_c = _flat(contrib)
        if result is None:
            result = flat_c
        else:
            result = [a + b for a, b in zip(result, flat_c)]
    # subtract 2*ndim*input to get true Laplacian
    flat_in = _flat(input)
    result = [r - 2.0 * ndim * v for r, v in zip(result, flat_in)]
    return _make(result, shape)


def sobel(input: Tensor, axis: int = -1, mode: str = 'reflect', cval: float = 0.0) -> Tensor:
    """Sobel edge detection filter along one axis."""
    shape = _shape(input)
    ndim = len(shape)
    if axis < 0:
        axis = ndim + axis
    # Sobel = [-1, 0, 1] along target axis, [1, 2, 1] along others
    result = _convolve_axis(input, [-1.0, 0.0, 1.0], axis, mode, cval)
    for ax in range(ndim):
        if ax != axis:
            result = _convolve_axis(result, [1.0, 2.0, 1.0], ax, mode, cval)
    return result


def prewitt(input: Tensor, axis: int = -1, mode: str = 'reflect', cval: float = 0.0) -> Tensor:
    """Prewitt edge detection filter."""
    shape = _shape(input)
    ndim = len(shape)
    if axis < 0:
        axis = ndim + axis
    result = _convolve_axis(input, [-1.0, 0.0, 1.0], axis, mode, cval)
    for ax in range(ndim):
        if ax != axis:
            result = _convolve_axis(result, [1.0, 1.0, 1.0], ax, mode, cval)
    return result


def convolve(input: Tensor, weights: Tensor, mode: str = 'reflect', cval: float = 0.0) -> Tensor:
    """N-D convolution with arbitrary kernel tensor."""
    shape = _shape(input)
    kshape = _shape(weights)
    ndim = len(shape)
    flat_in = _flat(input)
    flat_k = _flat(weights)

    strides_in = [1] * ndim
    for i in range(ndim - 2, -1, -1):
        strides_in[i] = strides_in[i + 1] * shape[i + 1]

    total = 1
    for s in shape:
        total *= s

    radii = [ks // 2 for ks in kshape]
    strides_k = [1] * ndim
    for i in range(ndim - 2, -1, -1):
        strides_k[i] = strides_k[i + 1] * kshape[i + 1]

    out = []
    for linear in range(total):
        rem = linear
        indices = []
        for s in reversed(shape):
            indices.append(rem % s)
            rem //= s
        indices.reverse()
        val = 0.0
        import itertools
        for kidx in itertools.product(*[range(ks) for ks in kshape]):
            kpos = sum(kidx[i] * strides_k[i] for i in range(ndim))
            kw = flat_k[kpos]
            src = []
            valid = True
            for i in range(ndim):
                s_idx = indices[i] + kidx[i] - radii[i]
                n = shape[i]
                if s_idx < 0 or s_idx >= n:
                    if mode == 'constant':
                        kw_contrib = kw * cval
                        val += kw_contrib
                        valid = False
                        break
                    elif mode == 'reflect':
                        if s_idx < 0:
                            s_idx = -s_idx
                        else:
                            s_idx = 2 * n - s_idx - 2
                        s_idx = max(0, min(n - 1, s_idx))
                    elif mode == 'nearest':
                        s_idx = max(0, min(n - 1, s_idx))
                    elif mode == 'wrap':
                        s_idx = s_idx % n
                src.append(s_idx)
            else:
                pos = sum(src[i] * strides_in[i] for i in range(ndim))
                val += kw * flat_in[pos]
        out.append(val)

    return _make(out, shape)


def zoom(input: Tensor, zoom_factor: float, mode: str = 'nearest') -> Tensor:
    """Zoom (rescale) an N-D tensor."""
    shape = _shape(input)
    flat = _flat(input)
    ndim = len(shape)
    new_shape = tuple(max(1, int(round(s * zoom_factor))) for s in shape)

    strides = [1] * ndim
    for i in range(ndim - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]

    total_new = 1
    for s in new_shape:
        total_new *= s

    out = []
    for linear in range(total_new):
        rem = linear
        new_idx = []
        for s in reversed(new_shape):
            new_idx.append(rem % s)
            rem //= s
        new_idx.reverse()
        src = []
        for i in range(ndim):
            fi = new_idx[i] / zoom_factor
            si = int(round(fi)) if mode == 'nearest' else int(fi)
            si = max(0, min(shape[i] - 1, si))
            src.append(si)
        pos = sum(src[i] * strides[i] for i in range(ndim))
        out.append(flat[pos])

    return _make(out, new_shape)


def rotate(input: Tensor, angle: float, reshape: bool = True,
           mode: str = 'constant', cval: float = 0.0) -> Tensor:
    """Rotate a 2D (or last 2 dims of N-D) tensor by angle degrees."""
    shape = _shape(input)
    if len(shape) < 2:
        return input
    H, W = shape[-2], shape[-1]
    flat = _flat(input)
    angle_rad = math.radians(angle)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    cx, cy = W / 2.0, H / 2.0

    out = []
    for h in range(H):
        for w in range(W):
            dx, dy = w - cx, h - cy
            src_x = cos_a * dx + sin_a * dy + cx
            src_y = -sin_a * dx + cos_a * dy + cy
            x0 = int(math.floor(src_x))
            y0 = int(math.floor(src_y))
            if 0 <= x0 < W - 1 and 0 <= y0 < H - 1:
                fx, fy = src_x - x0, src_y - y0
                v = (flat[y0 * W + x0]       * (1 - fy) * (1 - fx) +
                     flat[y0 * W + x0 + 1]   * (1 - fy) * fx +
                     flat[(y0+1)*W + x0]      * fy * (1 - fx) +
                     flat[(y0+1)*W + x0 + 1]  * fy * fx)
            else:
                v = cval
            out.append(v)

    return _make(out, (H, W))


def label(input: Tensor, structure=None) -> tuple:
    """Label connected components in a binary 2D tensor."""
    shape = _shape(input)
    if len(shape) != 2:
        raise ValueError("label currently supports 2D tensors only")
    H, W = shape
    flat = _flat(input)
    labels = [0] * (H * W)
    current = 0
    for h in range(H):
        for w in range(W):
            if flat[h * W + w] != 0 and labels[h * W + w] == 0:
                current += 1
                # BFS flood fill
                queue = [(h, w)]
                while queue:
                    ch, cw = queue.pop()
                    if ch < 0 or ch >= H or cw < 0 or cw >= W:
                        continue
                    if labels[ch * W + cw] != 0 or flat[ch * W + cw] == 0:
                        continue
                    labels[ch * W + cw] = current
                    queue.extend([(ch+1, cw), (ch-1, cw), (ch, cw+1), (ch, cw-1)])
    nd = _nd([float(v) for v in labels], dtype='float32').reshape((H, W))
    return Tensor(nd), current


def binary_erosion(input: Tensor, structure=None, iterations: int = 1) -> Tensor:
    """Binary erosion using a 3×3 structuring element (2D)."""
    shape = _shape(input)
    if len(shape) != 2:
        raise ValueError("binary_erosion currently supports 2D tensors only")
    H, W = shape
    flat = _flat(input)
    for _ in range(iterations):
        out = []
        for h in range(H):
            for w in range(W):
                val = 1
                for dh in [-1, 0, 1]:
                    for dw in [-1, 0, 1]:
                        nh, nw = h + dh, w + dw
                        if nh < 0 or nh >= H or nw < 0 or nw >= W:
                            val = 0
                        elif flat[nh * W + nw] == 0:
                            val = 0
                out.append(float(val))
        flat = out
    nd = _nd(flat, dtype='float32').reshape((H, W))
    return Tensor(nd)


def binary_dilation(input: Tensor, structure=None, iterations: int = 1) -> Tensor:
    """Binary dilation using a 3×3 structuring element (2D)."""
    shape = _shape(input)
    if len(shape) != 2:
        raise ValueError("binary_dilation currently supports 2D tensors only")
    H, W = shape
    flat = _flat(input)
    for _ in range(iterations):
        out = []
        for h in range(H):
            for w in range(W):
                val = 0
                for dh in [-1, 0, 1]:
                    for dw in [-1, 0, 1]:
                        nh, nw = h + dh, w + dw
                        if 0 <= nh < H and 0 <= nw < W:
                            if flat[nh * W + nw] != 0:
                                val = 1
                                break
                    if val:
                        break
                out.append(float(val))
        flat = out
    nd = _nd(flat, dtype='float32').reshape((H, W))
    return Tensor(nd)


def binary_opening(input: Tensor, structure=None, iterations: int = 1) -> Tensor:
    """Binary opening = erosion then dilation."""
    return binary_dilation(binary_erosion(input, structure, iterations), structure, iterations)


def binary_closing(input: Tensor, structure=None, iterations: int = 1) -> Tensor:
    """Binary closing = dilation then erosion."""
    return binary_erosion(binary_dilation(input, structure, iterations), structure, iterations)


__all__ = [
    'gaussian_filter', 'gaussian_filter1d',
    'uniform_filter', 'uniform_filter1d',
    'median_filter', 'maximum_filter', 'minimum_filter',
    'laplace', 'sobel', 'prewitt',
    'convolve', 'zoom', 'rotate',
    'label', 'binary_erosion', 'binary_dilation',
    'binary_opening', 'binary_closing',
]
