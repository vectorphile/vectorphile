"""Vision transforms — works with numpy arrays, NDArray, and nested lists.

For compatibility, when input is a numpy array, output is also a numpy array.
When input is NDArray, output is NDArray.
"""
import math
import random
import random as _rng
from typing import List


def _is_numpy(img):
    t = type(img)
    return t.__module__ == 'numpy' or (hasattr(img, 'flatten') and hasattr(img, 'dtype') and not hasattr(img, 'to_flat_list'))


def _make_out(C, H, W):
    return [[[0.0] * W for _ in range(H)] for _ in range(C)]


def _to_list3(img):
    """Convert to nested list [C][H][W]."""
    if _is_numpy(img):
        C, H, W = img.shape
        return [[[float(img[c, h, w]) for w in range(W)] for h in range(H)] for c in range(C)]
    if hasattr(img, 'to_flat_list') and hasattr(img, 'shape'):
        sh = img.shape; C, H, W = sh
        flat = list(img.to_flat_list())
        return [[[float(flat[c * H * W + h * W + w]) for w in range(W)] for h in range(H)] for c in range(C)]
    return img  # already a list


def _from_list3(lst, orig_img):
    """Convert nested list back to same type as orig_img."""
    C = len(lst); H = len(lst[0]); W = len(lst[0][0])
    if _is_numpy(orig_img):
        import numpy as _np
        arr = _np.empty((C, H, W), dtype=_np.float32)
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    arr[c, h, w] = lst[c][h][w]
        return arr
    if hasattr(orig_img, 'to_flat_list') and hasattr(orig_img, 'shape'):
        from vectorphile._core.ndarray import NDArray, _c_strides, _DType
        import array as _arr
        flat = [lst[c][h][w] for c in range(C) for h in range(H) for w in range(W)]
        buf = _arr.array('f', [float(v) for v in flat])
        sh = (C, H, W)
        return NDArray(buf, sh, _c_strides(sh), _DType('float32'))
    return lst


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class RandomCrop:
    """Crop a (C, H, W) image to (C, size, size)."""
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        s = self.size
        if _is_numpy(img):
            C, H, W = img.shape
            top = random.randint(0, H - s)
            left = random.randint(0, W - s)
            return img[:, top:top + s, left:left + s].copy()
        lst = _to_list3(img)
        C, H, W = len(lst), len(lst[0]), len(lst[0][0])
        top = random.randint(0, H - s)
        left = random.randint(0, W - s)
        out = [[[lst[c][top + h][left + w] for w in range(s)] for h in range(s)] for c in range(C)]
        return _from_list3(out, img)


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img):
        if random.random() >= self.p:
            return img
        if _is_numpy(img):
            return img[:, :, ::-1].copy()
        lst = _to_list3(img)
        C, H, W = len(lst), len(lst[0]), len(lst[0][0])
        out = [[[lst[c][h][W - 1 - w] for w in range(W)] for h in range(H)] for c in range(C)]
        return _from_list3(out, img)


class Normalize:
    """Channel-wise normalize: (img - mean) / std."""
    def __init__(self, mean, std):
        self.mean = list(mean)
        self.std = list(std)

    def __call__(self, img):
        if _is_numpy(img):
            import numpy as _np
            m = _np.array(self.mean, dtype=_np.float32)[:, _np.newaxis, _np.newaxis]
            s = _np.array(self.std, dtype=_np.float32)[:, _np.newaxis, _np.newaxis]
            return ((img.astype(_np.float32) - m) / s).astype(img.dtype)
        lst = _to_list3(img)
        C, H, W = len(lst), len(lst[0]), len(lst[0][0])
        m = self.mean; s = self.std
        out = [[[( lst[c][h][w] - m[c]) / s[c] for w in range(W)] for h in range(H)] for c in range(C)]
        return _from_list3(out, img)


class ColorJitter:
    """Random brightness and contrast."""
    def __init__(self, brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, img):
        if _is_numpy(img):
            import numpy as _np
            out = img.copy().astype(_np.float32)
            if self.brightness > 0:
                out = out * (1.0 + random.uniform(-self.brightness, self.brightness))
            if self.contrast > 0:
                cf = 1.0 + random.uniform(-self.contrast, self.contrast)
                mean = out.mean(axis=(1, 2), keepdims=True)
                out = mean + cf * (out - mean)
            return _np.clip(out, 0.0, 1.0).astype(img.dtype)
        lst = _to_list3(img)
        C, H, W = len(lst), len(lst[0]), len(lst[0][0])
        bf = 1.0 + random.uniform(-self.brightness, self.brightness) if self.brightness > 0 else 1.0
        if self.contrast > 0:
            cf = 1.0 + random.uniform(-self.contrast, self.contrast)
            means = [sum(lst[c][h][w] for h in range(H) for w in range(W)) / (H * W) for c in range(C)]
            out = [[[min(1.0, max(0.0, means[c] + cf * (lst[c][h][w] * bf - means[c])))
                     for w in range(W)] for h in range(H)] for c in range(C)]
        else:
            out = [[[min(1.0, max(0.0, lst[c][h][w] * bf)) for w in range(W)] for h in range(H)] for c in range(C)]
        return _from_list3(out, img)


class Resize:
    """Bilinear resize to (H_new, W_new). Input: (C, H, W)."""
    def __init__(self, size):
        if isinstance(size, int):
            self.h = self.w = size
        else:
            self.h, self.w = size

    def __call__(self, img):
        lst = _to_list3(img)
        C, H, W = len(lst), len(lst[0]), len(lst[0][0])
        h_new, w_new = self.h, self.w
        ry = [i * (H - 1) / max(1, h_new - 1) for i in range(h_new)]
        rx = [i * (W - 1) / max(1, w_new - 1) for i in range(w_new)]
        out = _make_out(C, h_new, w_new)
        for c in range(C):
            for hi, yr in enumerate(ry):
                r0 = max(0, min(H - 2, int(yr)))
                r1 = r0 + 1
                dr = yr - r0
                for wi, xr in enumerate(rx):
                    c0 = max(0, min(W - 2, int(xr)))
                    c1 = c0 + 1
                    dc = xr - c0
                    v = ((1 - dr) * (1 - dc) * lst[c][r0][c0] +
                         (1 - dr) * dc * lst[c][r0][c1] +
                         dr * (1 - dc) * lst[c][r1][c0] +
                         dr * dc * lst[c][r1][c1])
                    out[c][hi][wi] = v
        return _from_list3(out, img)


class RandomErasing:
    """Randomly erase a rectangular patch."""
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0.0):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value

    def __call__(self, img):
        if random.random() >= self.p:
            return img
        if _is_numpy(img):
            import numpy as _np
            out = img.copy()
            C, H, W = out.shape
            area = H * W
            for _ in range(10):
                er_area = area * random.uniform(*self.scale)
                r = math.exp(random.uniform(math.log(self.ratio[0]), math.log(self.ratio[1])))
                er_h = int(round(math.sqrt(er_area * r)))
                er_w = int(round(math.sqrt(er_area / r)))
                if er_h >= H or er_w >= W:
                    continue
                top = random.randint(0, H - er_h)
                left = random.randint(0, W - er_w)
                out[:, top:top + er_h, left:left + er_w] = self.value
                return out
            return out
        lst = _to_list3(img)
        C, H, W = len(lst), len(lst[0]), len(lst[0][0])
        # deep copy
        lst = [[[lst[c][h][w] for w in range(W)] for h in range(H)] for c in range(C)]
        area = H * W
        for _ in range(10):
            er_area = area * random.uniform(*self.scale)
            r = math.exp(random.uniform(math.log(self.ratio[0]), math.log(self.ratio[1])))
            er_h = int(round(math.sqrt(er_area * r)))
            er_w = int(round(math.sqrt(er_area / r)))
            if er_h >= H or er_w >= W:
                continue
            top = random.randint(0, H - er_h)
            left = random.randint(0, W - er_w)
            for c in range(C):
                for h in range(top, top + er_h):
                    for w in range(left, left + er_w):
                        lst[c][h][w] = self.value
            break
        return _from_list3(lst, img)


class GaussianBlur:
    """Separable Gaussian blur."""
    def __init__(self, kernel_size: int = 5, sigma: float = 1.0):
        self.ks = kernel_size
        self.sigma = sigma

    def _kernel1d(self):
        k = self.ks
        half = k // 2
        g = [math.exp(-(i - half) ** 2 / (2 * self.sigma ** 2)) for i in range(k)]
        total = sum(g)
        return [v / total for v in g]

    def __call__(self, img):
        lst = _to_list3(img)
        C, H, W = len(lst), len(lst[0]), len(lst[0][0])
        k = self._kernel1d()
        pad = self.ks // 2
        out = _make_out(C, H, W)
        for c in range(C):
            tmp = [[0.0] * W for _ in range(H)]
            for h in range(H):
                for w in range(W):
                    acc = 0.0
                    for ki, kv in enumerate(k):
                        ww = max(0, min(W - 1, w + ki - pad))
                        acc += kv * lst[c][h][ww]
                    tmp[h][w] = acc
            for h in range(H):
                for w in range(W):
                    acc = 0.0
                    for ki, kv in enumerate(k):
                        hh = max(0, min(H - 1, h + ki - pad))
                        acc += kv * tmp[hh][w]
                    out[c][h][w] = acc
        return _from_list3(out, img)


class ToTensor:
    """Convert HWC uint8/float image to CHW float32 NDArray in [0,1]."""
    def __call__(self, img):
        if _is_numpy(img):
            import numpy as _np
            if img.ndim == 2:
                img = img[:, :, _np.newaxis]
            arr = img.transpose(2, 0, 1).astype(_np.float32)
            if 'int' in str(img.dtype):
                arr = arr / 255.0
            return arr
        # Already NDArray or list
        return img


class CenterCrop:
    """Crop the center of an image."""
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = tuple(size)

    def __call__(self, img):
        if _is_numpy(img):
            import numpy as _np
            h, w = img.shape[:2]
            th, tw = self.size
            top = (h - th) // 2
            left = (w - tw) // 2
            return img[top:top+th, left:left+tw]
        return img


class RandomVerticalFlip:
    """Flip image vertically with probability p."""
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img):
        if _rng.random() < self.p:
            if _is_numpy(img):
                return img[::-1]
            if isinstance(img, list):
                return list(reversed(img))
        return img


class Pad:
    """Pad image with constant value."""
    def __init__(self, padding, fill=0):
        if isinstance(padding, int):
            self.padding = (padding, padding, padding, padding)
        elif len(padding) == 2:
            self.padding = (padding[0], padding[1], padding[0], padding[1])
        else:
            self.padding = tuple(padding)
        self.fill = fill

    def __call__(self, img):
        if _is_numpy(img):
            import numpy as _np
            top, bottom, left, right = self.padding
            if img.ndim == 3:
                return _np.pad(img, ((top, bottom), (left, right), (0, 0)), constant_values=self.fill)
            return _np.pad(img, ((top, bottom), (left, right)), constant_values=self.fill)
        return img


class RandomGrayscale:
    """Convert image to grayscale with probability p."""
    def __init__(self, p: float = 0.1):
        self.p = p

    def __call__(self, img):
        if _rng.random() < self.p:
            if _is_numpy(img) and img.ndim == 3:
                import numpy as _np
                gray = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(img.dtype)
                return _np.stack([gray, gray, gray], axis=-1)
        return img


class RandomResizedCrop:
    """Randomly crop and resize to target size."""
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3/4, 4/3)):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = tuple(size)
        self.scale = scale
        self.ratio = ratio

    def __call__(self, img):
        if _is_numpy(img):
            import numpy as _np
            h, w = img.shape[:2]
            area = h * w
            for _ in range(10):
                target_area = area * _rng.uniform(*self.scale)
                log_ratio = (_np.log(self.ratio[0]), _np.log(self.ratio[1]))
                aspect = _np.exp(_rng.uniform(*log_ratio))
                new_w = int(round((target_area * aspect) ** 0.5))
                new_h = int(round((target_area / aspect) ** 0.5))
                if 0 < new_w <= w and 0 < new_h <= h:
                    top = _rng.randint(0, h - new_h)
                    left = _rng.randint(0, w - new_w)
                    cropped = img[top:top+new_h, left:left+new_w]
                    # Simple nearest-neighbor resize
                    th, tw = self.size
                    r_y = new_h / th
                    r_x = new_w / tw
                    out = _np.zeros((th, tw) + img.shape[2:], dtype=img.dtype)
                    for y in range(th):
                        for x in range(tw):
                            sy, sx = int(y * r_y), int(x * r_x)
                            out[y, x] = cropped[min(sy, new_h-1), min(sx, new_w-1)]
                    return out
            # Fallback: center crop + resize handled by Resize
            rc = RandomCrop(min(h, w))
            return Resize(self.size)(rc(img))
        return img


class RandomRotation:
    """Rotate image by a random angle in [-degrees, degrees]."""
    def __init__(self, degrees, expand=False, fill=0):
        self.degrees = degrees if hasattr(degrees, '__len__') else (-degrees, degrees)
        self.expand = expand
        self.fill = fill

    def __call__(self, img):
        import math as _m
        angle = _rng.uniform(*self.degrees)
        if _is_numpy(img):
            import numpy as _np
            h, w = img.shape[:2]
            cx, cy = w / 2, h / 2
            rad = _m.radians(angle)
            cos_a, sin_a = _m.cos(rad), _m.sin(rad)
            if self.expand:
                new_w = int(abs(w * cos_a) + abs(h * sin_a))
                new_h = int(abs(h * cos_a) + abs(w * sin_a))
            else:
                new_w, new_h = w, h
            ncx, ncy = new_w / 2, new_h / 2
            channels = img.shape[2] if img.ndim == 3 else 1
            out_shape = (new_h, new_w, channels) if img.ndim == 3 else (new_h, new_w)
            out = _np.full(out_shape, self.fill, dtype=img.dtype)
            for y in range(new_h):
                for x in range(new_w):
                    sx = int((x - ncx) * cos_a + (y - ncy) * sin_a + cx)
                    sy = int(-(x - ncx) * sin_a + (y - ncy) * cos_a + cy)
                    if 0 <= sx < w and 0 <= sy < h:
                        out[y, x] = img[sy, sx]
            return out
        return img


class Grayscale:
    """Convert image to grayscale. If num_output_channels=3, replicate channel."""
    def __init__(self, num_output_channels=1):
        self.num_output_channels = num_output_channels

    def __call__(self, img):
        if _is_numpy(img):
            import numpy as _np
            if img.ndim == 2:
                gray = img
            else:
                gray = (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(img.dtype)
            if self.num_output_channels == 3:
                return _np.stack([gray, gray, gray], axis=2)
            return gray
        return img


class RandomPerspective:
    """Random perspective distortion with given probability."""
    def __init__(self, distortion_scale=0.5, p=0.5, fill=0):
        self.distortion_scale = distortion_scale
        self.p = p
        self.fill = fill

    def __call__(self, img):
        if _rng.random() > self.p:
            return img
        if _is_numpy(img):
            import numpy as _np
            h, w = img.shape[:2]
            half_h = h * self.distortion_scale * 0.5
            half_w = w * self.distortion_scale * 0.5
            # Apply a simple shear as approximation
            shear = _rng.uniform(-self.distortion_scale, self.distortion_scale)
            out = _np.full_like(img, self.fill)
            for y in range(h):
                for x in range(w):
                    sx = int(x + shear * (y - h/2))
                    if 0 <= sx < w:
                        out[y, x] = img[y, sx]
            return out
        return img


class RandomSolarize:
    """Solarize image with probability p — invert pixels above threshold."""
    def __init__(self, threshold=128, p=0.5):
        self.threshold = threshold
        self.p = p

    def __call__(self, img):
        if _rng.random() > self.p:
            return img
        if _is_numpy(img):
            import numpy as _np
            out = img.copy()
            mask = img >= self.threshold
            out[mask] = 255 - img[mask]
            return out
        return img


class RandomPosterize:
    """Reduce bits per channel with probability p."""
    def __init__(self, bits=4, p=0.5):
        self.bits = bits
        self.p = p

    def __call__(self, img):
        if _rng.random() > self.p:
            return img
        if _is_numpy(img):
            import numpy as _np
            shift = 8 - self.bits
            return ((img.astype(_np.uint8) >> shift) << shift)
        return img


class RandomSharpness:
    """Randomly adjust image sharpness."""
    def __init__(self, sharpness_factor, p=0.5):
        self.sharpness_factor = sharpness_factor if hasattr(sharpness_factor, '__len__') else (0, sharpness_factor)
        self.p = p

    def __call__(self, img):
        if _rng.random() > self.p:
            return img
        if _is_numpy(img):
            import numpy as _np
            factor = _rng.uniform(*self.sharpness_factor)
            # Blend between blurred and original
            kernel = _np.ones((3,3), dtype='float32') / 9.0
            if img.ndim == 2:
                h, w = img.shape
                blurred = _np.zeros_like(img, dtype='float32')
                for ky in range(3):
                    for kx in range(3):
                        blurred += _np.pad(img, ((1,1),(1,1)), mode='edge')[ky:ky+h, kx:kx+w] * kernel[ky,kx]
            else:
                h, w, c = img.shape
                blurred = _np.zeros_like(img, dtype='float32')
                for ky in range(3):
                    for kx in range(3):
                        blurred += _np.pad(img, ((1,1),(1,1),(0,0)), mode='edge')[ky:ky+h, kx:kx+w] * kernel[ky,kx]
            result = _np.clip(blurred + factor * (img.astype('float32') - blurred), 0, 255)
            return result.astype(img.dtype)
        return img


__all__ = [
    'Compose', 'RandomCrop', 'CenterCrop', 'RandomHorizontalFlip', 'RandomVerticalFlip',
    'Normalize', 'Pad', 'ColorJitter', 'Resize', 'RandomErasing', 'GaussianBlur',
    'ToTensor', 'RandomGrayscale', 'RandomResizedCrop',
    'RandomRotation', 'Grayscale', 'RandomPerspective',
    'RandomSolarize', 'RandomPosterize', 'RandomSharpness',
]
