"""vectorphile.vision — Vision transforms, augmentation, and utilities."""
from __future__ import annotations
import math
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union


# ── Base transform ─────────────────────────────────────────────────────────────

class Transform:
    """Base class for all vision transforms."""

    def __call__(self, x):
        raise NotImplementedError

    def __repr__(self):
        return f'{self.__class__.__name__}()'


# ── Compose ───────────────────────────────────────────────────────────────────

class Compose(Transform):
    """Chain multiple transforms."""

    def __init__(self, transforms: List[Transform]):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self):
        lines = ['Compose(['] + [f'  {t},' for t in self.transforms] + ['])']
        return '\n'.join(lines)


# ── Basic spatial transforms ──────────────────────────────────────────────────

class Resize(Transform):
    """Resize image to (height, width) using bilinear interpolation.

    Accepts HWC numpy arrays (uint8 or float32).
    """

    def __init__(self, size: Union[int, Tuple[int, int]]):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = tuple(size)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        h_out, w_out = self.size
        h_in, w_in = x.shape[:2]
        if h_in == h_out and w_in == w_out:
            return x
        # Bilinear via scipy if available, else nearest-neighbor
        try:
            from scipy.ndimage import zoom
            factors = (h_out / h_in, w_out / w_in) + (1,) * (x.ndim - 2)
            return zoom(x, factors, order=1).astype(x.dtype)
        except ImportError:
            # Nearest-neighbor fallback
            row_idx = (np.arange(h_out) * h_in / h_out).astype(int)
            col_idx = (np.arange(w_out) * w_in / w_out).astype(int)
            return x[row_idx[:, None], col_idx[None, :]]

    def __repr__(self):
        return f'Resize(size={self.size})'


class CenterCrop(Transform):
    """Crop the center of the image."""

    def __init__(self, size: Union[int, Tuple[int, int]]):
        self.size = (size, size) if isinstance(size, int) else tuple(size)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h, w = x.shape[:2]
        th, tw = self.size
        top = (h - th) // 2
        left = (w - tw) // 2
        return x[top:top+th, left:left+tw]

    def __repr__(self):
        return f'CenterCrop(size={self.size})'


class RandomCrop(Transform):
    """Random crop with optional padding."""

    def __init__(self, size: Union[int, Tuple[int, int]], padding: int = 0,
                 pad_mode: str = 'constant'):
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        self.padding = padding
        self.pad_mode = pad_mode

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.padding > 0:
            p = self.padding
            pad_width = [(p, p), (p, p)] + [(0, 0)] * (x.ndim - 2)
            x = np.pad(x, pad_width, mode=self.pad_mode)
        h, w = x.shape[:2]
        th, tw = self.size
        top = np.random.randint(0, h - th + 1)
        left = np.random.randint(0, w - tw + 1)
        return x[top:top+th, left:left+tw]


class RandomHorizontalFlip(Transform):
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.fliplr(x) if np.random.random() < self.p else x


class RandomVerticalFlip(Transform):
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.flipud(x) if np.random.random() < self.p else x


class RandomRotation(Transform):
    """Random rotation by degrees in [-degrees, +degrees]."""

    def __init__(self, degrees: Union[float, Tuple[float, float]]):
        if isinstance(degrees, (int, float)):
            self.range = (-degrees, degrees)
        else:
            self.range = tuple(degrees)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        angle = np.random.uniform(*self.range)
        try:
            from scipy.ndimage import rotate
            return rotate(x, angle, reshape=False, mode='nearest').astype(x.dtype)
        except ImportError:
            return x  # no-op without scipy


class Pad(Transform):
    """Pad with zeros or constant value."""

    def __init__(self, padding: Union[int, Tuple[int, ...]], fill: float = 0,
                 mode: str = 'constant'):
        if isinstance(padding, int):
            self.padding = (padding, padding, padding, padding)  # T, B, L, R
        elif len(padding) == 2:
            self.padding = (padding[0], padding[0], padding[1], padding[1])
        else:
            self.padding = tuple(padding)
        self.fill = fill
        self.mode = mode

    def __call__(self, x: np.ndarray) -> np.ndarray:
        t, b, l, r = self.padding
        pad_width = [(t, b), (l, r)] + [(0, 0)] * (x.ndim - 2)
        return np.pad(x, pad_width, mode=self.mode,
                      constant_values=self.fill if self.mode == 'constant' else 0)


# ── Color transforms ──────────────────────────────────────────────────────────

class ToTensor(Transform):
    """Convert HWC uint8 [0,255] → CHW float32 [0,1]."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.dtype == np.uint8:
            x = x.astype(np.float32) / 255.0
        if x.ndim == 2:
            x = x[np.newaxis, :, :]
        elif x.ndim == 3:
            x = np.transpose(x, (2, 0, 1))
        return x.astype(np.float32)


class ToPILImage(Transform):
    """Convert CHW float32 [0,1] → HWC uint8 [0,255] numpy array."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.ndim == 3 and x.shape[0] in (1, 3, 4):
            x = np.transpose(x, (1, 2, 0))
        x = np.clip(x * 255.0, 0, 255).astype(np.uint8)
        if x.shape[-1] == 1:
            x = x.squeeze(-1)
        return x


class Normalize(Transform):
    """Normalize CHW tensor: (x - mean) / std per channel."""

    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = np.array(mean, dtype=np.float32)[:, np.newaxis, np.newaxis]
        self.std = np.array(std, dtype=np.float32)[:, np.newaxis, np.newaxis]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float32) - self.mean) / (self.std + 1e-8)

    def __repr__(self):
        return f'Normalize(mean={self.mean.ravel()}, std={self.std.ravel()})'


class ColorJitter(Transform):
    """Random brightness, contrast, saturation, hue jitter.

    Expects HWC uint8 or float32 image.
    """

    def __init__(self, brightness: float = 0.0, contrast: float = 0.0,
                 saturation: float = 0.0, hue: float = 0.0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        is_uint = x.max() > 1.0
        if is_uint:
            x = x / 255.0

        order = np.random.permutation(4)
        for op in order:
            if op == 0 and self.brightness > 0:
                f = np.random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
                x = np.clip(x * f, 0, 1)
            elif op == 1 and self.contrast > 0:
                f = np.random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
                mean = x.mean()
                x = np.clip((x - mean) * f + mean, 0, 1)
            elif op == 2 and self.saturation > 0:
                f = np.random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
                gray = np.mean(x, axis=-1, keepdims=True)
                x = np.clip(gray + (x - gray) * f, 0, 1)
            elif op == 3 and self.hue > 0 and x.shape[-1] == 3:
                # Rotate hue channel (HSV space approximation)
                shift = np.random.uniform(-self.hue, self.hue) * 360
                x = _shift_hue(x, shift)

        if is_uint:
            x = np.clip(x * 255, 0, 255).astype(np.uint8)
        return x


def _shift_hue(x: np.ndarray, degrees: float) -> np.ndarray:
    """Shift hue by degrees in RGB image [0,1]."""
    # Simple RGB rotation approximation
    theta = np.radians(degrees)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    u = cos_t + (1 - cos_t) / 3
    w = sin_t * math.sqrt(1 / 3)
    mat = np.array([
        [u + w, 1/3*(1-u) - w, 1/3*(1-u) + w],
        [1/3*(1-u) + w, u + 1/3*(1-u), 1/3*(1-u) - w],  # corrected
        [1/3*(1-u) - w, 1/3*(1-u) + w, u + 1/3*(1-u)],
    ], dtype=np.float32)
    return np.clip(x @ mat.T, 0, 1)


class GaussianBlur(Transform):
    """Gaussian blur with random kernel size."""

    def __init__(self, kernel_size: int = 3, sigma: Tuple[float, float] = (0.1, 2.0)):
        self.kernel_size = kernel_size
        self.sigma_range = sigma

    def __call__(self, x: np.ndarray) -> np.ndarray:
        sigma = np.random.uniform(*self.sigma_range)
        try:
            from scipy.ndimage import gaussian_filter
            return gaussian_filter(x, sigma=[sigma, sigma, 0] if x.ndim == 3 else sigma)
        except ImportError:
            return x


class RandomGrayscale(Transform):
    def __init__(self, p: float = 0.1):
        self.p = p

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.random() < self.p and x.ndim == 3 and x.shape[-1] == 3:
            gray = np.mean(x, axis=-1, keepdims=True)
            return np.repeat(gray, 3, axis=-1).astype(x.dtype)
        return x


class RandomErasing(Transform):
    """Randomly erase a rectangular patch."""

    def __init__(self, p: float = 0.5, scale: Tuple = (0.02, 0.33),
                 ratio: Tuple = (0.3, 3.3), value: float = 0.0):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.random() >= self.p:
            return x
        x = x.copy()
        h, w = x.shape[:2] if x.ndim == 3 else (x.shape[-2], x.shape[-1])
        area = h * w
        for _ in range(10):
            erase_area = np.random.uniform(*self.scale) * area
            aspect = np.random.uniform(*self.ratio)
            eh = int(math.sqrt(erase_area / aspect))
            ew = int(math.sqrt(erase_area * aspect))
            if eh < h and ew < w:
                top = np.random.randint(0, h - eh)
                left = np.random.randint(0, w - ew)
                if x.ndim == 3:
                    x[top:top+eh, left:left+ew] = self.value
                else:
                    x[..., top:top+eh, left:left+ew] = self.value
                break
        return x


# ── AugMix ───────────────────────────────────────────────────────────────────

class AugMix(Transform):
    """AugMix — Hendrycks et al. 2020 (simplified).

    Randomly mixes augmented versions of an image to improve robustness.
    """

    def __init__(self, severity: int = 3, width: int = 3, depth: int = -1,
                 alpha: float = 1.0):
        self.severity = severity
        self.width = width
        self.depth = depth  # -1 = random 1–3
        self.alpha = alpha
        self._aug_pool = [
            RandomHorizontalFlip(p=1.0),
            RandomRotation(30),
            ColorJitter(brightness=0.4),
            ColorJitter(contrast=0.4),
            ColorJitter(saturation=0.4),
        ]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        is_uint = x.max() > 1.0
        if is_uint:
            x = x / 255.0

        ws = np.random.dirichlet([self.alpha] * self.width)
        m = np.random.beta(self.alpha, self.alpha)

        mix = np.zeros_like(x)
        for i in range(self.width):
            x_aug = x.copy()
            d = np.random.randint(1, 4) if self.depth == -1 else self.depth
            for _ in range(d):
                op = self._aug_pool[np.random.randint(len(self._aug_pool))]
                x_aug = op(x_aug)
            mix += ws[i] * x_aug

        result = (1 - m) * x + m * mix
        if is_uint:
            result = np.clip(result * 255, 0, 255).astype(np.uint8)
        return result


# ── Standard ImageNet transforms ──────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def imagenet_train_transforms(image_size: int = 224) -> Compose:
    return Compose([
        Resize(int(image_size * 1.14)),
        RandomCrop(image_size),
        RandomHorizontalFlip(),
        ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        ToTensor(),
        Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def imagenet_val_transforms(image_size: int = 224) -> Compose:
    return Compose([
        Resize(int(image_size * 256 / 224)),
        CenterCrop(image_size),
        ToTensor(),
        Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ── Image I/O ─────────────────────────────────────────────────────────────────

def load_image(path: str, mode: str = 'RGB') -> np.ndarray:
    """Load image as HWC numpy array (uint8)."""
    try:
        from PIL import Image
        img = Image.open(path).convert(mode)
        return np.array(img)
    except ImportError:
        pass
    try:
        import imageio
        img = imageio.imread(path)
        return np.asarray(img)
    except ImportError:
        raise ImportError("install pillow or imageio to load images")


def save_image(x: np.ndarray, path: str):
    """Save CHW float [0,1] or HWC uint8 [0,255] array as image."""
    if x.ndim == 3 and x.shape[0] in (1, 3, 4):
        x = np.transpose(x, (1, 2, 0))
    if x.dtype != np.uint8:
        x = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    if x.shape[-1] == 1:
        x = x.squeeze(-1)
    try:
        from PIL import Image
        Image.fromarray(x).save(path)
        return
    except ImportError:
        pass
    try:
        import imageio
        imageio.imwrite(path, x)
        return
    except ImportError:
        raise ImportError("install pillow or imageio to save images")


def make_grid(
    images: np.ndarray,
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    value_range: Optional[Tuple] = None,
    pad_value: float = 0.0,
) -> np.ndarray:
    """Arrange a batch of CHW images into a grid.

    Args:
        images: array of shape (N, C, H, W)
        nrow: number of images per row
        padding: pixels between images
        normalize: scale to [0,1]

    Returns:
        CHW grid image (float32)
    """
    images = np.asarray(images, dtype=np.float32)
    if images.ndim == 3:
        images = images[np.newaxis]
    n, c, h, w = images.shape

    if normalize:
        if value_range is not None:
            vmin, vmax = value_range
        else:
            vmin, vmax = images.min(), images.max()
        images = (images - vmin) / max(vmax - vmin, 1e-8)

    ncol = math.ceil(n / nrow)
    grid_h = ncol * h + (ncol + 1) * padding
    grid_w = nrow * w + (nrow + 1) * padding
    grid = np.full((c, grid_h, grid_w), pad_value, dtype=np.float32)

    for idx in range(n):
        row = idx // nrow
        col = idx % nrow
        y1 = padding + row * (h + padding)
        x1 = padding + col * (w + padding)
        grid[:, y1:y1+h, x1:x1+w] = images[idx]

    return grid


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(
    model,
    images: np.ndarray,
    layer_name: Optional[str] = None,
    batch_size: int = 32,
) -> np.ndarray:
    """Extract features from a model for a batch of images.

    Args:
        model: a vectorphile Module
        images: (N, C, H, W) float32 array
        layer_name: optional layer to hook; if None, uses final output
        batch_size: mini-batch size for inference

    Returns:
        (N, D) feature array
    """
    features = []
    n = len(images)

    if layer_name is not None:
        from vectorphile.nn.hooks import FeatureExtractor
        extractor = FeatureExtractor(model, [layer_name])
    else:
        extractor = None

    for start in range(0, n, batch_size):
        batch = images[start:start + batch_size]
        out = model(batch)
        if extractor is not None:
            feat = extractor.features.get(layer_name, out)
            extractor.clear()
        else:
            feat = out
        if hasattr(feat, 'data'):
            feat = feat.data
        feat = np.asarray(feat)
        if feat.ndim > 2:
            feat = feat.reshape(len(batch), -1)
        features.append(feat)

    if extractor is not None:
        extractor.remove()

    return np.concatenate(features, axis=0)


__all__ = [
    'Transform', 'Compose',
    'Resize', 'CenterCrop', 'RandomCrop', 'RandomHorizontalFlip', 'RandomVerticalFlip',
    'RandomRotation', 'Pad',
    'ToTensor', 'ToPILImage', 'Normalize',
    'ColorJitter', 'GaussianBlur', 'RandomGrayscale', 'RandomErasing',
    'AugMix',
    'IMAGENET_MEAN', 'IMAGENET_STD',
    'imagenet_train_transforms', 'imagenet_val_transforms',
    'load_image', 'save_image', 'make_grid', 'extract_features',
]
