"""vectorphile.vision — Vision transforms, augmentation, and utilities (pure Python)."""
from __future__ import annotations
import math
import random as _rng
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union


def _chw(img) -> bool:
    return isinstance(img, list) and img and isinstance(img[0], list) and img[0] and isinstance(img[0][0], list)


def _hwc(img) -> bool:
    return isinstance(img, list) and img and isinstance(img[0], list) and img[0] and isinstance(img[0][0], (int, float))


def _shape(img):
    if _chw(img): return len(img), len(img[0]), len(img[0][0])
    if isinstance(img, list) and img and isinstance(img[0], list): return len(img), len(img[0])
    return (len(img),)


def _clamp(v, lo=0.0, hi=1.0): return max(lo, min(hi, v))


def _copy_img(img):
    if _chw(img): return [[row[:] for row in ch] for ch in img]
    if isinstance(img, list) and img and isinstance(img[0], list): return [row[:] for row in img]
    return img[:]


class Transform:
    def __call__(self, x): raise NotImplementedError
    def __repr__(self): return f'{self.__class__.__name__}()'


class Compose(Transform):
    def __init__(self, transforms): self.transforms = transforms
    def __call__(self, x):
        for t in self.transforms: x = t(x)
        return x
    def __repr__(self):
        lines = ['Compose(['] + [f'  {t},' for t in self.transforms] + ['])']
        return '\n'.join(lines)


class Resize(Transform):
    def __init__(self, size):
        self.size = (size, size) if isinstance(size, int) else tuple(size)

    def __call__(self, img):
        nH, nW = self.size
        if _chw(img):
            C, H, W = _shape(img)
            if H == nH and W == nW: return img
            result = []
            for c in range(C):
                ch = []
                for r in range(nH):
                    row = []
                    sr = r * (H - 1) / max(nH - 1, 1); r0 = int(sr); r1 = min(r0+1,H-1); dr = sr-r0
                    for col in range(nW):
                        sc = col * (W - 1) / max(nW - 1, 1); c0 = int(sc); c1 = min(c0+1,W-1); dc = sc-c0
                        v = (img[c][r0][c0]*(1-dr)*(1-dc) + img[c][r0][c1]*(1-dr)*dc +
                             img[c][r1][c0]*dr*(1-dc) + img[c][r1][c1]*dr*dc)
                        row.append(v)
                    ch.append(row)
                result.append(ch)
            return result
        H, W = _shape(img)
        return [[img[int(r*(H-1)/max(nH-1,1))][int(c*(W-1)/max(nW-1,1))]
                 for c in range(nW)] for r in range(nH)]

    def __repr__(self): return f'Resize(size={self.size})'


class CenterCrop(Transform):
    def __init__(self, size): self.size = (size, size) if isinstance(size, int) else tuple(size)

    def __call__(self, img):
        nH, nW = self.size
        if _chw(img):
            C, H, W = _shape(img); r = (H-nH)//2; c = (W-nW)//2
            return [[[img[ch][r+rr][c+cc] for cc in range(nW)] for rr in range(nH)] for ch in range(C)]
        H, W = _shape(img); r = (H-nH)//2; c = (W-nW)//2
        return [[img[r+rr][c+cc] for cc in range(nW)] for rr in range(nH)]

    def __repr__(self): return f'CenterCrop(size={self.size})'


def _pad_img(img, ph, pw, fill=0.0):
    if _chw(img):
        C, H, W = _shape(img); nH, nW = H+2*ph, W+2*pw
        result = [[[fill]*nW for _ in range(nH)] for _ in range(C)]
        for c in range(C):
            for r in range(H):
                for col in range(W): result[c][r+ph][col+pw] = img[c][r][col]
        return result
    H, W = _shape(img); nH, nW = H+2*ph, W+2*pw
    result = [[fill]*nW for _ in range(nH)]
    for r in range(H):
        for c in range(W): result[r+ph][c+pw] = img[r][c]
    return result


class RandomCrop(Transform):
    def __init__(self, size, padding=0, pad_mode='constant'):
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        self.padding = padding; self.pad_mode = pad_mode

    def __call__(self, img):
        p = self.padding
        if p > 0: img = _pad_img(img, p, p)
        nH, nW = self.size
        if _chw(img):
            C, H, W = _shape(img); r = _rng.randint(0, H-nH); c = _rng.randint(0, W-nW)
            return [[[img[ch][r+rr][c+cc] for cc in range(nW)] for rr in range(nH)] for ch in range(C)]
        H, W = _shape(img); r = _rng.randint(0, H-nH); c = _rng.randint(0, W-nW)
        return [[img[r+rr][c+cc] for cc in range(nW)] for rr in range(nH)]


class RandomHorizontalFlip(Transform):
    def __init__(self, p=0.5): self.p = p
    def __call__(self, img):
        if _rng.random() >= self.p: return img
        if _chw(img): return [[row[::-1] for row in ch] for ch in img]
        return [row[::-1] for row in img]


class RandomVerticalFlip(Transform):
    def __init__(self, p=0.5): self.p = p
    def __call__(self, img):
        if _rng.random() >= self.p: return img
        if _chw(img): return [ch[::-1] for ch in img]
        return img[::-1]


class RandomRotation(Transform):
    def __init__(self, degrees, fill=0.0):
        if isinstance(degrees, (int, float)): self.range = (-degrees, degrees)
        else: self.range = tuple(degrees)
        self.fill = fill

    def __call__(self, img):
        angle = _rng.uniform(*self.range)
        if _chw(img):
            C, H, W = _shape(img)
        else:
            H, W = _shape(img); C = None
        cx, cy = W/2.0, H/2.0
        rad = math.radians(angle); cos_a, sin_a = math.cos(rad), math.sin(rad)
        fill = self.fill
        if C is not None:
            result = [[[fill]*W for _ in range(H)] for _ in range(C)]
            for r in range(H):
                for col in range(W):
                    sx = cos_a*(col-cx)+sin_a*(r-cy)+cx; sy = -sin_a*(col-cx)+cos_a*(r-cy)+cy
                    x0 = int(sx); y0 = int(sy)
                    if 0 <= x0 < W-1 and 0 <= y0 < H-1:
                        dx = sx-x0; dy = sy-y0
                        for c in range(C):
                            result[c][r][col] = (img[c][y0][x0]*(1-dy)*(1-dx)+img[c][y0][x0+1]*(1-dy)*dx+
                                                  img[c][y0+1][x0]*dy*(1-dx)+img[c][y0+1][x0+1]*dy*dx)
            return result
        result = [[fill]*W for _ in range(H)]
        for r in range(H):
            for col in range(W):
                sx = cos_a*(col-cx)+sin_a*(r-cy)+cx; sy = -sin_a*(col-cx)+cos_a*(r-cy)+cy
                x0 = int(sx); y0 = int(sy)
                if 0 <= x0 < W-1 and 0 <= y0 < H-1:
                    dx = sx-x0; dy = sy-y0
                    result[r][col] = (img[y0][x0]*(1-dy)*(1-dx)+img[y0][x0+1]*(1-dy)*dx+
                                       img[y0+1][x0]*dy*(1-dx)+img[y0+1][x0+1]*dy*dx)
        return result


class Pad(Transform):
    def __init__(self, padding, fill=0, mode='constant'):
        if isinstance(padding, int): self.padding = (padding,)*4
        elif len(padding) == 2: self.padding = (padding[0],padding[0],padding[1],padding[1])
        else: self.padding = tuple(padding)
        self.fill = fill

    def __call__(self, img):
        t, b, l, r = self.padding
        return _pad_img(img, t, l, self.fill)


class ToTensor(Transform):
    def __call__(self, img):
        # HWC list → CHW list, scale to [0,1]
        if _chw(img): return img
        if isinstance(img, list) and img and isinstance(img[0], list):
            H, W = len(img), len(img[0])
            if img[0] and isinstance(img[0][0], (list, tuple)):
                C = len(img[0][0])
                return [[[img[h][w][c]/255.0 if isinstance(img[h][w][c],int) and img[h][w][c]>1 else float(img[h][w][c])
                           for w in range(W)] for h in range(H)] for c in range(C)]
            return [[[img[h][w]/255.0 if isinstance(img[h][w],int) and img[h][w]>1 else float(img[h][w])
                      for w in range(W)] for h in range(H)]]
        return img


class ToPILImage(Transform):
    def __call__(self, img):
        if not _chw(img): return img
        C, H, W = _shape(img)
        return [[[int(_clamp(img[c][h][w])*255) for c in range(C)] for w in range(W)] for h in range(H)]


class Normalize(Transform):
    def __init__(self, mean, std): self.mean = list(mean); self.std = list(std)
    def __call__(self, img):
        if not _chw(img): return img
        C, H, W = _shape(img)
        return [[[(img[c][r][col]-self.mean[c])/(self.std[c]+1e-8) for col in range(W)]
                  for r in range(H)] for c in range(min(C,len(self.mean)))]
    def __repr__(self): return f'Normalize(mean={self.mean}, std={self.std})'


class ColorJitter(Transform):
    def __init__(self, brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0):
        self.brightness = brightness; self.contrast = contrast
        self.saturation = saturation; self.hue = hue

    def __call__(self, img):
        if not _chw(img): return img
        C, H, W = _shape(img); result = _copy_img(img)
        ops = list(range(4)); _rng.shuffle(ops)
        for op in ops:
            if op == 0 and self.brightness > 0:
                f = _rng.uniform(max(0,1-self.brightness), 1+self.brightness)
                result = [[[_clamp(result[c][r][col]*f) for col in range(W)] for r in range(H)] for c in range(C)]
            elif op == 1 and self.contrast > 0:
                flat = [result[c][r][col] for c in range(C) for r in range(H) for col in range(W)]
                mean_v = sum(flat)/max(len(flat),1)
                f = _rng.uniform(max(0,1-self.contrast), 1+self.contrast)
                result = [[[_clamp(mean_v+f*(result[c][r][col]-mean_v)) for col in range(W)] for r in range(H)] for c in range(C)]
            elif op == 2 and self.saturation > 0 and C == 3:
                f = _rng.uniform(max(0,1-self.saturation), 1+self.saturation)
                new_r = []
                for c in range(C):
                    ch = []
                    for r in range(H):
                        row = []
                        for col in range(W):
                            gray = 0.299*result[0][r][col]+0.587*result[1][r][col]+0.114*result[2][r][col]
                            row.append(_clamp(gray+f*(result[c][r][col]-gray)))
                        ch.append(row)
                    new_r.append(ch)
                result = new_r
        return result


class GaussianBlur(Transform):
    def __init__(self, kernel_size=3, sigma=(0.1, 2.0)):
        self.kernel_size = kernel_size; self.sigma_range = sigma

    def __call__(self, img):
        sigma = _rng.uniform(*self.sigma_range)
        if not _chw(img): return img
        C, H, W = _shape(img)
        half = self.kernel_size // 2
        kern = [math.exp(-(x**2)/(2*sigma**2)) for x in range(-half, half+1)]
        s = sum(kern); kern = [v/s for v in kern]
        def conv1d(row):
            n = len(row)
            return [sum(row[max(0,min(n-1,i+j-half))*1]*kern[j] for j in range(len(kern))) for i in range(n)]
        result = []
        for c in range(C):
            temp = [conv1d(img[c][r]) for r in range(H)]
            ch = [[sum(temp[r2][col]*kern[j] for j,r2 in enumerate(range(max(0,r-half), max(0,r-half)+len(kern)))
                       if 0<=r2<H)/1 for col in range(W)] for r in range(H)]
            result.append(ch)
        return result


class RandomGrayscale(Transform):
    def __init__(self, p=0.1): self.p = p
    def __call__(self, img):
        if _rng.random() >= self.p or not _chw(img): return img
        C, H, W = _shape(img)
        if C != 3: return img
        gray = [[0.299*img[0][r][col]+0.587*img[1][r][col]+0.114*img[2][r][col] for col in range(W)] for r in range(H)]
        return [gray, [row[:] for row in gray], [row[:] for row in gray]]


class RandomErasing(Transform):
    def __init__(self, p=0.5, scale=(0.02,0.33), ratio=(0.3,3.3), value=0.0):
        self.p=p; self.scale=scale; self.ratio=ratio; self.value=value

    def __call__(self, img):
        if _rng.random() >= self.p: return img
        result = _copy_img(img)
        if _chw(img): C, H, W = _shape(img)
        else: H, W = _shape(img); C = None
        area = H * W
        for _ in range(10):
            ea = _rng.uniform(*self.scale)*area; asp = _rng.uniform(*self.ratio)
            eh = int(math.sqrt(ea/asp)); ew = int(math.sqrt(ea*asp))
            if eh < H and ew < W:
                top = _rng.randint(0, H-eh-1) if H-eh>1 else 0
                left = _rng.randint(0, W-ew-1) if W-ew>1 else 0
                if C is not None:
                    for c in range(C):
                        for r in range(eh):
                            for col in range(ew): result[c][top+r][left+col] = self.value
                else:
                    for r in range(eh):
                        for col in range(ew): result[top+r][left+col] = self.value
                break
        return result


class AugMix(Transform):
    def __init__(self, severity=3, width=3, depth=-1, alpha=1.0):
        self.severity=severity; self.width=width; self.depth=depth; self.alpha=alpha
        self._aug_pool = [
            RandomHorizontalFlip(p=1.0), RandomRotation(30),
            ColorJitter(brightness=0.4), ColorJitter(contrast=0.4), ColorJitter(saturation=0.4),
        ]

    def __call__(self, img):
        if not _chw(img): return img
        C, H, W = _shape(img)
        # Dirichlet-like weights via gamma approx
        ws = [_rng.random() for _ in range(self.width)]
        ws_sum = sum(ws); ws = [w/ws_sum for w in ws]
        m = _rng.betavariate(self.alpha, self.alpha) if self.alpha > 0 else 0.5
        mix = [[[0.0]*W for _ in range(H)] for _ in range(C)]
        for i in range(self.width):
            x_aug = _copy_img(img)
            d = _rng.randint(1, 3) if self.depth == -1 else self.depth
            for _ in range(d):
                op = self._aug_pool[_rng.randint(0, len(self._aug_pool)-1)]
                x_aug = op(x_aug)
            for c in range(C):
                for r in range(H):
                    for col in range(W): mix[c][r][col] += ws[i] * x_aug[c][r][col]
        result = [[[_clamp((1-m)*img[c][r][col]+m*mix[c][r][col]) for col in range(W)]
                    for r in range(H)] for c in range(C)]
        return result


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def imagenet_train_transforms(image_size=224) -> Compose:
    return Compose([Resize(int(image_size*1.14)), RandomCrop(image_size), RandomHorizontalFlip(),
                    ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4,hue=0.1),
                    ToTensor(), Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def imagenet_val_transforms(image_size=224) -> Compose:
    return Compose([Resize(int(image_size*256/224)), CenterCrop(image_size),
                    ToTensor(), Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def load_image(path: str, mode: str = 'RGB'):
    """Load image as list (HWC). Falls back to PIL if available."""
    try:
        from PIL import Image
        img = Image.open(path).convert(mode)
        pixels = list(img.getdata())
        W, H = img.size
        C = len(pixels[0]) if isinstance(pixels[0], tuple) else 1
        if C == 1:
            return [[pixels[r*W+c] for c in range(W)] for r in range(H)]
        return [[[pixels[r*W+c][ch] for c in range(W)] for r in range(H)] for ch in range(C)]
    except ImportError:
        raise ImportError("install pillow to load images")


def save_image(img, path: str):
    """Save image list to file."""
    try:
        from PIL import Image
        if _chw(img):
            C, H, W = _shape(img)
            pixels = [tuple(int(_clamp(img[c][r][col])*255) for c in range(C)) for r in range(H) for col in range(W)]
            mode = 'RGB' if C == 3 else ('RGBA' if C == 4 else 'L')
            pil_img = Image.new(mode, (W, H))
            pil_img.putdata([(p if isinstance(p, tuple) else (p,)*3) for p in pixels])
            pil_img.save(path)
        return
    except ImportError:
        raise ImportError("install pillow to save images")


def make_grid(images, nrow=8, padding=2, normalize=False, value_range=None, pad_value=0.0):
    """Arrange a list of CHW images into a grid."""
    if not images: return []
    if _chw(images[0]):
        C, H, W = _shape(images[0])
    else:
        return images
    n = len(images)
    if normalize:
        all_vals = [images[i][c][r][col] for i in range(n) for c in range(C) for r in range(H) for col in range(W)]
        vmin, vmax = (value_range if value_range else (min(all_vals), max(all_vals)))
        rng = max(vmax-vmin, 1e-8)
        images = [[[[(_clamp((images[i][c][r][col]-vmin)/rng)) for col in range(W)]
                     for r in range(H)] for c in range(C)] for i in range(n)]
    ncol = math.ceil(n/nrow)
    grid_h = ncol*H + (ncol+1)*padding
    grid_w = nrow*W + (nrow+1)*padding
    grid = [[[pad_value]*grid_w for _ in range(grid_h)] for _ in range(C)]
    for idx in range(n):
        row = idx // nrow; col = idx % nrow
        y1 = padding + row*(H+padding); x1 = padding + col*(W+padding)
        for c in range(C):
            for r in range(H):
                for pc in range(W): grid[c][y1+r][x1+pc] = images[idx][c][r][pc]
    return grid


def extract_features(model, images, layer_name=None, batch_size=32):
    features = []
    n = len(images)
    if layer_name is not None:
        from vectorphile.nn.hooks import FeatureExtractor
        extractor = FeatureExtractor(model, [layer_name])
    else:
        extractor = None
    for start in range(0, n, batch_size):
        batch = images[start:start+batch_size]
        out = model(batch)
        if extractor is not None:
            feat = extractor.features.get(layer_name, out); extractor.clear()
        else:
            feat = out
        if hasattr(feat, 'data'): feat = feat.data
        if hasattr(feat, 'to_flat_list'): feat = feat.to_flat_list()
        elif hasattr(feat, '_buf'): feat = list(feat._buf)
        features.append(feat)
    if extractor is not None: extractor.remove()
    return features


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
