"""vectorphile.cv — Computer Vision algorithms (OpenCV-compatible API).

Pure Python implementation — no external dependencies.

Usage::

    import vectorphile.cv as cv

    img = cv.imread('photo.jpg')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 100, 200)
    cv.imwrite('edges.png', edges)
"""
from __future__ import annotations
import math
from typing import List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COLOR_BGR2GRAY = 'bgr2gray'
COLOR_BGR2RGB = 'bgr2rgb'
COLOR_RGB2BGR = 'rgb2bgr'
COLOR_BGR2HSV = 'bgr2hsv'
COLOR_HSV2BGR = 'hsv2bgr'
COLOR_BGR2LAB = 'bgr2lab'
COLOR_LAB2BGR = 'lab2bgr'
COLOR_BGR2YCrCb = 'bgr2ycrcb'
COLOR_YCrCb2BGR = 'ycrcb2bgr'
COLOR_GRAY2BGR = 'gray2bgr'
COLOR_GRAY2RGB = 'gray2rgb'
COLOR_RGB2GRAY = 'rgb2gray'

INTER_NEAREST = 'nearest'
INTER_LINEAR = 'bilinear'
INTER_CUBIC = 'cubic'
INTER_AREA = 'area'

BORDER_CONSTANT = 0
BORDER_REFLECT = 1
BORDER_WRAP = 2

MORPH_RECT = 'rect'
MORPH_ELLIPSE = 'ellipse'
MORPH_CROSS = 'cross'

MORPH_ERODE = 0
MORPH_DILATE = 1
MORPH_OPEN = 2
MORPH_CLOSE = 3

ROTATE_90_CLOCKWISE = 0
ROTATE_180 = 1
ROTATE_90_COUNTERCLOCKWISE = 2

THRESH_BINARY = 0
THRESH_BINARY_INV = 1
THRESH_TRUNC = 2
THRESH_TOZERO = 3


# ---------------------------------------------------------------------------
# Image type: 2D list for gray, 3D list for color
# Img = List[List[int]] or List[List[List[int]]]
# ---------------------------------------------------------------------------

def _shape(img) -> tuple:
    if not img: return (0,)
    if isinstance(img[0], list):
        if img[0] and isinstance(img[0][0], list):
            return (len(img), len(img[0]), len(img[0][0]))
        return (len(img), len(img[0]))
    return (len(img),)


def _clamp(v, lo=0, hi=255) -> int:
    return max(lo, min(hi, int(round(v))))


def _copy(img):
    if not img: return []
    if isinstance(img[0], list):
        if img[0] and isinstance(img[0][0], list):
            return [[[img[r][c][ch] for ch in range(len(img[r][c]))]
                     for c in range(len(img[r]))] for r in range(len(img))]
        return [[img[r][c] for c in range(len(img[r]))] for r in range(len(img))]
    return list(img)


def _pad2d(img, ph, pw, mode='edge', cval=0):
    H = len(img); W = len(img[0]) if H else 0
    NH = H + 2*ph; NW = W + 2*pw
    out = [[0] * NW for _ in range(NH)]
    for r in range(NH):
        for c in range(NW):
            sr = r - ph; sc = c - pw
            if mode == 'constant':
                out[r][c] = img[max(0, min(H-1, sr))][max(0, min(W-1, sc))] if 0 <= sr < H and 0 <= sc < W else cval
            elif mode == 'edge':
                out[r][c] = img[max(0, min(H-1, sr))][max(0, min(W-1, sc))]
            elif mode == 'reflect':
                sr2 = sr if 0 <= sr < H else ((-sr - 1) if sr < 0 else (2*H - sr - 1))
                sc2 = sc if 0 <= sc < W else ((-sc - 1) if sc < 0 else (2*W - sc - 1))
                out[r][c] = img[max(0, min(H-1, sr2))][max(0, min(W-1, sc2))]
            else:
                out[r][c] = img[max(0, min(H-1, sr))][max(0, min(W-1, sc))]
    return out


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def imread(filename: str, flags: int = 1):
    """Read an image from file (BGR format, like OpenCV)."""
    try:
        from vectorphile.image import Image
        img = Image.open(filename)
        arr = img.to_array()
        # arr is a 3D list H x W x C (RGB)
        if _shape(arr)[2:] == (3,):
            # RGB -> BGR
            H, W, _ = _shape(arr)
            return [[[arr[r][c][2], arr[r][c][1], arr[r][c][0]] for c in range(W)] for r in range(H)]
        return arr
    except Exception:
        return None


def imwrite(filename: str, img) -> bool:
    """Write an image to file (expects BGR or grayscale)."""
    try:
        from vectorphile.image import Image
        sh = _shape(img)
        if len(sh) == 3 and sh[2] == 3:
            H, W, _ = sh
            rgb = [[[img[r][c][2], img[r][c][1], img[r][c][0]] for c in range(W)] for r in range(H)]
            Image.fromarray(rgb).save(filename)
        else:
            Image.fromarray(img).save(filename)
        return True
    except Exception:
        return False


def imshow(winname: str, mat) -> None:
    pass


def waitKey(delay: int = 0) -> int:
    return -1


# ---------------------------------------------------------------------------
# Color conversion
# ---------------------------------------------------------------------------

def cvtColor(src, code: str):
    sh = _shape(src)
    H, W = sh[0], sh[1]
    if code in (COLOR_BGR2GRAY, COLOR_RGB2GRAY):
        if len(sh) == 2:
            return _copy(src)
        b_w = 0.114 if code == COLOR_BGR2GRAY else 0.299
        g_w = 0.587
        r_w = 0.299 if code == COLOR_BGR2GRAY else 0.114
        return [[_clamp(b_w*src[r][c][0] + g_w*src[r][c][1] + r_w*src[r][c][2])
                 for c in range(W)] for r in range(H)]
    elif code in (COLOR_BGR2RGB, COLOR_RGB2BGR):
        return [[[src[r][c][2], src[r][c][1], src[r][c][0]] for c in range(W)] for r in range(H)]
    elif code in (COLOR_GRAY2BGR, COLOR_GRAY2RGB):
        return [[[src[r][c], src[r][c], src[r][c]] for c in range(W)] for r in range(H)]
    elif code == COLOR_BGR2HSV:
        return _bgr_to_hsv(src)
    elif code == COLOR_HSV2BGR:
        return _hsv_to_bgr(src)
    return _copy(src)


def _bgr_to_hsv(bgr):
    H, W = _shape(bgr)[:2]
    out = [[[0, 0, 0] for _ in range(W)] for _ in range(H)]
    for r in range(H):
        for c in range(W):
            b = bgr[r][c][0] / 255.0
            g = bgr[r][c][1] / 255.0
            rv = bgr[r][c][2] / 255.0
            cmax = max(rv, g, b); cmin = min(rv, g, b)
            delta = cmax - cmin + 1e-10
            if cmax == rv: h = (g - b) / delta % 6
            elif cmax == g: h = (b - rv) / delta + 2
            else: h = (rv - g) / delta + 4
            h = (h * 60) % 360
            s = delta / cmax if cmax > 0 else 0.0
            v = cmax
            out[r][c] = [_clamp(h), _clamp(s * 255), _clamp(v * 255)]
    return out


def _hsv_to_bgr(hsv):
    H, W = _shape(hsv)[:2]
    out = [[[0, 0, 0] for _ in range(W)] for _ in range(H)]
    for r in range(H):
        for c in range(W):
            h = hsv[r][c][0] / 1.0
            s = hsv[r][c][1] / 255.0
            v = hsv[r][c][2] / 255.0
            hi = int(h / 60) % 6
            f = h / 60 - math.floor(h / 60)
            p = v * (1 - s); q = v * (1 - s * f); t = v * (1 - s * (1 - f))
            if hi == 0: rv, g, b = v, t, p
            elif hi == 1: rv, g, b = q, v, p
            elif hi == 2: rv, g, b = p, v, t
            elif hi == 3: rv, g, b = p, q, v
            elif hi == 4: rv, g, b = t, p, v
            else: rv, g, b = v, p, q
            out[r][c] = [_clamp(b * 255), _clamp(g * 255), _clamp(rv * 255)]
    return out


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def _apply_filter2d(src, kernel: List[List[float]]):
    sh = _shape(src)
    H, W = sh[0], sh[1]
    kh = len(kernel); kw = len(kernel[0]) if kernel else 0
    ph, pw = kh // 2, kw // 2
    if len(sh) == 2:
        padded = _pad2d(src, ph, pw, mode='edge')
        out = [[0.0] * W for _ in range(H)]
        for ky in range(kh):
            for kx in range(kw):
                k = kernel[ky][kx]
                if k == 0: continue
                for r in range(H):
                    for c in range(W):
                        out[r][c] += k * padded[r + ky][c + kx]
        return [[_clamp(out[r][c]) for c in range(W)] for r in range(H)]
    else:
        C = sh[2]
        channels = []
        for ch in range(C):
            ch_img = [[src[r][c][ch] for c in range(W)] for r in range(H)]
            channels.append(_apply_filter2d(ch_img, kernel))
        return [[[channels[ch][r][c] for ch in range(C)] for c in range(W)] for r in range(H)]


def GaussianBlur(src, ksize: Tuple[int, int], sigmaX: float, sigmaY: float = 0):
    kw, kh = ksize
    sigmaY = sigmaY or sigmaX
    ax = [i - kw // 2 for i in range(kw)]
    ay = [i - kh // 2 for i in range(kh)]
    kx = [math.exp(-v**2 / (2 * sigmaX**2)) for v in ax]
    ky = [math.exp(-v**2 / (2 * sigmaY**2)) for v in ay]
    total = sum(ky[i] * kx[j] for i in range(kh) for j in range(kw))
    kernel = [[ky[i] * kx[j] / total for j in range(kw)] for i in range(kh)]
    return _apply_filter2d(src, kernel)


def blur(src, ksize: Tuple[int, int]):
    kw, kh = ksize
    k = 1.0 / (kh * kw)
    kernel = [[k] * kw for _ in range(kh)]
    return _apply_filter2d(src, kernel)


def medianBlur(src, ksize: int):
    sh = _shape(src)
    H, W = sh[0], sh[1]
    half = ksize // 2
    out = _copy(src)
    for r in range(H):
        for c in range(W):
            if len(sh) == 2:
                patch = []
                for dy in range(-half, half+1):
                    for dx in range(-half, half+1):
                        nr = max(0, min(H-1, r+dy)); nc = max(0, min(W-1, c+dx))
                        patch.append(src[nr][nc])
                patch.sort()
                out[r][c] = patch[len(patch)//2]
            else:
                C = sh[2]
                for ch in range(C):
                    patch = []
                    for dy in range(-half, half+1):
                        for dx in range(-half, half+1):
                            nr = max(0, min(H-1, r+dy)); nc = max(0, min(W-1, c+dx))
                            patch.append(src[nr][nc][ch])
                    patch.sort()
                    out[r][c][ch] = patch[len(patch)//2]
    return out


_SOBEL_X = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
_SOBEL_Y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
_LAPLACIAN = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]


def Sobel(src, ddepth: int, dx: int, dy: int, ksize: int = 3):
    if dx == 1 and dy == 0:
        kernel = _SOBEL_X
    elif dx == 0 and dy == 1:
        kernel = _SOBEL_Y
    else:
        kernel = [[1.0 / ksize**2] * ksize for _ in range(ksize)]
    return _apply_filter2d(src, kernel)


def Laplacian(src, ddepth: int, ksize: int = 1):
    return _apply_filter2d(src, _LAPLACIAN)


def filter2D(src, ddepth: int, kernel: List[List[float]]):
    return _apply_filter2d(src, kernel)


# ---------------------------------------------------------------------------
# Edge detection
# ---------------------------------------------------------------------------

def Canny(image, threshold1: float, threshold2: float,
          apertureSize: int = 3, L2gradient: bool = False):
    sh = _shape(image)
    H, W = sh[0], sh[1]
    smoothed = GaussianBlur(image if len(sh) == 2 else
                             [[sum(image[r][c][ch] for ch in range(sh[2])) // sh[2]
                               for c in range(W)] for r in range(H)],
                             (5, 5), 1.4)
    gx = _apply_filter2d(smoothed, _SOBEL_X)
    gy = _apply_filter2d(smoothed, _SOBEL_Y)
    out = [[0] * W for _ in range(H)]
    for r in range(H):
        for c in range(W):
            if L2gradient:
                mag = math.sqrt(gx[r][c]**2 + gy[r][c]**2)
            else:
                mag = abs(gx[r][c]) + abs(gy[r][c])
            out[r][c] = 255 if mag >= threshold2 else 0
    return out


# ---------------------------------------------------------------------------
# Morphological operations
# ---------------------------------------------------------------------------

def getStructuringElement(shape: str, ksize: Tuple[int, int]) -> List[List[int]]:
    kh, kw = ksize
    if shape == MORPH_RECT:
        return [[1] * kw for _ in range(kh)]
    elif shape == MORPH_ELLIPSE:
        cx, cy = kw // 2, kh // 2
        return [[1 if ((c-cx)/(cx+0.5))**2 + ((r-cy)/(cy+0.5))**2 <= 1 else 0
                 for c in range(kw)] for r in range(kh)]
    elif shape == MORPH_CROSS:
        kernel = [[0] * kw for _ in range(kh)]
        for c in range(kw): kernel[kh//2][c] = 1
        for r in range(kh): kernel[r][kw//2] = 1
        return kernel
    return [[1] * kw for _ in range(kh)]


def erode(src, kernel: List[List[int]], iterations: int = 1):
    img = _copy(src)
    for _ in range(iterations):
        sh = _shape(img); H, W = sh[0], sh[1]
        kh = len(kernel); kw = len(kernel[0]) if kernel else 0
        ph, pw = kh // 2, kw // 2
        padded = _pad2d(img, ph, pw, mode='constant', cval=255)
        out = _copy(img)
        for r in range(H):
            for c in range(W):
                mn = 255
                for ky in range(kh):
                    for kx in range(kw):
                        if kernel[ky][kx]:
                            mn = min(mn, padded[r+ky][c+kx])
                out[r][c] = mn
        img = out
    return img


def dilate(src, kernel: List[List[int]], iterations: int = 1):
    img = _copy(src)
    for _ in range(iterations):
        sh = _shape(img); H, W = sh[0], sh[1]
        kh = len(kernel); kw = len(kernel[0]) if kernel else 0
        ph, pw = kh // 2, kw // 2
        padded = _pad2d(img, ph, pw, mode='constant', cval=0)
        out = _copy(img)
        for r in range(H):
            for c in range(W):
                mx = 0
                for ky in range(kh):
                    for kx in range(kw):
                        if kernel[ky][kx]:
                            mx = max(mx, padded[r+ky][c+kx])
                out[r][c] = mx
        img = out
    return img


def morphologyEx(src, op: int, kernel: List[List[int]]):
    if op == MORPH_ERODE: return erode(src, kernel)
    elif op == MORPH_DILATE: return dilate(src, kernel)
    elif op == MORPH_OPEN: return dilate(erode(src, kernel), kernel)
    elif op == MORPH_CLOSE: return erode(dilate(src, kernel), kernel)
    return _copy(src)


# ---------------------------------------------------------------------------
# Geometric transforms
# ---------------------------------------------------------------------------

def resize(src, dsize: Optional[Tuple[int, int]] = None,
           fx: float = 0, fy: float = 0, interpolation: str = INTER_LINEAR):
    sh = _shape(src)
    H, W = sh[0], sh[1]
    if dsize:
        new_W, new_H = dsize
    else:
        new_W, new_H = int(W * fx), int(H * fy)
    out = [[None] * new_W for _ in range(new_H)]
    for r in range(new_H):
        for c in range(new_W):
            sr = r * (H - 1) / max(new_H - 1, 1)
            sc = c * (W - 1) / max(new_W - 1, 1)
            if interpolation == INTER_NEAREST:
                out[r][c] = src[int(round(sr))][int(round(sc))]
            else:
                r0, c0 = int(sr), int(sc)
                r1, c1 = min(r0 + 1, H - 1), min(c0 + 1, W - 1)
                fr, fc = sr - r0, sc - c0
                if len(sh) == 2:
                    out[r][c] = _clamp(
                        src[r0][c0] * (1-fr) * (1-fc) + src[r0][c1] * (1-fr) * fc +
                        src[r1][c0] * fr * (1-fc) + src[r1][c1] * fr * fc)
                else:
                    C = sh[2]
                    out[r][c] = [_clamp(
                        src[r0][c0][ch] * (1-fr) * (1-fc) + src[r0][c1][ch] * (1-fr) * fc +
                        src[r1][c0][ch] * fr * (1-fc) + src[r1][c1][ch] * fr * fc)
                        for ch in range(C)]
    return out


def flip(src, flipCode: int):
    sh = _shape(src)
    H, W = sh[0], sh[1]
    if flipCode == 0:  # vertical
        return [src[H - 1 - r] for r in range(H)]
    elif flipCode == 1:  # horizontal
        if len(sh) == 2:
            return [[src[r][W - 1 - c] for c in range(W)] for r in range(H)]
        return [[src[r][W - 1 - c] for c in range(W)] for r in range(H)]
    else:
        return [src[H - 1 - r][::- 1] if isinstance(src[0], list) else src[H-1-r]
                for r in range(H)]


def rotate(src, rotateCode: int):
    sh = _shape(src)
    H, W = sh[0], sh[1]
    if rotateCode == ROTATE_90_CLOCKWISE:
        return [[src[H - 1 - r][c] for r in range(H)] for c in range(W)]
    elif rotateCode == ROTATE_180:
        return [[src[H - 1 - r][W - 1 - c] for c in range(W)] for r in range(H)]
    else:  # 90 CCW
        return [[src[r][W - 1 - c] for r in range(H)] for c in range(W)]


# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------

def calcHist(images, channels: List[int], mask, histSize: List[int],
             ranges: List[float]):
    img = images[0]
    chan = channels[0]
    sh = _shape(img)
    if len(sh) == 2:
        flat = [img[r][c] for r in range(sh[0]) for c in range(sh[1])]
    else:
        flat = [img[r][c][chan] for r in range(sh[0]) for c in range(sh[1])]
    lo, hi = ranges[0], ranges[1]
    n_bins = histSize[0]
    hist = [0.0] * n_bins
    step = (hi - lo) / n_bins
    for v in flat:
        idx = min(int((v - lo) / step), n_bins - 1)
        if 0 <= idx < n_bins:
            hist[idx] += 1.0
    return [[v] for v in hist]


def equalizeHist(src):
    sh = _shape(src)
    H, W = sh[0], sh[1]
    flat = [src[r][c] for r in range(H) for c in range(W)]
    hist = [0] * 256
    for v in flat: hist[v] += 1
    cdf = [0] * 256
    cdf[0] = hist[0]
    for i in range(1, 256): cdf[i] = cdf[i-1] + hist[i]
    cdf_min = next((v for v in cdf if v > 0), 1)
    n = H * W
    lut = [_clamp(round((cdf[i] - cdf_min) / (n - cdf_min + 1e-8) * 255)) for i in range(256)]
    return [[lut[src[r][c]] for c in range(W)] for r in range(H)]


# ---------------------------------------------------------------------------
# Geometric features
# ---------------------------------------------------------------------------

def boundingRect(points) -> Tuple[int, int, int, int]:
    xs = [p[0] for p in points]; ys = [p[1] for p in points]
    x, y = min(xs), min(ys)
    return int(x), int(y), int(max(xs) - x), int(max(ys) - y)


def contourArea(contour) -> float:
    pts = [(p[0], p[1]) for p in contour]
    n = len(pts); area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1]
    return abs(area) / 2.0


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def rectangle(img, pt1: Tuple[int, int], pt2: Tuple[int, int], color,
              thickness: int = 1):
    img = _copy(img)
    x1, y1 = pt1; x2, y2 = pt2
    sh = _shape(img); H, W = sh[0], sh[1]
    def _set(r, c):
        if 0 <= r < H and 0 <= c < W:
            img[r][c] = list(color) if isinstance(color, (list, tuple)) else color
    if thickness < 0:
        for r in range(y1, y2):
            for c in range(x1, x2): _set(r, c)
    else:
        for c in range(x1, x2):
            for t in range(thickness): _set(y1+t, c); _set(y2-1-t, c)
        for r in range(y1, y2):
            for t in range(thickness): _set(r, x1+t); _set(r, x2-1-t)
    return img


def circle(img, center: Tuple[int, int], radius: int, color, thickness: int = 1):
    img = _copy(img)
    cx, cy = center
    sh = _shape(img); H, W = sh[0], sh[1]
    def _set(r, c):
        if 0 <= r < H and 0 <= c < W:
            img[r][c] = list(color) if isinstance(color, (list, tuple)) else color
    for angle in range(360):
        rad = math.radians(angle)
        for t in range(max(1, thickness)):
            r_circ = radius - t
            x = int(cx + r_circ * math.cos(rad))
            y = int(cy + r_circ * math.sin(rad))
            _set(y, x)
    return img


def line(img, pt1: Tuple[int, int], pt2: Tuple[int, int], color, thickness: int = 1):
    img = _copy(img)
    x1, y1 = pt1; x2, y2 = pt2
    sh = _shape(img); H, W = sh[0], sh[1]
    def _set(r, c):
        if 0 <= r < H and 0 <= c < W:
            img[r][c] = list(color) if isinstance(color, (list, tuple)) else color
    dx, dy = abs(x2-x1), abs(y2-y1)
    sx = 1 if x1 < x2 else -1; sy = 1 if y1 < y2 else -1
    err = dx - dy
    while True:
        _set(y1, x1)
        if x1 == x2 and y1 == y2: break
        e2 = 2 * err
        if e2 > -dy: err -= dy; x1 += sx
        if e2 < dx: err += dx; y1 += sy
    return img


def putText(img, text: str, org, fontFace: int, fontScale: float, color,
            thickness: int = 1):
    return _copy(img)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def threshold(src, thresh: float, maxval: float, type_: int = 0):
    sh = _shape(src); H, W = sh[0], sh[1]
    if type_ == THRESH_BINARY:
        dst = [[int(maxval) if src[r][c] >= thresh else 0 for c in range(W)] for r in range(H)]
    elif type_ == THRESH_BINARY_INV:
        dst = [[int(maxval) if src[r][c] < thresh else 0 for c in range(W)] for r in range(H)]
    else:
        dst = [[int(maxval) if src[r][c] >= thresh else 0 for c in range(W)] for r in range(H)]
    return float(thresh), dst


def addWeighted(src1, alpha, src2, beta, gamma):
    sh = _shape(src1); H, W = sh[0], sh[1]
    if len(sh) == 2:
        return [[_clamp(alpha * src1[r][c] + beta * src2[r][c] + gamma)
                 for c in range(W)] for r in range(H)]
    C = sh[2]
    return [[[_clamp(alpha * src1[r][c][ch] + beta * src2[r][c][ch] + gamma)
              for ch in range(C)] for c in range(W)] for r in range(H)]


__all__ = [
    'imread', 'imwrite', 'imshow', 'waitKey',
    'cvtColor',
    'GaussianBlur', 'blur', 'medianBlur', 'Sobel', 'Laplacian', 'filter2D',
    'Canny',
    'erode', 'dilate', 'morphologyEx', 'getStructuringElement',
    'resize', 'flip', 'rotate',
    'calcHist', 'equalizeHist',
    'boundingRect', 'contourArea',
    'rectangle', 'circle', 'line', 'putText',
    'threshold', 'addWeighted',
    'COLOR_BGR2GRAY', 'COLOR_BGR2RGB', 'COLOR_RGB2BGR', 'COLOR_BGR2HSV',
    'COLOR_HSV2BGR', 'COLOR_GRAY2BGR', 'COLOR_GRAY2RGB', 'COLOR_RGB2GRAY',
    'INTER_NEAREST', 'INTER_LINEAR', 'INTER_CUBIC', 'INTER_AREA',
    'MORPH_RECT', 'MORPH_ELLIPSE', 'MORPH_CROSS',
    'MORPH_ERODE', 'MORPH_DILATE', 'MORPH_OPEN', 'MORPH_CLOSE',
    'BORDER_CONSTANT', 'BORDER_REFLECT', 'BORDER_WRAP',
    'ROTATE_90_CLOCKWISE', 'ROTATE_180', 'ROTATE_90_COUNTERCLOCKWISE',
    'THRESH_BINARY', 'THRESH_BINARY_INV', 'THRESH_TRUNC', 'THRESH_TOZERO',
]
