"""vectorphile.image — PIL/Pillow-compatible image processing.

Provides image loading, saving, and processing operations built from scratch.
Where format I/O is needed, uses Python's built-in struct/zlib for PNG,
and stdlib imghdr for detection.

Usage::

    from vectorphile.image import Image, open, fromarray

    img = Image.open('photo.jpg')
    img_resized = img.resize((224, 224))
    arr = img.to_list()   # → nested Python list H×W×C

    # Create from list
    img2 = Image.fromarray(arr)
    img2.save('output.png')
"""
from __future__ import annotations
import os
import struct
import zlib
import math
from typing import Optional, Tuple, Union, List


# ---------------------------------------------------------------------------
# Color mode helpers
# ---------------------------------------------------------------------------

MODES = {'L', 'RGB', 'RGBA', 'F', '1'}


def _clamp(v: float, lo: int = 0, hi: int = 255) -> int:
    return max(lo, min(hi, int(v)))


def _shape(data) -> tuple:
    """Return (H, W) or (H, W, C) for nested-list image data."""
    H = len(data)
    if H == 0:
        return (0, 0)
    W = len(data[0])
    if isinstance(data[0][0], (list, tuple)):
        C = len(data[0][0])
        return (H, W, C)
    return (H, W)


def _channels(data) -> int:
    sh = _shape(data)
    return sh[2] if len(sh) == 3 else 1


def _copy(data):
    """Deep copy of nested list image."""
    if isinstance(data[0][0], (list, tuple)):
        return [[list(px) for px in row] for row in data]
    return [[v for v in row] for row in data]


def _get(data, y: int, x: int):
    """Get pixel value (int or list)."""
    return data[y][x]


def _set(data, y: int, x: int, val) -> None:
    data[y][x] = val


def _ensure_uint8_list(data):
    """Clamp all pixel values to [0, 255] ints in a new nested list."""
    sh = _shape(data)
    if len(sh) == 2:
        return [[_clamp(v) for v in row] for row in data]
    return [[[ _clamp(ch) for ch in px] for px in row] for row in data]


def _zeros_data(H: int, W: int, C: int):
    if C == 1:
        return [[0] * W for _ in range(H)]
    return [[[0] * C for _ in range(W)] for _ in range(H)]


def _full_data(H: int, W: int, C: int, val):
    if C == 1:
        return [[int(val)] * W for _ in range(H)]
    if isinstance(val, (list, tuple)):
        return [[list(val) for _ in range(W)] for _ in range(H)]
    return [[[int(val)] * C for _ in range(W)] for _ in range(H)]


# ---------------------------------------------------------------------------
# Image class
# ---------------------------------------------------------------------------

class Image:
    """PIL-compatible image class backed by pure Python nested lists.

    _data layout:
      - Grayscale ('L', 'F', '1'): List[List[int]]
      - Color ('RGB', 'RGBA'):      List[List[List[int]]]  (H×W×C)

    Attributes:
        mode: color mode string ('L', 'RGB', 'RGBA', 'F')
        size: (width, height) tuple
        width, height: image dimensions
    """

    def __init__(self, data, mode: str = 'RGB'):
        # Accept nested lists directly
        if isinstance(data, list):
            self._data = data
        else:
            # Fallback: try to convert from something iterable
            try:
                self._data = [list(row) for row in data]
            except Exception:
                self._data = []
        self.mode = mode

    @property
    def size(self) -> Tuple[int, int]:
        sh = _shape(self._data)
        if len(sh) >= 2:
            return (sh[1], sh[0])  # (W, H)
        return (0, 0)

    @property
    def width(self) -> int:
        return self.size[0]

    @property
    def height(self) -> int:
        return self.size[1]

    def to_list(self):
        """Return copy as nested Python list."""
        return _copy(self._data)

    def to_array(self):
        """Return as nested list (pure Python replacement for numpy array)."""
        return _copy(self._data)

    # ── Resizing ───────────────────────────────────────────────────────────

    def resize(self, size: Tuple[int, int], resample: str = 'bilinear') -> 'Image':
        """Resize image. size = (width, height)."""
        new_w, new_h = size
        src = self._data
        sh = _shape(src)
        H, W = sh[0], sh[1]
        C = sh[2] if len(sh) == 3 else 1
        x_ratio = W / new_w
        y_ratio = H / new_h

        if C == 1:
            out = []
            for ny in range(new_h):
                yi = (ny + 0.5) * y_ratio - 0.5
                y0 = max(0, min(H - 1, int(math.floor(yi))))
                y1 = max(0, min(H - 1, y0 + 1))
                dy = yi - math.floor(yi)
                row = []
                for nx in range(new_w):
                    xi = (nx + 0.5) * x_ratio - 0.5
                    x0 = max(0, min(W - 1, int(math.floor(xi))))
                    x1 = max(0, min(W - 1, x0 + 1))
                    dx = xi - math.floor(xi)
                    v = (src[y0][x0] * (1 - dx) * (1 - dy) +
                         src[y0][x1] * dx * (1 - dy) +
                         src[y1][x0] * (1 - dx) * dy +
                         src[y1][x1] * dx * dy)
                    row.append(_clamp(v))
                out.append(row)
        else:
            out = []
            for ny in range(new_h):
                yi = (ny + 0.5) * y_ratio - 0.5
                y0 = max(0, min(H - 1, int(math.floor(yi))))
                y1 = max(0, min(H - 1, y0 + 1))
                dy = yi - math.floor(yi)
                row = []
                for nx in range(new_w):
                    xi = (nx + 0.5) * x_ratio - 0.5
                    x0 = max(0, min(W - 1, int(math.floor(xi))))
                    x1 = max(0, min(W - 1, x0 + 1))
                    dx = xi - math.floor(xi)
                    px = []
                    for c in range(C):
                        v = (src[y0][x0][c] * (1 - dx) * (1 - dy) +
                             src[y0][x1][c] * dx * (1 - dy) +
                             src[y1][x0][c] * (1 - dx) * dy +
                             src[y1][x1][c] * dx * dy)
                        px.append(_clamp(v))
                    row.append(px)
                out.append(row)
        return Image(out, mode=self.mode)

    # ── Crop ───────────────────────────────────────────────────────────────

    def crop(self, box: Tuple[int, int, int, int]) -> 'Image':
        """Crop to (left, upper, right, lower)."""
        l, u, r, lo = box
        sh = _shape(self._data)
        if len(sh) == 2:
            out = [row[l:r] for row in self._data[u:lo]]
        else:
            out = [[px for px in row[l:r]] for row in self._data[u:lo]]
        return Image(out, mode=self.mode)

    # ── Flip / Rotate ──────────────────────────────────────────────────────

    def transpose(self, method: int) -> 'Image':
        """PIL-compatible transpose (FLIP_LEFT_RIGHT=0, FLIP_TOP_BOTTOM=1,
        ROTATE_90=2, ROTATE_180=3, ROTATE_270=4)."""
        src = self._data
        sh = _shape(src)
        H, W = sh[0], sh[1]
        C = sh[2] if len(sh) == 3 else 1

        if method == 0:  # FLIP_LEFT_RIGHT
            if C == 1:
                out = [list(reversed(row)) for row in src]
            else:
                out = [list(reversed(row)) for row in src]
        elif method == 1:  # FLIP_TOP_BOTTOM
            out = list(reversed(src))
            if C > 1:
                out = [list(row) for row in out]
        elif method == 2:  # ROTATE_90 (CCW)
            if C == 1:
                out = [[src[H - 1 - x][y] for x in range(H)] for y in range(W)]
            else:
                out = [[list(src[H - 1 - x][y]) for x in range(H)] for y in range(W)]
        elif method == 3:  # ROTATE_180
            if C == 1:
                out = [list(reversed(row)) for row in reversed(src)]
            else:
                out = [list(reversed(row)) for row in reversed(src)]
        elif method == 4:  # ROTATE_270 (CW)
            if C == 1:
                out = [[src[x][W - 1 - y] for x in range(H)] for y in range(W)]
            else:
                out = [[list(src[x][W - 1 - y]) for x in range(H)] for y in range(W)]
        else:
            return Image(_copy(src), self.mode)
        return Image(out, self.mode)

    # Constants
    FLIP_LEFT_RIGHT = 0
    FLIP_TOP_BOTTOM = 1
    ROTATE_90 = 2
    ROTATE_180 = 3
    ROTATE_270 = 4

    def rotate(self, angle: float, expand: bool = False, fillcolor=0) -> 'Image':
        """Rotate image by angle degrees (counter-clockwise)."""
        src = self._data
        sh = _shape(src)
        H, W = sh[0], sh[1]
        C = sh[2] if len(sh) == 3 else 1
        cx, cy = W / 2.0, H / 2.0
        rad = math.radians(-angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        fill_px = [int(fillcolor)] * C if C > 1 else int(fillcolor)

        if C == 1:
            out = []
            for y in range(H):
                row = []
                for x in range(W):
                    xs = cos_a * (x - cx) - sin_a * (y - cy) + cx
                    ys = sin_a * (x - cx) + cos_a * (y - cy) + cy
                    if 0 <= xs < W and 0 <= ys < H:
                        row.append(src[int(ys)][int(xs)])
                    else:
                        row.append(fill_px)
                out.append(row)
        else:
            out = []
            for y in range(H):
                row = []
                for x in range(W):
                    xs = cos_a * (x - cx) - sin_a * (y - cy) + cx
                    ys = sin_a * (x - cx) + cos_a * (y - cy) + cy
                    if 0 <= xs < W and 0 <= ys < H:
                        row.append(list(src[int(ys)][int(xs)]))
                    else:
                        row.append(list(fill_px))
                out.append(row)
        return Image(out, self.mode)

    # ── Color conversion ───────────────────────────────────────────────────

    def convert(self, mode: str) -> 'Image':
        """Convert to a different mode (e.g. 'L', 'RGB', 'RGBA')."""
        src = self._data
        sh = _shape(src)
        H, W = sh[0], sh[1]
        C = sh[2] if len(sh) == 3 else 1

        if mode == self.mode:
            return Image(_copy(src), mode)

        if mode == 'L':
            if C >= 3:
                out = [[_clamp(0.299 * src[y][x][0] + 0.587 * src[y][x][1] +
                                0.114 * src[y][x][2]) for x in range(W)] for y in range(H)]
            else:
                out = _copy(src)
            return Image(out, 'L')

        elif mode == 'RGB':
            if C == 1:
                out = [[[src[y][x], src[y][x], src[y][x]] for x in range(W)] for y in range(H)]
            elif C == 4:
                out = [[list(src[y][x][:3]) for x in range(W)] for y in range(H)]
            else:
                out = _copy(src)
            return Image(out, 'RGB')

        elif mode == 'RGBA':
            if C == 1:
                out = [[[src[y][x], src[y][x], src[y][x], 255] for x in range(W)] for y in range(H)]
            elif C == 3:
                out = [[src[y][x] + [255] for x in range(W)] for y in range(H)]
            else:
                out = _copy(src)
            return Image(out, 'RGBA')

        return Image(_copy(src), self.mode)

    # ── Pixel access ───────────────────────────────────────────────────────

    def getpixel(self, xy: Tuple[int, int]):
        x, y = xy
        px = self._data[y][x]
        return tuple(px) if isinstance(px, list) else int(px)

    def putpixel(self, xy: Tuple[int, int], value) -> None:
        x, y = xy
        if isinstance(value, (list, tuple)):
            self._data[y][x] = list(value)
        else:
            self._data[y][x] = int(value)

    def getdata(self):
        sh = _shape(self._data)
        H, W = sh[0], sh[1]
        result = []
        for y in range(H):
            for x in range(W):
                px = self._data[y][x]
                result.append(tuple(px) if isinstance(px, list) else int(px))
        return result

    # ── File I/O ───────────────────────────────────────────────────────────

    @classmethod
    def open(cls, fp: str) -> 'Image':
        """Open image from file. Supports PNG (native) and formats via PIL fallback."""
        try:
            from PIL import Image as PILImage
            import array as _arr
            pil_img = PILImage.open(fp)
            mode = pil_img.mode
            W, H = pil_img.size
            raw = list(pil_img.getdata())
            if mode == 'L':
                data = [[int(raw[y * W + x]) for x in range(W)] for y in range(H)]
            else:
                C = len(raw[0]) if isinstance(raw[0], tuple) else 1
                data = [[list(raw[y * W + x]) for x in range(W)] for y in range(H)]
            return cls(data, mode)
        except ImportError:
            pass
        if fp.lower().endswith('.png'):
            return cls._read_png(fp)
        if fp.lower().endswith('.ppm'):
            return cls._read_ppm(fp)
        raise ValueError(f'Cannot read image format: {fp}. Install Pillow for full format support.')

    def save(self, fp: str, format: Optional[str] = None) -> None:
        """Save image to file."""
        try:
            from PIL import Image as PILImage
            sh = _shape(self._data)
            H, W = sh[0], sh[1]
            C = sh[2] if len(sh) == 3 else 1
            if C == 1:
                flat = bytes(_clamp(self._data[y][x]) for y in range(H) for x in range(W))
                pil_img = PILImage.frombytes(self.mode, (W, H), flat)
            else:
                flat = bytes(_clamp(ch) for y in range(H) for x in range(W) for ch in self._data[y][x])
                pil_img = PILImage.frombytes(self.mode, (W, H), flat)
            pil_img.save(fp, format=format)
            return
        except ImportError:
            pass
        if fp.lower().endswith('.png') or format == 'PNG':
            self._write_png(fp)
        elif fp.lower().endswith('.ppm') or format == 'PPM':
            self._write_ppm(fp)
        else:
            raise ValueError(f'Cannot write format: {fp}. Install Pillow for full support.')

    @classmethod
    def fromarray(cls, arr, mode: Optional[str] = None) -> 'Image':
        """Create Image from nested list or any iterable-of-iterables."""
        # Handle nested list directly
        if isinstance(arr, list):
            data = arr
            sh = _shape(data)
        else:
            # Try to iterate rows
            try:
                rows = list(arr)
                if rows and isinstance(rows[0], (list, tuple)):
                    if isinstance(rows[0][0], (list, tuple)):
                        data = [[list(px) for px in row] for row in rows]
                    else:
                        data = [[v for v in row] for row in rows]
                else:
                    data = rows
            except Exception:
                data = []
            sh = _shape(data)

        if mode is None:
            if len(sh) == 2:
                mode = 'L'
            elif len(sh) == 3 and sh[2] == 3:
                mode = 'RGB'
            elif len(sh) == 3 and sh[2] == 4:
                mode = 'RGBA'
            else:
                mode = 'L'
        return cls(data, mode)

    @classmethod
    def new(cls, mode: str, size: Tuple[int, int], color=0) -> 'Image':
        w, h = size
        if mode == 'L':
            data = [[_clamp(color)] * w for _ in range(h)]
        elif mode == 'RGB':
            if isinstance(color, (list, tuple)):
                data = [[list(color[:3]) for _ in range(w)] for _ in range(h)]
            else:
                data = [[[_clamp(color)] * 3 for _ in range(w)] for _ in range(h)]
        elif mode == 'RGBA':
            if isinstance(color, (list, tuple)):
                c = list(color) + [255] * (4 - len(color))
                data = [[c[:4] for _ in range(w)] for _ in range(h)]
            else:
                data = [[[_clamp(color)] * 4 for _ in range(w)] for _ in range(h)]
        else:
            data = [[0] * w for _ in range(h)]
        return cls(data, mode)

    # ── PNG codec (minimal) ────────────────────────────────────────────────

    @staticmethod
    def _read_png(path: str) -> 'Image':
        with open(path, 'rb') as f:
            sig = f.read(8)
            assert sig == b'\x89PNG\r\n\x1a\n', 'Not a PNG file'
            idat_data = b''
            width = height = bit_depth = color_type = 0
            while True:
                length_bytes = f.read(4)
                if len(length_bytes) < 4:
                    break
                length = struct.unpack('>I', length_bytes)[0]
                chunk_type = f.read(4)
                data = f.read(length)
                f.read(4)  # CRC
                if chunk_type == b'IHDR':
                    width, height = struct.unpack('>II', data[:8])
                    bit_depth, color_type = data[8], data[9]
                elif chunk_type == b'IDAT':
                    idat_data += data
                elif chunk_type == b'IEND':
                    break
        raw = zlib.decompress(idat_data)
        channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color_type, 3)
        stride = width * channels + 1
        rows_bytes = []
        prev = bytes(width * channels)
        for row_i in range(height):
            row_start = row_i * stride
            filter_byte = raw[row_start]
            row = bytearray(raw[row_start + 1:row_start + stride])
            if filter_byte == 1:  # Sub
                for i in range(channels, len(row)):
                    row[i] = (row[i] + row[i - channels]) & 0xFF
            elif filter_byte == 2:  # Up
                for i in range(len(row)):
                    row[i] = (row[i] + prev[i]) & 0xFF
            elif filter_byte == 3:  # Average
                for i in range(len(row)):
                    a = row[i - channels] if i >= channels else 0
                    b = prev[i]
                    row[i] = (row[i] + (a + b) // 2) & 0xFF
            elif filter_byte == 4:  # Paeth
                for i in range(len(row)):
                    a = row[i - channels] if i >= channels else 0
                    b = prev[i]
                    c = prev[i - channels] if i >= channels else 0
                    p = a + b - c
                    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                    pr = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                    row[i] = (row[i] + pr) & 0xFF
            rows_bytes.append(bytes(row))
            prev = bytes(row)

        all_bytes = b''.join(rows_bytes)
        mode = {1: 'L', 3: 'RGB', 4: 'RGBA'}.get(channels, 'RGB')
        if channels == 1:
            data = [[all_bytes[y * width + x] for x in range(width)] for y in range(height)]
        else:
            data = [[list(all_bytes[y * width * channels + x * channels:
                                    y * width * channels + x * channels + channels])
                     for x in range(width)] for y in range(height)]
        return Image(data, mode)

    def _write_ppm(self, path: str) -> None:
        sh = _shape(self._data)
        H, W = sh[0], sh[1]
        C = sh[2] if len(sh) == 3 else 1
        if C == 1:
            with open(path, 'wb') as f:
                f.write(f'P5\n{W} {H}\n255\n'.encode())
                f.write(bytes(_clamp(self._data[y][x]) for y in range(H) for x in range(W)))
        else:
            with open(path, 'wb') as f:
                f.write(f'P6\n{W} {H}\n255\n'.encode())
                f.write(bytes(_clamp(ch) for y in range(H) for x in range(W)
                              for ch in self._data[y][x][:3]))

    @staticmethod
    def _read_ppm(path: str) -> 'Image':
        with open(path, 'rb') as f:
            header = f.readline().strip()
            while True:
                line = f.readline().strip()
                if not line.startswith(b'#'):
                    break
            w, h = map(int, line.split())
            f.readline()  # maxval
            raw = f.read()
        if header == b'P6':
            data = [[list(raw[y * w * 3 + x * 3: y * w * 3 + x * 3 + 3])
                     for x in range(w)] for y in range(h)]
            return Image(data, 'RGB')
        else:
            data = [[raw[y * w + x] for x in range(w)] for y in range(h)]
            return Image(data, 'L')

    def _write_png(self, path: str) -> None:
        sh = _shape(self._data)
        H, W = sh[0], sh[1]
        C = sh[2] if len(sh) == 3 else 1
        color_type = {1: 0, 3: 2, 4: 6}.get(C, 2)
        rows = []
        for y in range(H):
            if C == 1:
                row_bytes = bytes(_clamp(self._data[y][x]) for x in range(W))
            else:
                row_bytes = bytes(_clamp(ch) for x in range(W) for ch in self._data[y][x])
            rows.append(b'\x00' + row_bytes)
        raw = b''.join(rows)
        compressed = zlib.compress(raw, level=6)

        def chunk(name: bytes, data: bytes) -> bytes:
            length = struct.pack('>I', len(data))
            crc = struct.pack('>I', zlib.crc32(name + data) & 0xFFFFFFFF)
            return length + name + data + crc

        ihdr_data = struct.pack('>IIBBBBB', W, H, 8, color_type, 0, 0, 0)
        with open(path, 'wb') as f:
            f.write(b'\x89PNG\r\n\x1a\n')
            f.write(chunk(b'IHDR', ihdr_data))
            f.write(chunk(b'IDAT', compressed))
            f.write(chunk(b'IEND', b''))

    # ── Operations ─────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f'<Image mode={self.mode!r} size={self.size}>'

    def copy(self) -> 'Image':
        return Image(_copy(self._data), self.mode)

    def paste(self, im: 'Image', box: Optional[Tuple] = None,
              mask: Optional['Image'] = None) -> None:
        """Paste another image at given position (in-place)."""
        x0, y0 = box[:2] if box else (0, 0)
        sh = _shape(im._data)
        h, w = sh[0], sh[1]
        C = sh[2] if len(sh) == 3 else 1
        if mask is not None:
            msh = _shape(mask._data)
            for dy in range(h):
                for dx in range(w):
                    ty, tx = y0 + dy, x0 + dx
                    if 0 <= ty < self.height and 0 <= tx < self.width:
                        m_px = mask._data[dy][dx]
                        m = (m_px if isinstance(m_px, (int, float)) else m_px[0]) / 255.0
                        src_px = im._data[dy][dx]
                        dst_px = self._data[ty][tx]
                        if isinstance(src_px, list):
                            self._data[ty][tx] = [
                                _clamp(src_px[c] * m + dst_px[c] * (1 - m))
                                for c in range(len(src_px))
                            ]
                        else:
                            self._data[ty][tx] = _clamp(src_px * m + dst_px * (1 - m))
        else:
            for dy in range(h):
                for dx in range(w):
                    ty, tx = y0 + dy, x0 + dx
                    if 0 <= ty < self.height and 0 <= tx < self.width:
                        px = im._data[dy][dx]
                        self._data[ty][tx] = list(px) if isinstance(px, list) else px

    def thumbnail(self, size: Tuple[int, int]) -> None:
        """Resize in-place to fit within size, preserving aspect ratio."""
        w, h = self.size
        max_w, max_h = size
        ratio = min(max_w / w, max_h / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        resized = self.resize((new_w, new_h))
        self._data = resized._data

    def split(self):
        """Split into individual channel images."""
        sh = _shape(self._data)
        H, W = sh[0], sh[1]
        C = sh[2] if len(sh) == 3 else 1
        if C == 1:
            return (Image(_copy(self._data), 'L'),)
        return tuple(
            Image([[self._data[y][x][c] for x in range(W)] for y in range(H)], 'L')
            for c in range(C)
        )

    @staticmethod
    def merge(mode: str, bands) -> 'Image':
        """Merge channel images back."""
        sh = _shape(bands[0]._data)
        H, W = sh[0], sh[1]
        data = [[
            [bands[c]._data[y][x] for c in range(len(bands))]
            for x in range(W)
        ] for y in range(H)]
        return Image(data, mode)


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def open(fp: str) -> Image:
    return Image.open(fp)


def fromarray(arr, mode: Optional[str] = None) -> Image:
    return Image.fromarray(arr, mode)


def new(mode: str, size: Tuple[int, int], color=0) -> Image:
    return Image.new(mode, size, color)


__all__ = [
    'Image', 'open', 'fromarray', 'new',
    'MODES',
]
