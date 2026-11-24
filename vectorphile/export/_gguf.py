"""GGUF format reader/writer — llama.cpp compatible (pure Python).

Binary layout:
  [4]  magic  b'GGUF'
  [4]  version  uint32LE = 3
  [8]  tensor_count  uint64LE
  [8]  metadata_kv_count  uint64LE
  [*]  metadata key-value pairs
  [*]  tensor info records
  [*]  tensor data  (page-aligned to 32 bytes)
"""
from __future__ import annotations
import os
import math
import struct
import array as _arr
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# ── GGUF type constants ────────────────────────────────────────────────────────
GGUF_MAGIC = b'GGUF'
GGUF_VERSION = 3

GGUF_VAL_UINT8   = 0
GGUF_VAL_INT8    = 1
GGUF_VAL_UINT16  = 2
GGUF_VAL_INT16   = 3
GGUF_VAL_UINT32  = 4
GGUF_VAL_INT32   = 5
GGUF_VAL_FLOAT32 = 6
GGUF_VAL_BOOL    = 7
GGUF_VAL_STRING  = 8
GGUF_VAL_ARRAY   = 9
GGUF_VAL_UINT64  = 10
GGUF_VAL_INT64   = 11
GGUF_VAL_FLOAT64 = 12

GGML_TYPE_F32  = 0
GGML_TYPE_F16  = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q8_0 = 7
GGML_TYPE_Q8_1 = 8

_VAL_FMT = {
    GGUF_VAL_UINT8:   ('B', 1),
    GGUF_VAL_INT8:    ('b', 1),
    GGUF_VAL_UINT16:  ('H', 2),
    GGUF_VAL_INT16:   ('h', 2),
    GGUF_VAL_UINT32:  ('I', 4),
    GGUF_VAL_INT32:   ('i', 4),
    GGUF_VAL_FLOAT32: ('f', 4),
    GGUF_VAL_BOOL:    ('?', 1),
    GGUF_VAL_UINT64:  ('Q', 8),
    GGUF_VAL_INT64:   ('q', 8),
    GGUF_VAL_FLOAT64: ('d', 8),
}

BLOCK_SIZE = 32


# ── Pure-Python quantization ──────────────────────────────────────────────────

def _prod(shape) -> int:
    n = 1
    for s in shape: n *= s
    return n

def _to_f32_list(data) -> list:
    """Extract flat float list from NDArray/Tensor/list."""
    if hasattr(data, '_data') and hasattr(data._data, 'to_flat_list'):
        return data._data.to_flat_list()
    if hasattr(data, 'to_flat_list'):
        return data.to_flat_list()
    if hasattr(data, '_buf'):
        return list(data._buf)
    if isinstance(data, (list, tuple)):
        out = []
        def _r(v):
            if isinstance(v, (list,tuple)):
                for i in v: _r(i)
            else: out.append(float(v))
        _r(data); return out
    return [float(data)]

def _shape_of(data) -> tuple:
    if hasattr(data, '_data') and hasattr(data._data, 'shape'): return data._data.shape
    if hasattr(data, 'shape'): return data.shape
    return (len(_to_f32_list(data)),)


def quantize_q8_0(data) -> Tuple[list, list]:
    """Q8_0: blocks of 32 int8 + float16 scale per block.
    Returns: (quantized: list[int], scales: list[float])
    """
    flat = _to_f32_list(data)
    pad = (-len(flat)) % BLOCK_SIZE
    flat = flat + [0.0]*pad
    n_blocks = len(flat)//BLOCK_SIZE
    quant = []; scales = []
    for b in range(n_blocks):
        block = flat[b*BLOCK_SIZE:(b+1)*BLOCK_SIZE]
        mx = max(abs(v) for v in block)
        scale = max(mx/127.0, 1e-8)
        scales.append(scale)
        for v in block:
            q = int(round(v/scale))
            quant.append(max(-128, min(127, q)))
    return quant, scales


def dequantize_q8_0(quantized: list, scales: list) -> list:
    n_blocks = len(scales); result = []
    for b in range(n_blocks):
        s = float(scales[b])
        for k in range(BLOCK_SIZE):
            result.append(float(quantized[b*BLOCK_SIZE+k])*s)
    return result


def quantize_q4_0(data) -> Tuple[list, list]:
    """Q4_0: blocks of 32, 4-bit values packed 2 per byte + float16 scale.
    Returns: (packed: list[int] uint8, scales: list[float])
    """
    flat = _to_f32_list(data)
    pad = (-len(flat)) % BLOCK_SIZE
    flat = flat + [0.0]*pad
    n_blocks = len(flat)//BLOCK_SIZE
    packed = []; scales = []
    for b in range(n_blocks):
        block = flat[b*BLOCK_SIZE:(b+1)*BLOCK_SIZE]
        mx = max(abs(v) for v in block)
        scale = max(mx/7.0, 1e-8)
        scales.append(scale)
        q = [max(-8, min(7, int(round(v/scale)))) for v in block]
        q_u = [(v+8) & 0x0F for v in q]
        for i in range(0, BLOCK_SIZE, 2):
            packed.append((q_u[i] & 0x0F) | ((q_u[i+1] & 0x0F) << 4))
    return packed, scales


def dequantize_q4_0(packed: list, scales: list) -> list:
    n_blocks = len(scales); result = []
    for b in range(n_blocks):
        s = float(scales[b])
        for i in range(16):
            lo = (packed[b*16+i] & 0x0F) - 8
            hi = ((packed[b*16+i] >> 4) & 0x0F) - 8
            result.append(float(lo)*s)
            result.append(float(hi)*s)
    return result


# ── Binary helpers ─────────────────────────────────────────────────────────────

def _read_str(f) -> str:
    length = struct.unpack('<Q', f.read(8))[0]
    return f.read(length).decode('utf-8')


def _write_str(f, s: str) -> None:
    b = s.encode('utf-8')
    f.write(struct.pack('<Q', len(b)))
    f.write(b)


def _read_value(f, vtype: int):
    if vtype in _VAL_FMT:
        fmt, size = _VAL_FMT[vtype]
        return struct.unpack('<' + fmt, f.read(size))[0]
    if vtype == GGUF_VAL_STRING:
        return _read_str(f)
    if vtype == GGUF_VAL_ARRAY:
        elem_type = struct.unpack('<I', f.read(4))[0]
        count = struct.unpack('<Q', f.read(8))[0]
        return [_read_value(f, elem_type) for _ in range(count)]
    raise ValueError(f'Unknown GGUF value type: {vtype}')


def _write_value(f, vtype: int, value) -> None:
    if vtype in _VAL_FMT:
        fmt, _ = _VAL_FMT[vtype]
        f.write(struct.pack('<' + fmt, value))
    elif vtype == GGUF_VAL_STRING:
        _write_str(f, value)
    elif vtype == GGUF_VAL_ARRAY:
        elem_type, items = value
        f.write(struct.pack('<I', elem_type))
        f.write(struct.pack('<Q', len(items)))
        for item in items:
            _write_value(f, elem_type, item)


def _infer_vtype(value) -> int:
    if isinstance(value, bool): return GGUF_VAL_BOOL
    if isinstance(value, int): return GGUF_VAL_UINT32
    if isinstance(value, float): return GGUF_VAL_FLOAT32
    if isinstance(value, str): return GGUF_VAL_STRING
    if isinstance(value, (list, tuple)): return GGUF_VAL_ARRAY
    return GGUF_VAL_STRING


def _float32_to_float16_bytes(v: float) -> bytes:
    """Convert float32 to float16 bytes (lossy, pure Python)."""
    b = struct.pack('<f', float(v))
    u = struct.unpack('<I', b)[0]
    sign = (u >> 31) & 0x1
    exp = (u >> 23) & 0xFF
    mantissa = u & 0x7FFFFF
    if exp == 255:
        h = (sign << 15) | 0x7C00 | (1 if mantissa else 0)
    elif exp == 0:
        h = (sign << 15)
    else:
        exp16 = exp - 127 + 15
        if exp16 >= 31:
            h = (sign << 15) | 0x7C00
        elif exp16 <= 0:
            shift = 1 - exp16
            m = (mantissa | 0x800000) >> (shift + 13)
            h = (sign << 15) | (m & 0x3FF)
        else:
            h = (sign << 15) | (exp16 << 10) | (mantissa >> 13)
    return struct.pack('<H', h & 0xFFFF)


def _float16_bytes_to_float32(b2: bytes) -> float:
    """Convert float16 bytes to float32."""
    h = struct.unpack('<H', b2)[0]
    sign = (h >> 15) & 0x1
    exp = (h >> 10) & 0x1F
    mantissa = h & 0x3FF
    if exp == 31:
        return float('inf') * (-1 if sign else 1) if mantissa == 0 else float('nan')
    if exp == 0:
        v = mantissa / (1 << 24)
    else:
        v = (1.0 + mantissa / (1 << 10)) * (2.0 ** (exp - 15))
    return -v if sign else v


# ── GGUFReader ─────────────────────────────────────────────────────────────────

class GGUFReader:
    """Read llama.cpp compatible .gguf files."""

    def __init__(self, path: str):
        self._path = path
        self._metadata: Dict[str, Any] = {}
        self._tensor_infos: List[dict] = []
        self._data_offset: int = 0
        self._parsed = False

    def _parse(self):
        if self._parsed: return
        with open(self._path, 'rb') as f:
            magic = f.read(4)
            if magic != GGUF_MAGIC:
                raise ValueError(f'Not a GGUF file (magic={magic})')
            struct.unpack('<I', f.read(4))[0]  # version
            tensor_count = struct.unpack('<Q', f.read(8))[0]
            kv_count = struct.unpack('<Q', f.read(8))[0]
            for _ in range(kv_count):
                key = _read_str(f)
                vtype = struct.unpack('<I', f.read(4))[0]
                self._metadata[key] = _read_value(f, vtype)
            for _ in range(tensor_count):
                name = _read_str(f)
                n_dims = struct.unpack('<I', f.read(4))[0]
                shape = list(struct.unpack('<' + 'Q'*n_dims, f.read(8*n_dims)))
                dtype = struct.unpack('<I', f.read(4))[0]
                offset = struct.unpack('<Q', f.read(8))[0]
                self._tensor_infos.append({'name': name,'shape': shape,'dtype': dtype,'offset': offset})
            pos = f.tell()
            self._data_offset = (pos+31) & ~31
        self._parsed = True

    @property
    def metadata(self) -> dict:
        self._parse(); return self._metadata

    @property
    def arch(self) -> str:
        self._parse(); return self._metadata.get('general.architecture', 'unknown')

    def read_header(self) -> dict:
        self._parse(); return dict(self._metadata)

    def list_tensors(self) -> List[dict]:
        self._parse(); return list(self._tensor_infos)

    def read_tensor(self, name: str) -> list:
        self._parse()
        info = next((t for t in self._tensor_infos if t['name']==name), None)
        if info is None: raise KeyError(f'Tensor not found: {name}')
        return self._load_tensor(info)

    def _load_tensor(self, info: dict) -> list:
        dtype = info['dtype']; shape = info['shape']
        offset = self._data_offset + info['offset']
        n_elem = _prod(shape) if shape else 1
        with open(self._path, 'rb') as f:
            f.seek(offset)
            if dtype == GGML_TYPE_F32:
                raw = f.read(n_elem*4)
                data = list(struct.unpack(f'<{n_elem}f', raw))
            elif dtype == GGML_TYPE_F16:
                raw = f.read(n_elem*2)
                data = [_float16_bytes_to_float32(raw[i*2:i*2+2]) for i in range(n_elem)]
            elif dtype == GGML_TYPE_Q8_0:
                n_blocks = (n_elem+BLOCK_SIZE-1)//BLOCK_SIZE
                scales = [_float16_bytes_to_float32(f.read(2)) for _ in range(n_blocks)]
                raw = f.read(n_blocks*BLOCK_SIZE)
                q = list(struct.unpack(f'<{n_blocks*BLOCK_SIZE}b', raw))
                data = dequantize_q8_0(q, scales)[:n_elem]
            elif dtype == GGML_TYPE_Q4_0:
                n_blocks = (n_elem+BLOCK_SIZE-1)//BLOCK_SIZE
                scales = [_float16_bytes_to_float32(f.read(2)) for _ in range(n_blocks)]
                raw = f.read(n_blocks*16)
                packed = list(struct.unpack(f'<{n_blocks*16}B', raw))
                data = dequantize_q4_0(packed, scales)[:n_elem]
            else:
                data = [0.0]*n_elem
        return data

    def read_all(self) -> Dict[str, list]:
        self._parse()
        return {t['name']: self._load_tensor(t) for t in self._tensor_infos}


# ── GGUFWriter ─────────────────────────────────────────────────────────────────

class GGUFWriter:
    """Write llama.cpp compatible .gguf files."""

    def __init__(self, path: str, arch: str = 'llama'):
        self._path = path
        self._metadata: List[Tuple[str, int, Any]] = []
        self._tensors: List[dict] = []
        self.add_metadata('general.architecture', arch)
        self.add_metadata('general.file_type', 1)

    def add_metadata(self, key: str, value) -> None:
        vtype = _infer_vtype(value)
        self._metadata.append((key, vtype, value))

    def add_tensor(self, name: str, data, quantize: str = 'f16') -> None:
        self._tensors.append({'name': name, 'data': data, 'quantize': quantize})

    def write(self) -> None:
        import io as _io
        tensor_data_buf = _io.BytesIO(); offsets = []
        for t in self._tensors:
            pos = tensor_data_buf.tell(); pad = (-pos)%32
            if pad: tensor_data_buf.write(b'\x00'*pad)
            offsets.append(tensor_data_buf.tell())
            _write_tensor_data(tensor_data_buf, t['data'], t['quantize'])
        with open(self._path, 'wb') as f:
            f.write(GGUF_MAGIC); f.write(struct.pack('<I', GGUF_VERSION))
            f.write(struct.pack('<Q', len(self._tensors)))
            f.write(struct.pack('<Q', len(self._metadata)))
            for key, vtype, value in self._metadata:
                _write_str(f, key); f.write(struct.pack('<I', vtype)); _write_value(f, vtype, value)
            for t, offset in zip(self._tensors, offsets):
                _write_str(f, t['name'])
                flat = _to_f32_list(t['data']); sh = _shape_of(t['data'])
                f.write(struct.pack('<I', len(sh)))
                f.write(struct.pack('<'+'Q'*len(sh), *sh))
                f.write(struct.pack('<I', _quant_to_dtype(t['quantize'])))
                f.write(struct.pack('<Q', offset))
            pos = f.tell(); pad = (-pos)%32
            if pad: f.write(b'\x00'*pad)
            f.write(tensor_data_buf.getvalue())


def _quant_to_dtype(quantize: str) -> int:
    return {'f32': GGML_TYPE_F32, 'f16': GGML_TYPE_F16,
            'q8_0': GGML_TYPE_Q8_0, 'q4_0': GGML_TYPE_Q4_0}.get(quantize, GGML_TYPE_F16)


def _write_tensor_data(f, data, quantize: str) -> None:
    flat = _to_f32_list(data)
    if quantize == 'f32':
        f.write(struct.pack(f'<{len(flat)}f', *flat))
    elif quantize == 'f16':
        for v in flat: f.write(_float32_to_float16_bytes(v))
    elif quantize == 'q8_0':
        q, scales = quantize_q8_0(flat)
        n_blocks = len(scales)
        for i in range(n_blocks):
            f.write(_float32_to_float16_bytes(scales[i]))
            blk = q[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE]
            f.write(struct.pack(f'<{BLOCK_SIZE}b', *blk))
    elif quantize == 'q4_0':
        packed, scales = quantize_q4_0(flat)
        n_blocks = len(scales)
        for i in range(n_blocks):
            f.write(_float32_to_float16_bytes(scales[i]))
            blk = packed[i*16:(i+1)*16]
            f.write(struct.pack(f'<{16}B', *blk))
    else:
        for v in flat: f.write(_float32_to_float16_bytes(v))


# ── High-level helpers ────────────────────────────────────────────────────────

def to_gguf(model, path: str, quantize: str = 'f16', arch: str = None) -> None:
    """Export a vectorphile model to GGUF format."""
    if arch is None:
        arch = model.__class__.__name__.lower()
    writer = GGUFWriter(path, arch)
    writer.add_metadata('general.name', model.__class__.__name__)
    writer.add_metadata('general.quantization_version', 2)
    for name, param in model.named_parameters():
        writer.add_tensor(name.replace('.', '_'), param, quantize)
    writer.write()
    size_mb = os.path.getsize(path)/1e6
    print(f'Saved {len(list(model.named_parameters()))} tensors → {path} ({size_mb:.1f} MB)')


def from_gguf(path: str):
    """Load a GGUF file and return (weights_dict, metadata).

    Returns:
        weights: dict mapping tensor names to float32 lists
        metadata: dict of model metadata
    """
    reader = GGUFReader(path)
    weights = reader.read_all()
    return weights, reader.metadata
