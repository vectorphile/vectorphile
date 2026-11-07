"""GGUF format reader/writer — llama.cpp compatible.

Binary layout:
  [4]  magic  b'GGUF'
  [4]  version  uint32LE = 3
  [8]  tensor_count  uint64LE
  [8]  metadata_kv_count  uint64LE
  [*]  metadata key-value pairs
  [*]  tensor info records
  [*]  tensor data  (page-aligned to 32 bytes)

Value types used in metadata:
  0=UINT8 1=INT8 2=UINT16 3=INT16 4=UINT32 5=INT32
  6=FLOAT32 7=BOOL 8=STRING 9=ARRAY 10=UINT64 11=INT64 12=FLOAT64

Tensor dtype enum:
  0=F32 1=F16 2=Q4_0 3=Q4_1 7=Q8_0 8=Q8_1
"""
from __future__ import annotations
import os
import struct
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# ── GGUF type constants ────────────────────────────────────────────────────────
GGUF_MAGIC = b'GGUF'
GGUF_VERSION = 3

# Metadata value types
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

# Tensor types
GGML_TYPE_F32  = 0
GGML_TYPE_F16  = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q8_0 = 7
GGML_TYPE_Q8_1 = 8

GGML_TYPE_TO_NP = {
    GGML_TYPE_F32: np.float32,
    GGML_TYPE_F16: np.float16,
}

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

BLOCK_SIZE = 32  # Q4_0 and Q8_0 block size


# ── Quantization / dequantization ─────────────────────────────────────────────

def quantize_q8_0(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Q8_0: blocks of 32, int8 values + float16 scale per block.

    Returns:
        quantized: (n_blocks * 32,) int8
        scales:    (n_blocks,) float16
    """
    flat = data.flatten().astype(np.float32)
    pad = (-len(flat)) % BLOCK_SIZE
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
    blocks = flat.reshape(-1, BLOCK_SIZE)
    scales = np.abs(blocks).max(axis=1) / 127.0
    scales = np.where(scales == 0, 1e-8, scales).astype(np.float16)
    q = np.round(blocks / scales[:, None].astype(np.float32)).clip(-128, 127).astype(np.int8)
    return q.flatten(), scales


def dequantize_q8_0(quantized: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Dequantize Q8_0 blocks back to float32."""
    q = quantized.reshape(-1, BLOCK_SIZE).astype(np.float32)
    return (q * scales.astype(np.float32)[:, None]).flatten()


def quantize_q4_0(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Q4_0: blocks of 32, 4-bit values packed 2 per byte + float16 scale.

    Returns:
        packed: (n_blocks * 16,) uint8 — two 4-bit values per byte
        scales: (n_blocks,) float16
    """
    flat = data.flatten().astype(np.float32)
    pad = (-len(flat)) % BLOCK_SIZE
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
    blocks = flat.reshape(-1, BLOCK_SIZE)
    scales = np.abs(blocks).max(axis=1) / 7.0
    scales = np.where(scales == 0, 1e-8, scales).astype(np.float16)
    q = np.round(blocks / scales[:, None].astype(np.float32)).clip(-8, 7).astype(np.int8)
    # Offset by 8 to get [0,15]
    q_u = (q + 8).astype(np.uint8)  # (n_blocks, 32)
    # Pack pairs: low nibble = q_u[:, 0::2], high nibble = q_u[:, 1::2]
    low  = q_u[:, 0::2] & 0x0F
    high = (q_u[:, 1::2] & 0x0F) << 4
    packed = (low | high).astype(np.uint8)  # (n_blocks, 16)
    return packed.flatten(), scales


def dequantize_q4_0(packed: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Dequantize Q4_0 blocks back to float32."""
    n_blocks = len(scales)
    p = packed.reshape(n_blocks, 16).astype(np.uint8)
    low  = (p & 0x0F).astype(np.int8)
    high = ((p >> 4) & 0x0F).astype(np.int8)
    # Interleave low and high
    q = np.empty((n_blocks, 32), dtype=np.int8)
    q[:, 0::2] = low
    q[:, 1::2] = high
    # Undo offset
    q = (q.astype(np.int16) - 8).astype(np.int8)
    return (q.astype(np.float32) * scales.astype(np.float32)[:, None]).flatten()


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
    if isinstance(value, bool):
        return GGUF_VAL_BOOL
    if isinstance(value, int):
        return GGUF_VAL_UINT32
    if isinstance(value, float):
        return GGUF_VAL_FLOAT32
    if isinstance(value, str):
        return GGUF_VAL_STRING
    if isinstance(value, (list, tuple)):
        return GGUF_VAL_ARRAY
    return GGUF_VAL_STRING


# ── GGUFReader ─────────────────────────────────────────────────────────────────

class GGUFReader:
    """Read llama.cpp compatible .gguf files.

    Args:
        path: path to .gguf file
    """

    def __init__(self, path: str):
        self._path = path
        self._metadata: Dict[str, Any] = {}
        self._tensor_infos: List[dict] = []
        self._data_offset: int = 0
        self._parsed = False

    def _parse(self):
        if self._parsed:
            return
        with open(self._path, 'rb') as f:
            magic = f.read(4)
            if magic != GGUF_MAGIC:
                raise ValueError(f'Not a GGUF file (magic={magic})')
            version = struct.unpack('<I', f.read(4))[0]
            tensor_count = struct.unpack('<Q', f.read(8))[0]
            kv_count = struct.unpack('<Q', f.read(8))[0]

            # Read metadata
            for _ in range(kv_count):
                key = _read_str(f)
                vtype = struct.unpack('<I', f.read(4))[0]
                value = _read_value(f, vtype)
                self._metadata[key] = value

            # Read tensor infos
            for _ in range(tensor_count):
                name = _read_str(f)
                n_dims = struct.unpack('<I', f.read(4))[0]
                shape = list(struct.unpack('<' + 'Q' * n_dims, f.read(8 * n_dims)))
                dtype = struct.unpack('<I', f.read(4))[0]
                offset = struct.unpack('<Q', f.read(8))[0]
                self._tensor_infos.append({
                    'name': name,
                    'shape': shape,
                    'dtype': dtype,
                    'offset': offset,
                })

            # Data section starts at next 32-byte alignment
            pos = f.tell()
            aligned = (pos + 31) & ~31
            self._data_offset = aligned

        self._parsed = True

    @property
    def metadata(self) -> dict:
        self._parse()
        return self._metadata

    @property
    def arch(self) -> str:
        self._parse()
        return self._metadata.get('general.architecture', 'unknown')

    def read_header(self) -> dict:
        self._parse()
        return dict(self._metadata)

    def list_tensors(self) -> List[dict]:
        self._parse()
        return list(self._tensor_infos)

    def read_tensor(self, name: str) -> np.ndarray:
        """Read and dequantize a single tensor by name."""
        self._parse()
        info = next((t for t in self._tensor_infos if t['name'] == name), None)
        if info is None:
            raise KeyError(f'Tensor not found: {name}')
        return self._load_tensor(info)

    def _load_tensor(self, info: dict) -> np.ndarray:
        dtype = info['dtype']
        shape = info['shape']
        offset = self._data_offset + info['offset']
        n_elem = int(np.prod(shape)) if shape else 1

        with open(self._path, 'rb') as f:
            f.seek(offset)
            if dtype == GGML_TYPE_F32:
                data = np.frombuffer(f.read(n_elem * 4), dtype=np.float32).copy()
            elif dtype == GGML_TYPE_F16:
                data = np.frombuffer(f.read(n_elem * 2), dtype=np.float16).astype(np.float32)
            elif dtype == GGML_TYPE_Q8_0:
                n_blocks = (n_elem + BLOCK_SIZE - 1) // BLOCK_SIZE
                scales = np.frombuffer(f.read(n_blocks * 2), dtype=np.float16).copy()
                q = np.frombuffer(f.read(n_blocks * BLOCK_SIZE), dtype=np.int8).copy()
                data = dequantize_q8_0(q, scales)[:n_elem]
            elif dtype == GGML_TYPE_Q4_0:
                n_blocks = (n_elem + BLOCK_SIZE - 1) // BLOCK_SIZE
                scales = np.frombuffer(f.read(n_blocks * 2), dtype=np.float16).copy()
                packed = np.frombuffer(f.read(n_blocks * 16), dtype=np.uint8).copy()
                data = dequantize_q4_0(packed, scales)[:n_elem]
            else:
                # Unsupported quantization — read raw bytes, return zeros
                data = np.zeros(n_elem, dtype=np.float32)

        return data.reshape(shape) if shape else data

    def read_all(self) -> Dict[str, np.ndarray]:
        """Load all tensors into a dict."""
        self._parse()
        return {t['name']: self._load_tensor(t) for t in self._tensor_infos}


# ── GGUFWriter ─────────────────────────────────────────────────────────────────

class GGUFWriter:
    """Write llama.cpp compatible .gguf files.

    Args:
        path: output file path
        arch: model architecture string (e.g. 'llama', 'gpt2')
    """

    def __init__(self, path: str, arch: str = 'llama'):
        self._path = path
        self._metadata: List[Tuple[str, int, Any]] = []
        self._tensors: List[dict] = []
        self.add_metadata('general.architecture', arch)
        self.add_metadata('general.file_type', 1)  # F16

    def add_metadata(self, key: str, value) -> None:
        vtype = _infer_vtype(value)
        self._metadata.append((key, vtype, value))

    def add_tensor(self, name: str, data: np.ndarray, quantize: str = 'f16') -> None:
        """Stage a tensor for writing.

        Args:
            name: tensor name
            data: numpy array
            quantize: 'f32', 'f16', 'q8_0', 'q4_0'
        """
        self._tensors.append({'name': name, 'data': data, 'quantize': quantize})

    def write(self) -> None:
        """Finalise and write the GGUF file."""
        import io as _io

        # Serialise tensor data
        tensor_data_buf = _io.BytesIO()
        offsets = []
        for t in self._tensors:
            # Align each tensor to 32 bytes
            pos = tensor_data_buf.tell()
            pad = (-pos) % 32
            if pad:
                tensor_data_buf.write(b'\x00' * pad)
            offsets.append(tensor_data_buf.tell())
            _write_tensor_data(tensor_data_buf, t['data'], t['quantize'])

        with open(self._path, 'wb') as f:
            # Header
            f.write(GGUF_MAGIC)
            f.write(struct.pack('<I', GGUF_VERSION))
            f.write(struct.pack('<Q', len(self._tensors)))
            f.write(struct.pack('<Q', len(self._metadata)))

            # Metadata
            for key, vtype, value in self._metadata:
                _write_str(f, key)
                f.write(struct.pack('<I', vtype))
                _write_value(f, vtype, value)

            # Tensor infos
            for t, offset in zip(self._tensors, offsets):
                _write_str(f, t['name'])
                shape = list(t['data'].shape)
                f.write(struct.pack('<I', len(shape)))
                f.write(struct.pack('<' + 'Q' * len(shape), *shape))
                f.write(struct.pack('<I', _quant_to_dtype(t['quantize'])))
                f.write(struct.pack('<Q', offset))

            # Align data section
            pos = f.tell()
            pad = (-pos) % 32
            if pad:
                f.write(b'\x00' * pad)

            # Tensor data
            f.write(tensor_data_buf.getvalue())


def _quant_to_dtype(quantize: str) -> int:
    return {'f32': GGML_TYPE_F32, 'f16': GGML_TYPE_F16,
            'q8_0': GGML_TYPE_Q8_0, 'q4_0': GGML_TYPE_Q4_0}.get(quantize, GGML_TYPE_F16)


def _write_tensor_data(f, data: np.ndarray, quantize: str) -> None:
    flat = data.flatten().astype(np.float32)
    if quantize == 'f32':
        f.write(flat.tobytes())
    elif quantize == 'f16':
        f.write(flat.astype(np.float16).tobytes())
    elif quantize == 'q8_0':
        q, scales = quantize_q8_0(flat)
        # Interleave: scale (2 bytes) then block (32 bytes) per block
        n_blocks = len(scales)
        for i in range(n_blocks):
            f.write(scales[i:i+1].tobytes())
            f.write(q[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE].tobytes())
    elif quantize == 'q4_0':
        packed, scales = quantize_q4_0(flat)
        n_blocks = len(scales)
        for i in range(n_blocks):
            f.write(scales[i:i+1].tobytes())
            f.write(packed[i*16:(i+1)*16].tobytes())
    else:
        f.write(flat.astype(np.float16).tobytes())


# ── High-level helpers ────────────────────────────────────────────────────────

def to_gguf(model, path: str, quantize: str = 'f16', arch: str = None) -> None:
    """Export a vectorphile model to GGUF format.

    Args:
        model: vectorphile Module
        path: output .gguf file path
        quantize: tensor dtype ('f32', 'f16', 'q8_0', 'q4_0')
        arch: architecture string; inferred from class name if None
    """
    if arch is None:
        arch = model.__class__.__name__.lower()
    writer = GGUFWriter(path, arch)
    writer.add_metadata('general.name', model.__class__.__name__)
    writer.add_metadata('general.quantization_version', 2)
    for name, param in model.named_parameters():
        tensor_name = name.replace('.', '_')
        writer.add_tensor(tensor_name, param._data, quantize)
    writer.write()
    size_mb = os.path.getsize(path) / 1e6
    print(f'Saved {len(list(model.named_parameters()))} tensors → {path} ({size_mb:.1f} MB)')


def from_gguf(path: str):
    """Load a GGUF file and return (weights_dict, metadata).

    Returns:
        weights: dict mapping tensor names to float32 numpy arrays
        metadata: dict of model metadata
    """
    reader = GGUFReader(path)
    weights = reader.read_all()
    return weights, reader.metadata
