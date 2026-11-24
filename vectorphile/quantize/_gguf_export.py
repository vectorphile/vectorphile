"""GGUF export for llama.cpp / Ollama compatibility.

Implements the minimal GGUF v3 container format with Q4_0 and Q8_0
quantization schemes so vectorphile models can be loaded by llama.cpp
and Ollama without any extra tooling.

GGUF spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
"""
from __future__ import annotations
import struct
import math
import array as _a
from typing import Dict, Any, Optional


# ── GGUF constants ────────────────────────────────────────────────────────────

GGUF_MAGIC       = b'GGUF'
GGUF_VERSION     = 3

GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32  = 5
GGUF_TYPE_FLOAT32= 6
GGUF_TYPE_BOOL   = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY  = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64  = 11

GGML_TYPE_F32    = 0
GGML_TYPE_F16    = 1
GGML_TYPE_Q4_0   = 2
GGML_TYPE_Q8_0   = 8

# Q4_0: 32 weights per block, one fp16 scale + 16 bytes of packed nibbles
Q4_0_BLOCK_SIZE  = 32
Q4_0_BYTES       = 2 + 16   # fp16 scale + 16 bytes (32 nibbles)

# Q8_0: 32 weights per block, one fp16 scale + 32 int8 bytes
Q8_0_BLOCK_SIZE  = 32
Q8_0_BYTES       = 2 + 32   # fp16 scale + 32 int8


# ── Helpers ───────────────────────────────────────────────────────────────────

def _float32_to_float16_bytes(val: float) -> bytes:
    """Pack a float32 as 2-byte IEEE 754 float16 (little-endian)."""
    import struct as _s
    f32_bytes = _s.pack('<f', val)
    f32_int   = _s.unpack('<I', f32_bytes)[0]
    sign     = (f32_int >> 31) & 0x1
    exponent = (f32_int >> 23) & 0xFF
    mantissa =  f32_int & 0x7FFFFF
    if exponent == 0xFF:
        f16_exp = 0x1F; f16_man = mantissa >> 13
    elif exponent == 0:
        f16_exp = 0; f16_man = 0
    else:
        new_exp = exponent - 127 + 15
        if new_exp >= 31:
            f16_exp = 0x1F; f16_man = 0
        elif new_exp <= 0:
            f16_exp = 0; f16_man = 0
        else:
            f16_exp = new_exp; f16_man = mantissa >> 13
    f16 = (sign << 15) | (f16_exp << 10) | f16_man
    return _s.pack('<H', f16)


def _write_string(buf: bytearray, s: str) -> None:
    encoded = s.encode('utf-8')
    buf += struct.pack('<Q', len(encoded))
    buf += encoded


def _write_kv_string(buf: bytearray, key: str, value: str) -> None:
    _write_string(buf, key)
    buf += struct.pack('<I', GGUF_TYPE_STRING)
    _write_string(buf, value)


def _write_kv_uint32(buf: bytearray, key: str, value: int) -> None:
    _write_string(buf, key)
    buf += struct.pack('<I', GGUF_TYPE_UINT32)
    buf += struct.pack('<I', value)


def _write_kv_float32(buf: bytearray, key: str, value: float) -> None:
    _write_string(buf, key)
    buf += struct.pack('<I', GGUF_TYPE_FLOAT32)
    buf += struct.pack('<f', value)


# ── Quantizers ────────────────────────────────────────────────────────────────

def _quantize_q4_0(weights: list) -> bytes:
    """Quantize a flat list of float32 weights to Q4_0 blocks.

    Each block of 32 values → 2-byte fp16 scale + 16 packed nibble bytes.
    """
    n = len(weights)
    n_blocks = math.ceil(n / Q4_0_BLOCK_SIZE)
    out = bytearray()
    for b in range(n_blocks):
        start = b * Q4_0_BLOCK_SIZE
        block = weights[start: start + Q4_0_BLOCK_SIZE]
        # zero-pad last block
        while len(block) < Q4_0_BLOCK_SIZE:
            block.append(0.0)
        abs_max = max(abs(v) for v in block) or 1e-9
        scale   = abs_max / 7.0   # int4 range [-8,7] → use 7 for sym
        out += _float32_to_float16_bytes(scale)
        # pack two nibbles per byte (low nibble first)
        nibbles = []
        for v in block:
            q = max(-8, min(7, round(v / scale)))
            nibbles.append(q & 0xF)   # 4-bit unsigned representation
        for i in range(0, Q4_0_BLOCK_SIZE, 2):
            byte = (nibbles[i] & 0xF) | ((nibbles[i + 1] & 0xF) << 4)
            out.append(byte)
    return bytes(out)


def _quantize_q8_0(weights: list) -> bytes:
    """Quantize a flat list of float32 weights to Q8_0 blocks.

    Each block of 32 values → 2-byte fp16 scale + 32 int8 bytes.
    """
    n = len(weights)
    n_blocks = math.ceil(n / Q8_0_BLOCK_SIZE)
    out = bytearray()
    for b in range(n_blocks):
        start = b * Q8_0_BLOCK_SIZE
        block = weights[start: start + Q8_0_BLOCK_SIZE]
        while len(block) < Q8_0_BLOCK_SIZE:
            block.append(0.0)
        abs_max = max(abs(v) for v in block) or 1e-9
        scale   = abs_max / 127.0
        out += _float32_to_float16_bytes(scale)
        for v in block:
            q = max(-128, min(127, round(v / scale)))
            out.append(q & 0xFF)
    return bytes(out)


# ── Main export function ───────────────────────────────────────────────────────

def export_gguf(
    model,
    path: str,
    quantization: str = 'Q8_0',
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Export a vectorphile model to GGUF format.

    Produces a file loadable by llama.cpp (`llama-cli -m model.gguf`) and
    Ollama (`ollama create mymodel --file Modelfile`).

    Args:
        model        : vectorphile Module with named_parameters()
        path         : output file path, e.g. 'model.gguf'
        quantization : 'Q4_0' or 'Q8_0' (default 'Q8_0')
        metadata     : optional dict of extra GGUF key-value pairs

    Raises:
        ValueError   : if quantization is not 'Q4_0' or 'Q8_0'
    """
    if quantization not in ('Q4_0', 'Q8_0'):
        raise ValueError(f"quantization must be 'Q4_0' or 'Q8_0', got {quantization!r}")

    quant_fn   = _quantize_q4_0 if quantization == 'Q4_0' else _quantize_q8_0
    ggml_type  = GGML_TYPE_Q4_0  if quantization == 'Q4_0' else GGML_TYPE_Q8_0

    # Gather tensors
    named_params = list(model.named_parameters())
    n_tensors    = len(named_params)

    # Build key-value metadata
    kv_buf = bytearray()
    n_kv   = 0

    def add_kv_string(k, v):
        nonlocal n_kv
        _write_kv_string(kv_buf, k, v)
        n_kv += 1

    def add_kv_uint32(k, v):
        nonlocal n_kv
        _write_kv_uint32(kv_buf, k, v)
        n_kv += 1

    add_kv_string('general.architecture', 'vectorphile')
    add_kv_string('general.name',         getattr(model, '__class__', type(model)).__name__)
    add_kv_string('general.quantization_version', quantization)
    add_kv_uint32('general.tensor_count', n_tensors)

    if metadata:
        for k, v in metadata.items():
            if isinstance(v, str):
                add_kv_string(k, v)
            elif isinstance(v, int):
                add_kv_uint32(k, v)

    # Build tensor info
    tensor_info_buf  = bytearray()
    tensor_data_bufs = []

    for name, param in named_params:
        data_nd  = param._data
        shape    = data_nd.shape
        ndim     = len(shape)
        flat     = [float(v) for v in data_nd.flatten()._buf]

        # Quantize
        q_bytes  = quant_fn(flat)

        # Tensor info entry:
        #   name (string), n_dims (uint32), dims (uint64 × n_dims),
        #   type (uint32), offset (uint64)
        _write_string(tensor_info_buf, name)
        tensor_info_buf += struct.pack('<I', ndim)
        for d in shape:
            tensor_info_buf += struct.pack('<Q', d)
        tensor_info_buf += struct.pack('<I', ggml_type)
        # offset filled in later
        tensor_info_buf += struct.pack('<Q', 0)   # placeholder

        tensor_data_bufs.append(q_bytes)

    # Rewrite offsets now that we know them
    # (Simplified: recalculate offsets after all tensor-info is built)
    # For a proper implementation we'd need a two-pass approach.
    # Here we write header + fill offsets in a second pass.

    with open(path, 'wb') as f:
        # GGUF header
        f.write(GGUF_MAGIC)
        f.write(struct.pack('<I', GGUF_VERSION))
        f.write(struct.pack('<Q', n_tensors))
        f.write(struct.pack('<Q', n_kv))

        # Key-value metadata
        f.write(bytes(kv_buf))

        # Tensor info (with real offsets)
        offset = 0
        for idx, (name, param) in enumerate(named_params):
            data_nd  = param._data
            shape    = data_nd.shape
            ndim     = len(shape)
            flat     = [float(v) for v in data_nd.flatten()._buf]
            q_bytes  = tensor_data_bufs[idx]

            # write info
            encoded_name = name.encode('utf-8')
            f.write(struct.pack('<Q', len(encoded_name)))
            f.write(encoded_name)
            f.write(struct.pack('<I', ndim))
            for d in shape:
                f.write(struct.pack('<Q', d))
            f.write(struct.pack('<I', ggml_type))
            f.write(struct.pack('<Q', offset))
            offset += len(q_bytes)
            # align to 32 bytes
            offset = (offset + 31) & ~31

        # Tensor data
        current = 0
        for q_bytes in tensor_data_bufs:
            f.write(q_bytes)
            current += len(q_bytes)
            # pad to 32-byte alignment
            pad = ((current + 31) & ~31) - current
            if pad:
                f.write(b'\x00' * pad)
                current += pad

    size_mb = offset / (1024 * 1024)
    print(f'Exported {n_tensors} tensors to {path!r} ({quantization}, ~{size_mb:.1f} MB data)')
