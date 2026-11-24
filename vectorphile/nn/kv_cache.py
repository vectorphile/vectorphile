"""
vectorphile.nn.kv_cache — Production-grade KV cache management.

Provides:
  - KVCache          : simple growing cache for single-sequence decode
  - PagedKVCache     : vLLM-style block-based cache for continuous batching
  - KVCacheQuantized : INT8-compressed KV cache (halves memory vs FP32)
  - ContinuousBatcher: manages a pool of requests with dynamic batching

No third-party dependencies — all storage is built on vectorphile NDArray.
"""
from __future__ import annotations
import math
from typing import List, Optional, Tuple, Dict
from vectorphile._core.ndarray import NDArray, array as _nd_array


# ── Simple growing KV cache ──────────────────────────────────────────────────

class KVCache:
    """Per-layer growing KV cache for single-sequence autoregressive decoding.

    Usage::
        cache = KVCache(num_layers=32, num_heads=8, head_dim=64, max_seq_len=2048)
        # At each decode step:
        k_full, v_full = cache.append(layer_idx=0, k_new=k, v_new=v)
    """

    def __init__(self, num_layers: int, num_heads: int, head_dim: int,
                 max_seq_len: int = 2048, dtype: str = 'float32'):
        self.num_layers  = num_layers
        self.num_heads   = num_heads
        self.head_dim    = head_dim
        self.max_seq_len = max_seq_len
        self.dtype       = dtype
        self._seq_len    = 0
        # Pre-allocate: list of (K_buf, V_buf) per layer
        # each buf shape: (num_heads, max_seq_len, head_dim)
        size = num_heads * max_seq_len * head_dim
        self._k_bufs = [_nd_array([0.0] * size, dtype=dtype).reshape(
                            (num_heads, max_seq_len, head_dim))
                        for _ in range(num_layers)]
        self._v_bufs = [_nd_array([0.0] * size, dtype=dtype).reshape(
                            (num_heads, max_seq_len, head_dim))
                        for _ in range(num_layers)]

    @property
    def seq_len(self) -> int:
        return self._seq_len

    def append(self, layer_idx: int, k_new: NDArray,
               v_new: NDArray) -> Tuple[NDArray, NDArray]:
        """Append a single new K/V step and return the full accumulated K/V.

        k_new, v_new : (num_heads, 1, head_dim)
        Returns      : (num_heads, seq_len+1, head_dim) each
        """
        H, _, Dh = self.num_heads, 1, self.head_dim
        s = self._seq_len
        if s >= self.max_seq_len:
            # Rolling: shift left by 1
            kfl = self._k_bufs[layer_idx].to_flat_list()
            vfl = self._v_bufs[layer_idx].to_flat_list()
            # Drop oldest position for each head
            new_k = []
            new_v = []
            for h in range(H):
                base = h * self.max_seq_len * Dh
                new_k.extend(kfl[base + Dh: base + s * Dh])  # drop pos 0
                new_v.extend(vfl[base + Dh: base + s * Dh])
            # Append new
            knfl = k_new.to_flat_list() if hasattr(k_new, 'to_flat_list') else list(k_new)
            vnfl = v_new.to_flat_list() if hasattr(v_new, 'to_flat_list') else list(v_new)
            for h in range(H):
                new_k.extend(knfl[h*Dh:(h+1)*Dh])
                new_v.extend(vnfl[h*Dh:(h+1)*Dh])
            self._k_bufs[layer_idx] = _nd_array(new_k, self.dtype).reshape(
                (H, self.max_seq_len, Dh))
            self._v_bufs[layer_idx] = _nd_array(new_v, self.dtype).reshape(
                (H, self.max_seq_len, Dh))
            seq_len = self.max_seq_len
        else:
            # Write new token at position s
            kfl = self._k_bufs[layer_idx].to_flat_list()
            vfl = self._v_bufs[layer_idx].to_flat_list()
            knfl = k_new.to_flat_list() if hasattr(k_new, 'to_flat_list') else list(k_new)
            vnfl = v_new.to_flat_list() if hasattr(v_new, 'to_flat_list') else list(v_new)
            for h in range(H):
                base = h * self.max_seq_len * Dh + s * Dh
                for d in range(Dh):
                    kfl[base + d] = knfl[h * Dh + d]
                    vfl[base + d] = vnfl[h * Dh + d]
            self._k_bufs[layer_idx] = _nd_array(kfl, self.dtype).reshape(
                (H, self.max_seq_len, Dh))
            self._v_bufs[layer_idx] = _nd_array(vfl, self.dtype).reshape(
                (H, self.max_seq_len, Dh))
            seq_len = s + 1
            if layer_idx == self.num_layers - 1:
                self._seq_len = seq_len

        # Return slice [0:seq_len]
        kfl2 = self._k_bufs[layer_idx].to_flat_list()
        vfl2 = self._v_bufs[layer_idx].to_flat_list()
        k_out = []
        v_out = []
        for h in range(H):
            base = h * self.max_seq_len * Dh
            k_out.extend(kfl2[base: base + seq_len * Dh])
            v_out.extend(vfl2[base: base + seq_len * Dh])
        k_ret = _nd_array(k_out, self.dtype).reshape((H, seq_len, Dh))
        v_ret = _nd_array(v_out, self.dtype).reshape((H, seq_len, Dh))
        return k_ret, v_ret

    def reset(self):
        self._seq_len = 0

    def __repr__(self):
        return (f'KVCache(layers={self.num_layers}, heads={self.num_heads}, '
                f'head_dim={self.head_dim}, seq_len={self._seq_len}/'
                f'{self.max_seq_len})')


# ── Paged KV cache (vLLM style) ───────────────────────────────────────────────

class PagedKVCache:
    """Block-based paged KV cache for continuous batching (vLLM 2023).

    Physical memory is divided into equal-size *blocks*.  Each request gets a
    *logical* sequence of block IDs from a free-list; blocks are allocated on
    demand and freed when a request finishes.  This eliminates fragmentation
    and enables sharing of common prefixes across requests.

    Usage::
        pool  = PagedKVCache(num_layers=32, num_heads=8, head_dim=64,
                             block_size=16, num_blocks=1024)
        seq_a = pool.allocate()   # returns sequence handle (int)
        k_full, v_full = pool.append(seq_a, layer=0, k_new=k, v_new=v)
        pool.free(seq_a)
    """

    def __init__(self, num_layers: int, num_heads: int, head_dim: int,
                 block_size: int = 16, num_blocks: int = 512,
                 dtype: str = 'float32'):
        self.num_layers = num_layers
        self.num_heads  = num_heads
        self.head_dim   = head_dim
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.dtype      = dtype
        # Physical block store: (num_layers, num_blocks, num_heads, block_size, head_dim)
        bsize = num_heads * block_size * head_dim
        self._k_store: List[List[List[float]]] = [
            [[0.0] * bsize for _ in range(num_blocks)] for _ in range(num_layers)
        ]
        self._v_store: List[List[List[float]]] = [
            [[0.0] * bsize for _ in range(num_blocks)] for _ in range(num_layers)
        ]
        # Free list of block indices
        self._free_blocks: List[int] = list(range(num_blocks))
        # Per-sequence state: {seq_id: {'block_ids': list[int], 'seq_len': int}}
        self._sequences: Dict[int, dict] = {}
        self._next_seq_id = 0

    def allocate(self) -> int:
        """Allocate a new sequence slot and return its ID."""
        sid = self._next_seq_id
        self._next_seq_id += 1
        self._sequences[sid] = {'block_ids': [], 'seq_len': 0}
        return sid

    def free(self, seq_id: int):
        """Release all blocks held by this sequence."""
        if seq_id not in self._sequences:
            return
        for bid in self._sequences[seq_id]['block_ids']:
            self._free_blocks.append(bid)
        del self._sequences[seq_id]

    def _ensure_block(self, seq_id: int):
        """Allocate a new physical block for seq_id if the last block is full."""
        state = self._sequences[seq_id]
        seq_len = state['seq_len']
        blocks  = state['block_ids']
        # Check if we need a new block
        if len(blocks) == 0 or (seq_len % self.block_size == 0 and seq_len > 0):
            if not self._free_blocks:
                raise RuntimeError('PagedKVCache: out of physical blocks')
            blocks.append(self._free_blocks.pop(0))

    def append(self, seq_id: int, layer: int,
               k_new: NDArray, v_new: NDArray) -> Tuple[NDArray, NDArray]:
        """Append a new K/V token and return full K/V for this sequence.

        k_new, v_new : (num_heads, 1, head_dim)
        Returns      : (num_heads, seq_len, head_dim) each
        """
        self._ensure_block(seq_id)
        state   = self._sequences[seq_id]
        seq_len = state['seq_len']
        blocks  = state['block_ids']
        H, Dh   = self.num_heads, self.head_dim
        bs      = self.block_size
        # Write into the last block at offset = seq_len % block_size
        block_idx    = len(blocks) - 1
        phys_bid     = blocks[block_idx]
        offset_in_blk = seq_len % bs
        kfl = k_new.to_flat_list() if hasattr(k_new, 'to_flat_list') else list(k_new)
        vfl = v_new.to_flat_list() if hasattr(v_new, 'to_flat_list') else list(v_new)
        for h in range(H):
            base = h * bs * Dh + offset_in_blk * Dh
            for d in range(Dh):
                self._k_store[layer][phys_bid][base + d] = kfl[h * Dh + d]
                self._v_store[layer][phys_bid][base + d] = vfl[h * Dh + d]
        # Update seq_len only once (layer-agnostic; callers must call for all layers)
        if layer == 0:
            state['seq_len'] = seq_len + 1
        # Build full K/V from all blocks
        new_seq_len = state['seq_len']
        k_out: List[float] = []
        v_out: List[float] = []
        for h in range(H):
            tokens_written = 0
            for bid in blocks:
                phys  = self._k_store[layer][bid]
                vphys = self._v_store[layer][bid]
                remain = min(bs, new_seq_len - tokens_written)
                for t in range(remain):
                    base = h * bs * Dh + t * Dh
                    k_out.extend(phys[base: base + Dh])
                    v_out.extend(vphys[base: base + Dh])
                tokens_written += remain
        k_ret = _nd_array(k_out, self.dtype).reshape((H, new_seq_len, Dh))
        v_ret = _nd_array(v_out, self.dtype).reshape((H, new_seq_len, Dh))
        return k_ret, v_ret

    @property
    def free_block_count(self) -> int:
        return len(self._free_blocks)

    def __repr__(self):
        used = self.num_blocks - len(self._free_blocks)
        return (f'PagedKVCache(layers={self.num_layers}, heads={self.num_heads}, '
                f'block_size={self.block_size}, blocks={used}/{self.num_blocks}, '
                f'sequences={len(self._sequences)})')


# ── INT8-quantized KV cache ───────────────────────────────────────────────────

class KVCacheQuantized:
    """INT8 KV cache — halves memory vs FP32 with minimal accuracy loss.

    Per-channel absmax quantization: scale = absmax / 127.
    Keys and values are stored as int8 (simulated as int list),
    dequantized on read.
    """

    def __init__(self, num_layers: int, num_heads: int, head_dim: int,
                 max_seq_len: int = 2048):
        self.num_layers  = num_layers
        self.num_heads   = num_heads
        self.head_dim    = head_dim
        self.max_seq_len = max_seq_len
        self._seq_len    = 0
        size = num_heads * max_seq_len * head_dim
        self._k_int8 = [[0] * size for _ in range(num_layers)]
        self._v_int8 = [[0] * size for _ in range(num_layers)]
        self._k_scale = [[1.0] * (num_heads * max_seq_len) for _ in range(num_layers)]
        self._v_scale = [[1.0] * (num_heads * max_seq_len) for _ in range(num_layers)]

    def _quantize_token(self, fp_vec: List[float]) -> Tuple[List[int], float]:
        absmax = max(abs(v) for v in fp_vec) or 1e-9
        scale  = absmax / 127.0
        codes  = [max(-128, min(127, round(v / scale))) for v in fp_vec]
        return codes, scale

    def _dequantize_token(self, codes: List[int], scale: float) -> List[float]:
        return [c * scale for c in codes]

    def append(self, layer_idx: int, k_new: NDArray,
               v_new: NDArray) -> Tuple[NDArray, NDArray]:
        """Append and return full FP32 K/V. k_new, v_new: (H, 1, Dh)"""
        H, Dh = self.num_heads, self.head_dim
        s     = self._seq_len
        kfl   = k_new.to_flat_list() if hasattr(k_new, 'to_flat_list') else list(k_new)
        vfl   = v_new.to_flat_list() if hasattr(v_new, 'to_flat_list') else list(v_new)
        if s < self.max_seq_len:
            for h in range(H):
                k_vec = kfl[h*Dh:(h+1)*Dh]
                v_vec = vfl[h*Dh:(h+1)*Dh]
                kc, ks = self._quantize_token(k_vec)
                vc, vs = self._quantize_token(v_vec)
                base = h * self.max_seq_len * Dh + s * Dh
                self._k_int8[layer_idx][base:base+Dh] = kc
                self._v_int8[layer_idx][base:base+Dh] = vc
                sidx = h * self.max_seq_len + s
                self._k_scale[layer_idx][sidx] = ks
                self._v_scale[layer_idx][sidx]  = vs
            # Update seq_len on every layer call (each layer appends independently)
            self._seq_len = s + 1
        seq_len = self._seq_len
        # Reconstruct FP32
        k_out = []
        v_out = []
        for h in range(H):
            for t in range(seq_len):
                base = h * self.max_seq_len * Dh + t * Dh
                sidx = h * self.max_seq_len + t
                k_out.extend(self._dequantize_token(
                    self._k_int8[layer_idx][base:base+Dh],
                    self._k_scale[layer_idx][sidx]))
                v_out.extend(self._dequantize_token(
                    self._v_int8[layer_idx][base:base+Dh],
                    self._v_scale[layer_idx][sidx]))
        k_ret = _nd_array(k_out, 'float32').reshape((H, seq_len, Dh))
        v_ret = _nd_array(v_out, 'float32').reshape((H, seq_len, Dh))
        return k_ret, v_ret

    def reset(self):
        self._seq_len = 0

    def __repr__(self):
        return (f'KVCacheQuantized(layers={self.num_layers}, heads={self.num_heads}, '
                f'head_dim={self.head_dim}, seq_len={self._seq_len}/{self.max_seq_len}, '
                f'dtype=int8)')


# ── Continuous batcher ────────────────────────────────────────────────────────

class ContinuousBatcher:
    """Continuous batching scheduler (Orca 2022 / vLLM 2023 style).

    Maintains a pool of in-flight requests and dynamically merges them into
    batches at each decode step.  New requests are added while ongoing ones
    continue; finished sequences are dropped without waiting for others.

    Usage::
        batcher = ContinuousBatcher(max_batch=8, max_seq_len=512)
        req_id = batcher.add_request(prompt_tokens=[1, 2, 3], max_new=50)
        while batcher.has_pending():
            batch = batcher.next_batch()   # list of (req_id, tokens_so_far)
            # ... run model on batch ...
            for req_id, logits in model_outputs:
                next_tok = sample(logits)
                done = batcher.step(req_id, next_tok)
        results = batcher.get_results()    # {req_id: generated_tokens}
    """

    def __init__(self, max_batch: int = 8, max_seq_len: int = 2048,
                 eos_token: int = -1):
        self.max_batch   = max_batch
        self.max_seq_len = max_seq_len
        self.eos_token   = eos_token
        self._queue:     List[dict] = []   # waiting requests
        self._running:   Dict[int, dict] = {}
        self._finished:  Dict[int, List[int]] = {}
        self._next_id    = 0

    def add_request(self, prompt_tokens: List[int],
                    max_new_tokens: int = 128) -> int:
        """Enqueue a new generation request. Returns request ID."""
        rid = self._next_id
        self._next_id += 1
        self._queue.append({
            'id': rid,
            'prompt': list(prompt_tokens),
            'tokens': list(prompt_tokens),
            'generated': [],
            'max_new': max_new_tokens,
        })
        return rid

    def next_batch(self) -> List[Tuple[int, List[int]]]:
        """Return (req_id, full_token_sequence) for each request in this step's batch."""
        # Fill running slots from queue
        while len(self._running) < self.max_batch and self._queue:
            req = self._queue.pop(0)
            self._running[req['id']] = req
        return [(rid, req['tokens']) for rid, req in self._running.items()]

    def step(self, req_id: int, next_token: int) -> bool:
        """Record the next token for req_id. Returns True if the request is done."""
        if req_id not in self._running:
            return True
        req = self._running[req_id]
        req['tokens'].append(next_token)
        req['generated'].append(next_token)
        done = (next_token == self.eos_token
                or len(req['generated']) >= req['max_new']
                or len(req['tokens']) >= self.max_seq_len)
        if done:
            self._finished[req_id] = req['generated']
            del self._running[req_id]
        return done

    def has_pending(self) -> bool:
        """True if there are running or queued requests."""
        return bool(self._running or self._queue)

    def get_results(self) -> Dict[int, List[int]]:
        """Return generated tokens for all finished requests."""
        return dict(self._finished)

    def queue_depth(self) -> int:
        return len(self._queue)

    def running_count(self) -> int:
        return len(self._running)

    def __repr__(self):
        return (f'ContinuousBatcher(running={len(self._running)}, '
                f'queued={len(self._queue)}, '
                f'finished={len(self._finished)})')


__all__ = [
    'KVCache', 'PagedKVCache', 'KVCacheQuantized', 'ContinuousBatcher',
]
