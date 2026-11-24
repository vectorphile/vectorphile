"""Paged KV cache — inspired by vLLM's PagedAttention.

Splits KV cache memory into fixed-size blocks (pages). Each sequence
gets a block table mapping logical block indices to physical pages.
This eliminates fragmentation and enables efficient memory sharing for
parallel sampling / beam search.
"""
from __future__ import annotations
import array as _a
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from vectorphile._core.ndarray import NDArray


@dataclass
class KVCacheConfig:
    """Configuration for the paged KV cache.

    Args:
        num_layers   : number of transformer layers
        num_heads    : number of KV heads (use num_kv_heads for GQA)
        head_dim     : dimension per attention head
        block_size   : tokens per page (default 16)
        max_blocks   : total number of pages in the pool (controls max memory)
        dtype        : storage dtype string ('float16', 'float32', 'bfloat16')
    """
    num_layers: int   = 32
    num_heads:  int   = 8
    head_dim:   int   = 128
    block_size: int   = 16
    max_blocks: int   = 1024
    dtype:      str   = 'float16'

    @property
    def block_bytes(self) -> int:
        """Bytes per page (K + V, one layer)."""
        elem_size = 2 if 'float16' in self.dtype or 'bfloat16' in self.dtype else 4
        return 2 * self.num_heads * self.block_size * self.head_dim * elem_size

    @property
    def total_bytes(self) -> int:
        return self.num_layers * self.max_blocks * self.block_bytes


class KVCacheBlock:
    """One physical page of KV cache (covers all layers).

    Stores keys and values for `block_size` token positions across all
    transformer layers, all in one contiguous buffer.

    Shape: [num_layers, 2, num_heads, block_size, head_dim]
    """

    __slots__ = ('data', 'block_id', 'ref_count')

    def __init__(self, block_id: int, num_layers: int, num_heads: int,
                 block_size: int, head_dim: int):
        self.block_id  = block_id
        self.ref_count = 0
        total = num_layers * 2 * num_heads * block_size * head_dim
        self.data = _a.array('f', [0.0] * total)

    def read_kv(self, layer: int, num_heads: int, block_size: int,
                head_dim: int) -> Tuple[List, List]:
        """Return (keys, values) for given layer as flat lists."""
        layer_stride = 2 * num_heads * block_size * head_dim
        kv_stride    = num_heads * block_size * head_dim
        base         = layer * layer_stride
        k = list(self.data[base            : base + kv_stride])
        v = list(self.data[base + kv_stride: base + 2 * kv_stride])
        return k, v

    def write_kv(self, layer: int, pos: int, num_heads: int, block_size: int,
                 head_dim: int, k_vec: List[float], v_vec: List[float]) -> None:
        """Write K/V vectors for a single token position in this layer."""
        layer_stride = 2 * num_heads * block_size * head_dim
        kv_stride    = num_heads * block_size * head_dim
        head_stride  = block_size * head_dim
        base_k       = layer * layer_stride
        base_v       = base_k + kv_stride
        vec_len      = head_dim
        for h in range(num_heads):
            off_k = base_k + h * head_stride + pos * head_dim
            off_v = base_v + h * head_stride + pos * head_dim
            src_h = h * head_dim
            for d in range(head_dim):
                self.data[off_k + d] = k_vec[src_h + d] if src_h + d < len(k_vec) else 0.0
                self.data[off_v + d] = v_vec[src_h + d] if src_h + d < len(v_vec) else 0.0


class PagedKVCache:
    """Paged KV cache pool.

    Manages a pool of `KVCacheBlock` pages and allocates/frees them to
    sequences on demand. Each sequence gets a block_table: a list of
    block_ids representing its logical token positions in order.

    Usage::
        cache = PagedKVCache(KVCacheConfig(num_layers=32, num_heads=8, head_dim=128))
        seq_id = cache.allocate_sequence(max_tokens=512)
        cache.write_token(seq_id, layer=0, token_pos=0, k_vec=[...], v_vec=[...])
        k, v = cache.read_layer(seq_id, layer=0, start_pos=0, length=5)
        cache.free_sequence(seq_id)
    """

    def __init__(self, config: KVCacheConfig):
        self.cfg = config
        # Physical block pool
        self._pool: List[KVCacheBlock] = [
            KVCacheBlock(i, config.num_layers, config.num_heads,
                         config.block_size, config.head_dim)
            for i in range(config.max_blocks)
        ]
        self._free_blocks: List[int] = list(range(config.max_blocks))
        # sequence_id → list of physical block_ids
        self._seq_tables: Dict[int, List[int]] = {}
        self._seq_lengths: Dict[int, int]      = {}
        self._next_seq_id = 0

    # ── Allocation ─────────────────────────────────────────────────────────

    def allocate_sequence(self, max_tokens: int) -> int:
        """Allocate pages for a new sequence, return seq_id."""
        n_blocks_needed = (max_tokens + self.cfg.block_size - 1) // self.cfg.block_size
        if len(self._free_blocks) < n_blocks_needed:
            raise MemoryError(
                f'PagedKVCache: out of blocks (need {n_blocks_needed}, '
                f'have {len(self._free_blocks)} free)')
        seq_id = self._next_seq_id
        self._next_seq_id += 1
        allocated = []
        for _ in range(n_blocks_needed):
            bid = self._free_blocks.pop()
            self._pool[bid].ref_count += 1
            allocated.append(bid)
        self._seq_tables[seq_id]  = allocated
        self._seq_lengths[seq_id] = 0
        return seq_id

    def free_sequence(self, seq_id: int) -> None:
        """Return all pages belonging to this sequence back to the pool."""
        for bid in self._seq_tables.pop(seq_id, []):
            self._pool[bid].ref_count -= 1
            if self._pool[bid].ref_count == 0:
                self._free_blocks.append(bid)
        self._seq_lengths.pop(seq_id, None)

    # ── Read / write ────────────────────────────────────────────────────────

    def write_token(self, seq_id: int, layer: int, token_pos: int,
                    k_vec: List[float], v_vec: List[float]) -> None:
        """Write K/V for one token at position `token_pos` in `layer`."""
        block_idx  = token_pos // self.cfg.block_size
        pos_in_blk = token_pos  % self.cfg.block_size
        bid        = self._seq_tables[seq_id][block_idx]
        self._pool[bid].write_kv(
            layer, pos_in_blk, self.cfg.num_heads,
            self.cfg.block_size, self.cfg.head_dim, k_vec, v_vec)
        if token_pos >= self._seq_lengths[seq_id]:
            self._seq_lengths[seq_id] = token_pos + 1

    def read_layer(self, seq_id: int, layer: int,
                   start_pos: int = 0, length: Optional[int] = None
                   ) -> Tuple[List[float], List[float]]:
        """Read all cached K/V for `layer` as flat lists.

        Returns (k_flat, v_flat) where each has shape
        [num_heads * seq_len * head_dim] (flattened).
        """
        seq_len = self._seq_lengths[seq_id]
        end     = seq_len if length is None else min(start_pos + length, seq_len)
        k_all, v_all = [], []
        for pos in range(start_pos, end):
            block_idx  = pos // self.cfg.block_size
            pos_in_blk = pos  % self.cfg.block_size
            bid        = self._seq_tables[seq_id][block_idx]
            k_blk, v_blk = self._pool[bid].read_kv(
                layer, self.cfg.num_heads, self.cfg.block_size, self.cfg.head_dim)
            # extract single position from block
            stride = self.cfg.block_size * self.cfg.head_dim
            for h in range(self.cfg.num_heads):
                off = h * self.cfg.block_size * self.cfg.head_dim + pos_in_blk * self.cfg.head_dim
                k_all.extend(k_blk[off: off + self.cfg.head_dim])
                v_all.extend(v_blk[off: off + self.cfg.head_dim])
        return k_all, v_all

    # ── Utilities ───────────────────────────────────────────────────────────

    def sequence_length(self, seq_id: int) -> int:
        return self._seq_lengths.get(seq_id, 0)

    def free_blocks(self) -> int:
        return len(self._free_blocks)

    def used_blocks(self) -> int:
        return self.cfg.max_blocks - len(self._free_blocks)

    def memory_usage_mb(self) -> float:
        return self.used_blocks() * self.cfg.block_bytes * self.cfg.num_layers / (1024 * 1024)

    def __repr__(self):
        return (f'PagedKVCache(layers={self.cfg.num_layers}, '
                f'heads={self.cfg.num_heads}, head_dim={self.cfg.head_dim}, '
                f'block_size={self.cfg.block_size}, '
                f'free={self.free_blocks()}/{self.cfg.max_blocks} blocks)')
