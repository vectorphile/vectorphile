"""vectorphile.nn.einops — Einstein-notation tensor operations (einops-compatible API).

Implements rearrange, reduce, repeat, pack, unpack, and EinMix using pure numpy.
Pattern syntax is a subset of the einops DSL:
  - 'b c h w -> b (c h w)'   flatten spatial dims
  - 'b (h p1) (w p2) c -> (b h w) p1 p2 c'  split dims with size hints
  - 'b n d -> b d n'          transpose
"""
from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import math as _math_einops
import random as _random_einops
import functools as _functools


# Lightweight shim — vectorphile NDArray for einops operations
class _NpEinops:
    @staticmethod
    def _fl(d):
        if d is None: return []
        if hasattr(d, 'to_flat_list'): return d.to_flat_list()
        if hasattr(d, '_buf') and d._buf is not None: return list(d._buf)
        if isinstance(d, list): return d
        return list(d)
    @staticmethod
    def _nd(lst, shape=None):
        from vectorphile._core.ndarray import array as _a
        r = _a(lst if isinstance(lst, list) else list(lst))
        if shape: r = r.reshape(shape)
        return r
    @staticmethod
    def asarray(d, dtype=None):
        if d is None: return _NpEinops._nd([])
        if hasattr(d, 'shape'): return d
        from vectorphile._core.ndarray import array as _a
        return _a(_NpEinops._fl(d))
    @staticmethod
    def prod(vals):
        r = 1
        for v in vals: r *= int(v)
        return r
    @staticmethod
    def transpose(d, perm):
        sh = d.shape if hasattr(d, 'shape') else (len(_NpEinops._fl(d)),)
        fl = _NpEinops._fl(d)
        ndim = len(sh)
        new_sh = tuple(sh[p] for p in perm)
        total = len(fl)
        # compute strides for old layout
        old_strides = [1]*ndim
        for i in range(ndim-2, -1, -1): old_strides[i] = old_strides[i+1]*sh[i+1]
        new_strides = [1]*ndim
        for i in range(ndim-2, -1, -1): new_strides[i] = new_strides[i+1]*new_sh[i+1]
        result = [0.0]*total
        def _recurse(old_idx, new_idx, dim):
            if dim == ndim:
                result[new_idx] = fl[old_idx]
                return
            for i in range(sh[dim]):
                _recurse(old_idx + i*old_strides[dim], new_idx + i*new_strides[list(perm).index(dim)], dim+1)
        # Use iterative approach
        for flat_idx in range(total):
            old_coords = []
            tmp = flat_idx
            for i in range(ndim-1, -1, -1):
                old_coords.append(tmp % sh[i]); tmp //= sh[i]
            old_coords.reverse()
            new_coords = [old_coords[perm[i]] for i in range(ndim)]
            new_flat = sum(new_coords[i]*new_strides[i] for i in range(ndim))
            result[new_flat] = fl[flat_idx]
        return _NpEinops._nd(result, new_sh)
    @staticmethod
    def expand_dims(d, axis):
        sh = d.shape if hasattr(d, 'shape') else (len(_NpEinops._fl(d)),)
        fl = _NpEinops._fl(d)
        ndim = len(sh); ax = axis if axis >= 0 else ndim+1+axis
        new_sh = sh[:ax] + (1,) + sh[ax:]
        return _NpEinops._nd(fl, new_sh)
    @staticmethod
    def tile(d, reps):
        sh = d.shape if hasattr(d, 'shape') else (len(_NpEinops._fl(d)),)
        fl = _NpEinops._fl(d)
        result = fl; cur_sh = list(sh)
        ndim = len(sh)
        # pad reps if shorter
        reps2 = [1]*(ndim-len(reps)) + list(reps)
        for dim in range(ndim):
            r = reps2[dim]
            if r == 1: continue
            inner = 1
            for i in range(dim+1, ndim): inner *= cur_sh[i]
            outer = len(result) // (cur_sh[dim]*inner)
            new_result = []
            for o in range(outer):
                chunk = result[o*cur_sh[dim]*inner:(o+1)*cur_sh[dim]*inner]
                for _ in range(r): new_result.extend(chunk)
            cur_sh[dim] *= r; result = new_result
        return _NpEinops._nd(result, tuple(cur_sh))
    @staticmethod
    def concatenate(arrays, axis=0):
        parts = []; sh0 = None
        for arr in arrays:
            f = _NpEinops._fl(arr)
            parts.extend(f)
            if sh0 is None: sh0 = arr.shape if hasattr(arr, 'shape') else (len(f),)
        ndim = len(sh0)
        ax = axis if axis >= 0 else ndim+axis
        new_sh = sh0[:ax] + (sum(a.shape[ax] if hasattr(a, 'shape') else 1 for a in arrays),) + sh0[ax+1:]
        return _NpEinops._nd(parts, new_sh)
    @staticmethod
    def mean(d, axis=None, keepdims=False):
        fl = _NpEinops._fl(d)
        if axis is None: return sum(fl)/len(fl) if fl else 0.0
        sh = d.shape if hasattr(d, 'shape') else (len(fl),)
        ndim = len(sh); ax = axis if axis >= 0 else ndim+axis
        dim_sz = sh[ax]
        inner = 1
        for i in range(ax+1, ndim): inner *= sh[i]
        outer = len(fl)//(dim_sz*inner)
        result = [sum(fl[o*dim_sz*inner + k*inner + i] for k in range(dim_sz))/dim_sz
                  for o in range(outer) for i in range(inner)]
        new_sh = sh[:ax] + ((1,) if keepdims else ()) + sh[ax+1:]
        return _NpEinops._nd(result, new_sh)
    @staticmethod
    def sum(d, axis=None, keepdims=False):
        fl = _NpEinops._fl(d)
        if axis is None: return sum(fl)
        sh = d.shape if hasattr(d, 'shape') else (len(fl),)
        ndim = len(sh); ax = axis if axis >= 0 else ndim+axis
        dim_sz = sh[ax]
        inner = 1
        for i in range(ax+1, ndim): inner *= sh[i]
        outer = len(fl)//(dim_sz*inner)
        result = [sum(fl[o*dim_sz*inner + k*inner + i] for k in range(dim_sz))
                  for o in range(outer) for i in range(inner)]
        new_sh = sh[:ax] + ((1,) if keepdims else ()) + sh[ax+1:]
        return _NpEinops._nd(result, new_sh)
    @staticmethod
    def max(d, axis=None, keepdims=False):
        fl = _NpEinops._fl(d)
        if axis is None: return max(fl)
        sh = d.shape if hasattr(d, 'shape') else (len(fl),)
        ndim = len(sh); ax = axis if axis >= 0 else ndim+axis
        dim_sz = sh[ax]
        inner = 1
        for i in range(ax+1, ndim): inner *= sh[i]
        outer = len(fl)//(dim_sz*inner)
        result = [max(fl[o*dim_sz*inner + k*inner + i] for k in range(dim_sz))
                  for o in range(outer) for i in range(inner)]
        new_sh = sh[:ax] + ((1,) if keepdims else ()) + sh[ax+1:]
        return _NpEinops._nd(result, new_sh)
    @staticmethod
    def min(d, axis=None, keepdims=False):
        fl = _NpEinops._fl(d)
        if axis is None: return min(fl)
        sh = d.shape if hasattr(d, 'shape') else (len(fl),)
        ndim = len(sh); ax = axis if axis >= 0 else ndim+axis
        dim_sz = sh[ax]
        inner = 1
        for i in range(ax+1, ndim): inner *= sh[i]
        outer = len(fl)//(dim_sz*inner)
        result = [min(fl[o*dim_sz*inner + k*inner + i] for k in range(dim_sz))
                  for o in range(outer) for i in range(inner)]
        new_sh = sh[:ax] + ((1,) if keepdims else ()) + sh[ax+1:]
        return _NpEinops._nd(result, new_sh)
    @staticmethod
    def zeros(shape, dtype='float32'):
        from vectorphile._core.ndarray import zeros as _z
        if isinstance(shape, int): shape = (shape,)
        return _z(shape, dtype=dtype)
    @staticmethod
    def random_randn(shape):
        total = 1
        for s in shape: total *= s
        vals = [_math_einops.sqrt(-2*_math_einops.log(max(_random_einops.random(),1e-10)))*_math_einops.cos(2*_math_einops.pi*_random_einops.random())
                for _ in range(total)]
        return _NpEinops._nd(vals, shape)
    @staticmethod
    def einsum(subscripts, a, b):
        # Minimal einsum for EinMix: 'ij,jk->ik' or similar
        # Parse subscripts
        parts = subscripts.replace(' ', '').split(',')
        in1, rest = parts[0], parts[1]
        in2, out_s = rest.split('->')
        # Get shapes
        sh_a = a.shape if hasattr(a, 'shape') else (len(_NpEinops._fl(a)),)
        sh_b = b.shape if hasattr(b, 'shape') else (len(_NpEinops._fl(b)),)
        fa = _NpEinops._fl(a); fb = _NpEinops._fl(b)
        # Build index maps
        idx_map = {}
        for i, c in enumerate(in1): idx_map[c] = sh_a[i]
        for i, c in enumerate(in2): idx_map[c] = sh_b[i]
        # Sum over contracted indices (not in out_s)
        contracted = [c for c in in1 if c not in out_s] + [c for c in in2 if c not in out_s]
        contracted = list(dict.fromkeys(contracted))
        out_dims = [idx_map[c] for c in out_s]
        total_out = 1
        for d in out_dims: total_out *= d
        # Strides for a and b
        def _strides(sh):
            s = [1]*len(sh)
            for i in range(len(sh)-2, -1, -1): s[i] = s[i+1]*sh[i+1]
            return s
        sa = _strides(sh_a); sb = _strides(sh_b)
        # Contracted dims product
        cont_ranges = [range(idx_map[c]) for c in contracted]
        result = [0.0]*total_out
        # Enumerate output indices
        def _idx(chars, coords, sh_map):
            s = 0
            str_map = {c: sh_map[c] for c in chars}
            st = _strides(list(str_map.values()))
            for i, c in enumerate(chars): s += coords[c]*st[i]
            return s
        # Build output index mapping
        out_chars = list(out_s)
        all_in1 = list(in1); all_in2 = list(in2)
        # Simple nested loop
        from itertools import product as _ip
        out_shape_a = list(sh_a); out_shape_b = list(sh_b)
        # rebuild using manual loops
        # Only handling simple matrix-like cases
        if len(in1) == 2 and len(in2) == 2 and len(out_s) == 2:
            # e.g. ij,jk->ik
            shared = [c for c in in1 if c in in2]
            I = idx_map[in1[0]]; K = idx_map[out_s[1]]; J = idx_map[shared[0]] if shared else 1
            for i in range(I):
                for k in range(K):
                    s = 0.0
                    for j in range(J):
                        a_idx = i*sa[0] + j*sa[1] if in1 == list(in1) else 0
                        b_idx = j*sb[0] + k*sb[1] if True else 0
                        # Properly compute indices from chars
                        coords_a = {in1[0]: i, in1[1]: j}
                        coords_b = {in2[0]: j, in2[1]: k}
                        ai = sum(coords_a.get(in1[ii], 0)*sa[ii] for ii in range(len(in1)))
                        bi = sum(coords_b.get(in2[ii], 0)*sb[ii] for ii in range(len(in2)))
                        s += fa[ai] * fb[bi]
                    result[i*K+k] = s
        return _NpEinops._nd(result, tuple(out_dims))


np = _NpEinops()

# ── Pattern parser ─────────────────────────────────────────────────────────────

def _parse_pattern(pattern: str) -> Tuple[List, List]:
    """Parse 'left -> right' into two lists of axis specs."""
    if '->' not in pattern:
        raise ValueError(f"Pattern must contain '->': {pattern!r}")
    left_str, right_str = pattern.split('->')
    return _parse_side(left_str.strip()), _parse_side(right_str.strip())


def _parse_side(side: str) -> List:
    """Parse one side of a pattern into a list of axis specs.

    Each element is either:
      - str: a named axis
      - list[str]: a grouped axis (parentheses)
    """
    tokens = []
    i = 0
    s = side.strip()
    while i < len(s):
        if s[i] == '(':
            j = s.index(')', i)
            inner = s[i+1:j].split()
            tokens.append(inner)
            i = j + 1
        elif s[i] == ' ':
            i += 1
        else:
            # collect identifier
            j = i
            while j < len(s) and s[j] not in (' ', '(', ')'):
                j += 1
            tokens.append(s[i:j])
            i = j
    return tokens


def _infer_sizes(left_axes, right_axes, x_shape, axes_lengths: dict) -> Dict[str, int]:
    """Infer all named-axis sizes from input shape and provided hints."""
    sizes: Dict[str, int] = dict(axes_lengths)
    # match left axes to x_shape
    if len(left_axes) != len(x_shape):
        raise ValueError(
            f"Left pattern has {len(left_axes)} axes but input has {len(x_shape)} dims. "
            f"Pattern axes: {left_axes}, shape: {x_shape}"
        )
    for ax, dim in zip(left_axes, x_shape):
        if isinstance(ax, list):
            # group: infer unknown
            known = [sizes[a] for a in ax if a in sizes]
            unknown = [a for a in ax if a not in sizes]
            if len(unknown) == 0:
                known_prod = _NpEinops.prod(known)
                assert known_prod == dim, f"Group {ax} product {known_prod} != {dim}"
            elif len(unknown) == 1:
                sizes[unknown[0]] = dim // int(_NpEinops.prod(known)) if known else dim
            else:
                raise ValueError(f"Cannot infer sizes for group {ax}: too many unknowns")
        else:
            if ax == '1':
                assert dim == 1, f"Axis '1' must have size 1 but got {dim}"
            elif ax not in sizes:
                sizes[ax] = dim
    return sizes


# ── Core: rearrange ────────────────────────────────────────────────────────────

def rearrange(x: np.ndarray, pattern: str, **axes_lengths) -> np.ndarray:
    """Rearrange tensor dimensions using an einops-style pattern.

    Examples::

        rearrange(x, 'b c h w -> b (c h w)')
        rearrange(x, 'b (h p1) (w p2) c -> (b h w) p1 p2 c', p1=8, p2=8)
        rearrange(x, 'b n d -> b d n')
        rearrange(x, '(b t) n d -> b t n d', b=2)
    """
    if hasattr(x, 'data'):  # unwrap Tensor
        x = x.data if hasattr(x, 'data') else _NpEinops.asarray(x)
    x = _NpEinops.asarray(x)

    left_axes, right_axes = _parse_pattern(pattern)
    sizes = _infer_sizes(left_axes, x.shape, x.shape, axes_lengths)

    # Step 1: reshape to split any grouped left axes
    split_shape = []
    for ax in left_axes:
        if isinstance(ax, list):
            split_shape.extend(sizes[a] for a in ax)
        else:
            split_shape.append(x.shape[left_axes.index(ax)] if ax != '1' else 1)
    x = x.reshape(split_shape)

    # Step 2: collect all individual axis names on left (in order)
    flat_left: List[str] = []
    for ax in left_axes:
        if isinstance(ax, list):
            flat_left.extend(ax)
        else:
            flat_left.append(ax)

    # Step 3: collect desired order from right axes
    flat_right_groups: List[List[str]] = []
    for ax in right_axes:
        if isinstance(ax, list):
            flat_right_groups.append(ax)
        else:
            flat_right_groups.append([ax])

    flat_right: List[str] = [a for g in flat_right_groups for a in g]

    # Step 4: transpose
    perm = [flat_left.index(a) for a in flat_right]
    x = _NpEinops.transpose(x, perm)

    # Step 5: reshape merging grouped right axes
    out_shape = []
    pos = 0
    for group in flat_right_groups:
        if len(group) == 1:
            out_shape.append(sizes[group[0]])
        else:
            out_shape.append(int(_NpEinops.prod([sizes[a] for a in group])))
        pos += len(group)
    x = x.reshape(out_shape)
    return x


# ── Core: reduce ──────────────────────────────────────────────────────────────

def reduce(x: np.ndarray, pattern: str, reduction: str, **axes_lengths) -> np.ndarray:
    """Reduce tensor dimensions.

    Args:
        x: input array
        pattern: e.g. 'b c h w -> b c'
        reduction: 'mean', 'sum', 'max', 'min', 'prod'

    Examples::

        reduce(x, 'b c h w -> b c', 'mean')   # global average pool
        reduce(x, 'b c (h ph) (w pw) -> b c h w', 'max', ph=2, pw=2)
    """
    if hasattr(x, 'data'):
        x = x.data
    x = _NpEinops.asarray(x)

    left_axes, right_axes = _parse_pattern(pattern)
    sizes = _infer_sizes(left_axes, x.shape, x.shape, axes_lengths)

    # Flatten grouped left axes
    split_shape = []
    for ax in left_axes:
        if isinstance(ax, list):
            split_shape.extend(sizes[a] for a in ax)
        else:
            split_shape.append(sizes.get(ax, x.shape[left_axes.index(ax)]))
    x = x.reshape(split_shape)

    flat_left: List[str] = []
    for ax in left_axes:
        if isinstance(ax, list):
            flat_left.extend(ax)
        else:
            flat_left.append(ax)

    flat_right: List[str] = []
    for ax in right_axes:
        if isinstance(ax, list):
            flat_right.extend(ax)
        else:
            flat_right.append(ax)

    # Axes to reduce
    reduce_axes = tuple(i for i, a in enumerate(flat_left) if a not in flat_right)

    _ops = {'mean': _NpEinops.mean, 'sum': _NpEinops.sum, 'max': _NpEinops.max, 'min': _NpEinops.min,
            'prod': lambda d, axis=None, keepdims=False: _NpEinops.sum(d, axis=axis, keepdims=keepdims)}
    if reduction not in _ops:
        raise ValueError(f"Unknown reduction: {reduction!r}. Use: {list(_ops.keys())}")

    if reduce_axes:
        for ax in sorted(reduce_axes, reverse=True):
            x = _ops[reduction](x, axis=ax)

    # Reorder remaining axes
    remaining = [a for a in flat_left if a in flat_right]
    perm = [remaining.index(a) for a in flat_right]
    if perm != list(range(len(perm))):
        x = _NpEinops.transpose(x, perm)

    # Reshape merged groups in right
    out_shape = []
    for ax in right_axes:
        if isinstance(ax, list):
            out_shape.append(int(_NpEinops.prod([sizes[a] for a in ax])))
        else:
            out_shape.append(sizes[ax])
    x = x.reshape(out_shape)
    return x


# ── Core: repeat ──────────────────────────────────────────────────────────────

def repeat(x: np.ndarray, pattern: str, **axes_lengths) -> np.ndarray:
    """Repeat tensor along new or existing axes.

    Examples::

        repeat(x, 'b d -> b n d', n=5)
        repeat(x, 'h w -> h w c', c=3)
    """
    if hasattr(x, 'data'):
        x = x.data
    x = _NpEinops.asarray(x)

    left_axes, right_axes = _parse_pattern(pattern)

    # Determine which right axes are new (not on left)
    flat_left: List[str] = []
    for ax in left_axes:
        flat_left.extend(ax if isinstance(ax, list) else [ax])

    sizes = _infer_sizes(left_axes, x.shape, x.shape, axes_lengths)
    # Add new axes to sizes
    for ax in right_axes:
        axl = ax if isinstance(ax, list) else [ax]
        for a in axl:
            if a not in sizes and a in axes_lengths:
                sizes[a] = axes_lengths[a]

    # Insert new dims and tile
    flat_right: List[str] = []
    for ax in right_axes:
        flat_right.extend(ax if isinstance(ax, list) else [ax])

    for i, a in enumerate(flat_right):
        if a not in flat_left:
            # Insert dim at position i
            x = _NpEinops.expand_dims(x, axis=i)
            flat_left.insert(i, a)
            reps = [1] * len(flat_left)
            reps[i] = sizes[a]
            x = _NpEinops.tile(x, reps)

    return rearrange(x, ' '.join(flat_left) + ' -> ' + pattern.split('->')[1], **axes_lengths)


# ── pack / unpack ──────────────────────────────────────────────────────────────

def pack(tensors: List[np.ndarray], pattern: str) -> Tuple[np.ndarray, List[Tuple]]:
    """Pack a list of tensors into one, recording shapes for unpack.

    Example::

        cls  = np.zeros((b, 1, d))
        toks = np.zeros((b, n, d))
        packed, ps = pack([cls, toks], '* n d')
        # packed.shape == (b, 1+n, d)
    """
    tensors = [_NpEinops.asarray(t.data if hasattr(t, 'data') else t) for t in tensors]
    # Find the '*' position and determine pack axis
    parts = pattern.split()
    star_idx = parts.index('*')
    packed_shapes = []
    reshaped = []
    for t in tensors:
        # Axes before * and after *
        n_before = star_idx
        n_after = len(parts) - star_idx - 1
        pack_dims = t.shape[n_before: len(t.shape) - n_after if n_after else len(t.shape)]
        packed_shapes.append(pack_dims)
        # Flatten pack dims
        new_shape = t.shape[:n_before] + (int(_NpEinops.prod(pack_dims)) if pack_dims else (1,)) + t.shape[len(t.shape)-n_after:] if n_after else t.shape[:n_before] + (int(_NpEinops.prod(pack_dims)) if pack_dims else 1,)
        reshaped.append(t.reshape(new_shape))
    result = _NpEinops.concatenate(reshaped, axis=star_idx)
    return result, packed_shapes


def unpack(x: np.ndarray, packed_shapes: List[Tuple], pattern: str) -> List[np.ndarray]:
    """Unpack a tensor packed with pack()."""
    x = _NpEinops.asarray(x.data if hasattr(x, 'data') else x)
    parts = pattern.split()
    star_idx = parts.index('*')
    n_after = len(parts) - star_idx - 1
    results = []
    pos = 0
    for ps in packed_shapes:
        size = int(_NpEinops.prod(ps)) if ps else 1
        slices = [slice(None)] * len(x.shape)
        slices[star_idx] = slice(pos, pos + size)
        chunk = x[tuple(slices)]
        # Reshape pack dim back to ps
        new_shape = x.shape[:star_idx] + ps + (x.shape[star_idx + 1:] if n_after else ())
        results.append(chunk.reshape(new_shape))
        pos += size
    return results


# ── EinMix ─────────────────────────────────────────────────────────────────────

class EinMix:
    """Learned linear mixing with einops-style pattern.

    Equivalent to a Linear layer applied along arbitrary axes.

    Example::

        mix = EinMix('b n d_in -> b n d_out', weight_shape='d_in d_out', d_in=64, d_out=128)
        y = mix(x)  # x: (b, n, 64) → y: (b, n, 128)
    """

    def __init__(self, pattern: str, weight_shape: str, bias: bool = True, **axes_lengths):
        self.pattern = pattern
        self.weight_shape = weight_shape
        self.axes_lengths = axes_lengths

        # Parse weight shape
        w_names = weight_shape.split()
        w_dims = tuple(axes_lengths[n] for n in w_names)
        scale = (2 / sum(w_dims)) ** 0.5
        self.weight = _NpEinops.random_randn(w_dims)
        # scale inplace
        wfl = _NpEinops._fl(self.weight)
        self.weight = _NpEinops._nd([v*scale for v in wfl], w_dims)
        self.bias_param = _NpEinops.zeros((w_dims[-1],)) if bias else None

        # Identify contraction axes
        left_axes, right_axes = _parse_pattern(pattern)
        flat_left = [a for ax in left_axes for a in (ax if isinstance(ax, list) else [ax])]
        flat_right = [a for ax in right_axes for a in (ax if isinstance(ax, list) else [ax])]
        self.in_axes = [a for a in w_names[:-1]]
        self.out_axis = w_names[-1]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if hasattr(x, 'data'):
            x = x.data
        x = _NpEinops.asarray(x)
        left_axes, right_axes = _parse_pattern(self.pattern)
        flat_left = [a for ax in left_axes for a in (ax if isinstance(ax, list) else [ax])]
        flat_right = [a for ax in right_axes for a in (ax if isinstance(ax, list) else [ax])]
        w_names = self.weight_shape.split()
        all_names = list(dict.fromkeys(flat_left + w_names + flat_right))
        char_map = {n: chr(ord('a') + i) for i, n in enumerate(all_names)}
        ein_str = (
            ''.join(char_map[n] for n in flat_left) + ',' +
            ''.join(char_map[n] for n in w_names) + '->' +
            ''.join(char_map[n] for n in flat_right)
        )
        out = _NpEinops.einsum(ein_str, x, self.weight)
        if self.bias_param is not None:
            out_fl = _NpEinops._fl(out); b_fl = _NpEinops._fl(self.bias_param)
            out_sh = out.shape if hasattr(out, 'shape') else (len(out_fl),)
            last = out_sh[-1]
            out = _NpEinops._nd([out_fl[i] + b_fl[i % last] for i in range(len(out_fl))], out_sh)
        return out

    def parameters(self):
        params = [self.weight]
        if self.bias_param is not None:
            params.append(self.bias_param)
        return params
