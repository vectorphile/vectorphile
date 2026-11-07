"""vectorphile.nn.einops — Einstein-notation tensor operations (einops-compatible API).

Implements rearrange, reduce, repeat, pack, unpack, and EinMix using pure numpy.
Pattern syntax is a subset of the einops DSL:
  - 'b c h w -> b (c h w)'   flatten spatial dims
  - 'b (h p1) (w p2) c -> (b h w) p1 p2 c'  split dims with size hints
  - 'b n d -> b d n'          transpose
"""
from __future__ import annotations
import re
import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


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
                assert np.prod(known) == dim, f"Group {ax} product {np.prod(known)} != {dim}"
            elif len(unknown) == 1:
                sizes[unknown[0]] = dim // int(np.prod(known)) if known else dim
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
        x = x.data if hasattr(x, 'data') else np.asarray(x)
    x = np.asarray(x)

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
    x = np.transpose(x, perm)

    # Step 5: reshape merging grouped right axes
    out_shape = []
    pos = 0
    for group in flat_right_groups:
        if len(group) == 1:
            out_shape.append(sizes[group[0]])
        else:
            out_shape.append(int(np.prod([sizes[a] for a in group])))
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
    x = np.asarray(x)

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

    ops = {'mean': np.mean, 'sum': np.sum, 'max': np.max, 'min': np.min, 'prod': np.prod}
    if reduction not in ops:
        raise ValueError(f"Unknown reduction: {reduction!r}. Use: {list(ops.keys())}")

    if reduce_axes:
        x = ops[reduction](x, axis=reduce_axes)

    # Reorder remaining axes
    remaining = [a for a in flat_left if a in flat_right]
    perm = [remaining.index(a) for a in flat_right]
    if perm != list(range(len(perm))):
        x = np.transpose(x, perm)

    # Reshape merged groups in right
    out_shape = []
    for ax in right_axes:
        if isinstance(ax, list):
            out_shape.append(int(np.prod([sizes[a] for a in ax])))
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
    x = np.asarray(x)

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
            x = np.expand_dims(x, axis=i)
            flat_left.insert(i, a)
            reps = [1] * len(flat_left)
            reps[i] = sizes[a]
            x = np.tile(x, reps)

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
    tensors = [np.asarray(t.data if hasattr(t, 'data') else t) for t in tensors]
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
        new_shape = t.shape[:n_before] + (int(np.prod(pack_dims)) if pack_dims else (1,)) + t.shape[len(t.shape)-n_after:] if n_after else t.shape[:n_before] + (int(np.prod(pack_dims)) if pack_dims else 1,)
        reshaped.append(t.reshape(new_shape))
    result = np.concatenate(reshaped, axis=star_idx)
    return result, packed_shapes


def unpack(x: np.ndarray, packed_shapes: List[Tuple], pattern: str) -> List[np.ndarray]:
    """Unpack a tensor packed with pack()."""
    x = np.asarray(x.data if hasattr(x, 'data') else x)
    parts = pattern.split()
    star_idx = parts.index('*')
    n_after = len(parts) - star_idx - 1
    results = []
    pos = 0
    for ps in packed_shapes:
        size = int(np.prod(ps)) if ps else 1
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
        self.weight = np.random.randn(*w_dims).astype(np.float32) * (2 / sum(w_dims)) ** 0.5
        self.bias_param = np.zeros(w_dims[-1], dtype=np.float32) if bias else None

        # Identify contraction axes
        left_axes, right_axes = _parse_pattern(pattern)
        flat_left = [a for ax in left_axes for a in (ax if isinstance(ax, list) else [ax])]
        flat_right = [a for ax in right_axes for a in (ax if isinstance(ax, list) else [ax])]
        self.in_axes = [a for a in w_names[:-1]]
        self.out_axis = w_names[-1]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if hasattr(x, 'data'):
            x = x.data
        x = np.asarray(x)
        # Use np.tensordot or einsum for the contraction
        # Build einsum string from pattern
        left_axes, right_axes = _parse_pattern(self.pattern)
        flat_left = [a for ax in left_axes for a in (ax if isinstance(ax, list) else [ax])]
        flat_right = [a for ax in right_axes for a in (ax if isinstance(ax, list) else [ax])]
        w_names = self.weight_shape.split()

        # Build einsum: 'b n d_in , d_in d_out -> b n d_out'
        ein_left = ' '.join(flat_left)
        ein_w = ' '.join(w_names)
        ein_right = ' '.join(flat_right)
        # Convert to single-char for np.einsum
        all_names = list(dict.fromkeys(flat_left + w_names + flat_right))
        char_map = {n: chr(ord('a') + i) for i, n in enumerate(all_names)}
        ein_str = (
            ''.join(char_map[n] for n in flat_left) + ',' +
            ''.join(char_map[n] for n in w_names) + '->' +
            ''.join(char_map[n] for n in flat_right)
        )
        out = np.einsum(ein_str, x, self.weight)
        if self.bias_param is not None:
            out = out + self.bias_param
        return out

    def parameters(self):
        params = [self.weight]
        if self.bias_param is not None:
            params.append(self.bias_param)
        return params
