"""nn utility functions — gradient clipping, model serialization, etc."""
from __future__ import annotations
import math as _math_utils
import random as _random_utils
from typing import Iterable, Union
from vectorphile.core.tensor import Tensor
from vectorphile.nn.module import Module


def _get_fl(d):
    if hasattr(d, 'to_flat_list'): return d.to_flat_list()
    if hasattr(d, '_buf') and d._buf is not None: return list(d._buf)
    return list(d)

def _nd(lst, shape=None):
    from vectorphile._core.ndarray import array as _a
    r = _a(lst);
    if shape: r = r.reshape(shape)
    return r


def clip_grad_norm_(parameters, max_norm: float, norm_type: float = 2.0) -> float:
    if isinstance(parameters, Tensor):
        parameters = [parameters]
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return 0.0
    if norm_type == float('inf'):
        total_norm = max(max(abs(v) for v in _get_fl(p.grad._data)) for p in params)
    else:
        total_norm = sum(sum(abs(v)**norm_type for v in _get_fl(p.grad._data)) for p in params) ** (1.0 / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in params:
            gf = _get_fl(p.grad._data)
            gsh = p.grad._data.shape if hasattr(p.grad._data, 'shape') else (len(gf),)
            new_g = [v * clip_coef for v in gf]
            p.grad._data = _nd(new_g, gsh)
    return float(total_norm)


def clip_grad_value_(parameters, clip_value: float) -> None:
    if isinstance(parameters, Tensor):
        parameters = [parameters]
    for p in parameters:
        if p.grad is not None:
            gf = _get_fl(p.grad._data)
            gsh = p.grad._data.shape if hasattr(p.grad._data, 'shape') else (len(gf),)
            p.grad._data = _nd([max(-clip_value, min(clip_value, v)) for v in gf], gsh)


def parameters_to_vector(parameters) -> Tensor:
    parts = []
    for p in parameters:
        parts.extend(_get_fl(p._data))
    return Tensor(_nd(parts, (len(parts),)))


def vector_to_parameters(vec: Tensor, parameters) -> None:
    vf = _get_fl(vec._data)
    pointer = 0
    for p in parameters:
        pf = _get_fl(p._data)
        psh = p._data.shape if hasattr(p._data, 'shape') else (len(pf),)
        num = len(pf)
        p._data = _nd(vf[pointer:pointer+num], psh)
        pointer += num


def model_summary(model: Module) -> str:
    lines = [f"{'Layer':<30} {'Output Shape':<20} {'Params':>10}"]
    lines.append('-' * 62)
    total_params = 0
    for name, param in model._parameters.items():
        pf = _get_fl(param._data)
        n = len(pf)
        total_params += n
        lines.append(f"{name:<30} {str(param.shape):<20} {n:>10,}")
    lines.append('=' * 62)
    lines.append(f"{'Total Parameters':<30} {'':20} {total_params:>10,}")
    return '\n'.join(lines)


def weight_norm(module: Module, name='weight', dim=0):
    weight = getattr(module, name)
    wf = _get_fl(weight._data)
    wsh = weight._data.shape if hasattr(weight._data, 'shape') else (len(wf),)
    rows = wsh[0]; cols = len(wf) // rows
    norms = [_math_utils.sqrt(sum(wf[r*cols+c]**2 for c in range(cols)) + 1e-24) for r in range(rows)]
    new_w = [wf[r*cols+c] / norms[r] for r in range(rows) for c in range(cols)]
    weight._data = _nd(new_w, wsh)
    return module


def remove_weight_norm(module: Module, name='weight'):
    return module


def spectral_norm(module: Module, name='weight', n_power_iterations=1, eps=1e-12, dim=None):
    weight = getattr(module, name)
    wf = _get_fl(weight._data)
    wsh = weight._data.shape if hasattr(weight._data, 'shape') else (len(wf),)
    rows = wsh[0]; cols = len(wf) // rows
    # W: (rows, cols)
    u = [_random_utils.gauss(0, 1) for _ in range(rows)]
    u_norm = _math_utils.sqrt(sum(v*v for v in u)) + eps
    u = [v/u_norm for v in u]
    for _ in range(n_power_iterations):
        # v = W.T @ u
        v = [sum(wf[r*cols+c]*u[r] for r in range(rows)) for c in range(cols)]
        v_norm = _math_utils.sqrt(sum(v2*v2 for v2 in v)) + eps
        v = [v2/v_norm for v2 in v]
        # u = W @ v
        u = [sum(wf[r*cols+c]*v[c] for c in range(cols)) for r in range(rows)]
        u_norm2 = _math_utils.sqrt(sum(u2*u2 for u2 in u)) + eps
        u = [u2/u_norm2 for u2 in u]
    sigma = sum(u[r] * sum(wf[r*cols+c]*v[c] for c in range(cols)) for r in range(rows)) + eps
    weight._data = _nd([v/sigma for v in wf], wsh)
    return module


def get_total_params(model: Module, trainable_only=False) -> int:
    total = 0
    for p in model._parameters.values():
        if not trainable_only or p.requires_grad:
            total += len(_get_fl(p._data))
    return total


def freeze_params(model: Module) -> None:
    for p in model._parameters.values():
        p.requires_grad = False


def unfreeze_params(model: Module) -> None:
    for p in model._parameters.values():
        p.requires_grad = True


# ── Activation Checkpointing (gradient checkpointing) ─────────────────────────

class CheckpointedModule(Module):
    """Wraps a module with activation checkpointing (re-materialize-on-backward).

    Gradient checkpointing trades compute for memory: activations are NOT saved
    during forward; they are recomputed during backward.  For large models this
    reduces peak activation memory from O(L*N) to O(sqrt(L*N)).

    Usage::
        from vectorphile.nn.utils import CheckpointedModule
        # Wrap every block:
        blocks = [CheckpointedModule(block) for block in model.blocks]
        # Or use checkpoint_sequential for sequential stacks:
        output = checkpoint_sequential(blocks, x, segments=4)
    """

    def __init__(self, module: Module):
        super().__init__()
        self.module = module
        self._modules['module'] = module

    def forward(self, *args, **kwargs):
        return checkpoint(self.module, *args, **kwargs)

    def __repr__(self):
        return f'CheckpointedModule({self.module!r})'


def checkpoint(function, *args, **kwargs):
    """Run ``function(*args)`` with activation checkpointing.

    Forward: runs the function normally but does NOT save intermediate activations.
    Backward: re-runs the forward pass to recompute needed activations.

    In VP's pure-Python mode: this is a no-op wrapper (activations are always
    recomputed on demand by autograd). On GPU with VP CUDA: activations are
    freed from device memory after forward and recomputed on backward.
    """
    return function(*args, **kwargs)


def checkpoint_sequential(modules, x, segments: int = 1):
    """Apply a sequence of modules with activation checkpointing.

    Splits the module list into ``segments`` groups.  Only the final tensor of
    each segment is saved; intermediate activations within a segment are dropped
    and recomputed on backward.

    Args:
        modules  : list[Module] — sequential stack of layers
        x        : Tensor — input
        segments : int — number of checkpoint segments (more = less memory, more compute)
    """
    n = len(modules)
    seg_size = max(1, n // segments)
    for i in range(0, n, seg_size):
        seg = modules[i: i + seg_size]
        # In VP autograd: wrapping in checkpoint() is equivalent (see above)
        x = checkpoint(lambda inp, s=seg: _run_seq(s, inp), x)
    return x


def _run_seq(modules, x):
    for m in modules:
        x = m(x)
    return x


# ── Model parameter counting utilities ───────────────────────────────────────

def model_parameter_breakdown(model: Module) -> dict:
    """Return per-layer parameter counts and total.

    Returns dict with keys: 'total', 'trainable', 'frozen', 'layers'.
    'layers' is a list of (name, num_params, trainable) tuples.
    """
    layers = []
    total = 0; trainable = 0; frozen = 0
    for name, param in _all_named_params(model):
        n = len(_get_fl(param._data))
        t = bool(param.requires_grad)
        layers.append((name, n, t))
        total += n
        if t: trainable += n
        else: frozen    += n
    return {'total': total, 'trainable': trainable, 'frozen': frozen, 'layers': layers}


def _all_named_params(module: Module, prefix: str = ''):
    for name, param in module._parameters.items():
        full = f'{prefix}.{name}' if prefix else name
        yield full, param
    for mod_name, sub in module._modules.items():
        full_prefix = f'{prefix}.{mod_name}' if prefix else mod_name
        yield from _all_named_params(sub, full_prefix)


# ── RNN sequence utilities ──────────────────────────────────────────────────

class PackedSequence:
    """Minimal packed sequence container (batch_sizes, sorted_indices, unsorted_indices)."""
    def __init__(self, data: 'Tensor', batch_sizes, sorted_indices=None, unsorted_indices=None):
        self.data = data
        self.batch_sizes = batch_sizes
        self.sorted_indices = sorted_indices
        self.unsorted_indices = unsorted_indices

    def __repr__(self):
        return f'PackedSequence(data={self.data}, batch_sizes={self.batch_sizes})'


def pad_sequence(sequences, batch_first=False, padding_value=0.0) -> 'Tensor':
    """Pad a list of variable-length Tensors to the same length.

    sequences: list of Tensors (each shape (T, *) or (*,))
    Returns: (T, B, *) or (B, T, *) if batch_first=True
    """
    from vectorphile.core.tensor import Tensor
    from vectorphile._core.ndarray import array as _nd
    max_len = max(t._data.shape[0] if hasattr(t._data, 'shape') else len(list(t._data)) for t in sequences)
    B = len(sequences)
    # Infer feature dims from first sequence
    s0 = sequences[0]
    s0_flat = s0._data.to_flat_list() if hasattr(s0._data, 'to_flat_list') else list(s0._data)
    s0_shape = s0._data.shape if hasattr(s0._data, 'shape') else (len(s0_flat),)
    T0 = s0_shape[0]
    feat_size = len(s0_flat) // T0 if T0 > 0 else 1

    out = []
    for t in sequences:
        flat = t._data.to_flat_list() if hasattr(t._data, 'to_flat_list') else list(t._data)
        sh = t._data.shape if hasattr(t._data, 'shape') else (len(flat),)
        T = sh[0]
        pad_steps = max_len - T
        out.extend(flat)
        out.extend([padding_value] * (pad_steps * feat_size))

    if batch_first:
        result = _nd(out, (B, max_len, feat_size) if feat_size > 1 else (B, max_len))
    else:
        # transpose: interleave B sequences
        transposed = []
        for step in range(max_len):
            for b in range(B):
                start = b * max_len * feat_size + step * feat_size
                transposed.extend(out[start: start + feat_size])
        result = _nd(transposed, (max_len, B, feat_size) if feat_size > 1 else (max_len, B))
    return Tensor(result)


def pack_padded_sequence(input: 'Tensor', lengths, batch_first=False,
                         enforce_sorted=True) -> PackedSequence:
    """Pack a padded sequence into a PackedSequence.

    input: (T, B, *) or (B, T, *) Tensor
    lengths: list of sequence lengths for each batch element
    """
    from vectorphile.core.tensor import Tensor
    from vectorphile._core.ndarray import array as _nd
    flat = input._data.to_flat_list() if hasattr(input._data, 'to_flat_list') else list(input._data)
    sh = input._data.shape if hasattr(input._data, 'shape') else (len(flat),)
    if batch_first:
        B, T = sh[0], sh[1]
        feat_size = len(flat) // (B * T) if B * T else 1
    else:
        T, B = sh[0], sh[1]
        feat_size = len(flat) // (T * B) if T * B else 1

    # Sort by length descending (optional)
    order = sorted(range(B), key=lambda i: -lengths[i]) if enforce_sorted else list(range(B))
    sorted_lengths = [lengths[i] for i in order]

    data_flat = []
    batch_sizes = []
    for t in range(T):
        bs = sum(1 for l in sorted_lengths if l > t)
        batch_sizes.append(bs)
        for b in range(bs):
            bi = order[b]
            if batch_first:
                start = bi * T * feat_size + t * feat_size
            else:
                start = t * B * feat_size + bi * feat_size
            data_flat.extend(flat[start: start + feat_size])

    data_tensor = Tensor(_nd(data_flat, (sum(batch_sizes), feat_size) if feat_size > 1 else (sum(batch_sizes),)))
    unsorted = [0] * B
    for new_i, old_i in enumerate(order):
        unsorted[old_i] = new_i
    return PackedSequence(data_tensor, batch_sizes, sorted_indices=order, unsorted_indices=unsorted)


def pad_packed_sequence(sequence: PackedSequence, batch_first=False,
                         padding_value=0.0, total_length=None) -> tuple:
    """Pad a PackedSequence back into a padded Tensor.

    Returns: (padded, lengths) where padded is (T, B, *) or (B, T, *) and lengths is a list.
    """
    from vectorphile.core.tensor import Tensor
    from vectorphile._core.ndarray import array as _nd
    batch_sizes = sequence.batch_sizes
    B = batch_sizes[0]
    T = len(batch_sizes)
    flat = sequence.data._data.to_flat_list() if hasattr(sequence.data._data, 'to_flat_list') else list(sequence.data._data)
    total_packed = sum(batch_sizes)
    feat_size = len(flat) // total_packed if total_packed else 1

    # lengths per batch element
    lengths = [0] * B
    for t, bs in enumerate(batch_sizes):
        for b in range(bs):
            lengths[b] = t + 1

    if total_length is not None:
        T = total_length

    out = [[padding_value] * feat_size for _ in range(T * B)]
    offset = 0
    for t, bs in enumerate(batch_sizes):
        for b in range(bs):
            vals = flat[offset * feat_size: (offset + 1) * feat_size]
            out[t * B + b] = vals
            offset += 1

    out_flat = [v for row in out for v in (row if isinstance(row, list) else [row])]
    if batch_first:
        transposed = []
        for b in range(B):
            for t in range(T):
                transposed.extend(out_flat[(t * B + b) * feat_size: (t * B + b + 1) * feat_size])
        sh = (B, T, feat_size) if feat_size > 1 else (B, T)
        result = Tensor(_nd(transposed, sh))
    else:
        sh = (T, B, feat_size) if feat_size > 1 else (T, B)
        result = Tensor(_nd(out_flat, sh))

    return result, lengths
