"""vectorphile.nn.pruning — Structured and unstructured network pruning."""
from __future__ import annotations
import math
import random as _random_prune
from typing import Optional, List, Tuple, Union


def _fl(d):
    if hasattr(d, 'to_flat_list'): return d.to_flat_list()
    if hasattr(d, '_buf') and d._buf is not None: return list(d._buf)
    return list(d)

def _nd(lst, shape=None):
    from vectorphile._core.ndarray import array as _a
    r = _a(lst if isinstance(lst, list) else list(lst))
    if shape: r = r.reshape(shape)
    return r

def _get_param(module, name: str):
    p = getattr(module, name, None)
    if p is None:
        raise AttributeError(f"Module has no parameter '{name}'")
    return p


def prune_unstructured(module, amount: float, name: str = 'weight',
                       method: str = 'magnitude') -> None:
    p = _get_param(module, name)
    pf = _fl(p._data); psh = p._data.shape if hasattr(p._data, 'shape') else (len(pf),)
    n = len(pf); n_prune = int(n * amount)
    if n_prune == 0: return
    if method == 'magnitude':
        abs_vals = [abs(v) for v in pf]
        threshold = sorted(abs_vals)[n_prune - 1]
        mask = [1.0 if abs(v) >= threshold else 0.0 for v in pf]
    elif method == 'random':
        mask = [1.0] * n
        idx = _random_prune.sample(range(n), n_prune)
        for i in idx: mask[i] = 0.0
    else:
        raise ValueError(f'Unknown pruning method: {method}')
    p._data = _nd([pf[i] * mask[i] for i in range(n)], psh)
    module.__dict__[f'_{name}_mask'] = mask


def prune_structured(module, amount: float, name: str = 'weight',
                     dim: int = 0, norm: int = 1) -> None:
    p = _get_param(module, name)
    pf = _fl(p._data); psh = p._data.shape if hasattr(p._data, 'shape') else (len(pf),)
    n_total = psh[dim]; n_prune = int(n_total * amount)
    if n_prune == 0: return
    # Compute per-filter norm
    ndim = len(psh)
    filter_sz = 1
    for i in range(ndim):
        if i != dim: filter_sz *= psh[i]
    norms = []
    for fi in range(n_total):
        # gather all elements along dim==fi
        s = 0.0
        for idx in range(len(pf)):
            # check if dim-axis index == fi
            di = (idx // (filter_sz if dim < ndim - 1 else 1)) % psh[dim] if ndim > 1 else idx
            # simplified: stride along dim
            stride = 1
            for d2 in range(dim+1, ndim): stride *= psh[d2]
            di2 = (idx // stride) % n_total
            if di2 == fi:
                s += abs(pf[idx]) ** norm
        norms.append(s ** (1.0/norm))
    prune_idx = set(sorted(range(n_total), key=lambda i: norms[i])[:n_prune])
    new_pf = list(pf)
    for idx in range(len(pf)):
        stride = 1
        for d2 in range(dim+1, ndim): stride *= psh[d2]
        di2 = (idx // stride) % n_total
        if di2 in prune_idx: new_pf[idx] = 0.0
    p._data = _nd(new_pf, psh)


def remove_pruning(module, name: str = 'weight') -> None:
    mask_key = f'_{name}_mask'
    if mask_key in module.__dict__:
        del module.__dict__[mask_key]


def apply_mask(module, name: str = 'weight') -> None:
    mask_key = f'_{name}_mask'
    mask = module.__dict__.get(mask_key)
    if mask is None: return
    p = _get_param(module, name)
    pf = _fl(p._data); psh = p._data.shape if hasattr(p._data, 'shape') else (len(pf),)
    p._data = _nd([pf[i] * mask[i] for i in range(len(pf))], psh)


def global_unstructured_prune(model, amount: float, method: str = 'magnitude') -> None:
    all_params = [p for p in model.parameters() if len(p._data.shape if hasattr(p._data, 'shape') else (len(_fl(p._data)),)) >= 2]
    if not all_params: return
    all_weights = [abs(v) for p in all_params for v in _fl(p._data)]
    n_prune = int(len(all_weights) * amount)
    if n_prune == 0: return
    if method == 'magnitude':
        threshold = sorted(all_weights)[n_prune - 1]
        for p in all_params:
            pf = _fl(p._data); psh = p._data.shape if hasattr(p._data, 'shape') else (len(pf),)
            p._data = _nd([v if abs(v) >= threshold else 0.0 for v in pf], psh)
    elif method == 'random':
        prune_set = set(_random_prune.sample(range(len(all_weights)), n_prune))
        offset = 0
        for p in all_params:
            pf = _fl(p._data); psh = p._data.shape if hasattr(p._data, 'shape') else (len(pf),)
            new_pf = [0.0 if (offset+i) in prune_set else pf[i] for i in range(len(pf))]
            p._data = _nd(new_pf, psh); offset += len(pf)


def sparsity(model) -> float:
    total = 0; zeros = 0
    for p in model.parameters():
        pf = _fl(p._data); total += len(pf); zeros += sum(1 for v in pf if v == 0.0)
    return zeros / (total + 1e-10)


class GradualMagnitudePruner:
    def __init__(self, initial_sparsity: float = 0.0, final_sparsity: float = 0.9,
                 begin_step: int = 0, end_step: int = 100, frequency: int = 10):
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.begin_step = begin_step
        self.end_step = end_step
        self.frequency = frequency
        self._step = 0

    def _target_sparsity(self) -> float:
        t = self._step
        if t < self.begin_step: return self.initial_sparsity
        if t >= self.end_step: return self.final_sparsity
        progress = (t - self.begin_step) / (self.end_step - self.begin_step)
        return self.final_sparsity + (self.initial_sparsity - self.final_sparsity) * (1 - progress) ** 3

    def step(self, model) -> float:
        self._step += 1
        if (self._step - self.begin_step) % self.frequency != 0: return sparsity(model)
        global_unstructured_prune(model, amount=self._target_sparsity(), method='magnitude')
        return sparsity(model)


class LotteryTicketPruner:
    def __init__(self, model, prune_amount_per_round: float = 0.2):
        self.model = model
        self.prune_amount = prune_amount_per_round
        self._initial_weights = {id(p): list(_fl(p._data)) for p in model.parameters()}
        self._initial_shapes = {id(p): (p._data.shape if hasattr(p._data, 'shape') else (len(_fl(p._data)),)) for p in model.parameters()}
        self._masks = {id(p): None for p in model.parameters()}

    def prune_and_rewind(self):
        global_unstructured_prune(self.model, amount=self.prune_amount)
        for p in self.model.parameters():
            pf = _fl(p._data)
            self._masks[id(p)] = [1.0 if v != 0.0 else 0.0 for v in pf]
        for p in self.model.parameters():
            init = self._initial_weights[id(p)]
            sh = self._initial_shapes[id(p)]
            mask = self._masks[id(p)]
            if mask is not None:
                p._data = _nd([init[i]*mask[i] for i in range(len(init))], sh)
            else:
                p._data = _nd(init, sh)

    def sparsity(self) -> float:
        return sparsity(self.model)
