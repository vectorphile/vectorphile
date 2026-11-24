"""Text generation utilities — greedy, sampling, top-k/p filtering."""
from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class GenerationConfig:
    max_new_tokens: int = 50
    do_sample: bool = False
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    eos_token_id: Optional[int] = None
    pad_token_id: int = 0


def _is_numpy(x):
    t = type(x); return t.__module__ == 'numpy' or (hasattr(x, 'flatten') and hasattr(x, 'dtype') and not hasattr(x, 'to_flat_list'))


def _to_float_list(logits):
    """Convert Tensor/NDArray/list to plain Python list of floats."""
    if hasattr(logits, 'tolist'):
        v = logits.tolist()
    elif hasattr(logits, '_data'):
        v = logits._data.tolist()
    else:
        v = logits
    if isinstance(v, list) and v and isinstance(v[0], list):
        v = [item for row in v for item in row]
    return [float(x) for x in v]


def _softmax_1d(logits: List[float]) -> List[float]:
    m = max(logits)
    e = [math.exp(v - m) for v in logits]
    s = sum(e)
    return [v / s for v in e]


def top_k_filtering(logits, top_k: int = 50):
    """Zero out all logits except the top-k per row. Returns modified copy."""
    if hasattr(logits, 'to_flat_list') or hasattr(logits, 'grad_fn'):
        logits = _to_float_list(logits)
    if _is_numpy(logits):
        import numpy as _np
        logits = _np.array(logits, copy=True)
        if logits.ndim == 1:
            logits = logits[_np.newaxis]
            squeeze = True
        else:
            squeeze = False
        V = logits.shape[-1]
        if top_k <= 0 or top_k >= V:
            return logits.squeeze(0) if squeeze else logits
        for b in range(logits.shape[0]):
            indexed = sorted(range(V), key=lambda i: logits[b, i], reverse=True)
            keep = set(indexed[:top_k])
            for i in range(V):
                if i not in keep:
                    logits[b, i] = -1e9
        return logits.squeeze(0) if squeeze else logits
    # 1D list path
    if not logits or top_k <= 0 or top_k >= len(logits):
        return list(logits)
    indexed = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)
    keep = set(i for i, _ in indexed[:top_k])
    return [logits[i] if i in keep else -1e9 for i in range(len(logits))]


def top_p_filtering(logits, top_p: float = 0.9):
    """Nucleus (top-p) filtering. Returns modified copy."""
    if hasattr(logits, 'to_flat_list') or hasattr(logits, 'grad_fn'):
        logits = _to_float_list(logits)
    if _is_numpy(logits):
        import numpy as _np
        logits = _np.array(logits, copy=True)
        if top_p >= 1.0:
            return logits
        orig_shape = logits.shape
        if logits.ndim == 1:
            logits = logits[_np.newaxis]
        for b in range(logits.shape[0]):
            row = logits[b]
            V = len(row)
            sorted_idx = _np.argsort(-row)
            sorted_l = row[sorted_idx]
            e = _np.exp(sorted_l - sorted_l.max()); probs = e / e.sum()
            cumprob = 0.0
            for rank in range(V):
                cumprob += probs[rank]
                if cumprob > top_p and rank > 0:
                    for r2 in range(rank + 1, V):
                        logits[b, sorted_idx[r2]] = -1e9
                    break
        return logits.reshape(orig_shape)
    # 1D list path
    if not logits or top_p >= 1.0:
        return list(logits)
    sorted_idx = sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)
    probs = _softmax_1d([logits[i] for i in sorted_idx])
    result = list(logits)
    cumprob = 0.0
    for rank, (idx, prob) in enumerate(zip(sorted_idx, probs)):
        cumprob += prob
        if cumprob > top_p and rank > 0:
            for r2 in range(rank + 1, len(sorted_idx)):
                result[sorted_idx[r2]] = -1e9
            break
    return result


def _get_logits_row(out_data, b: int, t: int) -> List[float]:
    """Extract logit row from NDArray at [b, t, :]."""
    if hasattr(out_data, 'shape'):
        sh = out_data.shape
        if len(sh) == 3:
            B, T, V = sh
            flat = list(out_data.to_flat_list()) if hasattr(out_data, 'to_flat_list') else [float(x) for x in out_data]
            start = b * T * V + t * V
            return flat[start:start + V]
        elif len(sh) == 2:
            T, V = sh
            flat = list(out_data.to_flat_list()) if hasattr(out_data, 'to_flat_list') else [float(x) for x in out_data]
            return flat[t * V:t * V + V]
    return [float(x) for x in out_data]


def _argmax(lst: List[float]) -> int:
    return max(range(len(lst)), key=lambda i: lst[i])


def _sample_token(logits_row: List[float], temperature: float,
                  top_k: int, top_p: float) -> int:
    l = [v / max(temperature, 1e-8) for v in logits_row]
    if top_k > 0:
        l = top_k_filtering(l, top_k)
    if top_p < 1.0:
        l = top_p_filtering(l, top_p)
    probs = _softmax_1d(l)
    r = random.random(); cumsum = 0.0
    for i, p in enumerate(probs):
        cumsum += p
        if r <= cumsum:
            return i
    return len(probs) - 1


def _ids_to_nd(ids: List[List[int]]):
    from vectorphile._core.ndarray import NDArray, _c_strides, _DType
    import array as _arr
    B = len(ids); T = len(ids[0])
    flat = [float(ids[b][t]) for b in range(B) for t in range(T)]
    buf = _arr.array('f', flat)
    sh = (B, T)
    return NDArray(buf, sh, _c_strides(sh), _DType('float32'))


def _ids_list_to_numpy(ids: List[List[int]], dtype='int64'):
    import numpy as _np
    B = len(ids); T = len(ids[0])
    arr = _np.empty((B, T), dtype=dtype)
    for b in range(B):
        for t in range(T):
            arr[b, t] = ids[b][t]
    return arr


class GreedySearchDecoder:
    def __init__(self, config: GenerationConfig):
        self.cfg = config

    def __call__(self, model, input_ids):
        return_numpy = _is_numpy(input_ids)
        if _is_numpy(input_ids):
            import numpy as _np
            flat = input_ids.ravel().tolist()
            B, T = (input_ids.shape[0], input_ids.shape[1]) if input_ids.ndim == 2 else (1, len(flat))
        else:
            flat = [float(x) for row in (input_ids if isinstance(input_ids, list) and isinstance(input_ids[0], list) else [input_ids]) for x in row]
            B = len(input_ids) if isinstance(input_ids, list) and isinstance(input_ids[0], list) else 1
            T = len(input_ids[0]) if isinstance(input_ids, list) and isinstance(input_ids[0], list) else len(input_ids)
        ids = [[int(flat[b * T + t]) for t in range(T)] for b in range(B)]

        for _ in range(self.cfg.max_new_tokens):
            from vectorphile.core.tensor import Tensor
            nd = _ids_to_nd(ids)
            out = model(Tensor(nd))
            out_data = out._data
            cur_T = len(ids[0])
            next_toks = [_argmax(_get_logits_row(out_data, b, cur_T - 1)) for b in range(B)]
            for b in range(B):
                ids[b].append(next_toks[b])
            if self.cfg.eos_token_id is not None and all(t == self.cfg.eos_token_id for t in next_toks):
                break

        if return_numpy:
            return _ids_list_to_numpy(ids)
        return ids


class SamplingDecoder:
    def __init__(self, config: GenerationConfig):
        self.cfg = config

    def __call__(self, model, input_ids):
        return_numpy = _is_numpy(input_ids)
        if _is_numpy(input_ids):
            flat = input_ids.ravel().tolist()
            B, T = (input_ids.shape[0], input_ids.shape[1]) if input_ids.ndim == 2 else (1, len(flat))
        else:
            flat = [float(x) for row in (input_ids if isinstance(input_ids, list) and isinstance(input_ids[0], list) else [input_ids]) for x in row]
            B = 1; T = len(flat)
        ids = [[int(flat[b * T + t]) for t in range(T)] for b in range(B)]

        for _ in range(self.cfg.max_new_tokens):
            from vectorphile.core.tensor import Tensor
            nd = _ids_to_nd(ids)
            out = model(Tensor(nd))
            cur_T = len(ids[0])
            next_toks = [_sample_token(_get_logits_row(out._data, b, cur_T - 1),
                                       self.cfg.temperature, self.cfg.top_k, self.cfg.top_p)
                         for b in range(B)]
            for b in range(B):
                ids[b].append(next_toks[b])
            if self.cfg.eos_token_id is not None and all(t == self.cfg.eos_token_id for t in next_toks):
                break

        if return_numpy:
            return _ids_list_to_numpy(ids)
        return ids


class BeamSearchDecoder:
    """Beam search decoder for autoregressive models."""

    def __init__(self, config: GenerationConfig = None, num_beams: int = 4):
        self.config = config or GenerationConfig()
        self.num_beams = num_beams

    def __call__(self, model, input_ids, return_numpy: bool = False):
        return_numpy = return_numpy or _is_numpy(input_ids)
        if _is_numpy(input_ids):
            flat = input_ids.ravel().tolist()
            B = input_ids.shape[0] if input_ids.ndim == 2 else 1
            T = input_ids.shape[1] if input_ids.ndim == 2 else len(flat)
        elif isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
            flat = [x for row in input_ids for x in row]
            B, T = len(input_ids), len(input_ids[0])
        else:
            flat = [int(x) for x in input_ids]
            B, T = 1, len(flat)
        ids_list = [[int(flat[b * T + t]) for t in range(T)] for b in range(B)]
        # Beam search per batch item (B=1 common case)
        ids_list = ids_list[0] if B == 1 else ids_list[0]  # simplify to single sequence
        num_beams = self.num_beams
        # Beam: (score, sequence)
        beams = [(0.0, ids_list[:])]
        completed = []

        for _ in range(self.config.max_new_tokens):
            if not beams:
                break
            candidates = []
            for score, seq in beams:
                try:
                    from vectorphile.core.tensor import Tensor
                    nd = _ids_to_nd([[seq]])
                    logits = model(Tensor(nd))
                    logits_flat = _get_logits_row(logits._data, 0, len(seq) - 1)
                    # Apply temperature
                    t = max(self.config.temperature, 1e-7)
                    logits_flat = [v / t for v in logits_flat]
                    # Log-softmax
                    mx = max(logits_flat)
                    log_probs = [v - mx - math.log(sum(math.exp(lv - mx) for lv in logits_flat))
                                 for v in logits_flat]
                    # Top-num_beams candidates
                    top_k_idx = sorted(range(len(log_probs)), key=lambda i: log_probs[i], reverse=True)[:num_beams]
                    for tok_id in top_k_idx:
                        new_score = score + log_probs[tok_id]
                        new_seq = seq + [tok_id]
                        if self.config.eos_token_id is not None and tok_id == self.config.eos_token_id:
                            completed.append((new_score, new_seq))
                        else:
                            candidates.append((new_score, new_seq))
                except Exception:
                    candidates.append((score, seq + [0]))
            # Keep top num_beams beams
            beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:num_beams]

        all_seqs = completed + beams
        if not all_seqs:
            return ids_list
        best = max(all_seqs, key=lambda x: x[0])[1]
        if return_numpy:
            return _ids_list_to_numpy([best])
        return best


def min_p_filtering(logits, min_p: float = 0.05):
    """Min-p sampling (Nguyen et al. 2024): keep tokens with p >= min_p * max_p."""
    from vectorphile.core.tensor import Tensor as _T
    from vectorphile._core.ndarray import array as _nd
    # Determine original shape for Tensor input
    if isinstance(logits, _T):
        sh = logits._data.shape
        flat = logits._data.to_flat_list() if hasattr(logits._data, 'to_flat_list') else list(logits._data)
        flat = [float(v) for v in flat]
    else:
        flat = _to_float_list(logits)
        sh = (len(flat),)
    # Process row by row if 2D
    if len(sh) == 2:
        B, V = sh
        result = []
        for b in range(B):
            row = flat[b*V:(b+1)*V]
            result.extend(_min_p_1d(row, min_p))
        nd = _nd(result)
        nd = nd.reshape(sh)
        return _T(nd)
    # 1-D case
    result = _min_p_1d(flat, min_p)
    nd = _nd(result)
    if len(sh) > 1:
        nd = nd.reshape(sh)
    return _T(nd)


def _min_p_1d(logits_flat, min_p):
    import math as _m
    m = max(logits_flat)
    exps = [_m.exp(v - m) for v in logits_flat]
    s = sum(exps)
    probs = [e / s for e in exps]
    max_prob = max(probs)
    threshold = min_p * max_prob
    return [v if p >= threshold else float('-inf') for v, p in zip(logits_flat, probs)]


def _typical_1d(logits_flat, mass):
    import math as _m
    m = max(logits_flat)
    exps = [_m.exp(v - m) for v in logits_flat]
    s = sum(exps)
    probs = [e / s for e in exps]
    entropy = -sum(p * _m.log(max(p, 1e-10)) for p in probs)
    neg_log_p = [-_m.log(max(p, 1e-10)) for p in probs]
    deviations = [abs(nlp - entropy) for nlp in neg_log_p]
    sorted_idx = sorted(range(len(logits_flat)), key=lambda i: deviations[i])
    cumprob = 0.0; keep = set()
    for i in sorted_idx:
        keep.add(i)
        cumprob += probs[i]
        if cumprob >= mass:
            break
    return [v if i in keep else float('-inf') for i, v in enumerate(logits_flat)]


def typical_filtering(logits, mass: float = 0.9):
    """Typical sampling (Meister et al. 2023): filter by conditional entropy."""
    from vectorphile.core.tensor import Tensor as _T
    from vectorphile._core.ndarray import array as _nd
    if isinstance(logits, _T):
        sh = logits._data.shape
        flat = [float(v) for v in (logits._data.to_flat_list() if hasattr(logits._data, 'to_flat_list') else list(logits._data))]
    else:
        flat = _to_float_list(logits)
        sh = (len(flat),)
    if len(sh) == 2:
        B, V = sh
        result = []
        for b in range(B):
            result.extend(_typical_1d(flat[b*V:(b+1)*V], mass))
        nd = _nd(result).reshape(sh)
        return _T(nd)
    result = _typical_1d(flat, mass)
    nd = _nd(result)
    if len(sh) > 1:
        nd = nd.reshape(sh)
    return _T(nd)


def apply_repetition_penalty(logits: List[float], input_ids: List[int],
                              penalty: float = 1.1) -> List[float]:
    """Penalize tokens that already appeared: logit / penalty if > 0 else * penalty."""
    result = logits[:]
    seen = set(input_ids)
    for tok in seen:
        if tok < len(result):
            result[tok] = result[tok] / penalty if result[tok] > 0 else result[tok] * penalty
    return result


class SpeculativeDecoder:
    """Speculative decoding (Leviathan et al. 2023).

    Uses a small draft model to propose γ tokens, then verifies with the target model.
    Expected speedup: ~2-4x for well-matched draft/target pairs.

    Args:
        draft_model: small fast model (same interface as target)
        target_model: large accurate model
        gamma: number of speculative tokens per step (default 4)
        cfg: GenerationConfig (temperature, top_k, etc.)
    """

    def __init__(self, draft_model, target_model, gamma: int = 4,
                 cfg: GenerationConfig = None):
        self.draft = draft_model
        self.target = target_model
        self.gamma = gamma
        self.cfg = cfg or GenerationConfig()

    def _get_probs(self, model, ids):
        from vectorphile.core.tensor import Tensor
        from vectorphile._core.ndarray import array as _nd_arr
        ids_t = Tensor(_nd_arr([float(v) for v in ids], 'float32').reshape((1, len(ids))))
        try:
            out = model(ids_t)
            logits = out._data.to_flat_list() if hasattr(out._data, 'to_flat_list') else list(out._data)
        except Exception:
            return _softmax_1d([0.0] * 32000)
        sh = out._data.shape if hasattr(out._data, 'shape') else (len(logits),)
        last_row = logits[-sh[-1]:]
        return _softmax_1d(last_row)

    def __call__(self, input_ids: List[int], max_new_tokens: int = 50) -> List[int]:
        import math as _m, random as _r
        ids = list(input_ids)
        generated = 0
        while generated < max_new_tokens:
            # Draft phase: propose γ tokens
            draft_ids = []; draft_probs_list = []
            cur = ids[:]
            for _ in range(self.gamma):
                dprobs = self._get_probs(self.draft, cur)
                tok = _r.choices(range(len(dprobs)), weights=dprobs)[0]
                draft_ids.append(tok)
                draft_probs_list.append(dprobs[tok])
                cur.append(tok)
            # Verify phase: target model on all γ draft tokens at once
            tprobs_list = []
            for i, tok in enumerate(draft_ids):
                tprobs = self._get_probs(self.target, ids + draft_ids[:i])
                tprobs_list.append(tprobs[tok])
            # Accept/reject via ratio test
            accepted = 0
            for i, (tp, dp) in enumerate(zip(tprobs_list, draft_probs_list)):
                r = _r.random()
                if r < min(1.0, tp / max(dp, 1e-10)):
                    ids.append(draft_ids[i])
                    generated += 1
                    accepted += 1
                else:
                    # sample correction token from (q - p)+ distribution
                    tprobs = self._get_probs(self.target, ids)
                    corrected = [max(0.0, tprobs[j] - draft_probs_list[i]) for j in range(len(tprobs))]
                    s = sum(corrected)
                    if s > 0:
                        corrected = [v/s for v in corrected]
                        tok = _r.choices(range(len(corrected)), weights=corrected)[0]
                    else:
                        tok = tprobs.index(max(tprobs))
                    ids.append(tok)
                    generated += 1
                    break
            if accepted == 0:
                # No draft tokens accepted; sample from target directly
                tprobs = self._get_probs(self.target, ids)
                tok = tprobs.index(max(tprobs))
                ids.append(tok)
                generated += 1
        return ids[len(input_ids):]


def generate(model, input_ids, max_new_tokens: int = 50,
             strategy: str = 'greedy', num_beams: int = 1,
             draft_model=None, gamma: int = 4,
             repetition_penalty: float = 1.0, min_p: float = 0.0,
             typical_p: float = 0.0, **kwargs):
    """Generate tokens from a model.

    Args:
        model: VP Module with forward(input_ids) → logits
        input_ids: starting token ids (list or Tensor)
        max_new_tokens: how many new tokens to generate
        strategy: 'greedy' | 'sample' | 'beam' | 'speculative'
        num_beams: beam width (strategy='beam')
        draft_model: small draft model for speculative decoding
        gamma: draft tokens per step for speculative decoding
        repetition_penalty: penalty > 1 discourages repetition
        min_p: min-p filtering threshold (0 = disabled)
        typical_p: typical sampling mass (0 = disabled)
    """
    cfg = GenerationConfig(max_new_tokens=max_new_tokens, **kwargs)
    if strategy == 'speculative' and draft_model is not None:
        ids_list = (input_ids._data.to_flat_list() if hasattr(input_ids, '_data')
                    else list(input_ids))
        ids_list = [int(v) for v in ids_list]
        return SpeculativeDecoder(draft_model, model, gamma=gamma, cfg=cfg)(ids_list, max_new_tokens)
    if strategy == 'beam' or num_beams > 1:
        return BeamSearchDecoder(cfg, num_beams=num_beams)(model, input_ids)
    if strategy == 'greedy' or not cfg.do_sample:
        return GreedySearchDecoder(cfg)(model, input_ids)
    return SamplingDecoder(cfg)(model, input_ids)


__all__ = [
    'GenerationConfig', 'GreedySearchDecoder', 'SamplingDecoder', 'BeamSearchDecoder',
    'SpeculativeDecoder', 'generate',
    'top_k_filtering', 'top_p_filtering', 'min_p_filtering', 'typical_filtering',
    'apply_repetition_penalty',
]
