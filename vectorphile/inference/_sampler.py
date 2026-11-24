"""Token samplers for autoregressive generation."""
from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class SamplerConfig:
    """Sampling hyperparameters.

    Args:
        temperature      : logit scaling (1.0 = unscaled, <1.0 = sharper, >1.0 = flatter)
        top_k            : keep only top-k logits (0 = disabled)
        top_p            : nucleus sampling probability mass (1.0 = disabled)
        repetition_penalty: penalise already-generated tokens (1.0 = disabled)
        min_p            : minimum probability relative to max (0.0 = disabled)
        max_new_tokens   : hard stop on generated tokens
        eos_token_id     : stop token id
    """
    temperature:       float = 1.0
    top_k:             int   = 0
    top_p:             float = 1.0
    repetition_penalty:float = 1.0
    min_p:             float = 0.0
    max_new_tokens:    int   = 512
    eos_token_id:      Optional[int] = None


def temperature_scale(logits: List[float], temperature: float) -> List[float]:
    """Scale logits by temperature (in-place over a copy)."""
    if temperature == 1.0 or temperature <= 0:
        return list(logits)
    inv_t = 1.0 / temperature
    return [v * inv_t for v in logits]


def _softmax(logits: List[float]) -> List[float]:
    max_v = max(logits)
    exps  = [math.exp(v - max_v) for v in logits]
    s     = sum(exps)
    return [e / s for e in exps]


def greedy_sample(logits: List[float]) -> int:
    """Return argmax of logits."""
    best_i, best_v = 0, logits[0]
    for i, v in enumerate(logits):
        if v > best_v:
            best_v = v; best_i = i
    return best_i


def top_k_sample(logits: List[float], k: int, temperature: float = 1.0) -> int:
    """Sample from the top-k logits after temperature scaling."""
    logits = temperature_scale(logits, temperature)
    if k <= 0 or k >= len(logits):
        probs = _softmax(logits)
        return _sample_from_probs(probs)
    # find top-k indices
    indexed = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)
    top_k_idx   = [i for i, _ in indexed[:k]]
    top_k_vals  = [logits[i] for i in top_k_idx]
    probs       = _softmax(top_k_vals)
    local_i     = _sample_from_probs(probs)
    return top_k_idx[local_i]


def top_p_sample(logits: List[float], p: float, temperature: float = 1.0) -> int:
    """Nucleus (top-p) sampling."""
    logits = temperature_scale(logits, temperature)
    indexed = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)
    sorted_probs = _softmax([v for _, v in indexed])
    cumulative   = 0.0
    cutoff = len(sorted_probs)
    for i, prob in enumerate(sorted_probs):
        cumulative += prob
        if cumulative >= p:
            cutoff = i + 1
            break
    nucleus_idx  = [indexed[i][0] for i in range(cutoff)]
    nucleus_probs = _softmax([logits[i] for i in nucleus_idx])
    local_i      = _sample_from_probs(nucleus_probs)
    return nucleus_idx[local_i]


def _sample_from_probs(probs: List[float]) -> int:
    """Draw one sample from a probability distribution."""
    r = random.random()
    cumulative = 0.0
    for i, p in enumerate(probs):
        cumulative += p
        if r < cumulative:
            return i
    return len(probs) - 1


def apply_repetition_penalty(logits: List[float], generated: List[int],
                               penalty: float) -> List[float]:
    """Penalise tokens that have already been generated."""
    if penalty == 1.0:
        return logits
    logits = list(logits)
    for tok_id in set(generated):
        if 0 <= tok_id < len(logits):
            logits[tok_id] = (logits[tok_id] / penalty
                              if logits[tok_id] > 0
                              else logits[tok_id] * penalty)
    return logits


def beam_search(
    score_fn,            # callable(input_ids: List[int]) -> List[float] logits
    input_ids: List[int],
    num_beams: int = 4,
    max_new_tokens: int = 50,
    eos_token_id: Optional[int] = None,
    length_penalty: float = 1.0,
) -> List[int]:
    """Simple beam search decoder.

    Args:
        score_fn       : function that takes token ids and returns next-token logits
        input_ids      : initial prompt token ids
        num_beams      : beam width
        max_new_tokens : maximum tokens to generate
        eos_token_id   : stop token (None = generate until max_new_tokens)
        length_penalty : >1 favours longer sequences, <1 shorter

    Returns:
        Best token sequence (prompt + generated tokens).
    """
    # Each beam: (score, token_ids, done)
    beams: List[Tuple[float, List[int], bool]] = [(0.0, list(input_ids), False)]

    for _ in range(max_new_tokens):
        if all(done for _, _, done in beams):
            break

        candidates: List[Tuple[float, List[int], bool]] = []
        for score, ids, done in beams:
            if done:
                candidates.append((score, ids, True))
                continue
            logits = score_fn(ids)
            probs  = _softmax(logits)
            # take top-num_beams tokens
            top_toks = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:num_beams]
            for tok in top_toks:
                new_score = score + math.log(max(probs[tok], 1e-30))
                new_ids   = ids + [tok]
                is_done   = (tok == eos_token_id) if eos_token_id is not None else False
                candidates.append((new_score, new_ids, is_done))

        # keep top-num_beams by length-normalised score
        def _norm_score(item):
            sc, ids, _ = item
            gen_len = max(1, len(ids) - len(input_ids))
            return sc / (gen_len ** length_penalty)

        candidates.sort(key=_norm_score, reverse=True)
        beams = candidates[:num_beams]

    best_score, best_ids, _ = max(beams, key=lambda x: x[0])
    return best_ids
