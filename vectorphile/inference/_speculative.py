"""Speculative decoding (Leviathan et al. 2023 / Chen et al. 2023).

A small draft model generates K candidate tokens cheaply, then a large
target model verifies them all in one forward pass. Expected speedup is
2-3× when the draft model's acceptance rate is high.

Algorithm:
1. Draft model autoregressively generates K tokens: x_1, …, x_K
2. Target model scores the full sequence (prompt + K drafts) in parallel
3. Each draft token x_i is accepted with probability
       min(1, p_target(x_i | ctx) / p_draft(x_i | ctx))
4. If a token is rejected, sample a correction from the residual distribution
5. Repeat until EOS or max_new_tokens
"""
from __future__ import annotations
import math
import random
from typing import List, Optional, Callable


LogitsFn = Callable[[List[int]], List[float]]


class DraftModel:
    """Thin wrapper that makes any model act as a speculative draft model.

    Args:
        model         : vectorphile Module with a callable forward
        tokenizer     : must have encode/decode or be None (raw ids used)
        lookahead     : number of tokens to draft per step (default 4)
        temperature   : draft sampling temperature
    """

    def __init__(self, model, tokenizer=None, lookahead: int = 4,
                 temperature: float = 1.0):
        self.model       = model
        self.tokenizer   = tokenizer
        self.lookahead   = lookahead
        self.temperature = temperature

    def draft(self, input_ids: List[int]) -> tuple:
        """Generate `lookahead` draft tokens greedily.

        Returns:
            (draft_ids, draft_probs)
            draft_ids  : list of K token ids
            draft_probs: list of K probability values (for rejection sampling)
        """
        ids   = list(input_ids)
        draft_ids   = []
        draft_probs = []
        for _ in range(self.lookahead):
            logits = _model_logits(self.model, ids)
            probs  = _softmax(logits, self.temperature)
            tok    = _argmax(probs)
            draft_ids.append(tok)
            draft_probs.append(probs[tok])
            ids.append(tok)
        return draft_ids, draft_probs


def speculative_decode(
    target_model,
    draft_model: DraftModel,
    input_ids: List[int],
    max_new_tokens: int = 100,
    eos_token_id: Optional[int] = None,
    temperature: float = 1.0,
    verbose: bool = False,
) -> List[int]:
    """Generate tokens using speculative decoding.

    Args:
        target_model   : large (slow) verifier model
        draft_model    : DraftModel wrapping a small fast model
        input_ids      : prompt token ids
        max_new_tokens : maximum tokens to generate
        eos_token_id   : stop token
        temperature    : target sampling temperature
        verbose        : print acceptance rate stats

    Returns:
        Generated token sequence (prompt + new tokens).
    """
    generated    = list(input_ids)
    total_steps  = 0
    total_drafts = 0
    total_accept = 0

    while len(generated) - len(input_ids) < max_new_tokens:
        # 1. Draft K tokens
        K = draft_model.lookahead
        draft_ids, draft_probs = draft_model.draft(generated)
        total_drafts += K

        # 2. Run target model on extended context (one forward pass)
        # We need K+1 sets of logits: positions len(generated)-1 .. len(generated)+K-1
        ctx_with_drafts = generated + draft_ids
        # Get target logits for each draft position
        # (simplified: score each position independently — in practice, batch)
        accepted = []
        for i, (tok, q_prob) in enumerate(zip(draft_ids, draft_probs)):
            ctx_i     = ctx_with_drafts[:len(generated) + i]
            tgt_logits = _model_logits(target_model, ctx_i)
            tgt_probs  = _softmax(tgt_logits, temperature)
            p_target   = tgt_probs[tok] if tok < len(tgt_probs) else 1e-9
            accept_prob = min(1.0, p_target / max(q_prob, 1e-9))
            if random.random() < accept_prob:
                accepted.append(tok)
                total_accept += 1
            else:
                # Sample correction from residual distribution
                residual = [max(0.0, tgt_probs[j] - draft_probs[j] if j < len(draft_probs) else tgt_probs[j])
                            for j in range(len(tgt_probs))]
                s = sum(residual) or 1e-9
                residual = [v / s for v in residual]
                correction = _sample(residual)
                accepted.append(correction)
                break

        # 3. Append accepted tokens
        generated.extend(accepted)
        total_steps += 1

        # 4. If fewer than K accepted, one more target token
        if len(accepted) < K:
            tgt_logits = _model_logits(target_model, generated)
            tgt_probs  = _softmax(tgt_logits, temperature)
            bonus_tok  = _sample(tgt_probs)
            generated.append(bonus_tok)

        # EOS check
        if eos_token_id is not None and generated[-1] == eos_token_id:
            break

    if verbose and total_drafts > 0:
        rate = total_accept / total_drafts
        print(f'[speculative_decode] steps={total_steps}, '
              f'draft_tokens={total_drafts}, accepted={total_accept} '
              f'({rate*100:.1f}% acceptance)')

    return generated


# ── Helpers ───────────────────────────────────────────────────────────────────

def _model_logits(model, input_ids: List[int]) -> List[float]:
    """Run model forward and return last-position logits as a Python list."""
    import vectorphile as vp
    ids_t = vp.tensor([[input_ids]], dtype='int32')
    try:
        out = model(ids_t)
        # out shape: (1, T, vocab)  or  (1, vocab)
        data = out._data
        flat = data.flatten()
        n    = flat.shape[0]
        vocab = n // max(1, len(input_ids))
        # take last vocab logits
        return [float(flat[n - vocab + j]) for j in range(vocab)]
    except Exception:
        # fallback: uniform logits
        return [0.0] * 1000


def _softmax(logits: List[float], temperature: float = 1.0) -> List[float]:
    if temperature != 1.0 and temperature > 0:
        logits = [v / temperature for v in logits]
    max_v = max(logits)
    exps  = [math.exp(v - max_v) for v in logits]
    s     = sum(exps)
    return [e / s for e in exps]


def _argmax(probs: List[float]) -> int:
    best_i, best_v = 0, probs[0]
    for i, v in enumerate(probs):
        if v > best_v:
            best_v = v; best_i = i
    return best_i


def _sample(probs: List[float]) -> int:
    r = random.random()
    c = 0.0
    for i, p in enumerate(probs):
        c += p
        if r < c:
            return i
    return len(probs) - 1
