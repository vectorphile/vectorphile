"""vectorphile.text — Text generation utilities — pure Python."""
from __future__ import annotations
import math
import random as _rand
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


# ── Sampling helpers ──────────────────────────────────────────────────────────

def _to_logits(x) -> List[float]:
    if hasattr(x, 'to_flat_list'):
        return [float(v) for v in x.to_flat_list()]
    if hasattr(x, 'tolist'):
        v = x.tolist()
        if isinstance(v, list) and v and isinstance(v[0], list):
            return [float(item) for item in v[-1]]
        return [float(item) for item in v]
    if isinstance(x, list):
        if x and isinstance(x[0], list):
            return [float(item) for item in x[-1]]
        return [float(item) for item in x]
    return [float(x)]


def _softmax(x: List[float]) -> List[float]:
    mx = max(x)
    exps = [math.exp(v - mx) for v in x]
    s = sum(exps)
    return [e/s for e in exps]


def _log_softmax(x: List[float]) -> List[float]:
    mx = max(x)
    log_sum = math.log(sum(math.exp(v - mx) for v in x)) + mx
    return [v - log_sum for v in x]


def _sample(probs: List[float]) -> int:
    r = _rand.random()
    cumsum = 0.0
    for i, p in enumerate(probs):
        cumsum += p
        if r <= cumsum:
            return i
    return len(probs) - 1


def _top_k_filter(logits: List[float], k: int) -> List[float]:
    if k <= 0:
        return logits
    sorted_vals = sorted(logits, reverse=True)
    threshold = sorted_vals[k-1]
    return [v if v >= threshold else -math.inf for v in logits]


def _top_p_filter(logits: List[float], p: float) -> List[float]:
    if p >= 1.0:
        return logits
    sorted_idx = sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)
    probs = _softmax([logits[i] for i in sorted_idx])
    cum = 0.0; cutoff = len(sorted_idx)
    for i, prob in enumerate(probs):
        cum += prob
        if cum >= p:
            cutoff = i + 1
            break
    keep = set(sorted_idx[:cutoff])
    return [v if i in keep else -math.inf for i, v in enumerate(logits)]


def _argmax(lst: List[float]) -> int:
    return max(range(len(lst)), key=lambda i: lst[i])


# ── Greedy / temperature sampling ────────────────────────────────────────────

def greedy_search(logits_fn: Callable, input_ids, max_new_tokens: int = 50,
                  eos_token_id: Optional[int] = None) -> List[int]:
    if hasattr(input_ids, 'tolist'):
        ids = list(input_ids.tolist())
    else:
        ids = list(input_ids)
    for _ in range(max_new_tokens):
        logits = _to_logits(logits_fn(ids))
        next_token = _argmax(logits)
        ids.append(next_token)
        if eos_token_id is not None and next_token == eos_token_id:
            break
    return ids


def temperature_sampling(logits_fn: Callable, input_ids, max_new_tokens: int = 50,
                         temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0,
                         eos_token_id: Optional[int] = None) -> List[int]:
    if hasattr(input_ids, 'tolist'):
        ids = list(input_ids.tolist())
    else:
        ids = list(input_ids)
    for _ in range(max_new_tokens):
        logits = _to_logits(logits_fn(ids))
        logits = [v / max(temperature, 1e-9) for v in logits]
        logits = _top_k_filter(logits, top_k)
        logits = _top_p_filter(logits, top_p)
        probs = _softmax(logits)
        next_token = _sample(probs)
        ids.append(next_token)
        if eos_token_id is not None and next_token == eos_token_id:
            break
    return ids


def nucleus_sampling(logits_fn: Callable, input_ids, max_new_tokens: int = 50,
                     p: float = 0.9, temperature: float = 1.0,
                     eos_token_id: Optional[int] = None) -> List[int]:
    return temperature_sampling(logits_fn, input_ids, max_new_tokens,
                                temperature=temperature, top_p=p, eos_token_id=eos_token_id)


# ── Beam search ──────────────────────────────────────────────────────────────

def _get_banned_ngram_tokens(ids: List[int], n: int) -> set:
    if len(ids) < n:
        return set()
    prefix = tuple(ids[-(n-1):])
    banned = set()
    for i in range(len(ids) - n):
        if tuple(ids[i:i+n-1]) == prefix:
            banned.add(ids[i+n-1])
    return banned


def beam_search(logits_fn: Callable, input_ids, max_new_tokens: int = 50,
                num_beams: int = 4, no_repeat_ngram_size: int = 0,
                eos_token_id: Optional[int] = None,
                length_penalty: float = 1.0) -> List[int]:
    if hasattr(input_ids, 'tolist'):
        input_list = list(input_ids.tolist())
    else:
        input_list = list(input_ids)
    beams: List[Tuple[float, List[int]]] = [(0.0, input_list)]
    completed: List[Tuple[float, List[int]]] = []

    for _ in range(max_new_tokens):
        if not beams:
            break
        candidates: List[Tuple[float, List[int]]] = []
        for score, ids in beams:
            if eos_token_id is not None and ids[-1] == eos_token_id:
                completed.append((score, ids))
                continue
            logits = _to_logits(logits_fn(ids))
            lp = _log_softmax(logits)
            if no_repeat_ngram_size > 0 and len(ids) >= no_repeat_ngram_size:
                banned = _get_banned_ngram_tokens(ids, no_repeat_ngram_size)
                lp = [(-math.inf if i in banned else v) for i, v in enumerate(lp)]
            top_k = min(num_beams * 2, len(lp))
            top_tokens = sorted(range(len(lp)), key=lambda i: lp[i], reverse=True)[:top_k]
            for tok in top_tokens:
                new_score = score + lp[tok]
                candidates.append((new_score, ids + [int(tok)]))
        if not candidates:
            break
        candidates.sort(key=lambda x: x[0] / max(len(x[1]) ** length_penalty, 1e-9), reverse=True)
        beams = candidates[:num_beams]

    all_beams = completed + beams
    all_beams.sort(key=lambda x: x[0] / max(len(x[1]) ** length_penalty, 1e-9), reverse=True)
    return all_beams[0][1] if all_beams else input_list


# ── Contrastive search ────────────────────────────────────────────────────────

def _cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(y*y for y in b))
    return dot / (na * nb + 1e-9)

def contrastive_search(logits_fn: Callable, hidden_fn: Callable, input_ids,
                       max_new_tokens: int = 50, k: int = 5, alpha: float = 0.6,
                       eos_token_id: Optional[int] = None) -> List[int]:
    if hasattr(input_ids, 'tolist'):
        ids = list(input_ids.tolist())
    else:
        ids = list(input_ids)
    past_hiddens: List[List[float]] = []

    for _ in range(max_new_tokens):
        logits = _to_logits(logits_fn(ids))
        probs = _softmax(logits)
        top_k_tokens = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:k]

        if past_hiddens:
            best_score = -math.inf; best_tok = top_k_tokens[0]
            for tok in top_k_tokens:
                cand_h = _to_logits(hidden_fn(ids + [int(tok)]))
                degen = max(_cosine_sim(cand_h, ph) for ph in past_hiddens)
                score = (1 - alpha) * probs[tok] - alpha * degen
                if score > best_score:
                    best_score = score; best_tok = int(tok)
        else:
            best_tok = max(top_k_tokens, key=lambda i: probs[i])

        h = _to_logits(hidden_fn(ids))
        past_hiddens.append(h)
        ids.append(best_tok)
        if eos_token_id is not None and best_tok == eos_token_id:
            break
    return ids


# ── Speculative decoding ──────────────────────────────────────────────────────

def speculative_decoding(target_logits_fn: Callable, draft_logits_fn: Callable,
                         input_ids, max_new_tokens: int = 50, draft_steps: int = 4,
                         temperature: float = 1.0,
                         eos_token_id: Optional[int] = None) -> List[int]:
    if hasattr(input_ids, 'tolist'):
        ids = list(input_ids.tolist())
    else:
        ids = list(input_ids)
    start_len = len(ids)

    for _ in range(max_new_tokens // draft_steps + 1):
        if len(ids) >= start_len + max_new_tokens:
            break
        draft_ids = ids.copy(); draft_tokens = []; draft_probs_list = []
        for _ in range(draft_steps):
            d_logits = _to_logits(draft_logits_fn(draft_ids))
            d_logits = [v / max(temperature, 1e-9) for v in d_logits]
            d_probs = _softmax(d_logits)
            tok = _sample(d_probs)
            draft_tokens.append(tok); draft_probs_list.append(d_probs[tok]); draft_ids.append(tok)
        verify_ids = ids + draft_tokens
        t_logits_all = target_logits_fn(verify_ids)
        accepted = []
        for i, (tok, dp) in enumerate(zip(draft_tokens, draft_probs_list)):
            if hasattr(t_logits_all, '__getitem__'):
                t_logit = _to_logits(t_logits_all[len(ids) + i - 1])
            else:
                t_logit = _to_logits(t_logits_all)
            t_logit = [v / max(temperature, 1e-9) for v in t_logit]
            t_probs = _softmax(t_logit)
            tp = t_probs[tok] if tok < len(t_probs) else 0.0
            if _rand.random() < min(1.0, tp / max(dp, 1e-9)):
                accepted.append(tok)
            else:
                break
        ids.extend(accepted)
        if not accepted:
            t_logit = _to_logits(t_logits_all)
            t_probs = _softmax([v / max(temperature, 1e-9) for v in t_logit])
            ids.append(_sample(t_probs))
        if eos_token_id is not None and ids[-1] == eos_token_id:
            break
    return ids[:start_len + max_new_tokens]


# ── Chat template ─────────────────────────────────────────────────────────────

CHAT_TEMPLATES = {
    'chatml': (
        '<|im_start|>system\n{system}<|im_end|>\n'
        '{turns}'
        '<|im_start|>assistant\n'
    ),
    'llama2': '[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{turns}[/INST]',
    'llama3': (
        '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n'
        '{system}<|eot_id|>{turns}'
        '<|start_header_id|>assistant<|end_header_id|>\n\n'
    ),
    'mistral': '[INST] {turns} [/INST]',
    'alpaca': 'Below is an instruction.\n\n### Instruction:\n{turns}\n\n### Response:\n',
    'vicuna': 'USER: {turns}\nASSISTANT:',
}


def constrained_decode(logits_fn: Callable, input_ids, max_new_tokens: int = 128,
                       allowed_tokens_fn: Callable = None,
                       temperature: float = 1.0, eos_token_id: int = -1) -> List[int]:
    """Constrained (grammar-guided) token-by-token decoding.

    At each step, ``allowed_tokens_fn(current_ids)`` returns a list/set of
    token IDs that are currently valid under the grammar/constraint.  Logits
    for all other tokens are set to -inf before sampling.

    Usage::
        # Only allow tokens from a fixed vocabulary
        allowed_fn = lambda ids: list(range(vocab_size))   # unconstrained
        # Allow only JSON-valid continuations (custom grammar):
        allowed_fn = json_grammar.get_valid_tokens

        output = constrained_decode(model.forward, prompt_ids,
                                    allowed_tokens_fn=allowed_fn,
                                    max_new_tokens=100)
    """
    import math as _m
    ids = list(input_ids) if not isinstance(input_ids, list) else list(input_ids)
    start = len(ids)

    def _softmax_temp(logits, temp):
        scaled = [v / max(temp, 1e-9) for v in logits]
        mx = max(scaled); exps = [_m.exp(v-mx) for v in scaled]; s = sum(exps)
        return [e/s for e in exps]

    def _sample_probs(probs):
        import random as _r
        r = _r.random(); cum = 0.0
        for i, p in enumerate(probs):
            cum += p
            if r <= cum: return i
        return len(probs) - 1

    while len(ids) - start < max_new_tokens:
        raw_logits = logits_fn(ids)
        if hasattr(raw_logits, '_data'):
            lfl = raw_logits._data.to_flat_list() if hasattr(raw_logits._data, 'to_flat_list') else list(raw_logits._data)
        else:
            lfl = list(raw_logits)
        # Take last-position logits if 2D
        V = len(lfl)
        # Apply constraint mask
        if allowed_tokens_fn is not None:
            allowed = set(allowed_tokens_fn(ids))
            lfl = [lfl[i] if i in allowed else -1e9 for i in range(V)]
        probs = _softmax_temp(lfl, temperature)
        tok = _sample_probs(probs)
        ids.append(tok)
        if tok == eos_token_id:
            break
    return ids[start:]


def top_p_top_k_decode(logits_fn: Callable, input_ids, max_new_tokens: int = 128,
                       top_p: float = 0.9, top_k: int = 50,
                       temperature: float = 1.0, eos_token_id: int = -1,
                       repetition_penalty: float = 1.0) -> List[int]:
    """Top-p + Top-k sampling with repetition penalty.

    - top_k: keep only top-k highest probability tokens
    - top_p: keep smallest set of tokens whose cumulative probability >= top_p
    - repetition_penalty: divides logits by penalty for already-generated tokens
    """
    import math as _m
    ids = list(input_ids)
    start = len(ids)
    generated = []

    while len(generated) < max_new_tokens:
        raw = logits_fn(ids)
        if hasattr(raw, '_data'):
            lfl = raw._data.to_flat_list() if hasattr(raw._data, 'to_flat_list') else list(raw._data)
        else:
            lfl = list(raw)
        # Repetition penalty
        if repetition_penalty != 1.0:
            seen = set(ids[-64:])  # look at last 64 tokens
            lfl = [lfl[i] / repetition_penalty if i in seen else lfl[i]
                   for i in range(len(lfl))]
        # Temperature
        scaled = [v / max(temperature, 1e-9) for v in lfl]
        # Top-K filter
        if top_k > 0:
            sorted_indices = sorted(range(len(scaled)), key=lambda i: scaled[i], reverse=True)
            top_k_set = set(sorted_indices[:top_k])
            scaled = [scaled[i] if i in top_k_set else -1e9 for i in range(len(scaled))]
        # Softmax
        mx = max(scaled); exps = [_m.exp(v-mx) for v in scaled]; s = sum(exps)
        probs = [e/s for e in exps]
        # Top-P nucleus filter
        if top_p < 1.0:
            sorted_by_prob = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
            cum = 0.0; keep = set()
            for idx, p in sorted_by_prob:
                cum += p; keep.add(idx)
                if cum >= top_p: break
            probs = [probs[i] if i in keep else 0.0 for i in range(len(probs))]
            s2 = sum(probs); probs = [p/s2 for p in probs] if s2 > 0 else probs
        # Sample
        import random as _r
        r = _r.random(); cum = 0.0; tok = 0
        for i, p in enumerate(probs):
            cum += p
            if r <= cum: tok = i; break
        generated.append(tok); ids.append(tok)
        if tok == eos_token_id: break
    return generated


def apply_chat_template(messages: List[Dict[str, str]], template: str = 'chatml',
                        add_generation_prompt: bool = True) -> str:
    system = ''
    turns_parts = []
    for msg in messages:
        role, content = msg['role'], msg['content']
        if role == 'system':
            system = content
        elif role == 'user':
            if template == 'chatml':
                turns_parts.append(f'<|im_start|>user\n{content}<|im_end|>\n')
            elif template in ('llama2', 'mistral'):
                turns_parts.append(content)
            elif template == 'llama3':
                turns_parts.append(f'<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>')
            else:
                turns_parts.append(content)
        elif role == 'assistant':
            if template == 'chatml':
                turns_parts.append(f'<|im_start|>assistant\n{content}<|im_end|>\n')
            elif template == 'llama3':
                turns_parts.append(f'<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>')
            else:
                turns_parts.append(content)
    turns = ''.join(turns_parts)
    tmpl = CHAT_TEMPLATES.get(template, '{turns}')
    return tmpl.format(system=system, turns=turns)
