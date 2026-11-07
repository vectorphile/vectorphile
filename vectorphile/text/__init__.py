"""vectorphile.text — Text generation utilities (beam search, sampling, pipelines)."""
from __future__ import annotations
import math
import heapq
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


# ── Sampling helpers ──────────────────────────────────────────────────────────

def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return logits
    threshold = np.sort(logits)[-k]
    logits = np.where(logits < threshold, -np.inf, logits)
    return logits


def _top_p_filter(logits: np.ndarray, p: float) -> np.ndarray:
    if p >= 1.0:
        return logits
    sorted_idx = np.argsort(logits)[::-1]
    sorted_logits = logits[sorted_idx]
    cum_probs = np.cumsum(_softmax(sorted_logits))
    cutoff = np.searchsorted(cum_probs, p)
    remove = sorted_idx[cutoff + 1:]
    logits = logits.copy()
    logits[remove] = -np.inf
    return logits


# ── Greedy / temperature sampling ────────────────────────────────────────────

def greedy_search(
    logits_fn: Callable[[np.ndarray], np.ndarray],
    input_ids: np.ndarray,
    max_new_tokens: int = 50,
    eos_token_id: Optional[int] = None,
) -> np.ndarray:
    """Greedy decoding: always pick the highest-probability token."""
    ids = list(input_ids.tolist() if hasattr(input_ids, 'tolist') else input_ids)
    for _ in range(max_new_tokens):
        logits = logits_fn(np.array(ids))
        next_token = int(np.argmax(logits[-1]))
        ids.append(next_token)
        if eos_token_id is not None and next_token == eos_token_id:
            break
    return np.array(ids)


def temperature_sampling(
    logits_fn: Callable,
    input_ids: np.ndarray,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    eos_token_id: Optional[int] = None,
) -> np.ndarray:
    """Stochastic sampling with temperature, top-k, and top-p (nucleus) filtering."""
    ids = list(input_ids.tolist() if hasattr(input_ids, 'tolist') else input_ids)
    for _ in range(max_new_tokens):
        logits = logits_fn(np.array(ids))
        logits = logits[-1].astype(np.float64)
        logits = logits / max(temperature, 1e-9)
        logits = _top_k_filter(logits, top_k)
        logits = _top_p_filter(logits, top_p)
        probs = _softmax(logits)
        next_token = int(np.random.choice(len(probs), p=probs))
        ids.append(next_token)
        if eos_token_id is not None and next_token == eos_token_id:
            break
    return np.array(ids)


# ── Nucleus / top-p sampling (alias) ─────────────────────────────────────────

def nucleus_sampling(
    logits_fn: Callable,
    input_ids: np.ndarray,
    max_new_tokens: int = 50,
    p: float = 0.9,
    temperature: float = 1.0,
    eos_token_id: Optional[int] = None,
) -> np.ndarray:
    """Nucleus (top-p) sampling — Holtzman et al. 2020."""
    return temperature_sampling(
        logits_fn, input_ids, max_new_tokens,
        temperature=temperature, top_p=p, eos_token_id=eos_token_id,
    )


# ── Beam search ──────────────────────────────────────────────────────────────

def beam_search(
    logits_fn: Callable,
    input_ids: np.ndarray,
    max_new_tokens: int = 50,
    num_beams: int = 4,
    no_repeat_ngram_size: int = 0,
    eos_token_id: Optional[int] = None,
    length_penalty: float = 1.0,
) -> np.ndarray:
    """Beam search with optional n-gram blocking and length penalty.

    Args:
        logits_fn: callable(token_ids) → logits array [seq_len, vocab]
        input_ids: 1-D starting token ids
        num_beams: beam width
        no_repeat_ngram_size: block n-gram repeats (0 = disabled)
        length_penalty: >1 favors longer sequences, <1 shorter
    """
    input_list = list(input_ids.tolist() if hasattr(input_ids, 'tolist') else input_ids)

    # Each beam: (log_prob, token_list)
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
            logits = logits_fn(np.array(ids))
            log_probs = logits[-1] - np.log(np.sum(np.exp(logits[-1] - logits[-1].max()))) - logits[-1].max()
            # Simple log_softmax
            lp = logits[-1].astype(np.float64)
            lp = lp - lp.max()
            lp = lp - np.log(np.sum(np.exp(lp)))

            # n-gram blocking
            if no_repeat_ngram_size > 0 and len(ids) >= no_repeat_ngram_size:
                banned = _get_banned_ngram_tokens(ids, no_repeat_ngram_size)
                lp[list(banned)] = -np.inf

            # top num_beams tokens
            top_k = min(num_beams * 2, len(lp))
            top_tokens = np.argpartition(lp, -top_k)[-top_k:]
            for tok in top_tokens:
                new_score = score + float(lp[tok])
                candidates.append((new_score, ids + [int(tok)]))

        if not candidates:
            break
        # Keep top num_beams beams
        candidates.sort(key=lambda x: x[0] / (len(x[1]) ** length_penalty), reverse=True)
        beams = candidates[:num_beams]

    # Return best completed or best current beam
    all_beams = completed + beams
    all_beams.sort(key=lambda x: x[0] / (len(x[1]) ** length_penalty), reverse=True)
    return np.array(all_beams[0][1])


def _get_banned_ngram_tokens(ids: List[int], n: int) -> set:
    """Return tokens that would create a repeated n-gram."""
    if len(ids) < n:
        return set()
    ngram_prefix = tuple(ids[-(n-1):])
    banned = set()
    for i in range(len(ids) - n):
        if tuple(ids[i:i+n-1]) == ngram_prefix:
            banned.add(ids[i+n-1])
    return banned


# ── Contrastive search ────────────────────────────────────────────────────────

def contrastive_search(
    logits_fn: Callable,
    hidden_fn: Callable,
    input_ids: np.ndarray,
    max_new_tokens: int = 50,
    k: int = 5,
    alpha: float = 0.6,
    eos_token_id: Optional[int] = None,
) -> np.ndarray:
    """Contrastive search — Su et al. 2022.

    Balances model confidence with degeneration penalty:
      score(v) = (1-alpha) * p(v|ctx) - alpha * max_j cos_sim(h_v, h_j)

    Args:
        logits_fn: callable(ids) → logits [seq, vocab]
        hidden_fn: callable(ids) → hidden states [seq, dim]
        k: candidate pool size
        alpha: degeneration penalty weight (0=greedy, 1=pure diversity)
    """
    ids = list(input_ids.tolist() if hasattr(input_ids, 'tolist') else input_ids)
    past_hiddens = []

    for _ in range(max_new_tokens):
        logits = logits_fn(np.array(ids))
        probs = _softmax(logits[-1].astype(np.float64))
        top_k_tokens = np.argpartition(probs, -k)[-k:]

        if past_hiddens:
            # Degeneration penalty
            best_score = -np.inf
            best_tok = top_k_tokens[0]
            for tok in top_k_tokens:
                # Get hidden for this candidate
                cand_hidden = hidden_fn(np.array(ids + [int(tok)]))[-1]
                # Max cosine similarity to past hiddens
                sims = []
                for ph in past_hiddens:
                    cos = np.dot(cand_hidden, ph) / (
                        np.linalg.norm(cand_hidden) * np.linalg.norm(ph) + 1e-9
                    )
                    sims.append(cos)
                degen = max(sims)
                score = (1 - alpha) * float(probs[tok]) - alpha * degen
                if score > best_score:
                    best_score = score
                    best_tok = int(tok)
        else:
            best_tok = int(top_k_tokens[np.argmax(probs[top_k_tokens])])

        # Store hidden
        h = hidden_fn(np.array(ids))[-1]
        past_hiddens.append(h)

        ids.append(best_tok)
        if eos_token_id is not None and best_tok == eos_token_id:
            break

    return np.array(ids)


# ── Speculative decoding ──────────────────────────────────────────────────────

def speculative_decoding(
    target_logits_fn: Callable,
    draft_logits_fn: Callable,
    input_ids: np.ndarray,
    max_new_tokens: int = 50,
    draft_steps: int = 4,
    temperature: float = 1.0,
    eos_token_id: Optional[int] = None,
) -> np.ndarray:
    """Speculative decoding — Leviathan et al. 2023.

    Uses a small draft model to propose tokens, then verifies with the
    large target model in a single forward pass.
    """
    ids = list(input_ids.tolist() if hasattr(input_ids, 'tolist') else input_ids)

    for _ in range(max_new_tokens // draft_steps + 1):
        if len(ids) >= len(input_ids) + max_new_tokens:
            break

        # Draft phase: sample `draft_steps` tokens
        draft_ids = ids.copy()
        draft_tokens = []
        draft_probs = []
        for _ in range(draft_steps):
            d_logits = draft_logits_fn(np.array(draft_ids))[-1].astype(np.float64)
            d_logits /= max(temperature, 1e-9)
            d_probs = _softmax(d_logits)
            tok = int(np.random.choice(len(d_probs), p=d_probs))
            draft_tokens.append(tok)
            draft_probs.append(d_probs[tok])
            draft_ids.append(tok)

        # Verification: run target on full draft sequence at once
        verify_ids = ids + draft_tokens
        t_logits = target_logits_fn(np.array(verify_ids))
        # t_logits shape: (len(verify_ids), vocab)
        accepted = []
        for i, (tok, dp) in enumerate(zip(draft_tokens, draft_probs)):
            t_logit = t_logits[len(ids) + i - 1].astype(np.float64)
            t_logit /= max(temperature, 1e-9)
            t_probs = _softmax(t_logit)
            tp = t_probs[tok]
            # Accept with probability min(1, tp/dp)
            if np.random.random() < min(1.0, float(tp) / max(float(dp), 1e-9)):
                accepted.append(tok)
            else:
                # Reject — sample from corrected distribution
                corr = np.maximum(t_probs - _softmax(
                    draft_logits_fn(np.array(ids + accepted))[-1].astype(np.float64) / max(temperature, 1e-9)
                ), 0)
                s = corr.sum()
                if s > 0:
                    corr /= s
                    accepted.append(int(np.random.choice(len(corr), p=corr)))
                break

        ids.extend(accepted)
        if not accepted:
            # Sample one token from target
            t_logit = t_logits[len(ids) - 1].astype(np.float64)
            t_logit /= max(temperature, 1e-9)
            t_probs = _softmax(t_logit)
            ids.append(int(np.random.choice(len(t_probs), p=t_probs)))

        if eos_token_id is not None and ids[-1] == eos_token_id:
            break

    return np.array(ids[:len(input_ids) + max_new_tokens])


# ── Chat template ─────────────────────────────────────────────────────────────

CHAT_TEMPLATES = {
    'chatml': (
        '<|im_start|>system\n{system}<|im_end|>\n'
        '{turns}'
        '<|im_start|>assistant\n'
    ),
    'llama2': (
        '[INST] <<SYS>>\n{system}\n<</SYS>>\n\n'
        '{turns}[/INST]'
    ),
    'llama3': (
        '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n'
        '{system}<|eot_id|>'
        '{turns}'
        '<|start_header_id|>assistant<|end_header_id|>\n\n'
    ),
    'mistral': '[INST] {turns} [/INST]',
    'alpaca': 'Below is an instruction.\n\n### Instruction:\n{turns}\n\n### Response:\n',
    'vicuna': 'USER: {turns}\nASSISTANT:',
}


def apply_chat_template(
    messages: List[Dict[str, str]],
    template: str = 'chatml',
    add_generation_prompt: bool = True,
) -> str:
    """Format a list of messages using a chat template.

    Args:
        messages: list of {'role': 'system'|'user'|'assistant', 'content': str}
        template: template name ('chatml', 'llama2', 'llama3', 'mistral', 'alpaca')

    Returns:
        Formatted string ready for tokenization.
    """
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
                turns_parts.append(
                    f'<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>'
                )
            else:
                turns_parts.append(content)
        elif role == 'assistant':
            if template == 'chatml':
                turns_parts.append(f'<|im_start|>assistant\n{content}<|im_end|>\n')
            elif template == 'llama3':
                turns_parts.append(
                    f'<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>'
                )
            else:
                turns_parts.append(content + ' ')

    turns = ''.join(turns_parts)
    tmpl = CHAT_TEMPLATES.get(template, CHAT_TEMPLATES['chatml'])
    result = tmpl.format(system=system, turns=turns)
    return result


# ── TextGenerationPipeline ────────────────────────────────────────────────────

class TextGenerationPipeline:
    """High-level text generation pipeline.

    Example::

        pipe = TextGenerationPipeline(model, tokenizer)
        output = pipe("Once upon a time", max_new_tokens=100, temperature=0.8)
        print(output[0]['generated_text'])
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = 'cpu',
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def _logits_fn(self, ids: np.ndarray) -> np.ndarray:
        import numpy as np
        output = self.model(ids[np.newaxis, :])
        if hasattr(output, 'data'):
            output = output.data
        if hasattr(output, '__class__') and hasattr(output, 'logits'):
            return output.logits[0]
        # Assume model returns logits directly
        if isinstance(output, np.ndarray):
            return output[0] if output.ndim == 3 else output
        return np.asarray(output)[0]

    def __call__(
        self,
        text: Union[str, List[str]],
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        num_beams: int = 1,
        do_sample: bool = False,
        no_repeat_ngram_size: int = 0,
        eos_token_id: Optional[int] = None,
        **kwargs,
    ) -> List[Dict[str, str]]:
        single = isinstance(text, str)
        texts = [text] if single else text
        results = []

        for t in texts:
            ids = self.tokenizer.encode(t)
            if not isinstance(ids, np.ndarray):
                ids = np.array(ids)

            if eos_token_id is None and hasattr(self.tokenizer, 'eos_token_id'):
                eos_token_id = self.tokenizer.eos_token_id

            if num_beams > 1:
                out_ids = beam_search(
                    self._logits_fn, ids, max_new_tokens,
                    num_beams=num_beams,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    eos_token_id=eos_token_id,
                )
            elif do_sample:
                out_ids = temperature_sampling(
                    self._logits_fn, ids, max_new_tokens,
                    temperature=temperature, top_k=top_k, top_p=top_p,
                    eos_token_id=eos_token_id,
                )
            else:
                out_ids = greedy_search(
                    self._logits_fn, ids, max_new_tokens,
                    eos_token_id=eos_token_id,
                )

            generated = self.tokenizer.decode(out_ids[len(ids):].tolist())
            results.append({
                'generated_text': t + generated,
                'input_text': t,
                'generated_tokens': out_ids[len(ids):].tolist(),
            })

        return results[0] if single else results


__all__ = [
    'greedy_search',
    'temperature_sampling',
    'nucleus_sampling',
    'beam_search',
    'contrastive_search',
    'speculative_decoding',
    'apply_chat_template',
    'TextGenerationPipeline',
    'CHAT_TEMPLATES',
]
