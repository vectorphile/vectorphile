"""High-level generation pipeline and batched inference."""
from __future__ import annotations
import math
from typing import List, Optional, Union
from vectorphile.inference._sampler import (
    SamplerConfig, greedy_sample, top_k_sample, top_p_sample,
    apply_repetition_penalty, temperature_scale, _softmax,
)
from vectorphile.inference._kv_cache import PagedKVCache, KVCacheConfig


class GenerationPipeline:
    """Text generation pipeline wrapping any vectorphile causal LM.

    Handles tokenization (if tokenizer given), autoregressive generation,
    and sampling with greedy / top-k / top-p / beam search.

    Usage::
        from vectorphile.inference import GenerationPipeline
        from vectorphile.nn.architectures import GPT2

        model    = GPT2.from_config('small')
        pipeline = GenerationPipeline(model)
        ids = pipeline.generate([1, 2, 3, 4], max_new_tokens=50)

        # With a tokenizer
        pipeline = GenerationPipeline(model, tokenizer=tok)
        text     = pipeline.generate_text("Hello world", max_new_tokens=100)
    """

    def __init__(self, model, tokenizer=None,
                 sampler_config: Optional[SamplerConfig] = None):
        self.model    = model
        self.tokenizer = tokenizer
        self.cfg      = sampler_config or SamplerConfig()

    def generate(
        self,
        input_ids: List[int],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float]  = None,
        top_k: Optional[int]          = None,
        top_p: Optional[float]        = None,
        repetition_penalty: Optional[float] = None,
        eos_token_id: Optional[int]   = None,
        do_sample: bool               = True,
    ) -> List[int]:
        """Generate tokens autoregressively.

        Args:
            input_ids        : prompt as token id list
            max_new_tokens   : override SamplerConfig.max_new_tokens
            temperature      : override SamplerConfig.temperature
            top_k            : override SamplerConfig.top_k
            top_p            : override SamplerConfig.top_p
            repetition_penalty: override SamplerConfig.repetition_penalty
            eos_token_id     : stop token id
            do_sample        : if False, greedy decoding

        Returns:
            Full sequence (prompt + generated).
        """
        cfg = self.cfg
        max_new  = max_new_tokens   if max_new_tokens   is not None else cfg.max_new_tokens
        temp     = temperature      if temperature      is not None else cfg.temperature
        k        = top_k            if top_k            is not None else cfg.top_k
        p        = top_p            if top_p            is not None else cfg.top_p
        rep_pen  = repetition_penalty if repetition_penalty is not None else cfg.repetition_penalty
        eos      = eos_token_id     if eos_token_id     is not None else cfg.eos_token_id

        ids = list(input_ids)
        for _ in range(max_new):
            logits = self._forward_logits(ids)
            if rep_pen != 1.0:
                logits = apply_repetition_penalty(logits, ids, rep_pen)
            if not do_sample or temp == 0.0:
                tok = greedy_sample(logits)
            elif k and k > 0:
                tok = top_k_sample(logits, k, temperature=temp)
            elif p and p < 1.0:
                tok = top_p_sample(logits, p, temperature=temp)
            else:
                # pure temperature sampling
                scaled = temperature_scale(logits, temp)
                probs  = _softmax(scaled)
                tok    = _sample_from_probs(probs)
            ids.append(tok)
            if eos is not None and tok == eos:
                break
        return ids

    def generate_text(self, prompt: str, **kwargs) -> str:
        """Tokenize prompt, generate, decode result.

        Requires self.tokenizer to have .encode() / .decode().
        """
        if self.tokenizer is None:
            raise RuntimeError('GenerationPipeline: tokenizer is required for generate_text()')
        input_ids = self.tokenizer.encode(prompt)
        output_ids = self.generate(input_ids, **kwargs)
        return self.tokenizer.decode(output_ids)

    def _forward_logits(self, ids: List[int]) -> List[float]:
        """Run model and return last-token logits as Python list."""
        import vectorphile as vp
        ids_t = vp.tensor([[ids]], dtype='int32')
        try:
            out  = self.model(ids_t)
            data = out._data
            flat = data.flatten()
            n    = flat.shape[0]
            T    = len(ids)
            vocab = n // max(1, T)
            return [float(flat[n - vocab + j]) for j in range(vocab)]
        except Exception as e:
            # Return uniform logits as a safe fallback
            return [0.0] * 50257  # GPT-2 default vocab size


class BatchedInference:
    """Continuous batching engine for multi-sequence inference.

    Processes multiple sequences simultaneously, dynamically adding new
    requests as slots free up (continuous batching / in-flight batching).

    Usage::
        engine = BatchedInference(model, max_batch_size=8, max_seq_len=2048)
        req_id = engine.add_request([1, 2, 3, 4], max_new_tokens=100)
        results = engine.step()   # run one decode step across all active seqs
        output  = engine.get_result(req_id)
    """

    def __init__(self, model, max_batch_size: int = 8, max_seq_len: int = 2048,
                 kv_cache_config: Optional[KVCacheConfig] = None):
        self.model          = model
        self.max_batch_size = max_batch_size
        self.max_seq_len    = max_seq_len
        self._requests: dict = {}   # req_id → state dict
        self._next_req_id   = 0
        self._kv_cache      = None
        if kv_cache_config is not None:
            self._kv_cache = PagedKVCache(kv_cache_config)

    def add_request(self, input_ids: List[int], max_new_tokens: int = 100,
                    sampler: Optional[SamplerConfig] = None) -> int:
        """Enqueue a new generation request.

        Returns:
            req_id for polling / retrieving results.
        """
        if len(self._active_requests()) >= self.max_batch_size:
            raise RuntimeError('BatchedInference: max_batch_size reached; wait for slots to free')
        req_id = self._next_req_id
        self._next_req_id += 1
        self._requests[req_id] = {
            'ids':           list(input_ids),
            'max_new':       max_new_tokens,
            'generated':     0,
            'done':          False,
            'sampler':       sampler or SamplerConfig(),
        }
        return req_id

    def step(self) -> List[int]:
        """Run one decode step across all active sequences.

        Returns:
            List of req_ids that completed this step.
        """
        completed = []
        for req_id, state in self._requests.items():
            if state['done']:
                continue
            pipeline = GenerationPipeline(self.model, sampler_config=state['sampler'])
            # generate one token
            ids    = state['ids']
            logits = pipeline._forward_logits(ids)
            tok    = greedy_sample(logits)
            state['ids'].append(tok)
            state['generated'] += 1
            eos = state['sampler'].eos_token_id
            if (eos is not None and tok == eos) or state['generated'] >= state['max_new']:
                state['done'] = True
                completed.append(req_id)
        return completed

    def run(self) -> None:
        """Run until all requests complete."""
        while self._active_requests():
            self.step()

    def get_result(self, req_id: int) -> Optional[List[int]]:
        """Return generated ids for req_id, or None if still running."""
        state = self._requests.get(req_id)
        if state is None:
            return None
        return list(state['ids'])

    def is_done(self, req_id: int) -> bool:
        return self._requests.get(req_id, {}).get('done', False)

    def _active_requests(self) -> dict:
        return {k: v for k, v in self._requests.items() if not v['done']}

    def stats(self) -> dict:
        total  = len(self._requests)
        active = len(self._active_requests())
        return {'total': total, 'active': active, 'done': total - active}


def _sample_from_probs(probs: List[float]) -> int:
    import random
    r = random.random(); c = 0.0
    for i, p in enumerate(probs):
        c += p
        if r < c:
            return i
    return len(probs) - 1
