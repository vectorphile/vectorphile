"""vectorphile.inference — High-performance inference engine.

Provides:
  - Paged KV cache (efficient memory management for long contexts)
  - Speculative decoding (draft + verify for 2-3× throughput)
  - Batched inference (continuous batching, dynamic batch sizes)
  - Greedy / top-k / top-p / beam search samplers
  - Text generation pipeline

Usage::
    from vectorphile.inference import GenerationPipeline, PagedKVCache, speculative_decode

    pipe = GenerationPipeline(model, tokenizer)
    output = pipe.generate("Hello, world", max_new_tokens=100, temperature=0.8)
"""

from vectorphile.inference._kv_cache import PagedKVCache, KVCacheBlock, KVCacheConfig
from vectorphile.inference._sampler import (
    greedy_sample, top_k_sample, top_p_sample, temperature_scale,
    beam_search, SamplerConfig,
)
from vectorphile.inference._speculative import speculative_decode, DraftModel
from vectorphile.inference._pipeline import GenerationPipeline, BatchedInference

__all__ = [
    # KV cache
    'PagedKVCache', 'KVCacheBlock', 'KVCacheConfig',
    # Samplers
    'greedy_sample', 'top_k_sample', 'top_p_sample', 'temperature_scale',
    'beam_search', 'SamplerConfig',
    # Speculative decoding
    'speculative_decode', 'DraftModel',
    # Pipelines
    'GenerationPipeline', 'BatchedInference',
]
