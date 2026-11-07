"""Load HuggingFace config.json and convert to vectorphile model kwargs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_hf_config(model_dir: Path) -> dict:
    """Read config.json from a downloaded model directory."""
    cfg_path = model_dir / 'config.json'
    if not cfg_path.exists():
        return {}
    with open(cfg_path, encoding='utf-8') as f:
        return json.load(f)


def hf_config_to_vectorphile(hf_cfg: dict, model_class: str) -> dict:
    """Translate a HuggingFace config dict to vectorphile constructor kwargs.

    Only the keys that differ between HF naming and vectorphile naming are
    translated; unknown keys are ignored rather than passed through.
    """
    mc = model_class.upper()

    if mc == 'GPT2':
        return dict(
            vocab_size    = hf_cfg.get('vocab_size', 50257),
            embed_dim     = hf_cfg.get('n_embd', hf_cfg.get('hidden_size', 768)),
            num_heads     = hf_cfg.get('n_head', hf_cfg.get('num_attention_heads', 12)),
            num_layers    = hf_cfg.get('n_layer', hf_cfg.get('num_hidden_layers', 12)),
            max_seq_len   = hf_cfg.get('n_positions', hf_cfg.get('max_position_embeddings', 1024)),
            dropout       = hf_cfg.get('resid_pdrop', 0.1),
        )

    if mc == 'BERT':
        return dict(
            vocab_size    = hf_cfg.get('vocab_size', 30522),
            embed_dim     = hf_cfg.get('hidden_size', 768),
            num_heads     = hf_cfg.get('num_attention_heads', 12),
            num_layers    = hf_cfg.get('num_hidden_layers', 12),
            max_seq_len   = hf_cfg.get('max_position_embeddings', 512),
            dropout       = hf_cfg.get('hidden_dropout_prob', 0.1),
        )

    if mc == 'T5':
        return dict(
            vocab_size    = hf_cfg.get('vocab_size', 32128),
            embed_dim     = hf_cfg.get('d_model', 512),
            num_heads     = hf_cfg.get('num_heads', 8),
            num_layers    = hf_cfg.get('num_layers', 6),
            d_ff          = hf_cfg.get('d_ff', 2048),
            dropout       = hf_cfg.get('dropout_rate', 0.1),
        )

    if mc == 'LLAMA':
        return dict(
            vocab_size    = hf_cfg.get('vocab_size', 32000),
            embed_dim     = hf_cfg.get('hidden_size', 4096),
            num_q_heads   = hf_cfg.get('num_attention_heads', 32),
            num_kv_heads  = hf_cfg.get('num_key_value_heads',
                                        hf_cfg.get('num_attention_heads', 32)),
            num_layers    = hf_cfg.get('num_hidden_layers', 32),
            ffn_dim       = hf_cfg.get('intermediate_size', 11008),
            rope_base     = hf_cfg.get('rope_theta', 10000),
        )

    if mc == 'MISTRAL':
        return dict(
            vocab_size    = hf_cfg.get('vocab_size', 32000),
            embed_dim     = hf_cfg.get('hidden_size', 4096),
            num_q_heads   = hf_cfg.get('num_attention_heads', 32),
            num_kv_heads  = hf_cfg.get('num_key_value_heads', 8),
            num_layers    = hf_cfg.get('num_hidden_layers', 32),
            ffn_dim       = hf_cfg.get('intermediate_size', 14336),
            window_size   = hf_cfg.get('sliding_window', 4096),
        )

    if mc == 'VIT':
        return dict(
            image_size    = hf_cfg.get('image_size', 224),
            patch_size    = hf_cfg.get('patch_size', 16),
            in_channels   = hf_cfg.get('num_channels', 3),
            embed_dim     = hf_cfg.get('hidden_size', 768),
            num_heads     = hf_cfg.get('num_attention_heads', 12),
            num_layers    = hf_cfg.get('num_hidden_layers', 12),
            mlp_dim       = hf_cfg.get('intermediate_size', 3072),
            num_classes   = hf_cfg.get('num_labels', 1000),
            dropout       = hf_cfg.get('hidden_dropout_prob', 0.0),
        )

    # Generic fallback
    return {}
