"""Model registry — maps short names to HuggingFace repo IDs, configs, and mappers."""

REGISTRY = {
    # ── GPT-2 ──────────────────────────────────────────────────────────────────
    'gpt2': {
        'hf_id': 'openai-community/gpt2',
        'model_class': 'GPT2',
        'config': {
            'vocab_size': 50257, 'embed_dim': 768, 'num_heads': 12,
            'num_layers': 12, 'max_seq_len': 1024,
        },
        'mapper': 'map_gpt2_weights',
        'description': 'GPT-2 small (117M params)',
        'size_mb': 548,
        'requires_auth': False,
    },
    'gpt2-medium': {
        'hf_id': 'openai-community/gpt2-medium',
        'model_class': 'GPT2',
        'config': {
            'vocab_size': 50257, 'embed_dim': 1024, 'num_heads': 16,
            'num_layers': 24, 'max_seq_len': 1024,
        },
        'mapper': 'map_gpt2_weights',
        'description': 'GPT-2 medium (345M params)',
        'size_mb': 1420,
        'requires_auth': False,
    },
    'gpt2-large': {
        'hf_id': 'openai-community/gpt2-large',
        'model_class': 'GPT2',
        'config': {
            'vocab_size': 50257, 'embed_dim': 1280, 'num_heads': 20,
            'num_layers': 36, 'max_seq_len': 1024,
        },
        'mapper': 'map_gpt2_weights',
        'description': 'GPT-2 large (774M params)',
        'size_mb': 3100,
        'requires_auth': False,
    },
    'gpt2-xl': {
        'hf_id': 'openai-community/gpt2-xl',
        'model_class': 'GPT2',
        'config': {
            'vocab_size': 50257, 'embed_dim': 1600, 'num_heads': 25,
            'num_layers': 48, 'max_seq_len': 1024,
        },
        'mapper': 'map_gpt2_weights',
        'description': 'GPT-2 XL (1.5B params)',
        'size_mb': 6200,
        'requires_auth': False,
    },

    # ── BERT ───────────────────────────────────────────────────────────────────
    'bert-base-uncased': {
        'hf_id': 'google-bert/bert-base-uncased',
        'model_class': 'BERT',
        'config': {
            'vocab_size': 30522, 'embed_dim': 768, 'num_heads': 12,
            'num_layers': 12, 'max_seq_len': 512,
        },
        'mapper': 'map_bert_weights',
        'description': 'BERT base uncased (110M params)',
        'size_mb': 440,
        'requires_auth': False,
    },
    'bert-base-cased': {
        'hf_id': 'google-bert/bert-base-cased',
        'model_class': 'BERT',
        'config': {
            'vocab_size': 28996, 'embed_dim': 768, 'num_heads': 12,
            'num_layers': 12, 'max_seq_len': 512,
        },
        'mapper': 'map_bert_weights',
        'description': 'BERT base cased (110M params)',
        'size_mb': 440,
        'requires_auth': False,
    },
    'bert-large-uncased': {
        'hf_id': 'google-bert/bert-large-uncased',
        'model_class': 'BERT',
        'config': {
            'vocab_size': 30522, 'embed_dim': 1024, 'num_heads': 16,
            'num_layers': 24, 'max_seq_len': 512,
        },
        'mapper': 'map_bert_weights',
        'description': 'BERT large uncased (340M params)',
        'size_mb': 1340,
        'requires_auth': False,
    },
    'distilbert-base-uncased': {
        'hf_id': 'distilbert/distilbert-base-uncased',
        'model_class': 'BERT',
        'config': {
            'vocab_size': 30522, 'embed_dim': 768, 'num_heads': 12,
            'num_layers': 6, 'max_seq_len': 512,
        },
        'mapper': 'map_distilbert_weights',
        'description': 'DistilBERT base uncased (66M params)',
        'size_mb': 265,
        'requires_auth': False,
    },
    'roberta-base': {
        'hf_id': 'FacebookAI/roberta-base',
        'model_class': 'BERT',
        'config': {
            'vocab_size': 50265, 'embed_dim': 768, 'num_heads': 12,
            'num_layers': 12, 'max_seq_len': 514,
        },
        'mapper': 'map_roberta_weights',
        'description': 'RoBERTa base (125M params)',
        'size_mb': 500,
        'requires_auth': False,
    },
    'roberta-large': {
        'hf_id': 'FacebookAI/roberta-large',
        'model_class': 'BERT',
        'config': {
            'vocab_size': 50265, 'embed_dim': 1024, 'num_heads': 16,
            'num_layers': 24, 'max_seq_len': 514,
        },
        'mapper': 'map_roberta_weights',
        'description': 'RoBERTa large (355M params)',
        'size_mb': 1420,
        'requires_auth': False,
    },

    # ── T5 ─────────────────────────────────────────────────────────────────────
    't5-small': {
        'hf_id': 'google-t5/t5-small',
        'model_class': 'T5',
        'config': {
            'vocab_size': 32128, 'embed_dim': 512, 'num_heads': 8,
            'num_layers': 6, 'd_ff': 2048,
        },
        'mapper': 'map_t5_weights',
        'description': 'T5 small (60M params)',
        'size_mb': 242,
        'requires_auth': False,
    },
    't5-base': {
        'hf_id': 'google-t5/t5-base',
        'model_class': 'T5',
        'config': {
            'vocab_size': 32128, 'embed_dim': 768, 'num_heads': 12,
            'num_layers': 12, 'd_ff': 3072,
        },
        'mapper': 'map_t5_weights',
        'description': 'T5 base (220M params)',
        'size_mb': 892,
        'requires_auth': False,
    },
    't5-large': {
        'hf_id': 'google-t5/t5-large',
        'model_class': 'T5',
        'config': {
            'vocab_size': 32128, 'embed_dim': 1024, 'num_heads': 16,
            'num_layers': 24, 'd_ff': 4096,
        },
        'mapper': 'map_t5_weights',
        'description': 'T5 large (770M params)',
        'size_mb': 2950,
        'requires_auth': False,
    },
    't5-xl': {
        'hf_id': 'google-t5/t5-xl',
        'model_class': 'T5',
        'config': {
            'vocab_size': 32128, 'embed_dim': 2048, 'num_heads': 32,
            'num_layers': 24, 'd_ff': 5120,
        },
        'mapper': 'map_t5_weights',
        'description': 'T5 XL (3B params)',
        'size_mb': 11400,
        'requires_auth': False,
    },

    # ── ViT ────────────────────────────────────────────────────────────────────
    'vit-base-patch16-224': {
        'hf_id': 'google/vit-base-patch16-224',
        'model_class': 'ViT',
        'config': {
            'image_size': 224, 'patch_size': 16, 'embed_dim': 768,
            'num_heads': 12, 'num_layers': 12, 'mlp_dim': 3072, 'num_classes': 1000,
        },
        'mapper': 'map_vit_weights',
        'description': 'ViT base patch16 (86M params)',
        'size_mb': 346,
        'requires_auth': False,
    },
    'vit-large-patch16-224': {
        'hf_id': 'google/vit-large-patch16-224',
        'model_class': 'ViT',
        'config': {
            'image_size': 224, 'patch_size': 16, 'embed_dim': 1024,
            'num_heads': 16, 'num_layers': 24, 'mlp_dim': 4096, 'num_classes': 1000,
        },
        'mapper': 'map_vit_weights',
        'description': 'ViT large patch16 (307M params)',
        'size_mb': 1200,
        'requires_auth': False,
    },
    'vit-base-patch32-224': {
        'hf_id': 'google/vit-base-patch32-224-in21k',
        'model_class': 'ViT',
        'config': {
            'image_size': 224, 'patch_size': 32, 'embed_dim': 768,
            'num_heads': 12, 'num_layers': 12, 'mlp_dim': 3072, 'num_classes': 21843,
        },
        'mapper': 'map_vit_weights',
        'description': 'ViT base patch32 ImageNet-21k (86M params)',
        'size_mb': 346,
        'requires_auth': False,
    },

    # ── LLaMA (gated) ──────────────────────────────────────────────────────────
    'meta-llama/Llama-2-7b-hf': {
        'hf_id': 'meta-llama/Llama-2-7b-hf',
        'model_class': 'LLaMA',
        'config': {
            'vocab_size': 32000, 'embed_dim': 4096, 'num_q_heads': 32,
            'num_kv_heads': 32, 'num_layers': 32, 'ffn_dim': 11008,
        },
        'mapper': 'map_llama_weights',
        'description': 'LLaMA 2 7B — gated (requires HF token)',
        'size_mb': 13476,
        'requires_auth': True,
    },
    'meta-llama/Llama-2-13b-hf': {
        'hf_id': 'meta-llama/Llama-2-13b-hf',
        'model_class': 'LLaMA',
        'config': {
            'vocab_size': 32000, 'embed_dim': 5120, 'num_q_heads': 40,
            'num_kv_heads': 40, 'num_layers': 40, 'ffn_dim': 13824,
        },
        'mapper': 'map_llama_weights',
        'description': 'LLaMA 2 13B — gated (requires HF token)',
        'size_mb': 26030,
        'requires_auth': True,
    },
    'llama-3-8b': {
        'hf_id': 'meta-llama/Meta-Llama-3-8B',
        'model_class': 'LLaMA',
        'config': {
            'vocab_size': 128256, 'embed_dim': 4096, 'num_q_heads': 32,
            'num_kv_heads': 8, 'num_layers': 32, 'ffn_dim': 14336,
        },
        'mapper': 'map_llama_weights',
        'description': 'LLaMA 3 8B — gated (requires HF token)',
        'size_mb': 16100,
        'requires_auth': True,
    },
    'mistralai/Mistral-7B-v0.1': {
        'hf_id': 'mistralai/Mistral-7B-v0.1',
        'model_class': 'Mistral',
        'config': {
            'vocab_size': 32000, 'embed_dim': 4096, 'num_q_heads': 32,
            'num_kv_heads': 8, 'num_layers': 32, 'ffn_dim': 14336,
            'window_size': 4096,
        },
        'mapper': 'map_mistral_weights',
        'description': 'Mistral 7B v0.1',
        'size_mb': 14480,
        'requires_auth': False,
    },
}

# Aliases
REGISTRY['llama-2-7b'] = REGISTRY['meta-llama/Llama-2-7b-hf']
REGISTRY['llama-2-13b'] = REGISTRY['meta-llama/Llama-2-13b-hf']
REGISTRY['mistral-7b'] = REGISTRY['mistralai/Mistral-7B-v0.1']
REGISTRY['distilbert'] = REGISTRY['distilbert-base-uncased']
