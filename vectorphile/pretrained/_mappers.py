"""Weight mappers — translate HuggingFace parameter names to vectorphile names."""
from __future__ import annotations

import warnings
from typing import Dict, Tuple

import numpy as np

from vectorphile.core.tensor import Tensor
from vectorphile.nn.module import Module


def _set(model: Module, attr_path: str, value: np.ndarray) -> bool:
    """Set a parameter on a (possibly nested) module via dot-path.

    Returns True on success, False if path not found.
    """
    parts = attr_path.split('.')
    obj = model
    for part in parts[:-1]:
        if hasattr(obj, part):
            obj = getattr(obj, part)
        elif hasattr(obj, '_modules') and part in obj._modules:
            obj = obj._modules[part]
        else:
            return False
    leaf = parts[-1]
    t = Tensor(value.astype(np.float32))
    t.requires_grad = True
    if hasattr(obj, '_parameters'):
        obj._parameters[leaf] = t
    setattr(obj, leaf, t)
    return True


def _report(name: str, mapped: int, total: int) -> None:
    pct = 100 * mapped / max(total, 1)
    print(f'[pretrained] {name}: mapped {mapped}/{total} weights ({pct:.1f}%)')


# ── GPT-2 ──────────────────────────────────────────────────────────────────────

def map_gpt2_weights(model: Module, hf: Dict[str, np.ndarray]) -> Module:
    """Map HuggingFace GPT-2 state dict to vectorphile GPT2."""
    mapped = 0
    total  = len(hf)

    def try_set(vp_path: str, arr: np.ndarray) -> None:
        nonlocal mapped
        if _set(model, vp_path, arr):
            mapped += 1
        else:
            warnings.warn(f'[GPT-2 mapper] could not set {vp_path}', stacklevel=3)

    # Token + positional embeddings
    if 'transformer.wte.weight' in hf:
        try_set('token_emb', hf['transformer.wte.weight'])
    if 'transformer.wpe.weight' in hf:
        try_set('pos_emb', hf['transformer.wpe.weight'])

    # Final layer norm
    if 'transformer.ln_f.weight' in hf:
        try_set('final_ln_w', hf['transformer.ln_f.weight'])
    if 'transformer.ln_f.bias' in hf:
        try_set('final_ln_b', hf['transformer.ln_f.bias'])

    # LM head (may be tied to wte)
    if 'lm_head.weight' in hf:
        try_set('lm_head_w', hf['lm_head.weight'])

    # Per-layer weights
    num_layers = getattr(model, 'num_layers', 12)
    for i in range(num_layers):
        prefix = f'transformer.h.{i}'
        # Layer norms
        for ln, vp in [('ln_1', f'blocks.{i}.ln1_w'), ('ln_2', f'blocks.{i}.ln2_w')]:
            if f'{prefix}.{ln}.weight' in hf:
                try_set(vp, hf[f'{prefix}.{ln}.weight'])
            vp_b = vp.replace('_w', '_b')
            if f'{prefix}.{ln}.bias' in hf:
                try_set(vp_b, hf[f'{prefix}.{ln}.bias'])

        # Attention: c_attn is a fused QKV projection (3*D, D)
        if f'{prefix}.attn.c_attn.weight' in hf:
            w = hf[f'{prefix}.attn.c_attn.weight']   # (3*D, D) or (D, 3*D)
            # GPT-2 stores as (D, 3*D) — transpose to (3*D, D)
            if w.ndim == 2 and w.shape[1] == 3 * w.shape[0]:
                w = w.T
            D = w.shape[1]
            try_set(f'blocks.{i}.attn.w_q', w[:D].T)
            try_set(f'blocks.{i}.attn.w_k', w[D:2*D].T)
            try_set(f'blocks.{i}.attn.w_v', w[2*D:].T)
        if f'{prefix}.attn.c_attn.bias' in hf:
            b = hf[f'{prefix}.attn.c_attn.bias']
            D = b.shape[0] // 3
            try_set(f'blocks.{i}.attn.b_q', b[:D])
            try_set(f'blocks.{i}.attn.b_k', b[D:2*D])
            try_set(f'blocks.{i}.attn.b_v', b[2*D:])
        if f'{prefix}.attn.c_proj.weight' in hf:
            w = hf[f'{prefix}.attn.c_proj.weight']
            try_set(f'blocks.{i}.attn.w_o', w if w.shape[0] < w.shape[1] else w.T)
        if f'{prefix}.attn.c_proj.bias' in hf:
            try_set(f'blocks.{i}.attn.b_o', hf[f'{prefix}.attn.c_proj.bias'])

        # MLP
        for src, dst in [
            (f'{prefix}.mlp.c_fc.weight',   f'blocks.{i}.mlp.fc1_w'),
            (f'{prefix}.mlp.c_fc.bias',     f'blocks.{i}.mlp.fc1_b'),
            (f'{prefix}.mlp.c_proj.weight', f'blocks.{i}.mlp.fc2_w'),
            (f'{prefix}.mlp.c_proj.bias',   f'blocks.{i}.mlp.fc2_b'),
        ]:
            if src in hf:
                w = hf[src]
                # c_fc.weight is (4*D, D) in HF — may need transpose
                if w.ndim == 2 and 'weight' in src:
                    # vectorphile expects (in, out)
                    if w.shape[0] > w.shape[1]:
                        w = w.T
                try_set(dst, w)

    _report('GPT-2', mapped, total)
    return model


# ── BERT ───────────────────────────────────────────────────────────────────────

def map_bert_weights(model: Module, hf: Dict[str, np.ndarray]) -> Module:
    """Map HuggingFace BERT state dict to vectorphile BERT."""
    mapped = 0
    total  = len(hf)

    def ts(vp: str, arr: np.ndarray) -> None:
        nonlocal mapped
        if _set(model, vp, arr): mapped += 1
        else: warnings.warn(f'[BERT mapper] could not set {vp}', stacklevel=3)

    # Embeddings
    emb_map = {
        'bert.embeddings.word_embeddings.weight':     'embeddings.word_emb',
        'bert.embeddings.position_embeddings.weight': 'embeddings.pos_emb',
        'bert.embeddings.token_type_embeddings.weight': 'embeddings.seg_emb',
        'bert.embeddings.LayerNorm.weight':            'embeddings.ln_w',
        'bert.embeddings.LayerNorm.bias':              'embeddings.ln_b',
    }
    for src, dst in emb_map.items():
        if src in hf:
            ts(dst, hf[src])

    # Pooler
    if 'bert.pooler.dense.weight' in hf:
        ts('pooler_w', hf['bert.pooler.dense.weight'].T)
    if 'bert.pooler.dense.bias' in hf:
        ts('pooler_b', hf['bert.pooler.dense.bias'])

    num_layers = getattr(model, 'num_layers', 12)
    for i in range(num_layers):
        p = f'bert.encoder.layer.{i}'
        layer_map = {
            f'{p}.attention.self.query.weight':     (f'layers.{i}.attn.w_q', True),
            f'{p}.attention.self.query.bias':       (f'layers.{i}.attn.b_q', False),
            f'{p}.attention.self.key.weight':       (f'layers.{i}.attn.w_k', True),
            f'{p}.attention.self.key.bias':         (f'layers.{i}.attn.b_k', False),
            f'{p}.attention.self.value.weight':     (f'layers.{i}.attn.w_v', True),
            f'{p}.attention.self.value.bias':       (f'layers.{i}.attn.b_v', False),
            f'{p}.attention.output.dense.weight':   (f'layers.{i}.attn.w_o', True),
            f'{p}.attention.output.dense.bias':     (f'layers.{i}.attn.b_o', False),
            f'{p}.attention.output.LayerNorm.weight': (f'layers.{i}.ln1_w', False),
            f'{p}.attention.output.LayerNorm.bias':   (f'layers.{i}.ln1_b', False),
            f'{p}.intermediate.dense.weight':       (f'layers.{i}.mlp.fc1_w', True),
            f'{p}.intermediate.dense.bias':         (f'layers.{i}.mlp.fc1_b', False),
            f'{p}.output.dense.weight':             (f'layers.{i}.mlp.fc2_w', True),
            f'{p}.output.dense.bias':               (f'layers.{i}.mlp.fc2_b', False),
            f'{p}.output.LayerNorm.weight':         (f'layers.{i}.ln2_w', False),
            f'{p}.output.LayerNorm.bias':           (f'layers.{i}.ln2_b', False),
        }
        for src, (dst, transpose) in layer_map.items():
            if src in hf:
                w = hf[src].T if (transpose and hf[src].ndim == 2) else hf[src]
                ts(dst, w)

    _report('BERT', mapped, total)
    return model


def map_distilbert_weights(model: Module, hf: Dict[str, np.ndarray]) -> Module:
    """Map DistilBERT (no pooler, different key prefix)."""
    # Remap distilbert.* keys to bert.* then reuse bert mapper
    remapped = {}
    for k, v in hf.items():
        new_k = k.replace('distilbert.transformer.layer.', 'bert.encoder.layer.')
        new_k = new_k.replace('distilbert.embeddings.', 'bert.embeddings.')
        new_k = new_k.replace('.attention.q_lin.', '.attention.self.query.')
        new_k = new_k.replace('.attention.k_lin.', '.attention.self.key.')
        new_k = new_k.replace('.attention.v_lin.', '.attention.self.value.')
        new_k = new_k.replace('.attention.out_lin.', '.attention.output.dense.')
        new_k = new_k.replace('.ffn.lin1.', '.intermediate.dense.')
        new_k = new_k.replace('.ffn.lin2.', '.output.dense.')
        new_k = new_k.replace('.sa_layer_norm.', '.attention.output.LayerNorm.')
        new_k = new_k.replace('.output_layer_norm.', '.output.LayerNorm.')
        remapped[new_k] = v
    return map_bert_weights(model, remapped)


def map_roberta_weights(model: Module, hf: Dict[str, np.ndarray]) -> Module:
    """RoBERTa uses same structure as BERT with 'roberta.' prefix."""
    remapped = {k.replace('roberta.', 'bert.'): v for k, v in hf.items()}
    return map_bert_weights(model, remapped)


# ── T5 ─────────────────────────────────────────────────────────────────────────

def map_t5_weights(model: Module, hf: Dict[str, np.ndarray]) -> Module:
    """Map HuggingFace T5 state dict to vectorphile T5."""
    mapped = 0
    total  = len(hf)

    def ts(vp: str, arr: np.ndarray) -> None:
        nonlocal mapped
        if _set(model, vp, arr): mapped += 1
        else: warnings.warn(f'[T5 mapper] could not set {vp}', stacklevel=3)

    if 'shared.weight' in hf:
        ts('shared_emb', hf['shared.weight'])
    if 'lm_head.weight' in hf:
        ts('lm_head_w', hf['lm_head.weight'])

    num_layers = getattr(model, 'num_layers', 6)
    for stack, vp_prefix in [('encoder', 'encoder.blocks'), ('decoder', 'decoder.blocks')]:
        for i in range(num_layers):
            p = f'{stack}.block.{i}.layer'
            block_map = {
                f'{p}.0.SelfAttention.q.weight':        (f'{vp_prefix}.{i}.attn.w_q', True),
                f'{p}.0.SelfAttention.k.weight':        (f'{vp_prefix}.{i}.attn.w_k', True),
                f'{p}.0.SelfAttention.v.weight':        (f'{vp_prefix}.{i}.attn.w_v', True),
                f'{p}.0.SelfAttention.o.weight':        (f'{vp_prefix}.{i}.attn.w_o', True),
                f'{p}.0.layer_norm.weight':              (f'{vp_prefix}.{i}.ln1_w', False),
                f'{p}.1.DenseReluDense.wi.weight':       (f'{vp_prefix}.{i}.mlp.fc1_w', True),
                f'{p}.1.DenseReluDense.wo.weight':       (f'{vp_prefix}.{i}.mlp.fc2_w', True),
                f'{p}.1.layer_norm.weight':              (f'{vp_prefix}.{i}.ln2_w', False),
            }
            # Decoder has cross-attention at layer index 1
            if stack == 'decoder':
                block_map.update({
                    f'{p}.1.EncDecAttention.q.weight': (f'{vp_prefix}.{i}.cross_attn.w_q', True),
                    f'{p}.1.EncDecAttention.k.weight': (f'{vp_prefix}.{i}.cross_attn.w_k', True),
                    f'{p}.1.EncDecAttention.v.weight': (f'{vp_prefix}.{i}.cross_attn.w_v', True),
                    f'{p}.1.EncDecAttention.o.weight': (f'{vp_prefix}.{i}.cross_attn.w_o', True),
                    f'{p}.1.layer_norm.weight':         (f'{vp_prefix}.{i}.ln_cross_w', False),
                    f'{p}.2.DenseReluDense.wi.weight':  (f'{vp_prefix}.{i}.mlp.fc1_w', True),
                    f'{p}.2.DenseReluDense.wo.weight':  (f'{vp_prefix}.{i}.mlp.fc2_w', True),
                    f'{p}.2.layer_norm.weight':          (f'{vp_prefix}.{i}.ln2_w', False),
                })
            for src, (dst, transpose) in block_map.items():
                if src in hf:
                    w = hf[src].T if (transpose and hf[src].ndim == 2) else hf[src]
                    ts(dst, w)

    for norm_key, vp in [
        ('encoder.final_layer_norm.weight', 'encoder_final_ln_w'),
        ('decoder.final_layer_norm.weight', 'decoder_final_ln_w'),
    ]:
        if norm_key in hf:
            ts(vp, hf[norm_key])

    _report('T5', mapped, total)
    return model


# ── LLaMA ──────────────────────────────────────────────────────────────────────

def map_llama_weights(model: Module, hf: Dict[str, np.ndarray]) -> Module:
    """Map HuggingFace LLaMA state dict to vectorphile LLaMA."""
    mapped = 0
    total  = len(hf)

    def ts(vp: str, arr: np.ndarray) -> None:
        nonlocal mapped
        if _set(model, vp, arr): mapped += 1
        else: warnings.warn(f'[LLaMA mapper] could not set {vp}', stacklevel=3)

    if 'model.embed_tokens.weight' in hf:
        ts('token_emb', hf['model.embed_tokens.weight'])
    if 'model.norm.weight' in hf:
        ts('final_norm_w', hf['model.norm.weight'])
    if 'lm_head.weight' in hf:
        ts('lm_head_w', hf['lm_head.weight'])

    num_layers = getattr(model, 'num_layers', 32)
    for i in range(num_layers):
        p = f'model.layers.{i}'
        layer_map = {
            f'{p}.self_attn.q_proj.weight': (f'blocks.{i}.attn.w_q', True),
            f'{p}.self_attn.k_proj.weight': (f'blocks.{i}.attn.w_k', True),
            f'{p}.self_attn.v_proj.weight': (f'blocks.{i}.attn.w_v', True),
            f'{p}.self_attn.o_proj.weight': (f'blocks.{i}.attn.w_o', True),
            f'{p}.mlp.gate_proj.weight':    (f'blocks.{i}.ffn.w1', True),
            f'{p}.mlp.up_proj.weight':      (f'blocks.{i}.ffn.w3', True),
            f'{p}.mlp.down_proj.weight':    (f'blocks.{i}.ffn.w2', True),
            f'{p}.input_layernorm.weight':           (f'blocks.{i}.ln1_w', False),
            f'{p}.post_attention_layernorm.weight':  (f'blocks.{i}.ln2_w', False),
        }
        for src, (dst, transpose) in layer_map.items():
            if src in hf:
                w = hf[src].T if (transpose and hf[src].ndim == 2) else hf[src]
                ts(dst, w)

    _report('LLaMA', mapped, total)
    return model


# ── Mistral ────────────────────────────────────────────────────────────────────

def map_mistral_weights(model: Module, hf: Dict[str, np.ndarray]) -> Module:
    """Mistral uses same structure as LLaMA."""
    return map_llama_weights(model, hf)


# ── ViT ────────────────────────────────────────────────────────────────────────

def map_vit_weights(model: Module, hf: Dict[str, np.ndarray]) -> Module:
    """Map HuggingFace ViT state dict to vectorphile ViT."""
    mapped = 0
    total  = len(hf)

    def ts(vp: str, arr: np.ndarray) -> None:
        nonlocal mapped
        if _set(model, vp, arr): mapped += 1
        else: warnings.warn(f'[ViT mapper] could not set {vp}', stacklevel=3)

    # Patch embedding
    if 'vit.embeddings.patch_embeddings.projection.weight' in hf:
        # HF stores as (embed_dim, in_channels, pH, pW) — reshape to (patch_dim, embed_dim)
        w = hf['vit.embeddings.patch_embeddings.projection.weight']
        embed_dim = w.shape[0]
        patch_dim = w.shape[1] * w.shape[2] * w.shape[3]
        ts('patch_embed.proj_w', w.reshape(embed_dim, patch_dim).T)
    if 'vit.embeddings.patch_embeddings.projection.bias' in hf:
        ts('patch_embed.proj_b', hf['vit.embeddings.patch_embeddings.projection.bias'])

    if 'vit.embeddings.cls_token' in hf:
        ts('cls_token', hf['vit.embeddings.cls_token'])
    if 'vit.embeddings.position_embeddings' in hf:
        ts('pos_embed', hf['vit.embeddings.position_embeddings'])

    # Final layer norm
    if 'vit.layernorm.weight' in hf:
        ts('ln_w', hf['vit.layernorm.weight'])
    if 'vit.layernorm.bias' in hf:
        ts('ln_b', hf['vit.layernorm.bias'])

    # Classifier head
    if 'classifier.weight' in hf:
        ts('head_w', hf['classifier.weight'].T)
    if 'classifier.bias' in hf:
        ts('head_b', hf['classifier.bias'])

    num_layers = getattr(model, 'num_layers', 12) if hasattr(model, 'num_layers') \
                 else len(getattr(model, 'blocks', []))
    for i in range(num_layers):
        p = f'vit.encoder.layer.{i}'
        block_map = {
            f'{p}.attention.attention.query.weight': (f'blocks.{i}.attn.w_q', True),
            f'{p}.attention.attention.query.bias':   (f'blocks.{i}.attn.b_q', False),
            f'{p}.attention.attention.key.weight':   (f'blocks.{i}.attn.w_k', True),
            f'{p}.attention.attention.key.bias':     (f'blocks.{i}.attn.b_k', False),
            f'{p}.attention.attention.value.weight': (f'blocks.{i}.attn.w_v', True),
            f'{p}.attention.attention.value.bias':   (f'blocks.{i}.attn.b_v', False),
            f'{p}.attention.output.dense.weight':    (f'blocks.{i}.attn.w_o', True),
            f'{p}.attention.output.dense.bias':      (f'blocks.{i}.attn.b_o', False),
            f'{p}.layernorm_before.weight':          (f'blocks.{i}.ln1_w', False),
            f'{p}.layernorm_before.bias':            (f'blocks.{i}.ln1_b', False),
            f'{p}.layernorm_after.weight':           (f'blocks.{i}.ln2_w', False),
            f'{p}.layernorm_after.bias':             (f'blocks.{i}.ln2_b', False),
            f'{p}.intermediate.dense.weight':        (f'blocks.{i}.mlp.fc1_w', True),
            f'{p}.intermediate.dense.bias':          (f'blocks.{i}.mlp.fc1_b', False),
            f'{p}.output.dense.weight':              (f'blocks.{i}.mlp.fc2_w', True),
            f'{p}.output.dense.bias':                (f'blocks.{i}.mlp.fc2_b', False),
        }
        for src, (dst, transpose) in block_map.items():
            if src in hf:
                w = hf[src].T if (transpose and hf[src].ndim == 2) else hf[src]
                ts(dst, w)

    _report('ViT', mapped, total)
    return model


# ── Dispatcher ─────────────────────────────────────────────────────────────────

_MAPPER_FNS = {
    'map_gpt2_weights':      map_gpt2_weights,
    'map_bert_weights':      map_bert_weights,
    'map_distilbert_weights': map_distilbert_weights,
    'map_roberta_weights':   map_roberta_weights,
    'map_t5_weights':        map_t5_weights,
    'map_llama_weights':     map_llama_weights,
    'map_mistral_weights':   map_mistral_weights,
    'map_vit_weights':       map_vit_weights,
}


def get_mapper(name: str):
    fn = _MAPPER_FNS.get(name)
    if fn is None:
        raise ValueError(f'Unknown weight mapper: {name!r}. '
                         f'Available: {list(_MAPPER_FNS)}')
    return fn
