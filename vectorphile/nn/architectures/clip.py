"""CLIP — Contrastive Language-Image Pre-training (Radford et al. 2021, OpenAI).

Architecture:
  - Image encoder: ViT or ResNet backbone → projection head
  - Text encoder: transformer → projection head
  - Contrastive loss: symmetric cross-entropy over cosine similarities
"""
from __future__ import annotations
import math
from typing import Optional
from vectorphile.nn.module import Module
from vectorphile.core.tensor import Tensor
from vectorphile._core.ndarray import array as _nd_arr


def _randn(shape, scale=0.02):
    import random as _r
    n = 1
    for s in shape: n *= s
    flat = [math.sqrt(-2*math.log(max(_r.random(),1e-10)))*math.cos(2*math.pi*_r.random())*scale
            for _ in range(n)]
    return _nd_arr(flat, 'float32').reshape(shape)


def _zeros(shape):
    n = 1
    for s in shape: n *= s
    return _nd_arr([0.0]*n, 'float32').reshape(shape)


def _flat(nd):
    if hasattr(nd, 'to_flat_list'): return nd.to_flat_list()
    if hasattr(nd, 'tolist'): return nd.tolist()
    return list(nd)


class CLIPLinear(Module):
    def __init__(self, in_f, out_f, bias=True):
        super().__init__()
        self.weight = Tensor(_randn((out_f, in_f)))
        self.bias = Tensor(_zeros((out_f,))) if bias else None
        self.in_f = in_f; self.out_f = out_f

    def forward(self, x: Tensor) -> Tensor:
        sh = x._data.shape
        flat = _flat(x._data)
        n = len(flat) // self.in_f
        W = _flat(self.weight._data)
        result = []
        for i in range(n):
            row = flat[i*self.in_f:(i+1)*self.in_f]
            for o in range(self.out_f):
                v = sum(W[o*self.in_f+j]*row[j] for j in range(self.in_f))
                if self.bias is not None:
                    v += _flat(self.bias._data)[o]
                result.append(v)
        new_sh = sh[:-1] + (self.out_f,)
        return Tensor(_nd_arr(result,'float32').reshape(new_sh))


class CLIPLayerNorm(Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = Tensor(_nd_arr([1.0]*dim,'float32'))
        self.bias = Tensor(_nd_arr([0.0]*dim,'float32'))
        self.eps = eps; self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        sh = x._data.shape; flat = _flat(x._data)
        n = len(flat) // self.dim
        wf = _flat(self.weight._data); bf = _flat(self.bias._data)
        result = []
        for i in range(n):
            row = flat[i*self.dim:(i+1)*self.dim]
            mu = sum(row)/self.dim
            var = sum((v-mu)**2 for v in row)/self.dim
            std = math.sqrt(var+self.eps)
            result.extend([(v-mu)/std*wf[j]+bf[j] for j,v in enumerate(row)])
        return Tensor(_nd_arr(result,'float32').reshape(sh))


class CLIPAttention(Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim; self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q = CLIPLinear(embed_dim, embed_dim)
        self.k = CLIPLinear(embed_dim, embed_dim)
        self.v = CLIPLinear(embed_dim, embed_dim)
        self.out = CLIPLinear(embed_dim, embed_dim)

    def forward(self, x: Tensor, causal: bool = False) -> Tensor:
        sh = x._data.shape
        B = sh[0] if len(sh)==3 else 1; T = sh[-2]; D = sh[-1]
        Q = self.q(x); K = self.k(x); V = self.v(x)
        scale = 1.0/math.sqrt(self.head_dim)
        qf = _flat(Q._data); kf = _flat(K._data); vf = _flat(V._data)
        result = []
        for b in range(B):
            for t in range(T):
                q_vec = qf[(b*T+t)*D:(b*T+t)*D+D]
                scores = []
                for s in range(T):
                    if causal and s > t:
                        scores.append(-1e9)
                        continue
                    k_vec = kf[(b*T+s)*D:(b*T+s)*D+D]
                    score = sum(q_vec[i]*k_vec[i] for i in range(D))*scale
                    scores.append(score)
                mx = max(scores); exps = [math.exp(v-mx) for v in scores]
                sm = sum(exps)+1e-9; weights = [v/sm for v in exps]
                attn_out = [0.0]*D
                for s in range(T):
                    v_vec = vf[(b*T+s)*D:(b*T+s)*D+D]
                    for i in range(D):
                        attn_out[i] += weights[s]*v_vec[i]
                result.extend(attn_out)
        new_sh = (B,T,D) if len(sh)==3 else (T,D)
        return self.out(Tensor(_nd_arr(result,'float32').reshape(new_sh)))


class CLIPEncoderLayer(Module):
    def __init__(self, embed_dim, num_heads, ffn_dim):
        super().__init__()
        self.ln1 = CLIPLayerNorm(embed_dim)
        self.attn = CLIPAttention(embed_dim, num_heads)
        self.ln2 = CLIPLayerNorm(embed_dim)
        self.fc1 = CLIPLinear(embed_dim, ffn_dim)
        self.fc2 = CLIPLinear(ffn_dim, embed_dim)

    def forward(self, x: Tensor, causal: bool = False) -> Tensor:
        x = x + self.attn(self.ln1(x), causal=causal)
        h = self.fc1(self.ln2(x))
        # QuickGELU (CLIP uses approximate GELU)
        hf = _flat(h._data)
        hf = [v / (1.0 + math.exp(-1.702*v)) for v in hf]
        h = Tensor(_nd_arr(hf,'float32').reshape(h._data.shape))
        return x + self.fc2(h)


class CLIPTextEncoder(Module):
    """CLIP text transformer encoder."""
    def __init__(self, vocab_size: int = 49408, context_length: int = 77,
                 embed_dim: int = 512, num_heads: int = 8, num_layers: int = 12,
                 ffn_dim: int = 2048, proj_dim: int = 512):
        super().__init__()
        self.token_emb = Tensor(_randn((vocab_size, embed_dim)))
        self.pos_emb = Tensor(_randn((context_length, embed_dim)))
        self.layers = [CLIPEncoderLayer(embed_dim, num_heads, ffn_dim) for _ in range(num_layers)]
        for i, l in enumerate(self.layers):
            setattr(self, f'layer_{i}', l)
        self.ln_final = CLIPLayerNorm(embed_dim)
        self.proj = CLIPLinear(embed_dim, proj_dim, bias=False)
        self.vocab_size = vocab_size; self.embed_dim = embed_dim
        self.context_length = context_length

    def forward(self, tokens: Tensor) -> Tensor:
        sh = tokens._data.shape
        B = sh[0] if len(sh)>1 else 1; T = sh[-1]
        flat_tok = _flat(tokens._data)
        emb_w = _flat(self.token_emb._data)
        pos_w = _flat(self.pos_emb._data)
        result = []
        for b in range(B):
            for t in range(T):
                tok_id = int(flat_tok[b*T+t]) % self.vocab_size
                for c in range(self.embed_dim):
                    result.append(emb_w[tok_id*self.embed_dim+c] + pos_w[t*self.embed_dim+c])
        x = Tensor(_nd_arr(result,'float32').reshape((B,T,self.embed_dim)))
        for layer in self.layers:
            x = layer(x, causal=True)
        x = self.ln_final(x)
        # Take [EOS] token (last non-padding = last position for simplicity)
        xf = _flat(x._data); D = self.embed_dim
        eos_vecs = []
        for b in range(B):
            eos_vecs.extend(xf[(b*T+(T-1))*D:(b*T+(T-1))*D+D])
        eos_t = Tensor(_nd_arr(eos_vecs,'float32').reshape((B,D)))
        return self.proj(eos_t)  # (B, proj_dim)


class CLIPVisionEncoder(Module):
    """CLIP ViT-style image encoder."""
    def __init__(self, image_size: int = 224, patch_size: int = 32,
                 embed_dim: int = 768, num_heads: int = 12, num_layers: int = 12,
                 ffn_dim: int = 3072, proj_dim: int = 512, in_channels: int = 3):
        super().__init__()
        self.patch_size = patch_size; self.embed_dim = embed_dim
        n_patches = (image_size // patch_size) ** 2
        self.patch_proj = CLIPLinear(in_channels * patch_size * patch_size, embed_dim, bias=False)
        self.cls_token = Tensor(_randn((1, 1, embed_dim)))
        self.pos_emb = Tensor(_randn((n_patches + 1, embed_dim)))
        self.pre_ln = CLIPLayerNorm(embed_dim)
        self.layers = [CLIPEncoderLayer(embed_dim, num_heads, ffn_dim) for _ in range(num_layers)]
        for i, l in enumerate(self.layers):
            setattr(self, f'layer_{i}', l)
        self.ln = CLIPLayerNorm(embed_dim)
        self.proj = CLIPLinear(embed_dim, proj_dim, bias=False)
        self.n_patches = n_patches

    def forward(self, x: Tensor) -> Tensor:
        sh = x._data.shape  # (B, C, H, W)
        B = sh[0]; C = sh[1]; H = sh[2]; W = sh[3]
        P = self.patch_size
        flat = _flat(x._data)
        # Extract patches
        patches = []
        for b in range(B):
            for ph in range(H // P):
                for pw in range(W // P):
                    patch = []
                    for c in range(C):
                        for pi in range(P):
                            for pj in range(P):
                                idx = b*C*H*W + c*H*W + (ph*P+pi)*W + (pw*P+pj)
                                patch.append(flat[idx])
                    patches.append(patch)
        # (B * n_patches, C*P*P)
        patch_flat = [v for p in patches for v in p]
        patch_t = Tensor(_nd_arr(patch_flat,'float32').reshape((B*self.n_patches, C*P*P)))
        patch_emb = self.patch_proj(patch_t)  # (B*n_p, embed_dim)
        # Reshape to (B, n_patches, embed_dim) and prepend CLS
        pf = _flat(patch_emb._data); D = self.embed_dim
        cls_f = _flat(self.cls_token._data)
        pos_f = _flat(self.pos_emb._data)
        result = []
        for b in range(B):
            # CLS token + pos 0
            result.extend([cls_f[i] + pos_f[i] for i in range(D)])
            # Patches
            for p in range(self.n_patches):
                result.extend([pf[(b*self.n_patches+p)*D+c] + pos_f[(p+1)*D+c] for c in range(D)])
        T = self.n_patches + 1
        x = Tensor(_nd_arr(result,'float32').reshape((B, T, D)))
        x = self.pre_ln(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln(x)
        # CLS token
        xf = _flat(x._data)
        cls_vecs = []
        for b in range(B):
            cls_vecs.extend(xf[b*T*D:b*T*D+D])
        cls_t = Tensor(_nd_arr(cls_vecs,'float32').reshape((B, D)))
        return self.proj(cls_t)  # (B, proj_dim)


class CLIP(Module):
    """CLIP model — joint image-text embedding with contrastive loss."""
    def __init__(self, vocab_size: int = 49408, context_length: int = 77,
                 image_size: int = 224, patch_size: int = 32,
                 embed_dim: int = 512, proj_dim: int = 512,
                 text_num_heads: int = 8, text_num_layers: int = 12,
                 vision_embed_dim: int = 768, vision_num_heads: int = 12,
                 vision_num_layers: int = 12):
        super().__init__()
        self.text_encoder = CLIPTextEncoder(
            vocab_size, context_length, embed_dim, text_num_heads, text_num_layers,
            embed_dim * 4, proj_dim
        )
        self.vision_encoder = CLIPVisionEncoder(
            image_size, patch_size, vision_embed_dim, vision_num_heads, vision_num_layers,
            vision_embed_dim * 4, proj_dim
        )
        # Learnable temperature
        self.logit_scale = Tensor(_nd_arr([math.log(1/0.07)],'float32'))

    def encode_text(self, tokens: Tensor) -> Tensor:
        return self._normalize(self.text_encoder(tokens))

    def encode_image(self, images: Tensor) -> Tensor:
        return self._normalize(self.vision_encoder(images))

    def _normalize(self, x: Tensor) -> Tensor:
        sh = x._data.shape
        flat = _flat(x._data)
        D = sh[-1]
        n = len(flat) // D
        result = []
        for i in range(n):
            row = flat[i*D:(i+1)*D]
            norm = math.sqrt(sum(v**2 for v in row)) + 1e-8
            result.extend([v/norm for v in row])
        return Tensor(_nd_arr(result,'float32').reshape(sh))

    def forward(self, images: Tensor, tokens: Tensor):
        img_feat = self.encode_image(images)   # (B, proj_dim)
        txt_feat = self.encode_text(tokens)    # (B, proj_dim)
        # Similarity matrix
        scale = math.exp(_flat(self.logit_scale._data)[0])
        img_f = _flat(img_feat._data); txt_f = _flat(txt_feat._data)
        sh = img_feat._data.shape
        B = sh[0]; D = sh[-1]
        logits = []
        for i in range(B):
            row = []
            for j in range(B):
                dot = sum(img_f[i*D+k]*txt_f[j*D+k] for k in range(D))
                row.append(dot * scale)
            logits.append(row)
        # Symmetric cross-entropy loss
        log_probs_i2t = []
        log_probs_t2i = []
        for i in range(B):
            mx_i = max(logits[i]); mx_t = max(logits[j][i] for j in range(B))
            denom_i = sum(math.exp(logits[i][j]-mx_i) for j in range(B))
            denom_t = sum(math.exp(logits[j][i]-mx_t) for j in range(B))
            log_probs_i2t.append(logits[i][i] - mx_i - math.log(denom_i+1e-9))
            log_probs_t2i.append(logits[i][i] - mx_t - math.log(denom_t+1e-9))
        loss = -0.5 * (sum(log_probs_i2t) + sum(log_probs_t2i)) / max(B, 1)
        return Tensor(_nd_arr([loss],'float32')), img_feat, txt_feat

    @classmethod
    def clip_vit_b_32(cls) -> 'CLIP':
        return cls(embed_dim=512, proj_dim=512, vision_embed_dim=768,
                   vision_num_heads=12, vision_num_layers=12, patch_size=32)

    @classmethod
    def clip_vit_l_14(cls) -> 'CLIP':
        return cls(embed_dim=768, proj_dim=768, vision_embed_dim=1024,
                   vision_num_heads=16, vision_num_layers=24, patch_size=14)


__all__ = [
    'CLIPLinear', 'CLIPLayerNorm', 'CLIPAttention', 'CLIPEncoderLayer',
    'CLIPTextEncoder', 'CLIPVisionEncoder', 'CLIP',
]
