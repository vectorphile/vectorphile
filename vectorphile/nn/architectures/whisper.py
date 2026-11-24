"""Whisper — OpenAI automatic speech recognition encoder-decoder (Radford et al. 2022).

Architecture:
  Encoder: log-mel spectrogram → 2x Conv1d (stride 1 & 2) → sinusoidal pos enc →
           N x TransformerEncoderLayer
  Decoder: token embeddings + learned pos enc → N x TransformerDecoderLayer → LM head
"""
from __future__ import annotations
import math
from typing import Optional, List
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


class WhisperLinear(Module):
    def __init__(self, in_f, out_f, bias=True):
        super().__init__()
        self.weight = Tensor(_randn((out_f, in_f)))
        self.bias = Tensor(_zeros((out_f,))) if bias else None
        self.in_f = in_f
        self.out_f = out_f

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
                    b = _flat(self.bias._data)
                    v += b[o]
                result.append(v)
        new_sh = sh[:-1] + (self.out_f,)
        return Tensor(_nd_arr(result,'float32').reshape(new_sh))


class WhisperConv1d(Module):
    """1-D causal convolution for Whisper encoder stem."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.weight = Tensor(_randn((out_channels, in_channels, kernel_size)))
        self.bias = Tensor(_zeros((out_channels,)))
        self.in_c = in_channels; self.out_c = out_channels
        self.ks = kernel_size; self.stride = stride; self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        sh = x._data.shape  # (batch, C, T) or (C, T)
        flat = _flat(x._data)
        if len(sh) == 2:
            C, T = sh
            B = 1
        else:
            B, C, T = sh
        T_pad = T + 2*self.padding
        W = _flat(self.weight._data)
        bias = _flat(self.bias._data)
        T_out = (T_pad - self.ks) // self.stride + 1
        result = []
        for b in range(B):
            for oc in range(self.out_c):
                for t in range(T_out):
                    v = bias[oc]
                    for ic in range(C):
                        for k in range(self.ks):
                            t_in = t * self.stride + k - self.padding
                            if 0 <= t_in < T:
                                src = b*C*T + ic*T + t_in
                                w_idx = oc*(C*self.ks) + ic*self.ks + k
                                v += W[w_idx] * flat[src]
                    result.append(v)
        # GELU activation
        result = [v * 0.5 * (1 + math.tanh(0.7978845608 * (v + 0.044715 * v**3))) for v in result]
        new_sh = (B, self.out_c, T_out) if len(sh) == 3 else (self.out_c, T_out)
        return Tensor(_nd_arr(result,'float32').reshape(new_sh))


class WhisperMHA(Module):
    """Multi-head attention for Whisper (supports cross-attention)."""
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = WhisperLinear(embed_dim, embed_dim, bias=True)
        self.k_proj = WhisperLinear(embed_dim, embed_dim, bias=False)
        self.v_proj = WhisperLinear(embed_dim, embed_dim, bias=True)
        self.out_proj = WhisperLinear(embed_dim, embed_dim, bias=True)

    def forward(self, x: Tensor, encoder_out: Optional[Tensor] = None,
                mask: Optional[Tensor] = None) -> Tensor:
        sh = x._data.shape  # (B, T, D) or (T, D)
        if len(sh) == 2:
            T, D = sh; B = 1
        else:
            B, T, D = sh
        kv_src = encoder_out if encoder_out is not None else x
        kv_sh = kv_src._data.shape
        S = kv_sh[-2] if len(kv_sh) >= 2 else kv_sh[0]
        Q = self.q_proj(x)
        K = self.k_proj(kv_src)
        V = self.v_proj(kv_src)
        scale = 1.0 / math.sqrt(self.head_dim)
        # Simple scaled dot-product (flatten heads for simplicity)
        qf = _flat(Q._data); kf = _flat(K._data); vf = _flat(V._data)
        H = self.num_heads; HD = self.head_dim
        result = []
        for b in range(B):
            for t in range(T):
                q_vec = qf[(b*T+t)*D:(b*T+t)*D+D]
                attn_out = [0.0]*D
                # compute attention scores over all S positions
                scores = []
                for s in range(S):
                    k_vec = kf[(b*S+s)*D:(b*S+s)*D+D]
                    score = sum(q_vec[i]*k_vec[i] for i in range(D)) * scale
                    scores.append(score)
                mx = max(scores)
                exps = [math.exp(v-mx) for v in scores]
                sm = sum(exps)+1e-9
                weights = [v/sm for v in exps]
                for s in range(S):
                    v_vec = vf[(b*S+s)*D:(b*S+s)*D+D]
                    for i in range(D):
                        attn_out[i] += weights[s]*v_vec[i]
                result.extend(attn_out)
        new_sh = (B,T,D) if len(sh)==3 else (T,D)
        attn_t = Tensor(_nd_arr(result,'float32').reshape(new_sh))
        return self.out_proj(attn_t)


class WhisperLayerNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = Tensor(_nd_arr([1.0]*dim,'float32'))
        self.bias = Tensor(_nd_arr([0.0]*dim,'float32'))
        self.eps = eps; self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        sh = x._data.shape
        flat = _flat(x._data)
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


class WhisperEncoderLayer(Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.self_attn = WhisperMHA(embed_dim, num_heads)
        self.self_attn_ln = WhisperLayerNorm(embed_dim)
        self.fc1 = WhisperLinear(embed_dim, ffn_dim)
        self.fc2 = WhisperLinear(ffn_dim, embed_dim)
        self.final_ln = WhisperLayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.self_attn_ln(x)
        x = self.self_attn(x)
        x = residual + x
        residual = x
        x = self.final_ln(x)
        h = self.fc1(x)
        # GELU
        hf = _flat(h._data)
        hf = [v * 0.5 * (1 + math.tanh(0.7978845608 * (v + 0.044715*v**3))) for v in hf]
        h = Tensor(_nd_arr(hf,'float32').reshape(h._data.shape))
        x = self.fc2(h)
        return residual + x


class WhisperDecoderLayer(Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.self_attn = WhisperMHA(embed_dim, num_heads)
        self.self_attn_ln = WhisperLayerNorm(embed_dim)
        self.cross_attn = WhisperMHA(embed_dim, num_heads)
        self.cross_attn_ln = WhisperLayerNorm(embed_dim)
        self.fc1 = WhisperLinear(embed_dim, ffn_dim)
        self.fc2 = WhisperLinear(ffn_dim, embed_dim)
        self.final_ln = WhisperLayerNorm(embed_dim)

    def forward(self, x: Tensor, encoder_out: Optional[Tensor] = None) -> Tensor:
        residual = x
        x = self.self_attn_ln(x)
        x = self.self_attn(x)
        x = residual + x
        if encoder_out is not None:
            residual = x
            x = self.cross_attn_ln(x)
            x = self.cross_attn(x, encoder_out=encoder_out)
            x = residual + x
        residual = x
        x = self.final_ln(x)
        h = self.fc1(x)
        hf = _flat(h._data)
        hf = [v * 0.5 * (1 + math.tanh(0.7978845608 * (v + 0.044715*v**3))) for v in hf]
        h = Tensor(_nd_arr(hf,'float32').reshape(h._data.shape))
        x = self.fc2(h)
        return residual + x


class WhisperEncoder(Module):
    """Whisper audio encoder: Conv1d stem + transformer layers."""
    def __init__(self, n_mels: int = 80, embed_dim: int = 512,
                 num_heads: int = 8, num_layers: int = 6, ffn_dim: int = 2048,
                 max_source_positions: int = 1500):
        super().__init__()
        self.conv1 = WhisperConv1d(n_mels, embed_dim, kernel_size=3, padding=1)
        self.conv2 = WhisperConv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        # Sinusoidal positional encoding
        pos = [[math.sin(p / 10000**(2*(i//2)/embed_dim)) if i%2==0
                else math.cos(p / 10000**(2*(i//2)/embed_dim))
                for i in range(embed_dim)] for p in range(max_source_positions)]
        flat_pos = [v for row in pos for v in row]
        self.register_buffer('pos_enc', Tensor(_nd_arr(flat_pos,'float32').reshape((max_source_positions, embed_dim))))
        self.layers = [WhisperEncoderLayer(embed_dim, num_heads, ffn_dim) for _ in range(num_layers)]
        for i, l in enumerate(self.layers):
            setattr(self, f'layer_{i}', l)
        self.ln_post = WhisperLayerNorm(embed_dim)
        self.embed_dim = embed_dim

    def register_buffer(self, name, tensor):
        setattr(self, name, tensor)

    def forward(self, mel: Tensor) -> Tensor:
        x = self.conv1(mel)
        x = self.conv2(x)
        # Transpose from (B, C, T) to (B, T, C)
        sh = x._data.shape
        if len(sh) == 3:
            B, C, T = sh
            flat = _flat(x._data)
            transposed = []
            for b in range(B):
                for t in range(T):
                    for c in range(C):
                        transposed.append(flat[b*C*T + c*T + t])
            x = Tensor(_nd_arr(transposed,'float32').reshape((B, T, C)))
        # Add positional encoding
        pf = _flat(self.pos_enc._data)
        xf = _flat(x._data)
        T_actual = sh[-1] if len(sh)==3 else 1
        new_flat = []
        for b in range(B):
            for t in range(min(T_actual, len(pf)//self.embed_dim)):
                for c in range(self.embed_dim):
                    new_flat.append(xf[b*T_actual*self.embed_dim + t*self.embed_dim + c] +
                                    pf[t*self.embed_dim + c])
        x = Tensor(_nd_arr(new_flat,'float32').reshape((B, T_actual, self.embed_dim)))
        for layer in self.layers:
            x = layer(x)
        return self.ln_post(x)


class WhisperDecoder(Module):
    """Whisper autoregressive decoder."""
    def __init__(self, vocab_size: int = 51865, embed_dim: int = 512,
                 num_heads: int = 8, num_layers: int = 6, ffn_dim: int = 2048,
                 max_target_positions: int = 448):
        super().__init__()
        self.token_emb = WhisperLinear(vocab_size, embed_dim, bias=False)
        self.pos_emb = Tensor(_randn((max_target_positions, embed_dim)))
        self.layers = [WhisperDecoderLayer(embed_dim, num_heads, ffn_dim) for _ in range(num_layers)]
        for i, l in enumerate(self.layers):
            setattr(self, f'layer_{i}', l)
        self.ln = WhisperLayerNorm(embed_dim)
        self.lm_head = WhisperLinear(embed_dim, vocab_size, bias=False)
        self.vocab_size = vocab_size; self.embed_dim = embed_dim

    def forward(self, tokens: Tensor, encoder_out: Optional[Tensor] = None) -> Tensor:
        sh = tokens._data.shape
        flat_tok = _flat(tokens._data)
        # one-hot → embed
        B = sh[0] if len(sh) > 1 else 1
        T = sh[-1]
        # embed via lookup (integer token ids)
        emb_flat = []
        wf = _flat(self.token_emb.weight._data)  # (vocab_size, embed_dim)
        # weight is (out_f, in_f) = (embed_dim, vocab_size) → transpose for embedding
        pf = _flat(self.pos_emb._data)
        for b in range(B):
            for t in range(T):
                tok_id = int(flat_tok[b*T+t]) % self.vocab_size
                for c in range(self.embed_dim):
                    emb_v = wf[c*self.vocab_size + tok_id] if c*self.vocab_size+tok_id < len(wf) else 0.0
                    pos_v = pf[t*self.embed_dim+c] if t*self.embed_dim+c < len(pf) else 0.0
                    emb_flat.append(emb_v + pos_v)
        x = Tensor(_nd_arr(emb_flat,'float32').reshape((B, T, self.embed_dim)))
        for layer in self.layers:
            x = layer(x, encoder_out=encoder_out)
        return self.lm_head(self.ln(x))


class Whisper(Module):
    """Full Whisper encoder-decoder ASR model."""
    def __init__(self, n_mels: int = 80, vocab_size: int = 51865,
                 embed_dim: int = 512, num_heads: int = 8,
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 ffn_dim: int = 2048):
        super().__init__()
        self.encoder = WhisperEncoder(n_mels, embed_dim, num_heads, num_encoder_layers, ffn_dim)
        self.decoder = WhisperDecoder(vocab_size, embed_dim, num_heads, num_decoder_layers, ffn_dim)

    def forward(self, mel: Tensor, tokens: Tensor) -> Tensor:
        enc_out = self.encoder(mel)
        return self.decoder(tokens, encoder_out=enc_out)

    @classmethod
    def tiny(cls) -> 'Whisper':
        return cls(embed_dim=384, num_heads=6, num_encoder_layers=4, num_decoder_layers=4, ffn_dim=1536)

    @classmethod
    def base(cls) -> 'Whisper':
        return cls(embed_dim=512, num_heads=8, num_encoder_layers=6, num_decoder_layers=6, ffn_dim=2048)

    @classmethod
    def small(cls) -> 'Whisper':
        return cls(embed_dim=768, num_heads=12, num_encoder_layers=12, num_decoder_layers=12, ffn_dim=3072)

    @classmethod
    def medium(cls) -> 'Whisper':
        return cls(embed_dim=1024, num_heads=16, num_encoder_layers=24, num_decoder_layers=24, ffn_dim=4096)

    @classmethod
    def large(cls) -> 'Whisper':
        return cls(embed_dim=1280, num_heads=20, num_encoder_layers=32, num_decoder_layers=32, ffn_dim=5120)


__all__ = [
    'WhisperLinear', 'WhisperConv1d', 'WhisperMHA', 'WhisperLayerNorm',
    'WhisperEncoderLayer', 'WhisperDecoderLayer',
    'WhisperEncoder', 'WhisperDecoder', 'Whisper',
]
