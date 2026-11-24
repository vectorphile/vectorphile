"""AdaLoRA, prefix tuning, low-bit weight storage and GRPO / policy-gradient objectives."""
from __future__ import annotations
import math
from vectorphile._core.ndarray import NDArray, array as _nd_array, zeros as _nd_zeros, ones as _nd_ones, full as _nd_full, empty as _nd_empty, eye as _nd_eye, arange as _nd_arange, linspace as _nd_linspace
import math as _math
from vectorphile.core.tensor import Tensor
from vectorphile.nn.module import Module
from vectorphile.nn.layers import Linear
import vectorphile.nn.functional as F
from vectorphile.autograd import complete as _ag
from vectorphile._vp_helpers import *



def _param(arr):
    return Tensor(_nd_array(arr, dtype='float32'), requires_grad=True)


# ── low-bit storage ──────────────────────────────────────────────────────────

class QuantizedLinear(Module):
    """Frozen Linear whose weight is stored as `bits`-bit integer codes (two codes per byte when
    bits <= 4) with a per-row offset and step. The float weight is rebuilt only during a forward pass."""

    def __init__(self, linear: Linear, bits=4):
        super().__init__()
        assert 2 <= bits <= 8
        w = linear.weight._data.astype('float32')
        self.bits, self.shape = bits, w.shape
        self.out_features, self.in_features = w.shape
        self.lo = w.min(axis=1, keepdims=True)
        self.step = _nd_maximum_fn(w.max(axis=1, keepdims=True) - self.lo, 1e-12) / (2 ** bits - 1)
        codes = _nd_round_fn((w - self.lo) / self.step).astype('uint8').reshape(-1)
        self._n = codes.size
        if bits <= 4:
            if codes.size % 2:
                codes = _np_append(codes, _nd_array([0], dtype='uint8'))
            codes = (codes[0::2] << 4) | codes[1::2]
        self.codes = codes
        self.bias_param = linear.bias_param
        if self.bias_param is not None:
            self.bias_param.requires_grad = False

    def _dequantize(self) -> NDArray:
        c = self.codes
        if self.bits <= 4:
            c = _nd_stack_fn([c >> 4, c & 0x0F], axis=1).reshape(-1)[:self._n]
        return c.reshape(self.shape).astype('float32') * self.step + self.lo

    @property
    def weight(self) -> Tensor:
        return Tensor(self._dequantize())

    @property
    def memory_bytes(self) -> int:
        def _nbytes(a):
            return len(a.tobytes()) if hasattr(a, 'tobytes') and not hasattr(a, 'nbytes') else a.nbytes
        return int(_nbytes(self.codes) + _nbytes(self.lo) + _nbytes(self.step))

    def forward(self, x):
        y = x @ Tensor(self._dequantize().T)
        return y + self.bias_param if self.bias_param is not None else y

    def to_linear(self) -> Linear:
        lin = Linear(self.in_features, self.out_features, bias=self.bias_param is not None)
        lin.weight._data[...] = self._dequantize()
        if self.bias_param is not None:
            lin.bias_param._data[...] = self.bias_param._data
        return lin


# ── AdaLoRA ──────────────────────────────────────────────────────────────────

class AdaLoRALinear(Module):
    """W + (alpha / r) * P diag(lambda) Q with an adaptive rank budget (Zhang et al., 2023)."""

    def __init__(self, base: Linear, r=12, alpha=24):
        super().__init__()
        self.base = base
        out_f, in_f = base.weight._data.shape
        self.r, self.scale = r, alpha / r
        self.lora_P = _param(_nd_randn_fn(out_f, r) * 0.02)
        self.lora_Q = _param(_nd_randn_fn(r, in_f) * 0.02)
        self.lora_E = _param(_nd_zeros(r))
        self.mask = _nd_ones(r, dtype='float32')
        self.importance = _nd_zeros(r, dtype='float32')
        for p in base.parameters():
            p.requires_grad = False

    def forward(self, x):
        lam = self.lora_E * Tensor(self.mask)
        return self.base(x) + ((x @ self.lora_Q.transpose(0, 1)) * lam) @ self.lora_P.transpose(0, 1) * self.scale

    def orth_loss(self):
        eye_r = Tensor(_nd_eye(self.r, dtype='float32'))
        a = self.lora_P.transpose(0, 1) @ self.lora_P - eye_r
        b = self.lora_Q @ self.lora_Q.transpose(0, 1) - eye_r
        return (a * a).sum() + (b * b).sum()

    def update_importance(self, beta=0.85):
        """Call after backward(): smoothed sensitivity |w * grad| of each (P column, lambda, Q row) triplet."""
        def s(t, axis=None):
            if t.grad is None:
                return 0.0
            v = _nd_abs_fn(t._data * t.grad._data)
            return v if axis is None else v.mean(axis=axis)
        score = s(self.lora_E) + s(self.lora_P, 0) + s(self.lora_Q, 1)
        self.importance = beta * self.importance + (1 - beta) * _nd_array(score, dtype='float32')

    def merged(self) -> Linear:
        self.base.weight._data += self.scale * (self.lora_P._data * (self.lora_E._data * self.mask)) @ self.lora_Q._data
        for p in self.base.parameters():
            p.requires_grad = True
        return self.base


def adalora_layers(model):
    return [m for m in model.modules() if isinstance(m, AdaLoRALinear)]


def adalora_orth_loss(model, weight=0.1):
    layers = adalora_layers(model)
    total = layers[0].orth_loss()
    for l in layers[1:]:
        total = total + l.orth_loss()
    return total * weight


def adalora_update(model, total_rank: int, beta=0.85) -> int:
    """After backward(): refresh importance scores and keep only the `total_rank` most important
    singular directions across all AdaLoRA layers. Returns the number of active directions."""
    layers = adalora_layers(model)
    for l in layers:
        l.update_importance(beta)
    scores = _nd_concatenate_fn([l.importance for l in layers])
    keep = min(total_rank, scores.size)
    thresh = _nd_sort_fn(scores)[-keep] if keep > 0 else float('inf')
    active = 0
    for l in layers:
        l.mask = (l.importance >= thresh).astype('float32')
        active += int(l.mask.sum())
    return active


# ── prefix tuning ────────────────────────────────────────────────────────────

class PrefixEncoder(Module):
    """Prefix tuning (Li & Liang, 2021): trainable key/value prefixes for every attention layer,
    reparameterised through a small MLP for stable training.

        prefixes = encoder()                       # list of (prefix_k, prefix_v), one per layer
        out, _ = attn(x, x, x, prefix_k=pk, prefix_v=pv)   # nn.MultiheadAttention
    """

    def __init__(self, num_layers: int, embed_dim: int, num_virtual_tokens=20, hidden=128):
        super().__init__()
        self.num_layers, self.embed_dim, self.n = num_layers, embed_dim, num_virtual_tokens
        self.embedding = _param(_nd_randn_fn(num_virtual_tokens, embed_dim) * 0.02)
        self.fc1 = Linear(embed_dim, hidden)
        self.fc2 = Linear(hidden, num_layers * 2 * embed_dim)

    def forward(self):
        h = self.fc2(self.fc1(self.embedding).tanh()).reshape(self.n, self.num_layers, 2, self.embed_dim)
        return [(h[:, i, 0], h[:, i, 1]) for i in range(self.num_layers)]


# ── reinforcement objectives ─────────────────────────────────────────────────

def group_advantages(rewards, eps=1e-6) -> NDArray:
    """rewards [groups, samples_per_group] -> advantages normalised within each group (GRPO)."""
    r = _nd_array(rewards._data if isinstance(rewards, Tensor) else rewards, dtype='float32')
    return (r - r.mean(axis=1, keepdims=True)) / (r.std(axis=1, keepdims=True) + eps)


def grpo_loss(logp_new: Tensor, logp_old, rewards, clip_eps=0.2, kl_coef=0.0, logp_ref=None) -> Tensor:
    """Group Relative Policy Optimization: clipped policy-gradient surrogate with group-normalised
    advantages. logp_* are sequence log-probs shaped [groups, samples_per_group]."""
    adv = Tensor(group_advantages(rewards))
    old = Tensor(_nd_array(logp_old._data if isinstance(logp_old, Tensor) else logp_old, dtype='float32'))
    ratio = (logp_new - old).clamp(-20.0, 20.0).exp()
    s1, s2 = ratio * adv, ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * adv
    loss = -_ag.where(s1._data < s2._data, s1, s2).mean()
    if kl_coef > 0 and logp_ref is not None:
        ref = Tensor(_nd_array(logp_ref._data if isinstance(logp_ref, Tensor) else logp_ref, dtype='float32'))
        d = ref - logp_new
        loss = loss + (d.exp() - d - 1.0).mean() * kl_coef
    return loss


def policy_gradient_loss(logp: Tensor, rewards, baseline=None) -> Tensor:
    """REINFORCE with an optional baseline (defaults to the batch mean reward)."""
    r = _nd_array(rewards._data if isinstance(rewards, Tensor) else rewards, dtype='float32')
    b = r.mean() if baseline is None else baseline
    return -(logp * Tensor(r - b)).mean()


def sample_tokens(logits: Tensor, temperature=1.0, top_k=0, rng=None) -> NDArray:
    """Sample next-token ids from logits [..., V] (for online rollouts in GRPO / RL fine-tuning)."""
    rng = rng or _vp_random.default_rng()
    z = logits._data.astype('float64') / max(temperature, 1e-6)
    if top_k and top_k < z.shape[-1]:
        kth = _nd_sort_fn(z, axis=-1)[..., -top_k][..., None]
        z = _nd_where_fn(z < kth, -float('inf'), z)
    p = _math_exp(z - z.max(axis=-1, keepdims=True))
    p /= p.sum(axis=-1, keepdims=True)
    flat = p.reshape(-1, p.shape[-1])
    ids = _nd_array([rng.choice(flat.shape[1], p=row) for row in flat])
    return ids.reshape(p.shape[:-1])
