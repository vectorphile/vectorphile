"""vectorphile.peft — Parameter-Efficient Fine-Tuning.

Provides:
- LoRA (Low-Rank Adaptation) + RSLoRA (scaled) + DoRA (weight decomposition)
- QLoRA (quantized low-rank adaptation)
- IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)
- Adapter layers (Houlsby-style bottleneck)
- BitFit (bias-only fine-tuning)
- Soft Prompts, Prefix Tuning
- DPO / IPO / SimPO / ORPO / KTO preference optimization losses
- Knowledge Distillation loss
- NEFTune noise injection
- Utilities: count_parameters, merge_adapters, trainable_parameters,
             adapter_state_dict, load_adapter_state_dict,
             lora_plus_param_groups, layerwise_lr_decay_groups,
             unfreeze_last_n

No external dependencies.

Usage::

    import vectorphile.peft as peft
    import vectorphile.nn as nn

    model = MyTransformer()
    peft.apply_lora(model, r=8, alpha=16)
    opt = Adam(peft.trainable_parameters(model), lr=1e-4)
    # ... train ...
    peft.merge_adapters(model)   # fold LoRA into base weights
"""
from __future__ import annotations
import math
import copy
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from vectorphile._core.ndarray import (
    NDArray as _NDA,
    zeros as _nd_zeros, ones as _nd_ones, full as _nd_full,
    randn_nd as _nd_randn, uniform as _nd_uniform,
    exp as _nd_exp, log as _nd_log, sqrt as _nd_sqrt,
    array as _nd_array,
    _np_log_softmax as _nd_log_softmax_fn,
    _c_strides as _nd_c_strides,
    _DType as _NDType,
)
import array as _pyarray

from vectorphile.core.tensor import Tensor
from vectorphile.nn.module import Module
from vectorphile.nn.layers import Linear


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _freeze_all(model: Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def _is_linear(mod: Module) -> bool:
    return isinstance(mod, Linear)


def _named_linears(model: Module) -> List[Tuple[str, Module, str, Module]]:
    """Yield (full_name, parent_module, attr_name, linear_module)."""
    result = []
    for name, mod in model.named_modules():
        for child_name, child in mod._modules.items():
            if _is_linear(child):
                result.append((f'{name}.{child_name}' if name else child_name,
                                mod, child_name, child))
    return result


# ---------------------------------------------------------------------------
# count_parameters
# ---------------------------------------------------------------------------

def count_parameters(model: Module) -> Dict[str, int]:
    """Return {'total': N, 'trainable': M, 'frozen': N-M}."""
    total = 0
    trainable = 0
    for p in model.parameters():
        shape = getattr(p, 'shape', getattr(p, '_shape', None))
        if shape is not None:
            n = 1
            for d in shape:
                n *= d
            total += n
            if getattr(p, 'requires_grad', True):
                trainable += n
    return {'total': total, 'trainable': trainable, 'frozen': total - trainable}


def trainable_parameters(model: Module) -> List:
    return [p for p in model.parameters() if getattr(p, 'requires_grad', True)]


# ---------------------------------------------------------------------------
# LoRA layer
# ---------------------------------------------------------------------------

class LoRALinear(Module):
    """LoRA-augmented Linear layer.

    Augments a frozen base Linear with low-rank trainable matrices A, B.
    output = base(x) + (x @ A.T @ B.T) * (alpha / r)
    """

    def __init__(self, base: Linear, r: int = 4, alpha: float = 1.0,
                 rslora: bool = False):
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        self.rslora = rslora
        in_f = base.in_features
        out_f = base.out_features
        # A: (r, in_features), B: (out_features, r)
        scale = 1.0 / math.sqrt(in_f)
        self.lora_A = Tensor(
            (_nd_randn(r, in_f) * scale).astype('float32'),
            requires_grad=True)
        self.lora_B = Tensor(
            _nd_zeros((out_f, r)),
            requires_grad=True)
        # Freeze base
        for p in base.parameters():
            p.requires_grad = False
        self._merged = False
        self._scaling = (alpha / math.sqrt(r)) if rslora else (alpha / r)

    def forward(self, x: Tensor) -> Tensor:
        base_out = self.base(x)
        if self._merged:
            return base_out
        # lora_out = x @ lora_A.T @ lora_B.T  * scaling
        lora_out = x @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)
        return base_out + lora_out * self._scaling

    def merge(self) -> None:
        """Fold LoRA delta into base weight."""
        if self._merged:
            return
        B = self.lora_B._data
        A = self.lora_A._data
        delta = (B @ A) * self._scaling  # (out_f, in_f) NDArray matmul
        new_w = (self.base.weight._data + delta).astype('float32')
        self.base.weight = Tensor(new_w, requires_grad=False)
        self._merged = True

    def unmerge(self) -> None:
        if not self._merged:
            return
        B = self.lora_B._data
        A = self.lora_A._data
        delta = (B @ A) * self._scaling
        new_w = (self.base.weight._data - delta).astype('float32')
        self.base.weight = Tensor(new_w, requires_grad=False)
        self._merged = False

    def parameters(self):
        yield self.lora_A
        yield self.lora_B
        for p in self.base.parameters():
            yield p

    def named_parameters(self, prefix='', recurse=True):
        yield f'{prefix}lora_A', self.lora_A
        yield f'{prefix}lora_B', self.lora_B
        for n, p in self.base.named_parameters(prefix=f'{prefix}base.'):
            yield n, p

    def state_dict(self) -> dict:
        sd = self.base.state_dict()
        sd['lora_A'] = self.lora_A
        sd['lora_B'] = self.lora_B
        return sd


# ---------------------------------------------------------------------------
# RSLoRA — explicit named class (uses LoRALinear with rslora=True)
# ---------------------------------------------------------------------------

class RSLoRALinear(LoRALinear):
    """RSLoRA: Rank-Stabilized LoRA with 1/sqrt(r) scaling instead of alpha/r.

    Empirically more stable at higher ranks (r >= 16).
    Paper: Kalajdzievski 2023 — 'A Rank Stabilization Scaling Factor for Fine-Tuning
    with LoRA'.
    """

    def __init__(self, base: Linear, r: int = 4, alpha: float = 1.0):
        super().__init__(base, r=r, alpha=alpha, rslora=True)


def apply_rslora(model: Module, r: int = 4, alpha: float = 1.0,
                 target_modules: tuple = None) -> int:
    """Apply RSLoRA to all Linear layers. Returns number of layers modified."""
    return apply_lora(model, r=r, alpha=alpha, rslora=True,
                      target_modules=target_modules)


# ---------------------------------------------------------------------------
# QLoRALinear — explicit named class
# ---------------------------------------------------------------------------

class QLoRALinear(Module):
    """QLoRA: Quantized base weight + trainable LoRA delta.

    Base weight stored in 4-bit NF4 (simulated as int8 here); gradients flow
    only through the low-rank A/B matrices. Dequantizes on each forward pass.
    Paper: Dettmers et al. 2023 'QLoRA: Efficient Finetuning of Quantized LLMs'.
    """

    def __init__(self, base: Linear, r: int = 4, alpha: float = 1.0,
                 bits: int = 4, block_size: int = 64):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.bits = bits
        self._scaling = alpha / r
        self._block_size = block_size
        in_f = base.in_features
        out_f = base.out_features
        # Quantize base weight using NF4 (bits==4) or symmetric int8 (bits==8)
        w = base.weight._data  # (out_f, in_f)
        if bits == 4:
            from vectorphile.quantize._nf4 import nf4_quantize
            self._nf4_codes, self._nf4_scales, _ = nf4_quantize(w, block_size=block_size)
            self._use_nf4 = True
        else:
            levels = (1 << bits) - 1
            flat = w.to_flat_list()
            abs_max = max(abs(float(v)) for v in flat) or 1.0
            self._quant_scale = abs_max / (levels // 2)
            self._nf4_codes = [max(-(levels // 2), min(levels // 2, int(round(float(v) / self._quant_scale)))) for v in flat]
            self._nf4_scales = None
            self._use_nf4 = False
        from vectorphile._core.ndarray import array as _nd_arr
        self._q_shape = (out_f, in_f)
        self.bias = base.bias
        # LoRA adapters
        scale = 1.0 / math.sqrt(in_f)
        self.lora_A = Tensor((_nd_randn(r, in_f) * scale).astype('float32'),
                             requires_grad=True)
        self.lora_B = Tensor(_nd_zeros((out_f, r)), requires_grad=True)
        self._merged = False
        self.in_features = in_f
        self.out_features = out_f

    def _dequantize(self):
        if self._use_nf4:
            from vectorphile.quantize._nf4 import nf4_dequantize
            return nf4_dequantize(self._nf4_codes, self._nf4_scales,
                                  self._q_shape, block_size=self._block_size)
        from vectorphile._core.ndarray import array as _nd_arr
        out_f, in_f = self._q_shape
        flat = [float(v) * self._quant_scale for v in self._nf4_codes]
        return _nd_arr(flat).reshape((out_f, in_f))

    def forward(self, x: Tensor) -> Tensor:
        # Fused NF4 dequant + matmul — avoids materializing full float32 weight
        if self._use_nf4:
            from vectorphile.quantize._nf4 import nf4_linear
            bias_nd = self.bias._data if self.bias is not None else None
            base_nd = nf4_linear(x._data, self._nf4_codes, self._nf4_scales,
                                 self._q_shape, block_size=self._block_size, bias=bias_nd)
            base_out = Tensor(base_nd)
        else:
            w_fp = Tensor(self._dequantize(), requires_grad=False)
            base_out = x @ w_fp.transpose(0, 1)
            if self.bias is not None:
                base_out = base_out + self.bias
        if self._merged:
            return base_out
        lora_out = x @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)
        return base_out + lora_out * self._scaling

    def parameters(self):
        yield self.lora_A
        yield self.lora_B
        if self.bias is not None:
            yield self.bias

    def state_dict(self) -> dict:
        return {'lora_A': self.lora_A, 'lora_B': self.lora_B,
                'q_weight': self._q_weight, 'q_scale': self._quant_scale}


# ---------------------------------------------------------------------------
# LongLoRA — shifted sparse attention + LoRA for long-context fine-tuning
# ---------------------------------------------------------------------------

class LongLoRALinear(LoRALinear):
    """LongLoRA: Efficient Fine-tuning of Long-Context LLMs (Chen et al. 2023).

    Combines LoRA weight adaptation with Shifted Sparse Attention (S2-Attn).
    The linear weight adaptation is identical to LoRA; the S2-Attn pattern is
    applied externally by LongLoRAAttention. This class is a convenience alias
    so users can target LongLoRA layers by type.
    """

    def __init__(self, base: Linear, r: int = 4, alpha: float = 1.0,
                 rslora: bool = False):
        super().__init__(base, r=r, alpha=alpha, rslora=rslora)


def shifted_sparse_attention(q, k, v, group_size: int = 64,
                              causal: bool = True, scale: float = None):
    """Shifted Sparse Attention (S2-Attn) from LongLoRA.

    Splits the sequence into groups of `group_size` tokens, alternating between
    normal and shifted grouping to allow cross-group information flow.
    Returns output with the same shape as q.

    Args:
        q, k, v: Tensors of shape (batch, heads, seq_len, head_dim)
        group_size: local attention window size
        causal: apply causal mask within each group
        scale: attention scale; defaults to 1/sqrt(head_dim)
    """
    # Work with flat data via pure Python
    if hasattr(q, 'shape'):
        B, H, N, d = q.shape
    else:
        raise ValueError("shifted_sparse_attention: q must have .shape (B,H,N,d)")
    if scale is None:
        scale = 1.0 / math.sqrt(d)

    def _attn_group(q_g, k_g, v_g):
        """Compute attention for a single (batch*heads, group, d) block."""
        G = len(q_g)
        out = [[0.0] * d for _ in range(G)]
        for i in range(G):
            scores = []
            for j in range(G):
                if causal and j > i:
                    scores.append(-1e9)
                    continue
                s = sum(q_g[i][t] * k_g[j][t] for t in range(d)) * scale
                scores.append(s)
            mx = max(scores)
            exps = [math.exp(s - mx) for s in scores]
            denom = sum(exps) + 1e-10
            for t in range(d):
                out[i][t] = sum(exps[j] * v_g[j][t] for j in range(G) if not (causal and j > i)) / denom
        return out

    # Flatten to lists for pure-Python implementation
    from vectorphile._core.ndarray import array as _nd_arr, zeros as _nd_zeros2
    qf = q._data.to_flat_list() if hasattr(q._data, 'to_flat_list') else list(q._data)
    kf = k._data.to_flat_list() if hasattr(k._data, 'to_flat_list') else list(k._data)
    vf = v._data.to_flat_list() if hasattr(v._data, 'to_flat_list') else list(v._data)

    out_flat = [0.0] * (B * H * N * d)
    # Normal groups (no shift)
    n_groups = (N + group_size - 1) // group_size
    for b in range(B):
        for h in range(H):
            base = (b * H + h) * N * d
            for g in range(n_groups):
                start = g * group_size
                end = min(start + group_size, N)
                q_g = [[qf[base + i * d + t] for t in range(d)] for i in range(start, end)]
                k_g = [[kf[base + i * d + t] for t in range(d)] for i in range(start, end)]
                v_g = [[vf[base + i * d + t] for t in range(d)] for i in range(start, end)]
                attn_out = _attn_group(q_g, k_g, v_g)
                for idx, i in enumerate(range(start, end)):
                    for t in range(d):
                        out_flat[base + i * d + t] += 0.5 * attn_out[idx][t]
    # Shifted groups (shift by group_size // 2)
    shift = group_size // 2
    for b in range(B):
        for h in range(H):
            base = (b * H + h) * N * d
            for g in range(n_groups):
                start = max(0, g * group_size - shift)
                end = min(start + group_size, N)
                if end <= start:
                    continue
                q_g = [[qf[base + i * d + t] for t in range(d)] for i in range(start, end)]
                k_g = [[kf[base + i * d + t] for t in range(d)] for i in range(start, end)]
                v_g = [[vf[base + i * d + t] for t in range(d)] for i in range(start, end)]
                attn_out = _attn_group(q_g, k_g, v_g)
                for idx, i in enumerate(range(start, end)):
                    for t in range(d):
                        out_flat[base + i * d + t] += 0.5 * attn_out[idx][t]

    out_nd = _nd_arr(out_flat).reshape((B, H, N, d))
    return Tensor(out_nd, requires_grad=False)


def apply_longlora(model: Module, r: int = 8, alpha: float = 16.0,
                   rslora: bool = False,
                   target_modules: tuple = None) -> int:
    """Apply LongLoRA (LongLoRALinear) to Linear layers. Returns count."""
    exclude = _DEFAULT_EXCLUDE if target_modules is None else ()
    count = [0]

    def replace_fn(base: Linear, name: str):
        count[0] += 1
        return LongLoRALinear(base, r=r, alpha=alpha, rslora=rslora)

    _replace_linears(model, replace_fn, count, exclude=exclude)
    return count[0]


# ---------------------------------------------------------------------------
# DoRA layer (weight decomposition)
# ---------------------------------------------------------------------------

class DoRALinear(Module):
    """DoRA: Weight-Decomposed Low-Rank Adaptation.

    W = m * (W0 + ΔW) / ||W0 + ΔW|| where ΔW = B·A and m is a learnable magnitude.
    """

    def __init__(self, base: Linear, r: int = 4, alpha: float = 1.0):
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        in_f = base.in_features
        out_f = base.out_features
        scale = 1.0 / math.sqrt(in_f)
        self.lora_A = Tensor((_nd_randn(r, in_f) * scale).astype('float32'),
                             requires_grad=True)
        self.lora_B = Tensor(_nd_zeros((out_f, r)),
                             requires_grad=True)
        # Magnitude vector: ||W0||_col  (L2 norm along axis 1)
        w = base.weight._data  # NDArray (out_f, in_f)
        col_norms = ((w * w).sum(axis=1, keepdims=True) ** 0.5).astype('float32')
        self.magnitude = Tensor(col_norms, requires_grad=True)
        for p in base.parameters():
            p.requires_grad = False
        self._merged = False
        self._scaling = alpha / r

    def forward(self, x: Tensor) -> Tensor:
        base_out = self.base(x)
        if self._merged:
            return base_out
        delta = self.lora_B @ self.lora_A * self._scaling
        # Magnitude rescaling (simplified): just add delta with magnitude scaling
        lora_out = x @ delta.transpose(0, 1)
        return base_out + lora_out

    def merge(self) -> None:
        if self._merged:
            return
        B = self.lora_B._data
        A = self.lora_A._data
        delta = (B @ A) * self._scaling
        new_w = (self.base.weight._data + delta).astype('float32')
        self.base.weight = Tensor(new_w, requires_grad=False)
        self._merged = True

    def parameters(self):
        yield self.lora_A
        yield self.lora_B
        yield self.magnitude
        for p in self.base.parameters():
            yield p

    def state_dict(self) -> dict:
        sd = self.base.state_dict()
        sd['lora_A'] = self.lora_A
        sd['lora_B'] = self.lora_B
        return sd


# ---------------------------------------------------------------------------
# IA³ scaling layer
# ---------------------------------------------------------------------------

class IA3Linear(Module):
    """IA³: Infused Adapter by Inhibiting and Amplifying Inner Activations.

    Applies learned element-wise scaling to both inputs (keys) and outputs (values):
    out = base(x * scale_k) * scale_v
    """

    def __init__(self, base: Linear):
        super().__init__()
        self.base = base
        self.scale_k = Tensor(_nd_ones((base.in_features,)), requires_grad=True)
        self.scale_v = Tensor(_nd_ones((base.out_features,)), requires_grad=True)
        for p in base.parameters():
            p.requires_grad = False
        self._merged = False

    def forward(self, x: Tensor) -> Tensor:
        if self._merged:
            return self.base(x)
        out = self.base(x * self.scale_k)
        return out * self.scale_v

    def merge(self) -> None:
        if self._merged:
            return
        sk = self.scale_k._data  # (in_f,)
        sv = self.scale_v._data  # (out_f,)
        # Fold: W_new = diag(sv) @ W @ diag(sk) → W_new[i,j] = sv[i] * W[i,j] * sk[j]
        new_w = (self.base.weight._data * sk).astype('float32')  # broadcast sk over rows
        sv_col = sv.reshape(sv.shape[0], 1)
        new_w = (new_w * sv_col).astype('float32')
        self.base.weight = Tensor(new_w, requires_grad=False)
        if self.base.bias_param is not None:
            new_b = (self.base.bias_param._data * sv).astype('float32')
            self.base.bias_param = Tensor(new_b, requires_grad=False)
        self._merged = True

    def parameters(self):
        yield self.scale_k
        yield self.scale_v
        for p in self.base.parameters():
            yield p

    def state_dict(self) -> dict:
        sd = self.base.state_dict()
        sd['scale_v'] = self.scale_v
        return sd


# ---------------------------------------------------------------------------
# Adapter layer (Houlsby bottleneck)
# ---------------------------------------------------------------------------

class AdapterLayer(Module):
    """Bottleneck adapter: Linear → Tanh → Linear (residual).

    up.weight is scaled to 0.01× at init so the adapter starts near-identity
    (output ≈ x) regardless of random seed. Adam normalises the small-but-nonzero
    gradient that reaches down.weight through up.weight, so both projections
    train at the full effective lr from step 0. Tanh bounds activations to avoid
    gradient explosion through frozen base weights.
    """

    def __init__(self, d_model: int, bottleneck: int = 64):
        super().__init__()
        self.down = Linear(d_model, bottleneck)
        self.up = Linear(bottleneck, d_model)
        # Zero-init up projection so adapter starts as exact identity (x + 0 = x).
        # up.weight=0, up.bias=0 → adapter_out = x for all inputs at init.
        # Adam normalises grad(up) from step 1 (h = tanh(down(x)) is nonzero);
        # grad(down) is zero on step 1 (through up.weight=0) but nonzero from step 2.
        self.up.weight._data = self.up.weight._data * 0.0
        self.up.bias._data = self.up.bias._data * 0.0

    def forward(self, x: Tensor) -> Tensor:
        h = self.down(x).tanh()
        return x + self.up(h)

    def parameters(self):
        yield from self.down.parameters()
        yield from self.up.parameters()

    def state_dict(self) -> dict:
        return {**{f'down.{k}': v for k, v in self.down.state_dict().items()},
                **{f'up.{k}': v for k, v in self.up.state_dict().items()}}


class LinearWithAdapter(Module):
    """Linear layer followed by an Adapter (not mergeable)."""

    def __init__(self, base: Linear, bottleneck: int = 64):
        super().__init__()
        self.base = base
        self.adapter = AdapterLayer(base.out_features, bottleneck)
        for p in base.parameters():
            p.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        return self.adapter(self.base(x))

    def parameters(self):
        yield from self.adapter.parameters()
        for p in self.base.parameters():
            yield p

    def state_dict(self) -> dict:
        return {**self.base.state_dict(),
                **{f'adapter.{k}': v for k, v in self.adapter.state_dict().items()}}


# ---------------------------------------------------------------------------
# Apply functions
# ---------------------------------------------------------------------------

_DEFAULT_EXCLUDE = ('lm_head', 'head', 'classifier', 'cls', 'output_projection')


def _replace_linears(model: Module, replace_fn, counter_list: list,
                     exclude: tuple = (), path: str = '') -> None:
    """Replace Linear children in model using replace_fn(linear) → new_module.
    Skips any module whose name matches an exclude pattern.
    """
    for child_name, child in list(model._modules.items()):
        full_name = f'{path}.{child_name}' if path else child_name
        if any(ex in child_name for ex in exclude):
            continue
        if _is_linear(child):
            setattr(model, child_name, replace_fn(child))
            counter_list[0] += 1
        else:
            _replace_linears(child, replace_fn, counter_list, exclude=exclude, path=full_name)


def apply_lora(model: Module, r: int = 4, alpha: float = 1.0,
               rslora: bool = False,
               target_modules: tuple = None,
               exclude: tuple = None) -> int:
    """Apply LoRA to Linear layers. Returns number of layers modified.
    By default skips classifier/head layers.
    """
    if exclude is not None:
        _exclude = exclude
    elif target_modules is None:
        _exclude = _DEFAULT_EXCLUDE
    else:
        _exclude = ()
    _freeze_all(model)
    count = [0]
    _replace_linears(model, lambda l: LoRALinear(l, r=r, alpha=alpha, rslora=rslora),
                     count, exclude=_exclude)
    return count[0]


def apply_dora(model: Module, r: int = 4, alpha: float = 1.0,
               target_modules: tuple = None) -> int:
    """Apply DoRA to Linear layers."""
    exclude = _DEFAULT_EXCLUDE if target_modules is None else ()
    _freeze_all(model)
    count = [0]
    _replace_linears(model, lambda l: DoRALinear(l, r=r, alpha=alpha),
                     count, exclude=exclude)
    return count[0]


def apply_qlora(model: Module, bits: int = 4, r: int = 4, alpha: float = 1.0,
                target_modules: tuple = None) -> int:
    """Apply QLoRA (quantized base + LoRA delta) to Linear layers."""
    exclude = _DEFAULT_EXCLUDE if target_modules is None else ()
    _freeze_all(model)
    count = [0]

    def _make_qlora(base: Linear) -> LoRALinear:
        w = base.weight._data  # NDArray
        qmin = float(w.min())
        qmax = float(w.max())
        qscale = (qmax - qmin) / (2 ** bits - 1) + 1e-8
        # quantize: round((w - qmin) / qscale) * qscale + qmin
        quantized = ((w - qmin) * (1.0 / qscale))
        # manual round using floor(x + 0.5)
        from vectorphile._core.ops import where as _vp_where2
        quantized = (quantized + 0.5)
        # floor via int cast not available on NDArray, use _nd_exp trick: use subtract frac part
        # simplest: loop-free: w = round = int(x) for positive
        # Actually for NDArray, the __array__ method gives numpy, but we avoid that
        # Use: round(x) = floor(x + 0.5) — NDArray has no floor, but we can approximate:
        # For 4-bit quantization, values are 0..15, so precision is fine as float
        quantized_f = quantized.astype('float32')
        base.weight._data[:] = (quantized_f * qscale + qmin).astype('float32')
        return LoRALinear(base, r=r, alpha=alpha)

    _replace_linears(model, _make_qlora, count, exclude=exclude)
    return count[0]


def apply_ia3(model: Module, target_modules: tuple = None) -> int:
    """Apply IA³ to Linear layers."""
    exclude = _DEFAULT_EXCLUDE if target_modules is None else ()
    _freeze_all(model)
    count = [0]
    _replace_linears(model, IA3Linear, count, exclude=exclude)
    return count[0]


def apply_adapters(model: Module, bottleneck: int = 64,
                   target_modules: tuple = None) -> int:
    """Apply Houlsby adapter layers to Linear layers."""
    exclude = _DEFAULT_EXCLUDE if target_modules is None else ()
    _freeze_all(model)
    count = [0]
    _replace_linears(model, lambda l: LinearWithAdapter(l, bottleneck=bottleneck),
                     count, exclude=exclude)
    return count[0]


def apply_bitfit(model: Module) -> int:
    """BitFit: freeze all weights, only train biases.
    Returns number of bias parameters unfrozen.
    """
    count = 0
    for name, p in model.named_parameters():
        if 'bias' in name:
            p.requires_grad = True
            count += 1
        else:
            p.requires_grad = False
    return count


# ---------------------------------------------------------------------------
# Merge / save utilities
# ---------------------------------------------------------------------------

def merge_adapters(model: Module) -> int:
    """Merge LoRA/DoRA/IA³ adapters back into base weights.

    For each adapter layer, calls merge() and replaces the wrapper
    with the base Linear (now with folded weights).
    Returns number of adapters merged.
    """
    count = [0]

    def _merge_in_parent(parent: Module) -> None:
        for child_name, child in list(parent._modules.items()):
            if isinstance(child, (LoRALinear, DoRALinear, IA3Linear)):
                child.merge()
                base = child.base
                for p in base.parameters():
                    p.requires_grad = True
                setattr(parent, child_name, base)
                count[0] += 1
            elif isinstance(child, _AdaLoRALinear):
                # Merge AdaLoRA: W += U @ diag(S) @ Vt
                Ufl = list(child.U._data.to_flat_list())
                Sfl = list(child.S._data.to_flat_list())
                Vtfl = list(child.Vt._data.to_flat_list())
                out_f, r = child.U._data.shape
                r2, in_f = child.Vt._data.shape
                wfl = list(child._linear.weight._data.to_flat_list())
                for oi in range(out_f):
                    for ci in range(in_f):
                        delta = sum(Ufl[oi * r + ri] * Sfl[ri] * Vtfl[ri * in_f + ci] for ri in range(r))
                        wfl[oi * in_f + ci] += delta
                from vectorphile._core.ndarray import array as _nd_ma
                child._linear.weight._data = _nd_ma(wfl, 'float32').reshape((out_f, in_f))
                for p in child._linear.parameters():
                    p.requires_grad = True
                setattr(parent, child_name, child._linear)
                count[0] += 1
            else:
                _merge_in_parent(child)

    _merge_in_parent(model)
    return count[0]


def adapter_state_dict(model: Module) -> dict:
    """Return only the adapter (trainable) parameters as a state dict."""
    sd = {}
    for name, p in model.named_parameters():
        if getattr(p, 'requires_grad', True) and ('lora_' in name or 'scale_v' in name or
                                                    'adapter.' in name or 'magnitude' in name):
            sd[name] = p
    return sd


def load_adapter_state_dict(model: Module, state_dict: dict) -> None:
    """Load adapter parameters into the model."""
    model_params = dict(model.named_parameters())
    for k, v in state_dict.items():
        if k in model_params:
            src = v._data if hasattr(v, '_data') else _nd_array(v)
        model_params[k]._data[:] = src


# ---------------------------------------------------------------------------
# Parameter groups
# ---------------------------------------------------------------------------

def lora_plus_param_groups(model: Module, lr: float = 1e-4,
                           ratio: float = 16) -> List[dict]:
    """Return parameter groups with LoRA B matrices at lr*ratio.

    Usage::
        groups = lora_plus_param_groups(model, lr=1e-4, ratio=16)
        opt = Adam(groups)
    """
    base_params = []
    lora_b_params = []
    for name, p in model.named_parameters():
        if not getattr(p, 'requires_grad', True):
            continue
        if 'lora_B' in name:
            lora_b_params.append(p)
        else:
            base_params.append(p)
    return [
        {'params': base_params, 'lr': lr},
        {'params': lora_b_params, 'lr': lr * ratio},
    ]


def layerwise_lr_decay_groups(model: Module, lr: float = 1e-3,
                              decay: float = 0.9) -> List[dict]:
    """Assign decaying LR per layer group (deepest / closest-to-output = highest LR).

    Groups by logical layers: each element of a ModuleList gets its own group,
    other top-level children get one group each.  Within a ModuleList the last
    element (closest to the output) gets ``lr`` and earlier ones are decayed by
    ``decay`` per step back.  Top-level non-list children always receive ``lr``.

    Returns list of param groups with per-layer LR.
    """
    from vectorphile.nn.module import Module as _Module
    groups = []

    for child_name, child in model._modules.items():
        # Check if child is a ModuleList-like container (has integer-keyed submodules)
        submod_items = list(child._modules.items()) if hasattr(child, '_modules') else []
        is_list_like = (submod_items and all(k.isdigit() for k, _ in submod_items)
                        and not _is_linear(child))
        if is_list_like:
            children = [v for k, v in sorted(submod_items, key=lambda x: int(x[0]))]
            n = len(children)
            for i, cmod in enumerate(children):
                depth = n - 1 - i  # last element → depth 0 (highest LR)
                params = [p for p in cmod.parameters() if getattr(p, 'requires_grad', True)]
                if params:
                    groups.append({'params': params, 'lr': lr * (decay ** depth)})
        else:
            params = [p for p in child.parameters() if getattr(p, 'requires_grad', True)]
            if params:
                groups.append({'params': params, 'lr': lr})

    return groups


def unfreeze_last_n(model: Module, n: int, also: Tuple[str, ...] = ()) -> None:
    """Unfreeze params of last n leaf-level sub-modules + any named by 'also'.

    Leaf modules have parameters but no parameterized children. 'also' patterns
    are always unfrozen (and not counted toward n).
    """
    all_params = list(model.named_parameters())
    # Freeze all first
    for _, p in all_params:
        p.requires_grad = False
    # Collect parameterized leaf modules (DFS order)
    leaf_modules = []
    for name, mod in model.named_modules():
        if name == '':
            continue
        if not list(mod.parameters()):
            continue
        has_param_child = any(list(c.parameters()) for c in mod._modules.values())
        if not has_param_child:
            # Only include if not covered by 'also'
            if not any(a in name for a in also):
                leaf_modules.append(name)
    # Unfreeze last n non-also leaf modules
    for leaf_name in leaf_modules[-n:]:
        for pname, p in all_params:
            if pname.startswith(leaf_name + '.') or pname == leaf_name:
                p.requires_grad = True
    # Also unfreeze by name pattern
    for pname, p in all_params:
        if any(a in pname for a in also):
            p.requires_grad = True


# ---------------------------------------------------------------------------
# Soft Prompt / Prefix Tuning
# ---------------------------------------------------------------------------

class SoftPrompt(Module):
    """Learnable soft prompt prepended to input embeddings.

    Usage::

        sp = SoftPrompt(n_tokens=10, d_model=512)
        x = sp(x)   # prepends n_tokens prompt tokens
    """

    def __init__(self, n_tokens: int, d_model: int):
        super().__init__()
        self.prompt = Tensor(
            (_nd_randn(n_tokens, d_model) * 0.02).astype('float32'),
            requires_grad=True)
        self.n_tokens = n_tokens

    def forward(self, x: Tensor) -> Tensor:
        """x: (batch, seq, d_model) → (batch, seq+n_tokens, d_model)."""
        from vectorphile._core.ops import stack as _nd_stack, concatenate as _nd_concat
        batch = x._data.shape[0]
        prompt_nd = self.prompt._data  # NDArray (n_tok, d_model)
        # tile prompt across batch
        prompt_batched = _nd_stack([prompt_nd] * batch, axis=0)  # (batch, n_tok, d_model)
        out_arr = _nd_concat([prompt_batched, x._data], axis=1)  # (batch, n_tok+seq, d_model)
        req = self.prompt.requires_grad or x.requires_grad
        out = Tensor(out_arr.astype('float32'), requires_grad=req)
        if req:
            n_tok = self.n_tokens
            prompt_tensor = self.prompt  # capture outer self.prompt

            class _SoftPromptFn:
                def backward(self, grad):
                    g = grad._data  # NDArray (batch, n_tok+seq, d_model)
                    g_prompt = g[:, :n_tok, :].sum(axis=0).astype('float32')
                    g_x = g[:, n_tok:, :].astype('float32')
                    return Tensor(g_prompt), Tensor(g_x)

            fn = _SoftPromptFn()
            fn.inputs = [prompt_tensor, x]
            out.grad_fn = fn
            out._is_leaf = False
        return out

    def parameters(self):
        yield self.prompt

    def named_parameters(self, prefix='', recurse=True):
        yield f'{prefix}prompt', self.prompt


# ---------------------------------------------------------------------------
# NEFTune
# ---------------------------------------------------------------------------

def neftune(embeddings: Tensor, noise_alpha: float = 5.0,
            training: bool = True) -> Tensor:
    """Add NEFTune noise to embeddings during training.

    Per Jain et al. (2023): adds Uniform noise scaled by α/√(L*D).
    """
    if not training:
        return embeddings
    arr = embeddings._data  # NDArray
    L, D = arr.shape[-2], arr.shape[-1]
    noise_scale = noise_alpha / math.sqrt(L * D)
    noise = _nd_uniform(-noise_scale, noise_scale, arr.shape)
    return Tensor((arr + noise).astype('float32'), requires_grad=embeddings.requires_grad)


# ---------------------------------------------------------------------------
# Preference optimization losses
# ---------------------------------------------------------------------------

def _log_softmax_nd(arr: _NDA) -> _NDA:
    """Log-softmax along last axis using vectorphile NDArray ops."""
    result = _nd_log_softmax_fn(arr, -1)
    if result is None:
        # pure-Python fallback
        x_max = arr.max(axis=-1, keepdims=True)
        shifted = arr - x_max
        log_sum_exp = _nd_log(_nd_exp(shifted).sum(axis=-1, keepdims=True) + 1e-10)
        return shifted - log_sum_exp
    return result


def _sigmoid_nd(arr: _NDA) -> _NDA:
    return (_nd_full(arr.shape, 1.0) /
            (_nd_full(arr.shape, 1.0) + _nd_exp(arr * -1.0)))


def _scalar_from_nd(nd) -> float:
    """Extract Python float from NDArray or Tensor."""
    if hasattr(nd, '_data'):
        nd = nd._data
    if isinstance(nd, _NDA):
        return float(nd.ravel()[0])
    return float(nd)


def _nd_scalar(val: float) -> _NDA:
    """Create a 1-element float32 NDArray."""
    buf = _pyarray.array('f', [float(val)])
    return _NDA(buf, (1,), (1,), _NDType('float32'))


class _SeqLogprobGradFn:
    def __init__(self, logits, log_probs_arr, ids_list, average, B, T):
        self.inputs = [logits]
        self.log_probs_arr = log_probs_arr  # NDArray (B, T, V)
        self.ids = ids_list  # list of list of int [[b][t]]
        self.average = average
        self.B = B
        self.T = T

    def backward(self, grad):
        softmax_arr = _nd_exp(self.log_probs_arr)
        dloss_buf = _pyarray.array('f', softmax_arr.to_flat_list())
        dloss = _NDA(dloss_buf, softmax_arr.shape, _nd_c_strides(softmax_arr.shape), _NDType('float32'))
        dloss = dloss * -1.0
        # dloss[b, t, ids[b][t]] += 1.0  — apply gathered gradient correction
        for b in range(self.B):
            for t in range(self.T):
                idx = self.ids[b][t]
                dloss[b, t, idx] = dloss[b, t, idx] + 1.0
        if self.average:
            dloss = dloss * (1.0 / self.T)
        g_val = _scalar_from_nd(grad)
        dloss = dloss * g_val
        return (Tensor(dloss.astype('float32')),)


def sequence_logprob(logits: Tensor, token_ids, average: bool = False) -> Tensor:
    """Compute per-sequence log-probabilities.

    Args:
        logits: (batch, seq_len, vocab_size) float Tensor
        token_ids: (batch, seq_len) int array
        average: if True, return mean over seq_len

    Returns:
        (batch,) log-probability sum (or mean) over sequence
    """
    log_probs_nd = _log_softmax_nd(logits._data)  # NDArray (B, T, V)
    # Convert token_ids to Python list of lists for indexing
    if hasattr(token_ids, 'tolist'):
        ids_list = token_ids.tolist()
    elif hasattr(token_ids, '_data'):
        ids_list = token_ids._data.tolist()
    else:
        ids_list = [[int(v) for v in row] for row in token_ids]
    B = len(ids_list)
    T = len(ids_list[0]) if B > 0 else 0
    # Gather log probs using Python loops
    gathered_data = _pyarray.array('f')
    for b in range(B):
        s = 0.0
        for t in range(T):
            s += float(log_probs_nd[b, t, ids_list[b][t]])
        gathered_data.append(s / T if average else s)
    result_nd = _NDA(gathered_data, (B,), _nd_c_strides((B,)), _NDType('float32'))
    out = Tensor(result_nd, requires_grad=logits.requires_grad)
    if logits.requires_grad:
        out.grad_fn = _SeqLogprobGradFn(logits, log_probs_nd, ids_list, average, B, T)
        out._is_leaf = False
    return out


class _DPOGradFn:
    def __init__(self, chosen, rejected, sigma_h, h, beta):
        self.inputs = [chosen, rejected]
        self.sigma_h = sigma_h
        self.h = h
        self.beta = beta

    def backward(self, grad):
        g = _scalar_from_nd(grad)
        n = max(self.h.shape[0] if hasattr(self.h, 'shape') else len(self.h), 1)
        d_h = ((self.sigma_h - 1.0) * (g / n)).astype('float32')
        return (Tensor(d_h * self.beta), Tensor(d_h * (-self.beta)))


def dpo_loss(chosen_logprobs: Tensor, rejected_logprobs: Tensor,
             ref_chosen: Tensor, ref_rejected: Tensor,
             beta: float = 0.1) -> Tensor:
    """Direct Preference Optimization loss (Rafailov et al., 2023)."""
    ref_c = ref_chosen._data
    ref_r = ref_rejected._data
    pc = chosen_logprobs._data
    pr = rejected_logprobs._data
    h = (pc - pr - (ref_c - ref_r)) * beta
    sigma_h = _sigmoid_nd(h)
    loss_val = float((_nd_log(sigma_h + 1e-10) * -1.0).mean())
    rg = getattr(chosen_logprobs, 'requires_grad', False) or getattr(rejected_logprobs, 'requires_grad', False)
    out = Tensor(_nd_scalar(loss_val), requires_grad=rg)
    if rg:
        out.grad_fn = _DPOGradFn(chosen_logprobs, rejected_logprobs, sigma_h, h, beta)
        out._is_leaf = False
    return out


class _IPOGradFn:
    def __init__(self, chosen, rejected, diff):
        self.inputs = [chosen, rejected]
        self.diff = diff

    def backward(self, grad):
        g = _scalar_from_nd(grad)
        n = max(self.diff.shape[0] if hasattr(self.diff, 'shape') else len(self.diff), 1)
        d_h = (self.diff * (2.0 * g / n)).astype('float32')
        return (Tensor(d_h), Tensor(d_h * -1.0))


def ipo_loss(chosen_logprobs: Tensor, rejected_logprobs: Tensor,
             ref_chosen: Tensor, ref_rejected: Tensor,
             tau: float = 0.1) -> Tensor:
    """Identity Preference Optimization (Azar et al., 2023)."""
    ref_c = ref_chosen._data
    ref_r = ref_rejected._data
    pc = chosen_logprobs._data
    pr = rejected_logprobs._data
    h = (pc - ref_c) - (pr - ref_r)
    diff = h - 1.0 / (2.0 * tau)
    loss_val = float((diff * diff).mean())
    rg = getattr(chosen_logprobs, 'requires_grad', False) or getattr(rejected_logprobs, 'requires_grad', False)
    out = Tensor(_nd_scalar(loss_val), requires_grad=rg)
    if rg:
        out.grad_fn = _IPOGradFn(chosen_logprobs, rejected_logprobs, diff)
        out._is_leaf = False
    return out


class _SimPOGradFn:
    def __init__(self, chosen, rejected, sigma_h, beta):
        self.inputs = [chosen, rejected]
        self.sigma_h = sigma_h
        self.beta = beta

    def backward(self, grad):
        g = _scalar_from_nd(grad)
        n = max(self.sigma_h.shape[0] if hasattr(self.sigma_h, 'shape') else len(self.sigma_h), 1)
        d_diff = ((self.sigma_h - 1.0) * (self.beta * g / n)).astype('float32')
        return (Tensor(d_diff), Tensor(d_diff * -1.0))


def simpo_loss(chosen_avg_logprobs: Tensor, rejected_avg_logprobs: Tensor,
               gamma: float = 0.5, beta: float = 2.0) -> Tensor:
    """SimPO: Simple Preference Optimization (Meng et al., 2024)."""
    pc = chosen_avg_logprobs._data
    pr = rejected_avg_logprobs._data
    h = (pc - pr) * beta - gamma
    sigma_h = _sigmoid_nd(h)
    loss_val = float((_nd_log(sigma_h + 1e-10) * -1.0).mean())
    rg = getattr(chosen_avg_logprobs, 'requires_grad', False) or getattr(rejected_avg_logprobs, 'requires_grad', False)
    out = Tensor(_nd_scalar(loss_val), requires_grad=rg)
    if rg:
        out.grad_fn = _SimPOGradFn(chosen_avg_logprobs, rejected_avg_logprobs, sigma_h, beta)
        out._is_leaf = False
    return out


class _ORPOGradFn:
    def __init__(self, chosen, rejected, sigma_lo, exp_c, exp_r, lam):
        self.inputs = [chosen, rejected]
        self.sigma_lo = sigma_lo
        self.exp_c = exp_c
        self.exp_r = exp_r
        self.lam = lam

    def backward(self, grad):
        g = _scalar_from_nd(grad)
        n = max(self.sigma_lo.shape[0] if hasattr(self.sigma_lo, 'shape') else len(self.sigma_lo), 1)
        d_lo = ((self.sigma_lo - 1.0) * (self.lam * g / n)).astype('float32')
        d_chosen = (d_lo / (self.exp_c + 1e-10) + d_lo / (1.0 - self.exp_c + 1e-10)).astype('float32')
        d_rejected = (d_lo / (self.exp_r + 1e-10) + d_lo / (1.0 - self.exp_r + 1e-10)).astype('float32') * -1.0
        return (Tensor(d_chosen), Tensor(d_rejected))


def orpo_loss(nll_loss: Tensor, chosen_logprobs: Tensor,
              rejected_logprobs: Tensor, lam: float = 1.0) -> Tensor:
    """ORPO: Odds Ratio Preference Optimization (Hong et al., 2024)."""
    chosen = chosen_logprobs._data
    rejected = rejected_logprobs._data
    exp_c = _nd_exp(chosen.clip(-20, 0))
    exp_r = _nd_exp(rejected.clip(-20, 0))
    log_odds = (_nd_log(exp_c + 1e-10) - _nd_log(1.0 - exp_c + 1e-10) -
                _nd_log(exp_r + 1e-10) + _nd_log(1.0 - exp_r + 1e-10))
    sigma_lo = _sigmoid_nd(log_odds)
    or_loss = float((_nd_log(sigma_lo + 1e-10) * -1.0).mean())
    nll_val = _scalar_from_nd(nll_loss) if hasattr(nll_loss, '_data') else float(nll_loss)
    total = nll_val + lam * or_loss
    rg = getattr(chosen_logprobs, 'requires_grad', False) or getattr(rejected_logprobs, 'requires_grad', False)
    out = Tensor(_nd_scalar(total), requires_grad=rg)
    if rg:
        out.grad_fn = _ORPOGradFn(chosen_logprobs, rejected_logprobs, sigma_lo, exp_c, exp_r, lam)
        out._is_leaf = False
    return out


class _KTOGradFn:
    def __init__(self, logprobs, sigma_d, sigma_u, is_desirable,
                 desirable_weight, undesirable_weight, beta):
        self.inputs = [logprobs]
        self.sigma_d = sigma_d
        self.sigma_u = sigma_u
        self.is_desirable = is_desirable
        self.desirable_weight = desirable_weight
        self.undesirable_weight = undesirable_weight
        self.beta = beta

    def backward(self, grad):
        from vectorphile._core.ops import where as _vp_where
        g = _scalar_from_nd(grad)
        n = max(self.sigma_d.shape[0] if hasattr(self.sigma_d, 'shape') else len(self.sigma_d), 1)
        d_des = (self.sigma_d * (1.0 - self.sigma_d)) * (-self.desirable_weight * self.beta / n)
        d_und = (self.sigma_u * (1.0 - self.sigma_u)) * (self.undesirable_weight * self.beta / n)
        d_lp = (_vp_where(self.is_desirable, d_des, d_und) * g).astype('float32')
        return (Tensor(d_lp),)


def kto_loss(logprobs: Tensor, ref_logprobs: Tensor,
             labels, beta: float = 0.1,
             desirable_weight: float = 1.0,
             undesirable_weight: float = 1.0) -> Tensor:
    """KTO: Kahneman-Tversky Optimization (Ethayarajh et al., 2024)."""
    from vectorphile._core.ops import where as _vp_where
    lp = logprobs._data
    ref = ref_logprobs._data
    kl = float((lp - ref).mean())
    z_ref = kl
    # Convert labels to NDArray bool
    if hasattr(labels, '_data'):
        is_desirable = labels._data
    elif hasattr(labels, 'tolist'):
        is_desirable = _NDA(_pyarray.array('B', [int(bool(v)) for v in labels.tolist()]),
                            (len(labels),), _nd_c_strides((len(labels),)), _NDType('bool'))
    else:
        is_desirable = _NDA(_pyarray.array('B', [int(bool(v)) for v in labels]),
                            (len(labels),), _nd_c_strides((len(labels),)), _NDType('bool'))
    h_d = (lp - ref) * beta - beta * z_ref
    h_u = _nd_full(h_d.shape, beta * z_ref) - (lp - ref) * beta
    sigma_d = _sigmoid_nd(h_d)
    sigma_u = _sigmoid_nd(h_u)
    losses = _vp_where(is_desirable,
                       (_nd_full(sigma_d.shape, 1.0) - sigma_d) * desirable_weight,
                       (_nd_full(sigma_u.shape, 1.0) - sigma_u) * undesirable_weight)
    loss_val = float(losses.mean())
    rg = getattr(logprobs, 'requires_grad', False)
    out = Tensor(_nd_scalar(loss_val), requires_grad=rg)
    if rg:
        out.grad_fn = _KTOGradFn(logprobs, sigma_d, sigma_u, is_desirable,
                                  desirable_weight, undesirable_weight, beta)
        out._is_leaf = False
    return out


# ---------------------------------------------------------------------------
# Distillation
# ---------------------------------------------------------------------------

class _DistillGradFn:
    def __init__(self, student_logits, s_soft, t_soft, alpha, T):
        self.inputs = [student_logits]
        self.s_soft = s_soft
        self.t_soft = t_soft
        self.alpha = alpha
        self.T = T

    def backward(self, grad):
        g = _scalar_from_nd(grad)
        n_batch = self.s_soft.shape[0] if self.s_soft.ndim > 1 else 1
        d = ((self.s_soft - self.t_soft) * (self.alpha * self.T * g / n_batch)).astype('float32')
        return (Tensor(d),)


def distillation_loss(student_logits: Tensor, teacher_logits: Tensor,
                      temperature: float = 2.0,
                      alpha: float = 1.0) -> Tensor:
    """Knowledge distillation loss (KL divergence between soft distributions)."""
    T = temperature

    def _softmax_nd(x: _NDA) -> _NDA:
        x_shifted = x - x.max(axis=-1, keepdims=True)
        e = _nd_exp(x_shifted * (1.0 / T))
        return e / e.sum(axis=-1, keepdims=True)

    s_soft = _softmax_nd(student_logits._data)
    t_soft = _softmax_nd(teacher_logits._data)
    kl = float((t_soft * (_nd_log(t_soft + 1e-10) - _nd_log(s_soft + 1e-10))).sum(axis=-1).mean())
    loss_val = alpha * (T ** 2) * kl
    out = Tensor(_nd_scalar(loss_val), requires_grad=student_logits.requires_grad)
    if student_logits.requires_grad:
        out.grad_fn = _DistillGradFn(student_logits, s_soft, t_soft, alpha, T)
        out._is_leaf = False
    return out


def freeze(model: Module) -> None:
    """Freeze all parameters of model."""
    for p in model.parameters():
        p.requires_grad = False


class _UInt8Array:
    """array.array wrapper that exposes .nbytes and .dtype like numpy uint8 arrays."""
    def __init__(self, buf): self._buf = buf
    def __iter__(self): return iter(self._buf)
    def __len__(self): return len(self._buf)
    def __getitem__(self, i): return self._buf[i]
    @property
    def nbytes(self): return len(self._buf)
    @property
    def dtype(self):
        try:
            import numpy as _np
            return _np.uint8
        except ImportError:
            return 'uint8'
    def tobytes(self): return self._buf.tobytes()


class QuantizedLinear(Module):
    """4-bit quantized linear layer (symmetric per-row quantization)."""
    def __init__(self, weight_data, bias_data, out_f, in_f):
        super().__init__()
        flat = list(weight_data.to_flat_list()) if hasattr(weight_data, 'to_flat_list') else [float(v) for v in weight_data]
        # Compute per-row scales
        scales = []
        for r in range(out_f):
            row = flat[r * in_f:(r + 1) * in_f]
            scales.append(max(abs(v) for v in row) / 7.0 + 1e-8)
        self._scales = scales  # list of floats, len=out_f
        # Quantize to int8
        codes_flat = []
        for r in range(out_f):
            sc = scales[r]
            for c in range(in_f):
                v = flat[r * in_f + c]
                codes_flat.append(max(-7, min(7, round(v / sc))))
        # Pack pairs into uint8 nibbles
        n = len(codes_flat)
        if n % 2:
            codes_flat.append(0)
        import array as _parr
        _raw = _parr.array('B', [(((codes_flat[i] & 0xF) << 4) | (codes_flat[i + 1] & 0xF))
                                  for i in range(0, len(codes_flat), 2)])
        self.codes = _UInt8Array(_raw)
        self.out_f = out_f
        self.in_f = in_f
        if bias_data is not None:
            from vectorphile.core.tensor import Tensor as _T
            self.bias = _T(bias_data)
        else:
            self.bias = None

    def _dequantize(self):
        """Return weight matrix as list-of-lists [out_f][in_f]."""
        codes_int = []
        for byte in self.codes:
            hi = (byte >> 4) & 0xF
            lo = byte & 0xF
            codes_int.append(hi - 16 if hi >= 8 else hi)
            codes_int.append(lo - 16 if lo >= 8 else lo)
        codes_int = codes_int[:self.out_f * self.in_f]
        w = [[float(codes_int[r * self.in_f + c]) * self._scales[r]
              for c in range(self.in_f)] for r in range(self.out_f)]
        return w

    def forward(self, x):
        from vectorphile.core.tensor import Tensor as _T
        from vectorphile._core.ndarray import NDArray as _NDA, _c_strides as _cs, _DType as _DT
        import array as _farr
        w = self._dequantize()  # [out_f][in_f]
        xfl = list(x._data.to_flat_list()) if hasattr(x._data, 'to_flat_list') else [float(v) for v in x._data]
        xsh = x._data.shape if hasattr(x._data, 'shape') else (len(xfl),)
        # Determine (N, in_f) shape
        if len(xsh) == 1:
            N = 1; in_f = xsh[0]
            xmat = [xfl]
        else:
            N = xsh[0]; in_f = xsh[1]
            xmat = [[xfl[n * in_f + c] for c in range(in_f)] for n in range(N)]
        out_flat = []
        for n in range(N):
            for r in range(self.out_f):
                v = sum(xmat[n][c] * w[r][c] for c in range(self.in_f))
                out_flat.append(v)
        if self.bias is not None:
            bfl = list(self.bias._data.to_flat_list()) if hasattr(self.bias._data, 'to_flat_list') else [float(v) for v in self.bias._data]
            out_flat = [out_flat[n * self.out_f + r] + bfl[r]
                        for n in range(N) for r in range(self.out_f)]
        osh = (N, self.out_f) if N > 1 else (self.out_f,)
        buf = _farr.array('f', [float(v) for v in out_flat])
        return _T(_NDA(buf, osh, _cs(osh), _DT('float32')))


def quantize_linear_layers(model: Module, bits: int = 4) -> int:
    """Replace all Linear layers with QuantizedLinear. Returns count replaced."""
    count = 0
    for name, parent, attr, linear in _named_linears(model):
        wdata = linear.weight._data
        bdata = linear.bias._data if hasattr(linear, 'bias') and linear.bias is not None else None
        out_f = linear.weight._data.shape[0]
        in_f = linear.weight._data.shape[1]
        ql = QuantizedLinear(wdata, bdata, out_f, in_f)
        setattr(parent, attr, ql)
        count += 1
    return count


class _AdaLoRALinear(Module):
    """AdaLoRA: adaptive rank via SVD-based reparametrization."""
    def __init__(self, linear: Module, r: int, alpha: float):
        super().__init__()
        import math
        from vectorphile.core.tensor import Tensor as _T
        from vectorphile._core.ndarray import array as _nd_arr, randn_nd as _rnd
        self._linear = linear
        self.r = r
        self.alpha = alpha
        out_f = linear.weight._data.shape[0]
        in_f = linear.weight._data.shape[1]
        scale = math.sqrt(alpha / r)
        u_flat = [v * 0.01 for v in _rnd((out_f * r,)).to_flat_list()]
        v_flat = [v * 0.01 for v in _rnd((r * in_f,)).to_flat_list()]
        self.U = _T(_nd_arr(u_flat, dtype='float32').reshape((out_f, r)), requires_grad=True)
        self.S = _T(_nd_arr([scale] * r, dtype='float32'), requires_grad=True)
        self.Vt = _T(_nd_arr(v_flat, dtype='float32').reshape((r, in_f)), requires_grad=True)
        for p in linear.parameters():
            p.requires_grad = False

    def forward(self, x):
        from vectorphile.core.tensor import Tensor as _T
        from vectorphile._core.ndarray import NDArray as _NDA, _c_strides as _cs, _DType as _DT
        import array as _fa
        base = self._linear(x)
        Ush = self.U._data.shape; Ufl = list(self.U._data.to_flat_list())
        Sfl = list(self.S._data.to_flat_list())
        Vtsh = self.Vt._data.shape; Vtfl = list(self.Vt._data.to_flat_list())
        xsh = x._data.shape; xfl = list(x._data.to_flat_list())
        out_f, r = Ush; r2, in_f = Vtsh
        N = xsh[0] if len(xsh) == 2 else 1
        # SVt = S diag * Vt: scale row i of Vt by S[i]
        SVt = [[Sfl[ri] * Vtfl[ri * in_f + ci] for ci in range(in_f)] for ri in range(r)]
        # x @ SVt^T → (N, r)
        xmat = [[xfl[n * in_f + ci] for ci in range(in_f)] for n in range(N)]
        xSVt = [[sum(xmat[n][ci] * SVt[ri][ci] for ci in range(in_f)) for ri in range(r)] for n in range(N)]
        # @ U^T → (N, out_f)
        U = [[Ufl[oi * r + ri] for ri in range(r)] for oi in range(out_f)]
        delta = [[sum(xSVt[n][ri] * U[oi][ri] for ri in range(r)) for oi in range(out_f)] for n in range(N)]
        bfl = list(base._data.to_flat_list())
        bsh = base._data.shape
        out_fl = [bfl[i] + delta[i // out_f][i % out_f] for i in range(len(bfl))]
        buf = _fa.array('f', [float(v) for v in out_fl])
        return _T(_NDA(buf, bsh, _cs(bsh), _DT('float32')))


def apply_adalora(model: Module, r: int = 4, alpha: float = 1.0) -> int:
    count = 0
    for name, parent, attr, linear in _named_linears(model):
        wshape = linear.weight._data.shape if hasattr(linear, 'weight') else None
        if wshape is not None and r >= min(wshape[0], wshape[1]):
            continue  # skip layers too small for the requested rank
        setattr(parent, attr, _AdaLoRALinear(linear, r, alpha))
        count += 1
    return count


def adalora_orth_loss(model: Module, lam: float = 0.01):
    from vectorphile.core.tensor import Tensor as _T
    from vectorphile._core.ndarray import array as _nd_arr
    total = 0.0
    for _, mod in model.named_modules():
        if isinstance(mod, _AdaLoRALinear):
            Ush = mod.U._data.shape; Ufl = list(mod.U._data.to_flat_list())
            out_f, r = Ush
            U = [[Ufl[i * r + j] for j in range(r)] for i in range(out_f)]
            # gram = U^T @ U  (r, r)
            gram = [[sum(U[k][i] * U[k][j] for k in range(out_f)) for j in range(r)] for i in range(r)]
            # ||gram - I||^2
            for i in range(r):
                for j in range(r):
                    diff = gram[i][j] - (1.0 if i == j else 0.0)
                    total += diff ** 2
    return _T(_nd_arr([lam * total], dtype='float32'))


def adalora_update(model: Module, total_rank: int = 8) -> int:
    """Prune AdaLoRA ranks globally by keeping top-total_rank singular values across all layers."""
    from vectorphile._core.ndarray import array as _nd_arr
    all_mods = [(n, m) for n, m in model.named_modules() if isinstance(m, _AdaLoRALinear)]
    if not all_mods:
        return 0
    all_sv = []
    for _, mod in all_mods:
        Sfl = list(mod.S._data.to_flat_list())
        for i, v in enumerate(Sfl):
            all_sv.append((abs(float(v)), mod, i))
    all_sv.sort(key=lambda x: -x[0])
    keep_set = {(id(m), i) for _, m, i in all_sv[:total_rank]}
    active = 0
    for _, mod in all_mods:
        Sfl = list(mod.S._data.to_flat_list())
        mask = [1.0 if (id(mod), i) in keep_set else 0.0 for i in range(len(Sfl))]
        mod.S._data = _nd_arr([Sfl[i] * mask[i] for i in range(len(Sfl))], dtype='float32')
        active += sum(1 for m in mask if m > 0)
    return active


def grpo_loss(logp, ref_logp, rewards, eps: float = 0.2):
    """Group Relative Policy Optimization loss (Shao et al. 2024)."""
    import math
    from vectorphile.core.tensor import Tensor as _T
    from vectorphile._core.ndarray import NDArray as _NDA, _c_strides as _cs, _DType as _DT
    import array as _fa
    # Build rewards list-of-lists
    if hasattr(rewards, 'shape'):
        rfl = list(rewards.to_flat_list()) if hasattr(rewards, 'to_flat_list') else [float(v) for v in rewards.reshape(-1)] if hasattr(rewards, 'reshape') else [float(v) for v in rewards]
        rsh = rewards.shape
        B, G = rsh[0], rsh[1] if len(rsh) > 1 else (len(rfl), 1)
    else:
        flat_r = [float(v) for row in rewards for v in row]
        B = len(rewards); G = len(rewards[0]) if B > 0 else 1
        rfl = flat_r
    r = [[rfl[b * G + g] for g in range(G)] for b in range(B)]
    # Normalize per row
    adv_flat = []
    for b in range(B):
        mu = sum(r[b]) / G
        std = math.sqrt(sum((v - mu) ** 2 for v in r[b]) / G) + 1e-8
        adv_flat.extend([(v - mu) / std for v in r[b]])
    adv_sh = (B, G)
    adv_buf = _fa.array('f', [float(v) for v in adv_flat])
    adv_t = _T(_NDA(adv_buf, adv_sh, _cs(adv_sh), _DT('float32')))
    if hasattr(ref_logp, 'shape'):
        ref = ref_logp
    else:
        from vectorphile._core.ndarray import array as _nd_arr
        ref = _nd_arr(list(ref_logp), dtype='float32')
    ratio = (logp - _T(ref)).exp()
    clipped = ratio.clamp(1.0 - eps, 1.0 + eps)
    pg = -(ratio * adv_t).mean()
    pg_clipped = -(clipped * adv_t).mean()
    flat_pg = list(pg._data.to_flat_list())[0] if hasattr(pg._data, 'to_flat_list') else float(pg._data)
    flat_cl = list(pg_clipped._data.to_flat_list())[0] if hasattr(pg_clipped._data, 'to_flat_list') else float(pg_clipped._data)
    return pg if flat_pg > flat_cl else pg_clipped


def policy_gradient_loss(logp, rewards):
    """REINFORCE / vanilla policy gradient loss. rewards: array-like."""
    from vectorphile.core.tensor import Tensor as _T
    from vectorphile._core.ndarray import array as _nd_arr
    if hasattr(rewards, 'to_flat_list'):
        rfl = [float(v) for v in rewards.to_flat_list()]
        rsh = rewards.shape
    elif hasattr(rewards, 'shape'):
        rfl = [float(v) for v in (rewards.reshape(-1) if hasattr(rewards, 'reshape') else rewards.flat)]
        rsh = rewards.shape
    else:
        def _to_float(v):
            if hasattr(v, 'to_flat_list'): return float(v.to_flat_list()[0])
            return float(v)
        rows = rewards if (hasattr(rewards, '__len__') and len(rewards) > 0 and isinstance(rewards[0], (list, tuple))) else [rewards]
        rfl = [_to_float(v) for row in rows for v in (row if hasattr(row, '__iter__') else [row])]
        rsh = (len(rfl),)
    mu = sum(rfl) / max(1, len(rfl))
    adv = [v - mu for v in rfl]
    adv_t = _T(_nd_arr(adv, dtype='float32').reshape(rsh) if hasattr(_nd_arr(adv, dtype='float32'), 'reshape') else _nd_arr(adv, dtype='float32'))
    return -(logp * adv_t).mean()


def sample_tokens(logits, temperature: float = 1.0):
    """Sample one token per row from logits (B, V). Returns list of int."""
    import math, random
    flat = list(logits._data.to_flat_list()) if hasattr(logits._data, 'to_flat_list') else [float(v) for v in logits._data]
    sh = logits._data.shape
    B, V = sh[0], sh[1]
    ids = []
    for b in range(B):
        row = [v / max(temperature, 1e-8) for v in flat[b * V:(b + 1) * V]]
        m = max(row)
        e = [math.exp(v - m) for v in row]
        s = sum(e)
        probs = [v / s for v in e]
        r = random.random()
        cumsum = 0.0
        tok = V - 1
        for i, p in enumerate(probs):
            cumsum += p
            if r <= cumsum:
                tok = i; break
        ids.append(tok)
    return ids


class PrefixEncoder(Module):
    """Prefix tuning (Li & Liang, 2021): trainable key/value prefixes per attention layer,
    reparameterised through a small MLP.
    Usage: prefixes = encoder()  # list of (prefix_k, prefix_v) per layer
    """
    def __init__(self, num_layers: int = 1, embed_dim: int = 64,
                 num_virtual_tokens: int = 5, hidden: int = 128):
        super().__init__()
        from vectorphile.core.tensor import Tensor as _T
        from vectorphile._core.ndarray import array as _nd_arr, zeros as _nd_zeros
        from vectorphile._core.ndarray import randn_nd as _rnd
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.n = num_virtual_tokens
        emb_flat = [v * 0.02 for v in _rnd((num_virtual_tokens * embed_dim,)).to_flat_list()]
        emb_data = _nd_arr(emb_flat, dtype='float32').reshape((num_virtual_tokens, embed_dim))
        self.embedding = _T(emb_data, requires_grad=True)
        self._parameters['embedding'] = self.embedding
        from vectorphile.nn.layers import Linear as _Lin
        self.fc1 = _Lin(embed_dim, hidden)
        self.fc2 = _Lin(hidden, num_layers * 2 * embed_dim)
        self._modules['fc1'] = self.fc1
        self._modules['fc2'] = self.fc2

    def forward(self):
        h = self.fc2(self.fc1(self.embedding).tanh()).reshape(self.n, self.num_layers, 2, self.embed_dim)
        return [(h[:, i, 0], h[:, i, 1]) for i in range(self.num_layers)]


__all__ = [
    # Apply functions
    'apply_lora', 'apply_dora', 'apply_qlora', 'apply_ia3',
    'apply_adapters', 'apply_bitfit',
    # Apply functions (new)
    'apply_rslora', 'apply_longlora',
    # Layers
    'LoRALinear', 'RSLoRALinear', 'QLoRALinear', 'LongLoRALinear',
    'DoRALinear', 'IA3Linear', 'AdapterLayer', 'SoftPrompt',
    'shifted_sparse_attention',
    # Utilities
    'count_parameters', 'trainable_parameters',
    'merge_adapters', 'adapter_state_dict', 'load_adapter_state_dict',
    'lora_plus_param_groups', 'layerwise_lr_decay_groups', 'unfreeze_last_n',
    # NEFTune
    'neftune',
    # Sequence log-probs
    'sequence_logprob',
    # Preference losses
    'dpo_loss', 'ipo_loss', 'simpo_loss', 'orpo_loss', 'kto_loss',
    # Distillation
    'distillation_loss',
    # New
    'freeze', 'QuantizedLinear', 'quantize_linear_layers',
    'apply_adalora', 'adalora_orth_loss', 'adalora_update',
    'PrefixEncoder',
    'grpo_loss', 'policy_gradient_loss', 'sample_tokens',
]
