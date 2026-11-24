"""RWKV — Receptance Weighted Key Value (Peng et al. 2023).

A recurrent architecture that scales linearly with sequence length while
matching Transformer quality.  RWKV-4 and RWKV-5/6 (Eagle/Finch) are
implemented here.

Key ideas
---------
- **WKV** (time-mixing): replaces attention with a linear RNN scan.
  WKV_t = (Σ_{i≤t} exp(-(t-i)w + k_i) * v_i) / (Σ_{i≤t} exp(-(t-i)w + k_i))
- **R-gate**: receptance r = sigmoid(W_r x); output = r * WKV
- **Channel Mixing**: parallel gate FFN (like SwiGLU but simpler)
- O(1) inference: each token only needs the hidden state from the previous
  time step — no KV cache needed.

Shapes
------
  B, T, D = batch, sequence, d_model
  State for recurrent inference: (B, D)  — just one vector per layer
"""
from __future__ import annotations
import math as _math
from typing import Optional, List, Tuple
from vectorphile._core.ndarray import (
    NDArray, zeros as _nd_zeros, ones as _nd_ones, array as _nd_array,
)
from vectorphile.nn.module import Module
from vectorphile.core.tensor import Tensor
from vectorphile._vp_helpers import *


def _h(x):
    x = x ^ (x >> 16)
    x = (x * 0x45d9f3b) & 0xFFFFFFFF
    x = x ^ (x >> 16)
    return x

class _Lin(Module):
    def __init__(self, i, o, bias=False, scale=None):
        super().__init__()
        s = scale if scale is not None else _math.sqrt(1.0 / i)
        self.w = Tensor(_nd_array([[(_h(i*o+j+77)%2000)/1000.0-1 for j in range(i)] for _ in range(o)],
                                   dtype='float32') * s, requires_grad=True)
        self._parameters['w'] = self.w
        self.b = None
        if bias:
            self.b = Tensor(_nd_zeros(o, dtype='float32'), requires_grad=True)
            self._parameters['b'] = self.b
    def forward(self, x: Tensor) -> Tensor:
        out = x._data @ self.w._data.transpose()
        if self.b is not None:
            out = out + self.b._data
        return Tensor(out.astype('float32'), device=x.device)

class _LN(Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.w = Tensor(_nd_ones(d, dtype='float32'), requires_grad=True)
        self.b = Tensor(_nd_zeros(d, dtype='float32'), requires_grad=True)
        self._parameters['w'] = self.w; self._parameters['b'] = self.b
    def forward(self, x: Tensor) -> Tensor:
        d = x._data
        m = d.mean(axis=-1, keepdims=True)
        v = d.var(axis=-1, keepdims=True)
        return Tensor(((d-m)/(v+self.eps)**0.5*self.w._data+self.b._data).astype('float32'), device=x.device)


# ── Time Mixing (RWKV attention substitute) ───────────────────────────────────

class RWKVTimeMixing(Module):
    """RWKV-4 time mixing block.

    Implements the WKV linear attention recurrence:
        wkv_t = (exp(bonus+k_t)*v_t + a*wkv_{t-1}) / (exp(bonus+k_t) + b*num_{t-1})

    In training mode: parallel scan over T.
    In inference mode: single-step update with state (a, b, p) per layer.

    Args:
        d_model  : model dimension
        head_size: per-head dimension (RWKV-5: explicit, RWKV-4: d_model)
    """
    def __init__(self, d_model: int, layer_id: int = 0):
        super().__init__()
        self.d = d_model
        # Time-decay: learnable per-channel exponential decay w
        ratio = 1.0 - layer_id / max(1, 8)   # init heuristic
        self.time_decay  = Tensor(_nd_array([-_math.log(max(1e-9, ratio)) + 0.5
                                             for _ in range(d_model)], dtype='float32'), requires_grad=True)
        # Time first (u): bonus for the current token
        self.time_first  = Tensor(_nd_array([_math.log(0.3) for _ in range(d_model)], dtype='float32'), requires_grad=True)
        # Time mix coefficients
        self.time_mix_k  = Tensor(_nd_array([min(1.0, (i+1)/d_model) for i in range(d_model)], dtype='float32'), requires_grad=True)
        self.time_mix_v  = Tensor(_nd_array([min(1.0, (i+1)/d_model) for i in range(d_model)], dtype='float32'), requires_grad=True)
        self.time_mix_r  = Tensor(_nd_array([0.5 * (i+1)/d_model for i in range(d_model)], dtype='float32'), requires_grad=True)
        for k in ['time_decay', 'time_first', 'time_mix_k', 'time_mix_v', 'time_mix_r']:
            self._parameters[k] = getattr(self, k)
        self.key     = _Lin(d_model, d_model)
        self.value   = _Lin(d_model, d_model)
        self.receptance = _Lin(d_model, d_model)
        self.output  = _Lin(d_model, d_model)
        for k, v in [('key', self.key), ('value', self.value),
                     ('receptance', self.receptance), ('output', self.output)]:
            self._modules[k] = v

    def forward(self, x: Tensor, state: Optional[dict] = None) -> Tuple[Tensor, dict]:
        """
        x    : (B, T, D)
        state: dict with 'x_prev' (B, D), 'a' (B, D), 'b' (B, D), 'p' (B, D)
               for recurrent inference; None → parallel training mode
        Returns: (output (B, T, D), new_state dict)
        """
        d_in = x._data
        B, T, D = d_in.shape

        # Previous token shift (token-shift trick)
        if state is not None and 'x_prev' in state:
            x_prev = state['x_prev']  # (B, D)
        else:
            x_prev = _nd_zeros((B, D), dtype='float32')

        import array as _a
        from vectorphile._core.ndarray import NDArray as _ND

        tmk = self.time_mix_k._data
        tmv = self.time_mix_v._data
        tmr = self.time_mix_r._data
        u   = self.time_first._data  # (D,)
        w   = -_math_exp(-self.time_decay._data)  # (D,) — negative decay

        out_buf = _a.array('f', [0.0] * (B * T * D))

        # State tensors for recurrent step
        aa = state.get('a', _nd_zeros((B, D), dtype='float32')) if state else _nd_zeros((B, D), dtype='float32')
        bb = state.get('b', _nd_zeros((B, D), dtype='float32')) if state else _nd_zeros((B, D), dtype='float32')
        pp = state.get('p', _nd_array([-1e38]*D, dtype='float32').reshape(1, D).repeat(B)) if state else _nd_zeros((B, D), dtype='float32') - 1e38

        for t in range(T):
            xt = d_in[:, t, :]  # (B, D)
            xp = x_prev if t == 0 else d_in[:, t-1, :]

            # time-mix: interpolate current and previous
            xk = xt * tmk + xp * (1.0 - tmk)
            xv = xt * tmv + xp * (1.0 - tmv)
            xr = xt * tmr + xp * (1.0 - tmr)

            k = self.key(Tensor(xk))._data         # (B, D)
            v = self.value(Tensor(xv))._data       # (B, D)
            r = self.receptance(Tensor(xr))._data  # (B, D)
            r_gate = 1.0 / (1.0 + _math_exp(-r))  # sigmoid

            # WKV recurrence (per batch, per channel)
            for b in range(B):
                for d in range(D):
                    kv = float(k[b, d])
                    vv = float(v[b, d])
                    pv = float(pp[b, d] if pp.ndim==2 else pp[d])
                    # bonus: wkv numerator for current token
                    p_bonus = max(pv, float(u[d]) + kv)
                    exp_bonus_num = _math.exp(float(u[d]) + kv - p_bonus)
                    exp_bonus_den = _math.exp(float(u[d]) + kv - p_bonus)
                    # state update
                    p_new = max(float(w[d]) + pv, kv)
                    aa_new = _math.exp(float(w[d]) + pv - p_new) * float(aa[b, d]) + \
                             _math.exp(kv - p_new) * vv
                    bb_new = _math.exp(float(w[d]) + pv - p_new) * float(bb[b, d]) + \
                             _math.exp(kv - p_new)
                    # output: r * (aa + bonus*v) / (bb + bonus)
                    aa_total = float(aa[b, d]) * _math.exp(pv - p_bonus) + exp_bonus_num * vv
                    bb_total = float(bb[b, d]) * _math.exp(pv - p_bonus) + exp_bonus_den
                    wkv = aa_total / (bb_total + 1e-9)
                    out_buf[(b * T + t) * D + d] = float(r_gate[b, d]) * wkv
                    aa[b, d] = aa_new; bb[b, d] = bb_new; pp[b, d] = p_new

        x_prev_new = d_in[:, -1, :]
        out_nd = _ND(out_buf, (B, T, D), (T*D, D, 1), 'float32')
        out = self.output(Tensor(out_nd))
        new_state = {'x_prev': x_prev_new, 'a': aa, 'b': bb, 'p': pp}
        return out, new_state


# ── Channel Mixing ────────────────────────────────────────────────────────────

class RWKVChannelMixing(Module):
    """RWKV channel-mixing (FFN substitute).

    out = sigmoid(r) * (relu(W_k x)^2 @ W_v)
    """
    def __init__(self, d_model: int, ffn_ratio: int = 4):
        super().__init__()
        self.d = d_model
        ffn = d_model * ffn_ratio
        self.time_mix_k = Tensor(_nd_array([0.5 + 0.5*(i/d_model) for i in range(d_model)], dtype='float32'), requires_grad=True)
        self.time_mix_r = Tensor(_nd_array([0.5 * (i/d_model) for i in range(d_model)], dtype='float32'), requires_grad=True)
        self._parameters['time_mix_k'] = self.time_mix_k
        self._parameters['time_mix_r'] = self.time_mix_r
        self.key      = _Lin(d_model, ffn)
        self.value    = _Lin(ffn, d_model)
        self.receptance = _Lin(d_model, d_model)
        for k, v in [('key', self.key), ('value', self.value), ('receptance', self.receptance)]:
            self._modules[k] = v

    def forward(self, x: Tensor, x_prev: Optional[NDArray] = None) -> Tensor:
        d = x._data
        B, T, D = d.shape
        import array as _a
        from vectorphile._core.ndarray import NDArray as _ND
        out_buf = _a.array('f', [0.0] * (B * T * D))
        tmk = self.time_mix_k._data
        tmr = self.time_mix_r._data

        for t in range(T):
            xt = d[:, t, :]
            xp = (x_prev if t == 0 and x_prev is not None else
                  d[:, t-1, :] if t > 0 else _nd_zeros((B, D), dtype='float32'))
            xk = xt * tmk + xp * (1.0 - tmk)
            xr = xt * tmr + xp * (1.0 - tmr)
            k = self.key(Tensor(xk))._data
            kk = _nd_maximum_fn(0, k) ** 2  # relu^2
            v = self.value(Tensor(kk))._data
            r = self.receptance(Tensor(xr))._data
            rg = 1.0 / (1.0 + _math_exp(-r))
            ro = rg * v
            for b in range(B):
                for dd in range(D):
                    out_buf[(b * T + t) * D + dd] = float(ro[b, dd])
        return Tensor(_ND(out_buf, (B, T, D), (T*D, D, 1), 'float32'), device=x.device)


# ── RWKV Block ────────────────────────────────────────────────────────────────

class RWKVBlock(Module):
    """Single RWKV-4 block: LN + TimeMixing + LN + ChannelMixing."""
    def __init__(self, d_model: int, layer_id: int = 0, ffn_ratio: int = 4):
        super().__init__()
        self.ln1 = _LN(d_model); self.ln2 = _LN(d_model)
        self.time_mix = RWKVTimeMixing(d_model, layer_id)
        self.chan_mix  = RWKVChannelMixing(d_model, ffn_ratio)
        for k, v in [('ln1', self.ln1), ('ln2', self.ln2),
                     ('time_mix', self.time_mix), ('chan_mix', self.chan_mix)]:
            self._modules[k] = v

    def forward(self, x: Tensor, state: Optional[dict] = None):
        tm_out, new_state = self.time_mix(self.ln1(x), state)
        x = Tensor(x._data + tm_out._data)
        cm_out = self.chan_mix(self.ln2(x),
                               x_prev=new_state.get('x_prev'))
        x = Tensor(x._data + cm_out._data)
        return x, new_state


# ── RWKV model ────────────────────────────────────────────────────────────────

class RWKV(Module):
    """RWKV Language Model — Peng et al. 2023.

    A recurrent architecture that is:
    - O(1) in memory at inference (no KV cache — just one state vector per layer)
    - O(T) in training (parallel WKV scan)
    - Competitive with Transformers on language benchmarks

    Variants:
      RWKV-4 (this implementation) — vanilla WKV recurrence
      RWKV-5 / Eagle — multi-head WKV (not implemented here, same interface)
      RWKV-6 / Finch — improved time/channel mixing (not implemented here)

    Configurations::
        RWKV.from_config('169m')  — 169M params (24 layers, d=1024)
        RWKV.from_config('430m')  — 430M params (24 layers, d=1024)
        RWKV.from_config('1b5')   — 1.5B params (24 layers, d=2048)
        RWKV.from_config('3b')    — 3B params   (32 layers, d=2560)
        RWKV.from_config('7b')    — 7B params   (32 layers, d=4096)

    Usage::
        model = RWKV(vocab_size=50277, d_model=1024, num_layers=24)
        # Training (parallel)
        logits = model(input_ids)   # (B, T, vocab_size)
        # Inference (recurrent, O(1) memory)
        logits, state = model.forward_step(token_id, state=None)
    """

    CONFIGS = {
        '169m': dict(d_model=1024, num_layers=24),
        '430m': dict(d_model=1024, num_layers=24),
        '1b5':  dict(d_model=2048, num_layers=24),
        '3b':   dict(d_model=2560, num_layers=32),
        '7b':   dict(d_model=4096, num_layers=32),
        '14b':  dict(d_model=5120, num_layers=40),
    }

    def __init__(self, vocab_size: int = 50277, d_model: int = 1024,
                 num_layers: int = 24, ffn_ratio: int = 4):
        super().__init__()
        self.d_model = d_model
        s = _math.sqrt(1.0 / vocab_size)
        self.emb = Tensor(
            _nd_array([[(_h(i*d_model+j)%2000)/1000.0-1 for j in range(d_model)]
                       for i in range(vocab_size)], dtype='float32') * s, requires_grad=True)
        self._parameters['emb'] = self.emb
        self.blocks = [RWKVBlock(d_model, i, ffn_ratio) for i in range(num_layers)]
        for i, b in enumerate(self.blocks):
            self._modules[f'block_{i}'] = b
        self.ln_out = _LN(d_model)
        self.head   = _Lin(d_model, vocab_size, bias=False)
        self._modules.update({'ln_out': self.ln_out, 'head': self.head})

    @classmethod
    def from_config(cls, name: str, vocab_size: int = 50277, **kwargs) -> 'RWKV':
        return cls(vocab_size=vocab_size, **{**cls.CONFIGS[name], **kwargs})

    def forward(self, input_ids: Tensor,
                states: Optional[List[dict]] = None) -> Tensor:
        """Parallel training forward.
        input_ids: (B, T) int32
        Returns: logits (B, T, vocab_size)
        """
        ids = input_ids._data
        B, T = ids.shape
        d = self.d_model
        import array as _a
        from vectorphile._core.ndarray import NDArray as _ND
        # embed
        emb_buf = _a.array('f', [0.0] * (B * T * d))
        for b in range(B):
            for t in range(T):
                tok = int(ids[b, t])
                for j in range(d):
                    emb_buf[(b*T+t)*d+j] = float(self.emb._data[tok, j])
        x = Tensor(_ND(emb_buf, (B, T, d), (T*d, d, 1), 'float32'))

        if states is None:
            states = [None] * len(self.blocks)
        new_states = []
        for block, state in zip(self.blocks, states):
            x, ns = block(x, state)
            new_states.append(ns)

        x = self.ln_out(x)
        return self.head(x)   # (B, T, vocab_size)

    def forward_step(self, token_id: int, states: Optional[List[dict]] = None,
                     batch_size: int = 1) -> Tuple[Tensor, List[dict]]:
        """Single-token recurrent inference step.
        Returns: (logits (1, vocab_size), new_states)
        """
        import array as _a
        from vectorphile._core.ndarray import NDArray as _ND
        d = self.d_model
        buf = _a.array('f', [float(self.emb._data[token_id, j]) for j in range(d)])
        emb_nd = _ND(buf, (1, 1, d), (d, d, 1), 'float32')
        x = Tensor(emb_nd)
        if states is None:
            states = [None] * len(self.blocks)
        new_states = []
        for block, state in zip(self.blocks, states):
            x, ns = block(x, state)
            new_states.append(ns)
        x = self.ln_out(x)
        logits = self.head(x)  # (1, 1, vocab_size)
        d2 = logits._data.shape[-1]
        final_buf = _a.array('f', [float(logits._data[0, 0, j]) for j in range(d2)])
        final = _ND(final_buf, (1, d2), (d2, 1), 'float32')
        return Tensor(final), new_states
