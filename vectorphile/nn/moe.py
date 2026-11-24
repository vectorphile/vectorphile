"""Mixture of Experts layers (Shazeer et al., 2017; Switch Transformer, 2021).

All implementations from scratch — no PyTorch/JAX dependencies.

Usage::

    from vectorphile.nn.moe import MixtureOfExperts, SparseMoE, SwitchFeedForward

    moe = SparseMoE(input_dim=512, num_experts=8, expert_dim=2048, top_k=2)
    out, aux_loss = moe(x)   # aux_loss for load balancing
"""
from __future__ import annotations
from typing import Optional, Tuple
import math as _math_moe
import random as _random_moe
from vectorphile.nn.module import Module
from vectorphile.nn.layers import Linear
from vectorphile.nn.activations import ReLU, GELU
from vectorphile.core.tensor import Tensor


class Expert(Module):
    """Single feed-forward expert."""
    def __init__(self, input_dim: int, expert_dim: int, output_dim: Optional[int] = None,
                 activation: str = 'relu', dropout: float = 0.0):
        super().__init__()
        out_dim = output_dim or input_dim
        self.fc1 = Linear(input_dim, expert_dim)
        self.fc2 = Linear(expert_dim, out_dim)
        self._act = ReLU() if activation == 'relu' else GELU()
        self._dropout = dropout

    def forward(self, x: Tensor) -> Tensor:
        h = self._act(self.fc1(x))
        return self.fc2(h)


class TopKRouter(Module):
    """Router that selects top-k experts per token."""
    def __init__(self, input_dim: int, num_experts: int, top_k: int = 2,
                 noise: float = 0.0):
        super().__init__()
        self.gate = Linear(input_dim, num_experts, bias=False)
        self.top_k = top_k
        self.noise = noise

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        from vectorphile._core.ndarray import array as _nd_moe
        def _fl(d):
            if hasattr(d, 'to_flat_list'): return d.to_flat_list()
            return list(d)
        logits = self.gate(x)
        lf = _fl(logits._data)
        lsh = logits._data.shape if hasattr(logits._data, 'shape') else (len(lf),)
        N_m, E_m = lsh
        if self.noise > 0 and self.training:
            lf = [v + _random_moe.gauss(0, self.noise) for v in lf]
        # Softmax per row
        probs = []
        for n in range(N_m):
            row = lf[n*E_m:(n+1)*E_m]
            mx = max(row); ex = [_math_moe.exp(v-mx) for v in row]; s = sum(ex)
            probs.append([v/(s+1e-10) for v in ex])
        # Top-k
        k = self.top_k
        top_idx = [sorted(range(E_m), key=lambda e: -probs[n][e])[:k] for n in range(N_m)]
        top_probs = [[probs[n][top_idx[n][ki]] for ki in range(k)] for n in range(N_m)]
        # Normalize
        top_probs = [[v/(sum(r)+1e-10) for v in r] for r in top_probs]
        tp_fl = [v for r in top_probs for v in r]
        ti_fl = [float(v) for r in top_idx for v in r]
        return Tensor(_nd_moe(tp_fl).reshape((N_m, k))), Tensor(_nd_moe(ti_fl).reshape((N_m, k))), logits


class SparseMoE(Module):
    """Sparse Mixture of Experts with top-k routing and load balancing.

    Args:
        input_dim    : token feature dimension
        num_experts  : total number of experts
        expert_dim   : hidden dimension inside each expert
        top_k        : number of experts selected per token
        output_dim   : output dimension (defaults to input_dim)
        noise        : jitter noise for routing (Shazeer 2017 noisy top-k)
        lb_alpha     : load-balancing loss coefficient
    """

    def __init__(self, input_dim: int, num_experts: int = 8, expert_dim: int = 512,
                 top_k: int = 2, output_dim: Optional[int] = None, noise: float = 0.01,
                 lb_alpha: float = 0.01):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.lb_alpha = lb_alpha
        self.router = TopKRouter(input_dim, num_experts, top_k, noise)
        self.experts = [Expert(input_dim, expert_dim, output_dim) for _ in range(num_experts)]
        # Register experts as submodules
        for i, exp in enumerate(self.experts):
            setattr(self, f'expert_{i}', exp)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            x: (B, T, D) or (N, D)

        Returns:
            output: same shape as input
            aux_loss: load-balancing auxiliary loss (scalar)
        """
        from vectorphile._core.ndarray import array as _nd_sparse, zeros as _nd_zeros_sparse
        def _fl(d):
            if hasattr(d, 'to_flat_list'): return d.to_flat_list()
            return list(d)
        xd = x._data; xsh = xd.shape if hasattr(xd, 'shape') else (len(_fl(xd)),)
        orig_shape = xsh
        xfl = _fl(xd)
        if len(xsh) == 3:
            B, T, D = xsh; N = B*T
            x_flat = xfl
        else:
            N, D = xsh; x_flat = xfl

        weights, indices, router_logits = self.router(Tensor(_nd_sparse(x_flat).reshape((N, D))))
        wfl = _fl(weights._data); ifl = [int(v) for v in _fl(indices._data)]
        out_D = D; output = [0.0]*(N*out_D)

        for k in range(self.top_k):
            for e in range(self.num_experts):
                tok_idx = [n for n in range(N) if ifl[n*self.top_k+k] == e]
                if not tok_idx: continue
                x_e = [x_flat[n*D+d] for n in tok_idx for d in range(D)]
                out_e_t = self.experts[e](Tensor(_nd_sparse(x_e).reshape((len(tok_idx), D))))
                ofl = _fl(out_e_t._data); oD = len(ofl)//len(tok_idx)
                if oD != out_D: out_D = oD; output = [0.0]*(N*out_D)
                for ii, n in enumerate(tok_idx):
                    w = wfl[n*self.top_k+k]
                    for d in range(out_D): output[n*out_D+d] += w * ofl[ii*out_D+d]

        # Load-balancing loss
        rlf = _fl(router_logits._data); rlsh = router_logits._data.shape if hasattr(router_logits._data, 'shape') else (N, self.num_experts)
        N_r, E_r = rlsh
        rp = []
        for n in range(N_r):
            row = rlf[n*E_r:(n+1)*E_r]; mx = max(row); ex = [_math_moe.exp(v-mx) for v in row]; s = sum(ex)
            rp.append([v/(s+1e-10) for v in ex])
        mean_p = [sum(rp[n][e] for n in range(N_r))/N_r for e in range(E_r)]
        lb_loss = float(self.num_experts * sum(v*v for v in mean_p)) * self.lb_alpha
        out_sh = orig_shape[:-1] + (out_D,) if len(orig_shape) == 3 else (N, out_D)
        return Tensor(_nd_sparse(output).reshape(out_sh)), Tensor(_nd_sparse([lb_loss]).reshape((1,)))


class MixtureOfExperts(SparseMoE):
    """Alias for SparseMoE (more Keras/HuggingFace-compatible naming)."""
    pass


class SwitchFeedForward(Module):
    """Switch Transformer-style MoE FFN layer (top-1 routing, Fedus et al., 2021).

    Uses capacity_factor to handle routing overflow.
    """

    def __init__(self, input_dim: int, num_experts: int = 8, expert_dim: int = 512,
                 capacity_factor: float = 1.25, dropout: float = 0.1):
        super().__init__()
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.gate = Linear(input_dim, num_experts, bias=False)
        self.experts = [Expert(input_dim, expert_dim, activation='relu')
                        for _ in range(num_experts)]
        for i, exp in enumerate(self.experts):
            setattr(self, f'expert_{i}', exp)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        from vectorphile._core.ndarray import array as _nd_sw
        def _fl(d):
            if hasattr(d, 'to_flat_list'): return d.to_flat_list()
            return list(d)
        xd = x._data; xsh = xd.shape if hasattr(xd, 'shape') else (len(_fl(xd)),)
        orig_shape = xsh; xfl = _fl(xd)
        if len(xsh) == 3:
            B, T, D = xsh; N = B*T
        else:
            N, D = xsh
        x_flat = xfl
        logits_t = self.gate(Tensor(_nd_sw(x_flat).reshape((N, D))))
        lf = _fl(logits_t._data); E = self.num_experts
        probs = []
        for n in range(N):
            row = lf[n*E:(n+1)*E]; mx = max(row); ex2 = [_math_moe.exp(v-mx) for v in row]; s = sum(ex2)
            probs.append([v/(s+1e-10) for v in ex2])
        expert_idx = [max(range(E), key=lambda e: probs[n][e]) for n in range(N)]
        expert_weights = [probs[n][expert_idx[n]] for n in range(N)]
        capacity = max(1, int(self.capacity_factor * N / E))
        output = [0.0]*(N*D)
        for e in range(E):
            tok_idx = [n for n in range(N) if expert_idx[n] == e][:capacity]
            if not tok_idx: continue
            x_e = [x_flat[n*D+d] for n in tok_idx for d in range(D)]
            out_e_t = self.experts[e](Tensor(_nd_sw(x_e).reshape((len(tok_idx), D))))
            ofl = _fl(out_e_t._data); oD = D
            for ii, n in enumerate(tok_idx):
                w = expert_weights[n]
                for d in range(oD): output[n*oD+d] += w * ofl[ii*oD+d]
        # Aux loss
        token_frac = [sum(1 for nn in range(N) if expert_idx[nn] == e)/N for e in range(E)]
        mean_p_e = [sum(probs[n][e] for n in range(N))/N for e in range(E)]
        aux = float(E * sum(token_frac[e]*mean_p_e[e] for e in range(E)))
        output2 = _nd_sw(output).reshape(orig_shape)
        return Tensor(output2), Tensor(_nd_sw([aux]).reshape((1,)))


# ── Sorted-dispatch SparseMoE (token sorting for cache efficiency) ───────────

class SortedDispatchMoE(Module):
    """SparseMoE with sorted token dispatch (Lepikhin et al. / ST-MoE 2022).

    Sorts tokens by their chosen expert before running the expert FFN, so each
    expert processes a contiguous slice of the token buffer — much more cache-
    friendly and parallelism-friendly than the scatter-gather approach.

    Performance advantages over naive SparseMoE:
    - Expert forward passes run on dense contiguous slices (better SIMD/cache)
    - Avoids per-token scatter; uses pre-sorted index arrays
    - Compatible with VP CUDA dispatch via backend matmul

    Usage::
        moe = SortedDispatchMoE(dim=512, num_experts=8, expert_dim=2048, top_k=2)
        output, aux_loss = moe(x)   # x: (B, T, D) or (N, D)
    """

    def __init__(self, dim: int, num_experts: int, expert_dim: int,
                 top_k: int = 2, capacity_factor: float = 1.25,
                 dropout: float = 0.0):
        super().__init__()
        self.dim            = dim
        self.num_experts    = num_experts
        self.expert_dim     = expert_dim
        self.top_k          = top_k
        self.capacity_factor = capacity_factor
        self.gate = Linear(dim, num_experts, bias=False)
        self.experts = Module.__new__(Module)
        self.experts._modules = {}
        for i in range(num_experts):
            setattr(self.experts, f'e{i}', Expert(dim, expert_dim, dim,
                                                   activation='gelu',
                                                   dropout=dropout))

    def _softmax_rows(self, logits_flat: list, N: int, E: int) -> list:
        import math as _m
        rows = []
        for n in range(N):
            row = logits_flat[n*E:(n+1)*E]
            mx  = max(row)
            exps = [_m.exp(v-mx) for v in row]
            s    = sum(exps) + 1e-10
            rows.append([v/s for v in exps])
        return rows

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        from vectorphile._core.ndarray import array as _nd_sd
        def _fl(d):
            if hasattr(d, 'to_flat_list'): return d.to_flat_list()
            return list(d)
        xfl   = _fl(x._data)
        xsh   = x._data.shape if hasattr(x._data, 'shape') else (len(xfl),)
        orig_shape = xsh
        D = self.dim; E = self.num_experts; K = self.top_k
        if len(xsh) == 3:
            B, T, _ = xsh; N = B * T
        else:
            N, _ = xsh

        # Router: (N, E) logits → top-K weights per token
        logits_t = self.gate(Tensor(_nd_sd(xfl).reshape((N, D))))
        lf = _fl(logits_t._data)
        rows = self._softmax_rows(lf, N, E)

        # For each token: select top-K experts
        # dispatch_map: list of (token_idx, expert_idx, weight)
        dispatch_map = []
        for n in range(N):
            top_k_idx = sorted(range(E), key=lambda e: rows[n][e], reverse=True)[:K]
            # renormalise top-K weights
            top_weights = [rows[n][e] for e in top_k_idx]
            tw_sum = sum(top_weights) + 1e-10
            for rank, e in enumerate(top_k_idx):
                dispatch_map.append((n, e, top_weights[rank] / tw_sum))

        # Sort dispatch_map by expert_idx (the key sorting step)
        dispatch_map.sort(key=lambda t: t[1])

        # Capacity: max tokens per expert
        capacity = max(1, int(self.capacity_factor * N * K / E))

        # Group into expert buckets (contiguous after sort)
        expert_buckets: list = [[] for _ in range(E)]
        for token_idx, exp_idx, weight in dispatch_map:
            if len(expert_buckets[exp_idx]) < capacity:
                expert_buckets[exp_idx].append((token_idx, weight))

        # Run each expert on its contiguous token slice
        output = [0.0] * (N * D)
        for e in range(E):
            bucket = expert_buckets[e]
            if not bucket: continue
            tok_indices = [b[0] for b in bucket]
            weights     = [b[1] for b in bucket]
            # Build contiguous input slice
            x_e = [xfl[n*D+d] for n in tok_indices for d in range(D)]
            expert_mod = getattr(self.experts, f'e{e}')
            out_e = expert_mod(Tensor(_nd_sd(x_e).reshape((len(tok_indices), D))))
            ofl   = _fl(out_e._data)
            for ii, (n, w) in enumerate(zip(tok_indices, weights)):
                for d in range(D):
                    output[n*D+d] += w * ofl[ii*D+d]

        # Auxiliary load balancing loss (same as Switch Transformer)
        token_frac = [sum(1 for _, e, _ in dispatch_map if e == exp) / (N * K)
                      for exp in range(E)]
        mean_p = [sum(rows[n][exp] for n in range(N)) / N for exp in range(E)]
        aux = float(E * sum(token_frac[e] * mean_p[e] for e in range(E)))

        return (Tensor(_nd_sd(output).reshape(orig_shape)),
                Tensor(_nd_sd([aux]).reshape((1,))))


# ── MoE with Expert Choice routing (Zhou et al. 2022) ────────────────────────

class MixtureOfDepths(Module):
    """Mixture of Depths (MoD) — Raposo et al. 2024.

    Routes only a fraction of tokens through the expensive FFN/attention block.
    Tokens are ranked by a learned scalar router; top-k% pass through the block,
    the rest use a residual bypass.  Reduces FLOPs by (1-k%) per layer with
    minimal accuracy loss.

    Usage::
        mod_layer = MixtureOfDepths(block=ffn_layer, dim=512, capacity=0.5)
        output = mod_layer(x)   # only 50% of tokens go through ffn_layer
    """

    def __init__(self, block: Module, dim: int, capacity: float = 0.5):
        super().__init__()
        self.block    = block
        self.dim      = dim
        self.capacity = capacity  # fraction of tokens to route
        self.router   = Linear(dim, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        from vectorphile._core.ndarray import array as _nd_mod
        def _fl(d):
            if hasattr(d, 'to_flat_list'): return d.to_flat_list()
            return list(d)
        xfl = _fl(x._data)
        xsh = x._data.shape if hasattr(x._data, 'shape') else (len(xfl),)
        D = self.dim
        N = len(xfl) // D

        # Scalar router score per token
        scores_t = self.router(x.reshape((N, D)) if len(xsh) > 1 else x)
        scores   = _fl(scores_t._data)  # (N, 1)

        k = max(1, int(self.capacity * N))
        # Top-k token indices by score
        top_indices = sorted(range(N), key=lambda i: scores[i], reverse=True)[:k]
        top_set     = set(top_indices)

        # Route top-k through block
        x_top = [xfl[i*D+d] for i in top_indices for d in range(D)]
        out_top_t = self.block(Tensor(_nd_mod(x_top).reshape((k, D))))
        out_top   = _fl(out_top_t._data)

        # Build output: replaced for top-k, residual for rest
        output = list(xfl)
        for ii, i in enumerate(top_indices):
            for d in range(D):
                output[i*D+d] = out_top[ii*D+d]
        return Tensor(_nd_mod(output).reshape(xsh))


class ExpertChoiceMoE(Module):
    """Expert Choice routing — experts select their top-k tokens (not vice versa).

    Guarantees perfect load balancing by construction: every expert processes
    exactly floor(capacity_factor × N / E) tokens. No auxiliary loss needed.

    Reference: "Mixture-of-Experts with Expert Choice Routing", Zhou et al. 2022
    """

    def __init__(self, dim: int, num_experts: int, expert_dim: int,
                 capacity_factor: float = 1.0):
        super().__init__()
        self.dim             = dim
        self.num_experts     = num_experts
        self.expert_dim      = expert_dim
        self.capacity_factor = capacity_factor
        self.gate = Linear(dim, num_experts, bias=False)
        for i in range(num_experts):
            setattr(self, f'expert_{i}', Expert(dim, expert_dim, dim, 'gelu'))

    def forward(self, x: Tensor) -> Tensor:
        """Returns output (same shape as x). No auxiliary loss needed."""
        import math as _m
        from vectorphile._core.ndarray import array as _nd_ec
        def _fl(d):
            if hasattr(d, 'to_flat_list'): return d.to_flat_list()
            return list(d)
        xfl = _fl(x._data)
        xsh = x._data.shape if hasattr(x._data, 'shape') else (len(xfl),)
        D   = self.dim; E = self.num_experts
        N   = len(xfl) // D
        capacity = max(1, int(self.capacity_factor * N / E))

        # Router: (N, E) → softmax over tokens per expert (column-wise softmax)
        logits_t = self.gate(Tensor(_nd_ec(xfl).reshape((N, D))))
        lf = _fl(logits_t._data)  # (N, E)

        # For each expert e: pick top-capacity tokens by score
        output = [0.0] * (N * D)
        for e in range(E):
            # column e scores across all tokens
            col = [lf[n * E + e] for n in range(N)]
            # softmax over tokens (expert-view)
            mx   = max(col)
            exps = [_m.exp(v - mx) for v in col]
            s    = sum(exps) + 1e-10
            scores = [v / s for v in exps]
            # top-capacity token indices
            top_indices = sorted(range(N), key=lambda n: scores[n], reverse=True)[:capacity]
            x_e = [xfl[n*D+d] for n in top_indices for d in range(D)]
            exp_mod = getattr(self, f'expert_{e}')
            out_e = exp_mod(Tensor(_nd_ec(x_e).reshape((len(top_indices), D))))
            ofl   = _fl(out_e._data)
            for ii, n in enumerate(top_indices):
                w = scores[n]
                for d in range(D): output[n*D+d] += w * ofl[ii*D+d]

        return Tensor(_nd_ec(output).reshape(xsh))
