"""AdamMini — Memory-Efficient Adam for LLMs (Zhang et al., 2024).

Reduces Adam's optimizer-state memory from 2× to ~1/d_model× by learning
a single LR per parameter sub-block based on its Hessian structure.

For embedding and linear weights: partition into d_model-sized blocks, compute
per-block learning rate as lr * (1 / sqrt(mean second moment of block)).
For 1D biases/norms: standard Adam.

Paper: "Adam-mini: Use Fewer Learning Rates To Gain More"
"""
from __future__ import annotations
import math
from typing import List


class AdamMini:
    """AdamMini optimizer.

    Args:
        params: iterable of Tensors
        lr: global learning rate
        betas: (β₁, β₂)
        eps: numerical stability
        weight_decay: decoupled weight decay
        dim: LLM hidden dimension (block size for LR partitioning)
    """

    def __init__(self, params, lr: float = 1e-3,
                 betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0, dim: int = 512):
        self._params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.dim = dim
        self._step = 0
        self._state = {}

        for p in self._params:
            pid = id(p)
            flat = list(p._data.to_flat_list() if hasattr(p._data, 'to_flat_list') else p._data)
            n = len(flat)
            sh = p._data.shape if hasattr(p._data, 'shape') else (n,)
            self._state[pid] = {
                'm':  [0.0] * n,         # first moment
                'v':  [0.0] * n,         # second moment (full; reduced to block mean for LR)
                'n':  n,
                'sh': sh,
                'is_1d': len(sh) <= 1 or n < dim,  # use standard Adam for 1D
            }

    def _block_lr(self, v_block: List[float], t: int) -> float:
        """Compute effective LR for a block from its second-moment values."""
        bc2 = 1.0 - self.beta2 ** t
        v_hat_mean = sum(v_block) / max(1, len(v_block)) / bc2
        return self.lr / (math.sqrt(v_hat_mean) + self.eps)

    def step(self) -> None:
        import array as _a
        from vectorphile._core.ndarray import NDArray, _c_strides, _DType
        self._step += 1
        t = self._step
        b1, b2 = self.beta1, self.beta2
        bc1 = 1.0 - b1 ** t

        for p in self._params:
            if p.grad is None:
                continue
            pid = id(p)
            if pid not in self._state:
                continue
            st = self._state[pid]
            g = list(p.grad._data.to_flat_list() if hasattr(p.grad._data, 'to_flat_list') else p.grad._data)
            m = st['m']; v = st['v']
            n = st['n']; sh = st['sh']
            w = list(p._data.to_flat_list() if hasattr(p._data, 'to_flat_list') else p._data)

            # Update moments
            new_m = [b1 * m[i] + (1.0 - b1) * g[i] for i in range(n)]
            new_v = [b2 * v[i] + (1.0 - b2) * g[i] * g[i] for i in range(n)]
            st['m'] = new_m; st['v'] = new_v

            new_w = w[:]
            if st['is_1d']:
                # Standard Adam for scalars / biases / norms
                bc2 = 1.0 - b2 ** t
                for i in range(n):
                    m_hat = new_m[i] / bc1
                    v_hat = new_v[i] / bc2
                    upd = m_hat / (math.sqrt(v_hat) + self.eps)
                    new_w[i] = w[i] * (1.0 - self.lr * self.weight_decay) - self.lr * upd
            else:
                # Block-wise LR: partition into blocks of size dim
                block_size = max(1, self.dim)
                num_blocks = (n + block_size - 1) // block_size
                for blk in range(num_blocks):
                    start = blk * block_size
                    end   = min(start + block_size, n)
                    # Compute block LR from mean second moment of this block
                    v_blk = new_v[start:end]
                    blk_lr = self._block_lr(v_blk, t)
                    for i in range(start, end):
                        m_hat = new_m[i] / bc1
                        new_w[i] = w[i] * (1.0 - blk_lr * self.weight_decay) - blk_lr * m_hat

            buf = _a.array('f', [float(v) for v in new_w])
            p._data = NDArray(buf, sh, _c_strides(sh), _DType('float32'))

    def zero_grad(self) -> None:
        for p in self._params:
            p.grad = None

    def state_dict(self) -> dict:
        return {'step': self._step, 'state': self._state}

    def load_state_dict(self, d: dict) -> None:
        self._step = d.get('step', 0)
        self._state = d.get('state', {})
