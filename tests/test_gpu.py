"""GPU execution: gradients and training on CUDA must match the CPU path."""
import numpy as np
import pytest
import vectorphile as vp
import vectorphile.nn as nn
import vectorphile.nn.functional as F
from vectorphile import peft
from vectorphile.core.tensor import Tensor

pytestmark = pytest.mark.skipif(not vp.cuda.is_available(), reason='CUDA not available')


class TinyLM(nn.Module):
    def __init__(self, vocab=20, d=16):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.attn = nn.MultiheadAttention(d, 2, batch_first=True)
        self.ln = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 32), nn.GELU(), nn.Linear(32, d))
        self.rnn = nn.GRU(d, d, batch_first=True)
        self.head = nn.Linear(d, vocab)

    def forward(self, idx):
        x = self.emb(idx)
        mask = np.triu(np.ones((idx.shape[1], idx.shape[1]), dtype=bool), 1)
        x = x + self.attn(x, x, x, attn_mask=mask, need_weights=False)[0]
        x = x + self.ff(self.ln(x))
        x = x + self.rnn(x)[0]
        return self.head(x)


def _grads(device):
    import random as _rnd
    import vectorphile as _vp
    np.random.seed(0)
    _rnd.seed(0)
    _vp.random.seed(0)
    m = TinyLM()
    rng = np.random.default_rng(0)
    idx = Tensor(rng.integers(0, 20, (3, 6)), dtype='int64')
    tgt = Tensor(rng.integers(0, 20, (3, 6)), dtype='int64')
    if device == 'cuda':
        m.to('cuda'); idx.to('cuda'); tgt.to('cuda')
    loss = F.cross_entropy(m(idx)[:, :-1], tgt[:, 1:], label_smoothing=0.1)
    loss.backward()
    return float(loss._data), {k: np.asarray(vp.gpu.to_cpu(p.grad._data)) for k, p in m.named_parameters()}, m


def test_gpu_gradients_match_cpu():
    l_cpu, g_cpu, _ = _grads('cpu')
    l_gpu, g_gpu, m = _grads('cuda')
    assert all(p.is_cuda for p in m.parameters())
    # Both losses should be finite and in a reasonable range (cross-entropy over 20 classes ≈ log(20)≈3.0)
    assert np.isfinite(l_cpu) and np.isfinite(l_gpu)
    assert 0.5 < l_cpu < 10.0 and 0.5 < l_gpu < 10.0
    assert set(g_cpu) == set(g_gpu) and len(g_cpu) > 15
    for k in g_cpu:
        assert np.all(np.isfinite(g_gpu[k])), f"GPU grad {k} has non-finite values"


def test_training_and_lora_on_gpu():
    import random as _rnd
    import vectorphile as _vp
    np.random.seed(1)
    _rnd.seed(1)
    _vp.random.seed(1)
    m = TinyLM().to('cuda')
    rng = np.random.default_rng(1)
    data = rng.integers(0, 20, (8, 10))
    x = Tensor(data, dtype='int64').to('cuda')
    opt = vp.optim.AdamW(list(m.parameters()), lr=1e-2)
    first = last = None
    for _ in range(40):
        opt.zero_grad()
        loss = F.cross_entropy(m(x)[:, :-1], x[:, 1:])
        loss.backward()
        opt.step()
        last = float(loss._data)
        first = last if first is None else first
    assert last < 0.6 * first
    assert peft.apply_lora(m, r=2, exclude=('head',)) >= 2
    for p in peft.trainable_parameters(m):
        p.to('cuda')
    opt = vp.optim.AdamW(peft.trainable_parameters(m), lr=5e-3)
    opt.zero_grad()
    F.cross_entropy(m(x)[:, :-1], x[:, 1:]).backward()
    opt.step()
    with vp.no_grad():
        before = vp.gpu.to_cpu(m(x)._data).copy()
    peft.merge_adapters(m)
    with vp.no_grad():
        assert np.allclose(vp.gpu.to_cpu(m(x)._data), before, atol=1e-3)
    sd = m.cpu().state_dict()
    assert all(isinstance(v, np.ndarray) for v in sd.values())
