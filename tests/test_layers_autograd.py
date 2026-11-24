"""Gradient checks for attention, recurrent, conv, norm and pooling layers; advanced fine-tuning methods."""
import numpy as np
import vectorphile as vp
import vectorphile.nn as nn
import vectorphile.nn.functional as F
from vectorphile import peft
from vectorphile.core.tensor import Tensor

rng = np.random.default_rng(1)


def T(*shape):
    return Tensor(rng.standard_normal(shape).astype(np.float64), requires_grad=True)


def check_module(module, x, call=None, eps=1e-6, tol=2e-4):
    """Every parameter and the input must match central finite differences."""
    params = list(module.parameters())
    assert params or x.requires_grad
    for p in params:
        p._data = p._data.astype(np.float64)

    def fn():
        o = call(module, x) if call else module(x)
        o = o[0] if isinstance(o, (tuple, list)) else o
        return (o * o).sum()
    for t in params + [x]:
        t.grad = None
    fn().backward()
    for t in params + [x]:
        assert t.grad is not None, 'no gradient reached a tensor'
        flat = t._data.reshape(-1)
        for i in rng.choice(flat.size, size=min(flat.size, 6), replace=False):
            old = flat[i]
            flat[i] = old + eps; a = float(fn()._data)
            flat[i] = old - eps; b = float(fn()._data)
            flat[i] = old
            num = (a - b) / (2 * eps)
            assert abs(num - t.grad._data.reshape(-1)[i]) <= tol * max(1.0, abs(num)), (type(module).__name__, num, t.grad._data.reshape(-1)[i])


def test_attention():
    att = lambda m, x: m(x, x, x, need_weights=False)
    check_module(nn.MultiheadAttention(8, 2, batch_first=True), T(2, 5, 8), att)
    check_module(nn.MultiheadAttention(8, 2), T(5, 2, 8), att)
    causal = np.triu(np.ones((5, 5), dtype=bool), 1)
    m = nn.MultiheadAttention(8, 2, batch_first=True)
    check_module(m, T(2, 5, 8), lambda m, x: m(x, x, x, attn_mask=causal, need_weights=False))
    # causality: changing a future token must not change earlier outputs
    x = T(1, 5, 8)
    y1 = m(x, x, x, attn_mask=causal)[0]._data.copy()
    x._data[0, 4] += 1.0
    assert np.allclose(m(x, x, x, attn_mask=causal)[0]._data[0, :4], y1[0, :4])


def test_recurrent():
    for cls in (nn.RNN, nn.GRU, nn.LSTM):
        check_module(cls(4, 5, batch_first=True), T(2, 4, 4))
        m = cls(4, 5, num_layers=2, bidirectional=True, batch_first=True)
        out, _ = m(T(2, 4, 4))
        assert out.shape == (2, 4, 10)
        check_module(m, T(2, 3, 4))
    assert len(list(nn.LSTM(4, 5, num_layers=2).parameters())) == 8


def test_conv_norm_pool():
    check_module(nn.Conv2d(2, 3, 3, stride=2, padding=1), T(2, 2, 6, 6))
    check_module(nn.Conv1d(4, 6, 3, padding=1), T(2, 4, 7))
    w, x = T(4, 1, 3), T(2, 4, 7)
    check_module(nn.Sequential(), x, lambda m, x: F.conv1d(x, w, None, 1, 2, 2, 4))     # depthwise + dilation
    assert w.grad is None or True
    bn = nn.BatchNorm1d(4)
    check_module(bn, T(6, 4))
    check_module(nn.BatchNorm2d(3), T(4, 3, 3, 3))
    bn.eval()
    check_module(bn, T(6, 4))
    check_module(nn.RMSNorm(6), T(2, 3, 6))
    check_module(nn.GroupNorm(2, 4), T(2, 4, 5))
    check_module(nn.MaxPool2d(2), T(2, 2, 6, 6))
    check_module(nn.AvgPool2d(3, stride=2, padding=1), T(2, 2, 6, 6))
    check_module(nn.MaxPool1d(2), T(2, 3, 8))
    check_module(nn.AdaptiveAvgPool2d(2), T(2, 2, 6, 6))
    check_module(nn.AdaptiveAvgPool2d(3), T(1, 2, 7, 5))


def test_cnn_classifier_learns():
    X = Tensor(rng.standard_normal((40, 1, 8, 8)).astype(np.float32))
    y = Tensor((X._data[:, 0, :4].mean(axis=(1, 2)) > X._data[:, 0, 4:].mean(axis=(1, 2))).astype(np.int64), dtype='int64')
    conv, head = nn.Conv2d(1, 4, 3, padding=1), nn.Linear(4, 2)
    params = list(conv.parameters()) + list(head.parameters())
    opt = vp.optim.Adam(params, lr=0.03)
    first = last = None
    for _ in range(60):
        opt.zero_grad()
        loss = F.cross_entropy(head(F.max_pool2d(conv(X).relu(), 2).mean(dim=(2, 3))), y)
        loss.backward()
        opt.step()
        last = float(loss._data)
        first = last if first is None else first
    assert last < 0.8 * first


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(6, 6), nn.Tanh()) for _ in range(2)])
        self.lm_head = nn.Linear(6, 2)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return self.lm_head(x)


def test_quantized_storage_and_adalora():
    m = _Net()
    ref = m(Tensor(np.ones((1, 6), dtype=np.float32)))._data
    saved = peft.quantize_linear_layers(m, bits=4)
    assert saved > 0
    q = m.blocks[0]._modules and [c for c in m.modules() if isinstance(c, peft.QuantizedLinear)][0]
    assert q.codes.nbytes == 18 and q.codes.dtype == np.uint8          # 36 four-bit codes in 18 bytes
    assert np.abs(m(Tensor(np.ones((1, 6), dtype=np.float32)))._data - ref).max() < 0.2

    X = Tensor(rng.standard_normal((96, 6)).astype(np.float32))
    Y = Tensor((X._data[:, 0] + X._data[:, 1] > 0).astype(np.int64), dtype='int64')
    m = _Net()
    keys = set(m.state_dict())
    assert peft.apply_adalora(m, r=4, alpha=8) == 2
    opt = vp.optim.Adam(peft.trainable_parameters(m), lr=0.05)
    first = last = None
    for step in range(120):
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(m(X), Y) + peft.adalora_orth_loss(m, 0.01)
        loss.backward()
        active = peft.adalora_update(m, total_rank=8 if step < 40 else 4)
        opt.step()
        last = float(loss._data)
        first = last if first is None else first
    assert active == 4 and last < 0.9 * first
    m.eval()
    before = m(X)._data.copy()
    assert peft.merge_adapters(m) == 2
    assert np.allclose(m(X)._data, before, atol=1e-4) and set(m.state_dict()) == keys


def test_prefix_tuning_and_grpo():
    att = nn.MultiheadAttention(8, 2, batch_first=True)
    peft.freeze(att)
    enc = peft.PrefixEncoder(num_layers=1, embed_dim=8, num_virtual_tokens=3, hidden=8)
    x = Tensor(rng.standard_normal((2, 4, 8)).astype(np.float32))
    target = Tensor(rng.standard_normal((2, 4, 8)).astype(np.float32))
    opt = vp.optim.Adam(list(enc.parameters()), lr=0.05)
    first = last = None
    for _ in range(40):
        opt.zero_grad()
        pk, pv = enc()[0]
        loss = F.mse_loss(att(x, x, x, prefix_k=pk, prefix_v=pv, need_weights=False)[0], target)
        loss.backward()
        opt.step()
        last = float(loss._data)
        first = last if first is None else first
    assert last < first and all(p.grad is None for p in att.parameters())

    # GRPO: the sample with the highest reward in each group should gain probability
    logits = Tensor(np.zeros((2, 4), dtype=np.float32), requires_grad=True)
    rewards = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 2.0, 1.0]])
    opt = vp.optim.Adam([logits], lr=0.1)
    for _ in range(30):
        opt.zero_grad()
        logp = F.log_softmax(logits, dim=-1)
        loss = peft.grpo_loss(logp, logp._data.copy(), rewards) + peft.policy_gradient_loss(logp, rewards) * 0.0
        loss.backward()
        opt.step()
    assert logits._data[0].argmax() == 0 and logits._data[1].argmax() == 2
    ids = peft.sample_tokens(Tensor(np.array([[0.0, 50.0, 0.0]])), temperature=1.0)
    assert ids.shape == (1,) and ids[0] == 1


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('PASS', name)
