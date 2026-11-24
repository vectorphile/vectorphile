"""Finite-difference gradient checks and learning tests for v0.5.0 autograd."""
import numpy as np
import vectorphile as vp
import vectorphile.nn as nn
import vectorphile.nn.functional as F
from vectorphile.core.tensor import Tensor

rng = np.random.default_rng(0)


def T(*shape):
    return Tensor(rng.standard_normal(shape).astype(np.float64), requires_grad=True)


def gradcheck(fn, *inputs, eps=1e-6, tol=1e-4):
    for t in inputs:
        t.grad = None
    out = fn(*inputs)
    out = out.sum() if out._data.size != 1 else out
    out.backward()
    for t in inputs:
        ana = np.zeros_like(t._data) if t.grad is None else t.grad._data
        flat = t._data.reshape(-1)
        for i in rng.choice(flat.size, size=min(flat.size, 10), replace=False):
            old = flat[i]
            flat[i] = old + eps; a = float(np.sum(fn(*inputs)._data))
            flat[i] = old - eps; b = float(np.sum(fn(*inputs)._data))
            flat[i] = old
            num = (a - b) / (2 * eps)
            assert abs(num - ana.reshape(-1)[i]) <= tol * max(1.0, abs(num)), (num, ana.reshape(-1)[i])


def test_tensor_ops():
    a, b = T(3, 4), T(3, 4)
    gradcheck(lambda x: (x ** 3).sum(), a)
    gradcheck(lambda x: (-x).exp().sum(), a)
    gradcheck(lambda x: (x * x + 1.0).log().sum(), a)
    gradcheck(lambda x: (x * x + 1.0).sqrt().sum(), a)
    gradcheck(lambda x: (x.sum(dim=1) ** 2).sum(), a)
    gradcheck(lambda x: (x.mean(dim=-1, keepdim=True) * x).sum(), a)
    gradcheck(lambda x: (x.reshape(4, 3) ** 2).sum(), a)
    gradcheck(lambda x: (x.view(2, 6) ** 2).sum(), a)
    gradcheck(lambda x, y: (x.transpose(0, 1) @ y).sum(), T(3, 4), T(3, 5))
    gradcheck(lambda x: (x.permute(1, 0, 2) ** 2).sum(), T(2, 3, 4))
    gradcheck(lambda x: (x[:, 1:] ** 2).sum(), a)
    gradcheck(lambda x: (x.unsqueeze(0).squeeze(0) ** 2).sum(), a)
    gradcheck(lambda x: (x.split(2, dim=1)[1] ** 2).sum(), a)
    gradcheck(lambda x, y: (vp.cat([x, y], dim=1) ** 2).sum(), a, b)
    gradcheck(lambda x, y: (vp.stack([x, y], dim=0) ** 2).sum(), a, b)
    gradcheck(lambda x, y: ((x @ y) ** 2).sum(), T(2, 2, 3, 4), T(2, 2, 4, 5))
    gradcheck(lambda x: (vp.softmax(x, dim=-1) ** 2).sum(), a)
    gradcheck(lambda x: (x.masked_fill(Tensor(np.triu(np.ones((3, 4)), 1)) == 1, -1e9).softmax(-1) ** 2).sum(), a)


def test_functional_and_layers():
    a = T(3, 4)
    tgt = Tensor(np.array([0, 1, -100, 3, -100, 2]), dtype='int64')
    gradcheck(lambda x: (F.log_softmax(x, dim=-1) ** 2).sum(), a)
    gradcheck(lambda x: F.cross_entropy(x, tgt), T(6, 5))
    gradcheck(lambda x: F.cross_entropy(x, tgt, label_smoothing=0.1), T(6, 5))
    gradcheck(lambda x: nn.CrossEntropyLoss()(x, Tensor(np.array([0, 1, 4, 3, 2, 2]), dtype='int64')), T(6, 5))
    gradcheck(lambda x: F.gelu(x).sum(), a)
    gradcheck(lambda x: F.silu(x).sum(), a)
    gradcheck(lambda x, w, b: (F.layer_norm(x, 4, w, b) ** 2).sum(), a, T(4), T(4))
    gradcheck(lambda w: (F.embedding(Tensor(np.array([[0, 2], [1, 1]]), dtype='int64'), w) ** 2).sum(), T(4, 3))
    gradcheck(lambda x, y: F.mse_loss(x, y), a, T(3, 4))
    lin = nn.Linear(4, 3)
    for p in lin.parameters():
        p._data = p._data.astype(np.float64)
    gradcheck(lambda x, w, b: (lin(x) ** 2).sum(), T(2, 5, 4), lin.weight, lin.bias_param)


def test_modulelist_registers_children():
    m = nn.ModuleList([nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2)) for _ in range(3)])
    assert len(list(m.parameters())) == 12


def test_extra_losses():
    pred, tgt = T(4, 1), T(4, 1)
    tgt._data[:] = rng.standard_normal((4, 1))
    gradcheck(lambda x: nn.MAELoss()(x, tgt), pred)
    gradcheck(lambda x: nn.HuberLoss()(x, tgt), pred)
    # BCEWithLogits: targets in [0,1], not differentiable w.r.t. target
    t01 = Tensor((tgt._data > 0).astype(np.float64))
    gradcheck(lambda x: nn.BCEWithLogitsLoss()(x, t01), pred)
    # KLDivLoss: input = log-probs, target = probs
    lp = T(3, 4); lp._data -= lp._data.max(-1, keepdims=True)
    lp._data -= np.log(np.exp(lp._data).sum(-1, keepdims=True))
    tp = Tensor(np.abs(rng.standard_normal((3, 4))) + 0.1)
    tp._data /= tp._data.sum(-1, keepdims=True)
    gradcheck(lambda x: nn.KLDivLoss()(x, tp), lp)
    # ReLU and LeakyReLU
    gradcheck(lambda x: nn.ReLU()(x).sum(), T(3, 4))
    gradcheck(lambda x: nn.LeakyReLU(0.1)(x).sum(), T(3, 4))


def test_classifier_learns():
    X = Tensor(rng.standard_normal((120, 4)).astype(np.float32))
    y = Tensor((X._data[:, 0] > 0).astype(np.int64) + (X._data[:, 1] > 0).astype(np.int64), dtype='int64')
    m = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 3))
    opt = vp.optim.Adam(list(m.parameters()), lr=0.05)
    first = last = None
    for _ in range(120):
        opt.zero_grad(); loss = nn.CrossEntropyLoss()(m(X), y); loss.backward(); opt.step()
        last = float(loss._data); first = last if first is None else first
    assert last < 0.6 * first


def test_classifier_exceeds_90pct():
    """Verify a ReLU network trained with Adam reaches >90% accuracy on a linearly separable task."""
    rng2 = np.random.default_rng(42)
    X_np = rng2.standard_normal((200, 4)).astype(np.float32)
    y_np = ((X_np[:, 0] + X_np[:, 1]) > 0).astype(np.int64)
    X = Tensor(X_np)
    y = Tensor(y_np, dtype='int64')
    m = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 2))
    opt = vp.optim.Adam(list(m.parameters()), lr=0.05)
    for _ in range(200):
        opt.zero_grad(); loss = nn.CrossEntropyLoss()(m(X), y); loss.backward(); opt.step()
    preds = m(X)._data.argmax(axis=1)
    acc = (preds == y_np).mean()
    assert acc > 0.90, f'Expected >90% accuracy, got {acc:.1%}'


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn(); print('PASS', name)
