"""Transformer wrappers, InstanceNorm, 3-D / adaptive-max pooling, transposed conv, no_grad, parameter de-duplication."""
import numpy as np
import vectorphile as vp
import vectorphile.nn as nn
import vectorphile.nn.functional as F
from vectorphile.core.tensor import Tensor
from tests.test_layers_autograd import check_module, T


def test_transformer_wrappers():
    enc_layer = nn.TransformerEncoderLayer(8, 2, 16, batch_first=True)
    ps = list(enc_layer.parameters())
    assert len(ps) == len({id(p) for p in ps}) == 14          # no parameter registered twice
    enc_layer.eval()
    check_module(enc_layer, T(2, 4, 8))
    enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(8, 2, 16, batch_first=True), 2)
    assert len(list(enc.parameters())) == 28
    enc.eval()
    check_module(enc, T(2, 4, 8))
    dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(8, 2, 16, batch_first=True), 2)
    assert len(list(dec.parameters())) == 44
    dec.eval()
    mem = Tensor(np.random.randn(2, 3, 8))
    causal = np.triu(np.ones((4, 4), dtype=bool), 1)
    check_module(dec, T(2, 4, 8), lambda m, x: m(x, mem, tgt_mask=causal))


def test_instance_norm_pool3d_adaptive_max_conv_transpose():
    check_module(nn.InstanceNorm1d(3, affine=True), T(2, 3, 6))
    check_module(nn.InstanceNorm2d(3, affine=True), T(2, 3, 4, 4))
    check_module(nn.MaxPool3d(2), T(1, 2, 4, 4, 4))
    check_module(nn.AvgPool3d(2), T(1, 2, 4, 4, 4))
    x = T(1, 1, 4, 4, 4)
    ref = x._data.reshape(1, 1, 2, 2, 2, 2, 2, 2).max(axis=(3, 5, 7))
    assert np.allclose(nn.MaxPool3d(2)(x)._data, ref)
    check_module(nn.AdaptiveMaxPool2d(2), T(2, 2, 6, 6))
    check_module(nn.AdaptiveMaxPool2d(3), T(1, 2, 7, 5))
    w = T(3, 2, 3, 3)
    check_module(nn.Sequential(), T(2, 3, 4, 4), lambda m, x: F.conv_transpose2d(x, w, None, 2, 1))
    assert w.grad is not None
    # transposed conv is the adjoint of conv: <conv(a), b> == <a, conv_transpose(b)>
    a, b = T(1, 2, 7, 7), T(1, 3, 4, 4)
    k = T(3, 2, 3, 3)
    lhs = float((F.conv2d(a, k, None, 2, 1) * b).sum()._data)
    rhs = float((a * F.conv_transpose2d(b, k, None, 2, 1)).sum()._data)
    assert abs(lhs - rhs) < 1e-8 * max(1.0, abs(lhs))


def test_no_grad_builds_no_graph():
    lin = nn.Linear(4, 2)
    with vp.no_grad():
        y = lin(Tensor(np.ones((1, 4), dtype=np.float32)))
    assert y.grad_fn is None and not y.requires_grad
    assert lin(Tensor(np.ones((1, 4), dtype=np.float32))).grad_fn is not None


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('PASS', name)
