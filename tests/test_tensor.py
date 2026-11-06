"""Tests for the core Tensor class."""
import numpy as np
import pytest
import vectorphile as vp
from vectorphile.core.tensor import Tensor


class TestTensorCreation:
    def test_from_list(self):
        t = vp.tensor([1, 2, 3])
        assert t.shape == (3,)

    def test_zeros(self):
        t = vp.zeros(3, 4)
        assert t.shape == (3, 4)
        assert np.all(t.numpy() == 0)

    def test_ones(self):
        t = vp.ones(2, 3)
        assert np.all(t.numpy() == 1)

    def test_rand(self):
        t = vp.rand(10, 10)
        assert t.shape == (10, 10)
        assert 0.0 <= t.numpy().min()
        assert t.numpy().max() <= 1.0

    def test_randn(self):
        t = vp.randn(1000)
        assert abs(float(t.mean().item())) < 0.2

    def test_arange(self):
        t = vp.arange(0, 10, 1)
        assert len(t) == 10

    def test_linspace(self):
        t = vp.linspace(0, 1, 100)
        assert len(t) == 100
        assert abs(float(t[-1].item()) - 1.0) < 1e-5

    def test_eye(self):
        t = vp.eye(3)
        assert t.shape == (3, 3)
        assert np.allclose(t.numpy(), np.eye(3))

    def test_from_numpy(self):
        arr = np.array([1.0, 2.0, 3.0])
        t = vp.from_numpy(arr)
        assert np.allclose(t.numpy(), arr)


class TestTensorOps:
    def test_add(self):
        a = vp.tensor([1.0, 2.0])
        b = vp.tensor([3.0, 4.0])
        c = a + b
        assert np.allclose(c.numpy(), [4.0, 6.0])

    def test_sub(self):
        a = vp.tensor([5.0, 3.0])
        b = vp.tensor([2.0, 1.0])
        assert np.allclose((a - b).numpy(), [3.0, 2.0])

    def test_mul(self):
        a = vp.tensor([2.0, 3.0])
        b = vp.tensor([4.0, 5.0])
        assert np.allclose((a * b).numpy(), [8.0, 15.0])

    def test_div(self):
        a = vp.tensor([10.0, 6.0])
        b = vp.tensor([2.0, 3.0])
        assert np.allclose((a / b).numpy(), [5.0, 2.0])

    def test_matmul(self):
        a = vp.randn(3, 4)
        b = vp.randn(4, 5)
        c = a @ b
        assert c.shape == (3, 5)

    def test_pow(self):
        a = vp.tensor([2.0, 3.0])
        assert np.allclose((a ** 2).numpy(), [4.0, 9.0])

    def test_neg(self):
        a = vp.tensor([1.0, -2.0])
        assert np.allclose((-a).numpy(), [-1.0, 2.0])

    def test_comparison(self):
        a = vp.tensor([1.0, 2.0, 3.0])
        b = vp.tensor([2.0, 2.0, 2.0])
        assert np.all((a < b).numpy() == [True, False, False])

    def test_reshape(self):
        t = vp.arange(0, 12)
        r = t.reshape(3, 4)
        assert r.shape == (3, 4)

    def test_transpose(self):
        t = vp.randn(3, 4)
        assert t.T.shape == (4, 3)

    def test_sum(self):
        t = vp.ones(3, 4)
        assert float(t.sum().item()) == 12.0

    def test_mean(self):
        t = vp.ones(4) * 2
        assert float(t.mean().item()) == 2.0

    def test_max_min(self):
        t = vp.tensor([3.0, 1.0, 4.0, 1.0, 5.0])
        assert float(t.max().item()) == 5.0
        assert float(t.min().item()) == 1.0

    def test_argmax(self):
        t = vp.tensor([1.0, 5.0, 3.0])
        assert int(t.argmax().item()) == 1

    def test_cat(self):
        a = vp.ones(3)
        b = vp.ones(3) * 2
        c = vp.cat([a, b])
        assert c.shape == (6,)

    def test_stack(self):
        a = vp.ones(3)
        b = vp.ones(3)
        c = vp.stack([a, b])
        assert c.shape == (2, 3)

    def test_squeeze_unsqueeze(self):
        t = vp.ones(3, 1, 4)
        assert t.squeeze(1).shape == (3, 4)
        assert t.squeeze().shape == (3, 4)
        assert vp.ones(3, 4).unsqueeze(0).shape == (1, 3, 4)

    def test_flatten(self):
        t = vp.ones(2, 3, 4)
        assert t.flatten().shape == (24,)


class TestTensorMath:
    def test_exp_log(self):
        t = vp.tensor([1.0, 2.0, 3.0])
        assert np.allclose(t.exp().log().numpy(), t.numpy(), atol=1e-5)

    def test_sqrt(self):
        t = vp.tensor([4.0, 9.0, 16.0])
        assert np.allclose(t.sqrt().numpy(), [2.0, 3.0, 4.0])

    def test_abs(self):
        t = vp.tensor([-1.0, 2.0, -3.0])
        assert np.allclose(t.abs().numpy(), [1.0, 2.0, 3.0])

    def test_sigmoid(self):
        t = vp.tensor([0.0])
        assert abs(float(t.sigmoid().item()) - 0.5) < 1e-5

    def test_relu(self):
        t = vp.tensor([-1.0, 0.0, 1.0])
        assert np.allclose(t.relu().numpy(), [0.0, 0.0, 1.0])

    def test_tanh(self):
        t = vp.tensor([0.0])
        assert abs(float(t.tanh().item())) < 1e-5

    def test_clamp(self):
        t = vp.tensor([-5.0, 0.5, 10.0])
        assert np.allclose(t.clamp(-1, 1).numpy(), [-1.0, 0.5, 1.0])

    def test_softmax(self):
        t = vp.tensor([[1.0, 2.0, 3.0]])
        s = vp.softmax(t)
        assert abs(float(s.sum().item()) - 1.0) < 1e-5

    def test_einsum(self):
        a = vp.randn(3, 4)
        b = vp.randn(4, 5)
        c = vp.einsum('ij,jk->ik', a, b)
        assert c.shape == (3, 5)
        assert np.allclose(c.numpy(), (a @ b).numpy(), atol=1e-4)


class TestTensorAutograd:
    def test_backward_add(self):
        x = vp.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = x + 1.0
        loss = y.sum()
        loss.backward()
        assert np.allclose(x.grad.numpy(), np.ones(3))

    def test_backward_mul(self):
        x = vp.tensor([2.0, 3.0], requires_grad=True)
        y = vp.tensor([4.0, 5.0], requires_grad=True)
        z = (x * y).sum()
        z.backward()
        assert np.allclose(x.grad.numpy(), [4.0, 5.0])
        assert np.allclose(y.grad.numpy(), [2.0, 3.0])

    def test_zero_grad(self):
        x = vp.tensor([1.0], requires_grad=True)
        loss = (x * 2).sum()
        loss.backward()
        assert x.grad is not None
        x.zero_grad()
        assert x.grad is None
