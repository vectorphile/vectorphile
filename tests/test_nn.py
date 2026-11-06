"""Tests for neural network components."""
import numpy as np
import pytest
import vectorphile as vp


class TestLayers:
    def test_linear_shape(self):
        layer = vp.nn.Linear(4, 8)
        x = vp.randn(16, 4)
        y = layer(x)
        assert y.shape == (16, 8)

    def test_linear_params(self):
        layer = vp.nn.Linear(4, 8)
        params = list(layer.parameters())
        assert len(params) == 2  # weight + bias

    def test_relu(self):
        relu = vp.nn.ReLU()
        x = vp.tensor([-1.0, 0.0, 1.0])
        y = relu(x)
        assert np.allclose(y.numpy(), [0.0, 0.0, 1.0])

    def test_dropout_train(self):
        drop = vp.nn.Dropout(p=0.5)
        drop.train()
        x = vp.ones(1000)
        y = drop(x)
        # Some elements should be zeroed
        assert float(y.sum().item()) < 1000.0

    def test_dropout_eval(self):
        drop = vp.nn.Dropout(p=0.5)
        drop.eval()
        x = vp.ones(100)
        y = drop(x)
        assert np.allclose(y.numpy(), x.numpy())

    def test_embedding(self):
        emb = vp.nn.Embedding(10, 4)
        idx = vp.tensor([0, 1, 2])
        out = emb(idx)
        assert out.shape == (3, 4)

    def test_layernorm(self):
        ln = vp.nn.LayerNorm(8)
        x = vp.randn(4, 8)
        y = ln(x)
        assert y.shape == (4, 8)
        # Mean should be ~0 per sample
        means = y._data.mean(axis=-1)
        assert np.allclose(means, 0, atol=1e-5)

    def test_batchnorm1d(self):
        bn = vp.nn.BatchNorm1d(4)
        x = vp.randn(16, 4)
        y = bn(x)
        assert y.shape == (16, 4)


class TestActivations:
    def test_gelu(self):
        gelu = vp.nn.GELU()
        x = vp.tensor([0.0])
        assert abs(float(gelu(x).item())) < 0.01

    def test_silu(self):
        silu = vp.nn.SiLU()
        x = vp.tensor([0.0])
        assert abs(float(silu(x).item())) < 0.01

    def test_softmax(self):
        sm = vp.nn.Softmax(dim=-1)
        x = vp.randn(4, 10)
        y = sm(x)
        sums = y._data.sum(axis=-1)
        assert np.allclose(sums, 1.0, atol=1e-5)

    def test_sigmoid(self):
        sig = vp.nn.Sigmoid()
        x = vp.tensor([0.0])
        assert abs(float(sig(x).item()) - 0.5) < 1e-5


class TestSequential:
    def test_basic_forward(self):
        model = vp.nn.Sequential(
            vp.nn.Linear(8, 16),
            vp.nn.ReLU(),
            vp.nn.Linear(16, 4),
        )
        x = vp.randn(10, 8)
        y = model(x)
        assert y.shape == (10, 4)

    def test_len(self):
        model = vp.nn.Sequential(vp.nn.Linear(4, 8), vp.nn.ReLU())
        assert len(model) == 2

    def test_parameters(self):
        model = vp.nn.Sequential(vp.nn.Linear(4, 8), vp.nn.Linear(8, 2))
        params = list(model.parameters())
        assert len(params) == 4  # 2 weights + 2 biases

    def test_train_eval(self):
        model = vp.nn.Sequential(vp.nn.Linear(4, 8), vp.nn.Dropout(0.5))
        model.eval()
        assert not model.training
        model.train()
        assert model.training


class TestLosses:
    def test_mse(self):
        loss = vp.nn.MSELoss()
        pred = vp.tensor([1.0, 2.0, 3.0])
        target = vp.tensor([1.0, 2.0, 3.0])
        assert float(loss(pred, target).item()) == 0.0

    def test_cross_entropy(self):
        loss = vp.nn.CrossEntropyLoss()
        logits = vp.randn(10, 5)
        labels = vp.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        val = loss(logits, labels)
        assert float(val.item()) > 0.0

    def test_bce_with_logits(self):
        loss = vp.nn.BCEWithLogitsLoss()
        logits = vp.randn(10)
        targets = vp.tensor(np.random.randint(0, 2, 10).astype(np.float32))
        val = loss(logits, targets)
        assert float(val.item()) > 0.0


class TestOptimizers:
    def _make_simple_model(self):
        model = vp.nn.Linear(4, 1)
        return model

    def test_sgd_step(self):
        model = self._make_simple_model()
        optimizer = vp.optim.SGD(model.parameters(), lr=0.01)
        X = vp.randn(16, 4)
        y = vp.randn(16, 1)
        pred = model(X)
        loss = vp.nn.MSELoss()(pred, y)
        loss.backward()
        w_before = model.weight._data.copy()
        optimizer.step()
        assert not np.allclose(model.weight._data, w_before)

    def test_adam_step(self):
        model = self._make_simple_model()
        optimizer = vp.optim.Adam(model.parameters(), lr=0.001)
        X = vp.randn(16, 4)
        y = vp.randn(16, 1)
        pred = model(X)
        loss = vp.nn.MSELoss()(pred, y)
        loss.backward()
        w_before = model.weight._data.copy()
        optimizer.step()
        assert not np.allclose(model.weight._data, w_before)

    def test_zero_grad(self):
        model = self._make_simple_model()
        optimizer = vp.optim.SGD(model.parameters(), lr=0.01)
        X = vp.randn(4, 4)
        y = vp.randn(4, 1)
        loss = vp.nn.MSELoss()(model(X), y)
        loss.backward()
        optimizer.zero_grad()
        for p in model.parameters():
            assert p.grad is None


class TestTrainingLoop:
    def test_mlp_learns_xor(self):
        """MLP should be able to fit XOR after a few hundred iterations."""
        np.random.seed(42)
        X = vp.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype='float32')
        y = vp.tensor([[0], [1], [1], [0]], dtype='float32')
        model = vp.nn.Sequential(
            vp.nn.Linear(2, 8),
            vp.nn.Tanh(),
            vp.nn.Linear(8, 1),
            vp.nn.Sigmoid(),
        )
        optimizer = vp.optim.Adam(model.parameters(), lr=0.05)
        loss_fn = vp.nn.BCELoss()
        for _ in range(1000):
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
        # Check predictions are correct side of 0.5
        preds = model(X)._data.flatten()
        assert preds[0] < 0.5  # 0 XOR 0 = 0
        assert preds[1] > 0.5  # 0 XOR 1 = 1
        assert preds[2] > 0.5  # 1 XOR 0 = 1
        assert preds[3] < 0.5  # 1 XOR 1 = 0
