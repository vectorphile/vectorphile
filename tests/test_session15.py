"""Session 15 feature tests: distributions, SAM, Lookahead."""
import pytest
import vectorphile as vp
from vectorphile.core.tensor import Tensor
from vectorphile._core.ndarray import array as _nd_arr
import math


# ── distributions ─────────────────────────────────────────────────────────────

class TestNormal:
    def test_sample_shape(self):
        from vectorphile.distributions import Normal
        n = Normal(0.0, 1.0)
        s = n.sample((5,))
        assert s._data.shape == (5,)

    def test_log_prob(self):
        from vectorphile.distributions import Normal
        n = Normal(0.0, 1.0)
        x = Tensor(_nd_arr([0.0], dtype='float32'))
        lp = n.log_prob(x)
        flat = list(lp._data.to_flat_list() if hasattr(lp._data, 'to_flat_list') else lp._data)
        # log_prob(0) for N(0,1) = -0.5*log(2*pi) ≈ -0.9189
        assert abs(flat[0] - (-0.9189)) < 0.01

    def test_entropy(self):
        from vectorphile.distributions import Normal
        n = Normal(0.0, 1.0)
        e = n.entropy()
        flat = list(e._data.to_flat_list() if hasattr(e._data, 'to_flat_list') else e._data)
        # H(N(0,1)) = 0.5*log(2*pi*e) ≈ 1.4189
        assert abs(flat[0] - 1.4189) < 0.01

    def test_kl_divergence(self):
        from vectorphile.distributions import Normal, kl_divergence
        p = Normal(0.0, 1.0)
        q = Normal(0.0, 1.0)
        kl = kl_divergence(p, q)
        flat = list(kl._data.to_flat_list() if hasattr(kl._data, 'to_flat_list') else kl._data)
        assert abs(flat[0]) < 1e-5  # KL(p||p) = 0


class TestBernoulli:
    def test_mean(self):
        from vectorphile.distributions import Bernoulli
        b = Bernoulli(probs=0.7)
        m = b.mean
        flat = list(m._data.to_flat_list() if hasattr(m._data, 'to_flat_list') else m._data)
        assert abs(flat[0] - 0.7) < 1e-5

    def test_sample_binary(self):
        from vectorphile.distributions import Bernoulli
        b = Bernoulli(probs=0.0)
        s = b.sample()
        flat = list(s._data.to_flat_list() if hasattr(s._data, 'to_flat_list') else s._data)
        assert flat[0] == 0.0

    def test_logits(self):
        from vectorphile.distributions import Bernoulli
        # logits=0 → p=0.5
        b = Bernoulli(logits=0.0)
        m = b.mean
        flat = list(m._data.to_flat_list() if hasattr(m._data, 'to_flat_list') else m._data)
        assert abs(flat[0] - 0.5) < 1e-5


class TestCategorical:
    def test_sample_range(self):
        from vectorphile.distributions import Categorical
        c = Categorical(probs=[0.1, 0.5, 0.4])
        for _ in range(10):
            s = c.sample()
            flat = list(s._data.to_flat_list() if hasattr(s._data, 'to_flat_list') else s._data)
            assert flat[0] in [0.0, 1.0, 2.0]

    def test_entropy_positive(self):
        from vectorphile.distributions import Categorical
        c = Categorical(probs=[0.5, 0.5])
        e = c.entropy()
        flat = list(e._data.to_flat_list() if hasattr(e._data, 'to_flat_list') else e._data)
        assert flat[0] > 0


class TestUniform:
    def test_mean(self):
        from vectorphile.distributions import Uniform
        u = Uniform(0.0, 2.0)
        m = u.mean
        flat = list(m._data.to_flat_list() if hasattr(m._data, 'to_flat_list') else m._data)
        assert abs(flat[0] - 1.0) < 1e-5

    def test_sample_in_range(self):
        from vectorphile.distributions import Uniform
        u = Uniform(2.0, 5.0)
        for _ in range(20):
            s = u.sample()
            flat = list(s._data.to_flat_list() if hasattr(s._data, 'to_flat_list') else s._data)
            assert 2.0 <= flat[0] < 5.0


class TestExponential:
    def test_mean(self):
        from vectorphile.distributions import Exponential
        e = Exponential(rate=2.0)
        m = e.mean
        flat = list(m._data.to_flat_list() if hasattr(m._data, 'to_flat_list') else m._data)
        assert abs(flat[0] - 0.5) < 1e-5

    def test_sample_positive(self):
        from vectorphile.distributions import Exponential
        e = Exponential(rate=1.0)
        s = e.sample((5,))
        flat = list(s._data.to_flat_list() if hasattr(s._data, 'to_flat_list') else s._data)
        assert all(v > 0 for v in flat)


class TestBeta:
    def test_mean(self):
        from vectorphile.distributions import Beta
        b = Beta(2.0, 2.0)
        m = b.mean
        flat = list(m._data.to_flat_list() if hasattr(m._data, 'to_flat_list') else m._data)
        assert abs(flat[0] - 0.5) < 1e-5

    def test_sample_in_01(self):
        from vectorphile.distributions import Beta
        b = Beta(1.0, 1.0)
        for _ in range(10):
            s = b.sample()
            flat = list(s._data.to_flat_list() if hasattr(s._data, 'to_flat_list') else s._data)
            assert 0.0 <= flat[0] <= 1.0


# ── SAM optimizer ─────────────────────────────────────────────────────────────

class TestSAM:
    def test_import(self):
        from vectorphile.optim import SAM
        assert SAM is not None

    def test_first_second_step(self):
        from vectorphile.optim import SAM, SGD
        m = vp.nn.Linear(4, 2)
        opt = SAM(m.parameters(), SGD, rho=0.05, lr=0.01)
        x = vp.randn(2, 4)
        # first forward
        y = m(x)
        loss = y.mean()
        loss.backward()
        opt.first_step(zero_grad=True)
        # second forward
        y2 = m(x)
        loss2 = y2.mean()
        loss2.backward()
        opt.second_step(zero_grad=True)


# ── Lookahead optimizer ───────────────────────────────────────────────────────

class TestLookahead:
    def test_import(self):
        from vectorphile.optim import Lookahead
        assert Lookahead is not None

    def test_step(self):
        from vectorphile.optim import Lookahead, Adam
        m = vp.nn.Linear(4, 2)
        base = Adam(m.parameters(), lr=1e-3)
        opt = Lookahead(base, k=3, alpha=0.5)
        for _ in range(6):
            x = vp.randn(2, 4)
            y = m(x)
            loss = y.mean()
            loss.backward()
            opt.step()
            opt.zero_grad()

    def test_state_dict(self):
        from vectorphile.optim import Lookahead, SGD
        m = vp.nn.Linear(2, 2)
        base = SGD(m.parameters(), lr=0.01)
        opt = Lookahead(base, k=2)
        for _ in range(2):
            x = vp.randn(2, 2)
            y = m(x)
            y.mean().backward()
            opt.step()
            opt.zero_grad()
        sd = opt.state_dict()
        assert sd['step_count'] == 2
