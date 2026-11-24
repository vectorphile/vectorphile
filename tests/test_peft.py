"""Fine-tuning methods: adapters train, merge exactly, and preference losses optimise."""
import numpy as np
import vectorphile as vp
import vectorphile.nn as nn
from vectorphile import peft
from vectorphile.core.tensor import Tensor

rng = np.random.default_rng(0)
X = Tensor(rng.standard_normal((96, 6)).astype(np.float32))
Y = Tensor((X._data[:, 0] + X._data[:, 1] > 0).astype(np.int64), dtype='int64')


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(6, 6), nn.Tanh()) for _ in range(2)])
        self.lm_head = nn.Linear(6, 2)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return self.lm_head(x)


def _train(model, params, steps=150, lr=0.05):
    opt = vp.optim.Adam(params, lr=lr)
    first = last = None
    for _ in range(steps):
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(model(X), Y)
        loss.backward()
        opt.step()
        last = float(loss._data)
        first = last if first is None else first
    return first, last


def _adapter_case(apply, mergeable=True, **kw):
    m = Net()
    base = {k: v.copy() for k, v in m.state_dict().items()}
    assert apply(m, **kw) == 2
    stats = peft.count_parameters(m)
    assert 0 < stats['trainable'] < stats['total']
    first, last = _train(m, peft.trainable_parameters(m))
    assert last < 0.9 * first, (first, last)
    m.eval()
    before = m(X)._data.copy()
    if mergeable:
        assert peft.merge_adapters(m) == 2
        assert np.allclose(m(X)._data, before, atol=1e-4)
        # plain keys again: the result reloads into the base architecture
        assert set(m.state_dict()) == set(base)
        assert np.allclose(m.state_dict()['lm_head.weight'], base['lm_head.weight'])


def test_lora():
    _adapter_case(peft.apply_lora, r=4, alpha=8)


def test_rslora():
    _adapter_case(peft.apply_lora, r=4, alpha=8, rslora=True)


def test_dora():
    _adapter_case(peft.apply_dora, r=4, alpha=8)


def test_qlora():
    _adapter_case(peft.apply_qlora, bits=4, r=4, alpha=8)


def test_ia3():
    _adapter_case(peft.apply_ia3)


def test_adapters():
    _adapter_case(peft.apply_adapters, mergeable=False, bottleneck=4)


def test_selection_and_groups():
    m = Net()
    assert peft.apply_bitfit(m) == 3
    peft.unfreeze_last_n(m, 1, also=('lm_head',))
    assert peft.count_parameters(m)['trainable'] == 42 + 14
    m = Net()
    peft.apply_lora(m, r=2)
    groups = peft.lora_plus_param_groups(m, lr=0.01, ratio=16)
    assert groups[1]['lr'] == 0.16 and len(groups[1]['params']) == 2
    opt = vp.optim.AdamW(groups)
    assert [g['lr'] for g in opt.param_groups] == [0.01, 0.16]
    sd = peft.adapter_state_dict(m)
    assert len(sd) == 4
    peft.load_adapter_state_dict(m, sd)
    m2 = Net()
    assert [round(g['lr'], 4) for g in peft.layerwise_lr_decay_groups(m2, 0.1, 0.5)] == [0.05, 0.1, 0.1]


def test_preference_losses_optimise():
    logits = Tensor(rng.standard_normal((4, 5, 7)).astype(np.float32), requires_grad=True)
    chosen = rng.integers(0, 7, (4, 5))
    rejected = rng.integers(0, 7, (4, 5))

    def margin():
        return float((peft.sequence_logprob(logits, chosen) - peft.sequence_logprob(logits, rejected))._data.mean())

    for name in ('dpo', 'ipo', 'simpo', 'orpo', 'kto'):
        logits._data[...] = rng.standard_normal(logits._data.shape)
        ref_c = Tensor(peft.sequence_logprob(Tensor(logits._data), chosen)._data)
        ref_r = Tensor(peft.sequence_logprob(Tensor(logits._data), rejected)._data)
        start = margin()
        opt = vp.optim.Adam([logits], lr=0.05)
        for _ in range(40):
            opt.zero_grad()
            pc, pr = peft.sequence_logprob(logits, chosen), peft.sequence_logprob(logits, rejected)
            pca = peft.sequence_logprob(logits, chosen, average=True)
            pra = peft.sequence_logprob(logits, rejected, average=True)
            if name == 'dpo':
                loss = peft.dpo_loss(pc, pr, ref_c, ref_r, 0.5)
            elif name == 'ipo':
                loss = peft.ipo_loss(pc, pr, ref_c, ref_r, 0.5)
            elif name == 'simpo':
                loss = peft.simpo_loss(pca, pra)
            elif name == 'orpo':
                loss = peft.orpo_loss(-pca.mean(), pca, pra, lam=1.0)
            else:
                loss = peft.kto_loss(vp.cat([pc, pr]), vp.cat([ref_c, ref_r]), np.array([1] * 4 + [0] * 4), beta=0.5)
            loss.backward()
            opt.step()
        assert margin() > start, name


def test_distillation_soft_prompt_neftune():
    s = Tensor(rng.standard_normal((8, 5)).astype(np.float32), requires_grad=True)
    t = Tensor(rng.standard_normal((8, 5)).astype(np.float32))
    opt = vp.optim.Adam([s], lr=0.1)
    first = last = None
    for _ in range(60):
        opt.zero_grad()
        loss = peft.distillation_loss(s, t)
        loss.backward()
        opt.step()
        last = float(loss._data)
        first = last if first is None else first
    assert last < 0.2 * first
    sp = peft.SoftPrompt(3, 4)
    out = sp(Tensor(rng.standard_normal((2, 5, 4)).astype(np.float32)))
    assert out.shape == (2, 8, 4)
    out.sum().backward()
    assert sp.prompt.grad is not None and np.allclose(sp.prompt.grad._data, 2.0)
    e = Tensor(np.zeros((2, 5, 4), dtype=np.float32))
    assert np.abs(peft.neftune(e, 5.0)._data).max() <= 5.0 / np.sqrt(20) + 1e-6
    assert peft.neftune(e, 5.0, training=False) is e


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('PASS', name)
