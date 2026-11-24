"""Adam and AdamW optimizers."""
import math as _math
import array as _arr
from vectorphile.optim.optimizer import Optimizer


def _to_flat(d):
    """Return a flat Python list of float32 values from NDArray or list."""
    from vectorphile._core.ndarray import NDArray as _NDA
    if isinstance(d, _NDA):
        if d._buf is not None:
            return list(d._buf)
        return [float(v) for v in d.to_flat_list()]
    if isinstance(d, list):
        return [float(v) for v in d]
    try:
        return [float(v) for v in d]
    except Exception:
        return [float(d)]


def _get_shape(d):
    from vectorphile._core.ndarray import NDArray as _NDA
    if isinstance(d, _NDA):
        return d.shape
    if isinstance(d, list):
        return (len(d),)
    return (1,)


def _to_np(d):
    """Return (flat_list, shape) of float values."""
    return _to_flat(d), _get_shape(d)


def _vec_op2(a, b, fn):
    return [fn(ai, bi) for ai, bi in zip(a, b)]


def _vec_scale(a, s):
    return [v * s for v in a]


def _vec_add(a, b):
    return [ai + bi for ai, bi in zip(a, b)]


def _vec_sub(a, b):
    return [ai - bi for ai, bi in zip(a, b)]


def _vec_sqrt(a):
    return [_math.sqrt(max(v, 0.0)) for v in a]


def _vec_div(a, b):
    return [ai / bi for ai, bi in zip(a, b)]


def _write_back(p, flat, shape):
    """Write flat list back into p._data."""
    from vectorphile._core.ndarray import NDArray as _NDA, _c_strides
    d = p._data
    if isinstance(d, _NDA):
        new_buf = _arr.array('f', [float(v) for v in flat])
        d._buf = new_buf
    else:
        p._data = flat


class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)
        self.state = {}

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            b1, b2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            amsgrad = group['amsgrad']
            for p in group['params']:
                if p.grad is None:
                    continue
                g, sh = _to_np(p.grad._data)
                pid = id(p)
                n = len(g)
                if pid not in self.state:
                    self.state[pid] = {
                        'step': 0,
                        'm': [0.0] * n,
                        'v': [0.0] * n,
                    }
                    if amsgrad:
                        self.state[pid]['v_max'] = [0.0] * n
                s = self.state[pid]
                s['step'] += 1
                t = s['step']
                # --- Triton fused Adam fast path (no amsgrad) ---
                if not amsgrad:
                    try:
                        from vectorphile.cuda.triton_kernels import fused_adam as _fa
                        from vectorphile._core.ndarray import array as _nd_opt
                        p_flat, p_sh = _to_np(p._data)
                        p_nd = _nd_opt(p_flat, 'float32')
                        g_nd = _nd_opt(g, 'float32')
                        m_nd = _nd_opt(s['m'], 'float32')
                        v_nd = _nd_opt(s['v'], 'float32')
                        p_new, m_new, v_new = _fa(p_nd, g_nd, m_nd, v_nd,
                                                   lr, b1, b2, eps, wd, t)
                        s['m'] = m_new.to_flat_list() if hasattr(m_new, 'to_flat_list') else list(m_new)
                        s['v'] = v_new.to_flat_list() if hasattr(v_new, 'to_flat_list') else list(v_new)
                        new_fl = p_new.to_flat_list() if hasattr(p_new, 'to_flat_list') else list(p_new)
                        _write_back(p, new_fl, p_sh)
                        continue
                    except Exception:
                        pass
                # --- Python fallback ---
                s['m'] = _vec_add(_vec_scale(s['m'], b1), _vec_scale(g, 1 - b1))
                s['v'] = _vec_add(_vec_scale(s['v'], b2), [b * v * v for b, v in zip([1 - b2] * n, g)])
                bias1 = 1 - b1 ** t
                bias2 = 1 - b2 ** t
                m_hat = _vec_scale(s['m'], 1.0 / bias1)
                v_hat = _vec_scale(s['v'], 1.0 / bias2)
                if amsgrad:
                    s['v_max'] = [max(a, b) for a, b in zip(s['v_max'], v_hat)]
                    denom = [_math.sqrt(max(v, 0.0)) + eps for v in s['v_max']]
                else:
                    denom = [_math.sqrt(max(v, 0.0)) + eps for v in v_hat]
                p_flat, p_sh = _to_np(p._data)
                if wd != 0:
                    p_flat = _vec_sub(p_flat, _vec_scale(p_flat, lr * wd))
                step_v = _vec_div(m_hat, denom)
                p_flat = _vec_sub(p_flat, _vec_scale(step_v, lr))
                _write_back(p, p_flat, p_sh)


class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.state = {}

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            b1, b2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                g, sh = _to_np(p.grad._data)
                pid = id(p)
                n = len(g)
                if pid not in self.state:
                    self.state[pid] = {'step': 0, 'm': [0.0] * n, 'v': [0.0] * n}
                s = self.state[pid]
                s['step'] += 1
                t = s['step']
                # Triton fused path
                try:
                    from vectorphile.cuda.triton_kernels import fused_adam as _fa2
                    from vectorphile._core.ndarray import array as _nd_opt2
                    p_flat2, p_sh2 = _to_np(p._data)
                    p_new2, m_new2, v_new2 = _fa2(
                        _nd_opt2(p_flat2, 'float32'), _nd_opt2(g, 'float32'),
                        _nd_opt2(s['m'], 'float32'), _nd_opt2(s['v'], 'float32'),
                        lr, b1, b2, eps, wd, t)
                    s['m'] = m_new2.to_flat_list() if hasattr(m_new2, 'to_flat_list') else list(m_new2)
                    s['v'] = v_new2.to_flat_list() if hasattr(v_new2, 'to_flat_list') else list(v_new2)
                    _write_back(p, p_new2.to_flat_list() if hasattr(p_new2, 'to_flat_list') else list(p_new2), p_sh2)
                    continue
                except Exception:
                    pass
                s['m'] = _vec_add(_vec_scale(s['m'], b1), _vec_scale(g, 1 - b1))
                s['v'] = _vec_add(_vec_scale(s['v'], b2), [(1 - b2) * v * v for v in g])
                m_hat = _vec_scale(s['m'], 1.0 / (1 - b1 ** t))
                v_hat = _vec_scale(s['v'], 1.0 / (1 - b2 ** t))
                p_flat, p_sh = _to_np(p._data)
                denom = [_math.sqrt(max(v, 0.0)) + eps for v in v_hat]
                p_flat = [pv * (1 - lr * wd) - lr * mv / dv
                          for pv, mv, dv in zip(p_flat, m_hat, denom)]
                _write_back(p, p_flat, p_sh)
