"""Loss functions — pure Python internals."""
import math
import array as _arr
from vectorphile.nn.module import Module
from vectorphile.core.tensor import Tensor

try:
    from vectorphile._core import _vp_core as _C_loss
    from vectorphile._core.ndarray import NDArray as _NDA_loss, _c_strides as _cs_loss, _DType as _DType_loss
    _LOSS_C_OK = (hasattr(_C_loss, 'ispc_cross_entropy_fwd_f32')
                  and hasattr(_C_loss, 'ispc_cross_entropy_bwd_f32'))
except ImportError:
    _C_loss = None
    _LOSS_C_OK = False

_ce_losses_cache: dict = {}
_ce_grad_cache: dict = {}
_ce_target_cache: dict = {}


def _get_flat(d):
    if hasattr(d, 'to_flat_list'): return d.to_flat_list()
    if hasattr(d, 'tolist'): return d.tolist()
    if isinstance(d, list): return d
    return list(d)


class MSELoss(Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        diff = pred - target; sq = diff * diff
        if self.reduction == 'mean': return sq.mean()
        if self.reduction == 'sum': return sq.sum()
        return sq


class MAELoss(Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        diff = pred - target; abs_diff = diff.abs()
        if self.reduction == 'mean': return abs_diff.mean()
        if self.reduction == 'sum': return abs_diff.sum()
        return abs_diff


# Alias: L1Loss == MAELoss (PyTorch compatibility)
class L1Loss(Module):
    """Mean Absolute Error loss (alias for MAELoss)."""
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        diff = pred - target; abs_diff = diff.abs()
        if self.reduction == 'mean': return abs_diff.mean()
        if self.reduction == 'sum': return abs_diff.sum()
        return abs_diff

    def __repr__(self): return f'L1Loss(reduction={self.reduction!r})'


class SmoothL1Loss(Module):
    """Smooth L1 (Huber) loss."""
    def __init__(self, reduction='mean', beta=1.0):
        super().__init__()
        self.reduction = reduction
        self.beta = beta

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        from vectorphile.nn.functional import smooth_l1_loss
        return smooth_l1_loss(pred, target, reduction=self.reduction, beta=self.beta)

    def __repr__(self): return f'SmoothL1Loss(reduction={self.reduction!r}, beta={self.beta})'


class _CEBackward:
    def __init__(self, logits_buf, targets_buf, N, C, reduction, ignore_index, logits_tensor=None):
        self.inputs = (logits_tensor,) if logits_tensor is not None else (None,)
        self.logits_buf = logits_buf
        self.targets_buf = targets_buf
        self.N = N; self.C = C
        self.reduction = reduction
        self.ignore_index = ignore_index

    def backward(self, grad):
        import struct as _struct
        gd = grad._data if hasattr(grad, '_data') else grad
        if hasattr(gd, '_buf') and gd._buf is not None:
            g_scale = float(gd._buf[0])
        elif hasattr(gd, 'to_flat_list'):
            g_scale = float(gd.to_flat_list()[0])
        else:
            try: g_scale = float(gd)
            except: g_scale = 1.0
        N, C = self.N, self.C
        raw = bytes(self.logits_buf)
        logits_flat = list(_struct.unpack_from(f'{N*C}f', raw))
        tgt_raw = bytes(self.targets_buf)
        targets = list(_struct.unpack_from(f'{N}i', tgt_raw))
        ignore = self.ignore_index
        valid = max(sum(1 for t in targets if t != ignore), 1) if ignore is not None else N
        if self.reduction != 'mean': valid = 1
        d_flat = [0.0] * (N * C)
        for n in range(N):
            row = logits_flat[n*C:(n+1)*C]
            mx = max(row); exps = [math.exp(v - mx) for v in row]; s = sum(exps)
            probs = [e / s for e in exps]
            tgt = targets[n]
            if ignore is not None and tgt == ignore: continue
            for c in range(C):
                p = probs[c] - (1.0 if c == tgt else 0.0)
                if self.reduction != 'none': p /= valid
                d_flat[n*C + c] = p * g_scale
        from vectorphile._core.ndarray import NDArray as _NDA_bwd, _c_strides as _cs_bwd, _DType as _DT_bwd
        out_buf = _arr.array('f', bytearray(4 * N * C))
        for i, v in enumerate(d_flat): out_buf[i] = v
        result_nd = _NDA_bwd(out_buf, (N, C), _cs_bwd((N, C)), _DT_bwd('float32'))
        return (Tensor(result_nd),)


class CrossEntropyLoss(Module):
    def __init__(self, reduction='mean', ignore_index=-100, label_smoothing=0.0):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        ld = logits._data; td = targets._data
        req_grad = logits.requires_grad

        if (_LOSS_C_OK and isinstance(ld, _NDA_loss) and ld._buf is not None
                and ld.ndim == 2 and ld.dtype == 'float32'
                and self.label_smoothing == 0.0 and self.ignore_index == -100):
            N, C = ld.shape
            if N not in _ce_target_cache:
                _ce_target_cache[N] = _arr.array('i', bytearray(4 * N))
            tgt_buf = _ce_target_cache[N]
            if isinstance(td, _NDA_loss) and td._buf is not None:
                for i in range(N): tgt_buf[i] = int(td._buf[i])
            else:
                for i, v in enumerate(_get_flat(td)): tgt_buf[i] = int(v)
            if N not in _ce_losses_cache:
                _ce_losses_cache[N] = _arr.array('f', bytearray(4 * N))
            losses_buf = _ce_losses_cache[N]
            _C_loss.ispc_cross_entropy_fwd_f32(ld._buf, tgt_buf, losses_buf, N, C)
            loss_val = _C_loss.sum_f32(losses_buf) / N
            from vectorphile._core.ndarray import array as _nd
            result = Tensor(_nd([loss_val]), device=logits.device, requires_grad=req_grad)
            if req_grad:
                result.grad_fn = _CEBackward(ld._buf, tgt_buf, N, C,
                                             self.reduction, self.ignore_index, logits)
                result._is_leaf = False
            return result

        # Triton GPU path (no label smoothing, no ignore_index)
        if self.label_smoothing == 0.0 and self.ignore_index == -100:
            try:
                from vectorphile.cuda.triton_kernels import (
                    fused_cross_entropy as _fce,
                    fused_cross_entropy_backward as _fce_bwd,
                )
                from vectorphile._core.ndarray import array as _nd_ce
                lsh2 = ld.shape if hasattr(ld, 'shape') else (1, 1)
                if len(lsh2) == 3:
                    N2, C2 = lsh2[0] * lsh2[1], lsh2[2]
                else:
                    N2, C2 = lsh2[0], lsh2[1] if len(lsh2) > 1 else 1
                loss_val2, ctx2 = _fce(ld, td, reduction=self.reduction)
                result2 = Tensor(_nd_ce([loss_val2]), device=logits.device, requires_grad=req_grad)
                if req_grad and ctx2 is not None:
                    _ctx2, _N2, _C2 = ctx2, N2, C2
                    class _CEBackwardTriton:
                        def __init__(s): s.inputs = (logits,)
                        def backward(s, grad):
                            g_scale = _get_flat(grad._data)[0] if hasattr(grad, '_data') else float(grad)
                            g_nd = _fce_bwd(_ctx2, g_scale, _N2, _C2)
                            return (Tensor(g_nd),)
                    result2.grad_fn = _CEBackwardTriton(); result2._is_leaf = False
                return result2
            except Exception:
                pass

        # Pure Python fallback
        lf = _get_flat(ld)
        lsh = ld.shape if hasattr(ld, 'shape') else (len(lf),)
        # Handle 3D logits (B, T, C) → flatten to (B*T, C)
        if len(lsh) == 3:
            C = lsh[2]; N = lsh[0] * lsh[1]
        else:
            N = lsh[0]; C = lsh[1] if len(lsh) > 1 else 1
        tf = [int(v) for v in _get_flat(td)]
        ls = self.label_smoothing; ig = self.ignore_index
        losses = []
        for n in range(N):
            row = lf[n*C:(n+1)*C]
            t = tf[n]
            if t == ig: losses.append(0.0); continue
            mx = max(row); exps = [math.exp(v - mx) for v in row]; s = sum(exps)
            log_probs = [v - mx - math.log(s) for v in row]
            nll = -log_probs[t]
            if ls > 0:
                nll = (1 - ls) * nll + ls * (-sum(log_probs) / C)
            losses.append(nll)
        valid = max(sum(1 for n in range(N) if tf[n] != ig), 1) if ig is not None else N
        if self.reduction == 'mean': loss_val = sum(losses) / valid
        elif self.reduction == 'sum': loss_val = sum(losses)
        else: loss_val = losses
        from vectorphile._core.ndarray import array as _nd
        _dt_celoss = str(ld.dtype) if hasattr(ld, 'dtype') else 'float32'
        result = Tensor(_nd([loss_val], dtype=_dt_celoss) if isinstance(loss_val, float) else _nd(loss_val, dtype=_dt_celoss).reshape((N,)),
                        device=logits.device, requires_grad=req_grad)
        if req_grad:
            _lf, _tf, _N, _C = lf, tf, N, C
            _ig, _red, _ls = ig, self.reduction, ls
            _orig_shape = lsh  # original logits shape (may be 3D)
            class _CEBackwardPy:
                def __init__(s): s.inputs = (logits,)
                def backward(s, grad):
                    gf = _get_flat(grad._data)
                    g_scale = gf[0] if len(gf) == 1 else 1.0
                    v2 = max(sum(1 for t in _tf if t != _ig), 1) if _ig is not None else _N
                    if _red != 'mean': v2 = 1
                    d = []
                    for n2 in range(_N):
                        row2 = _lf[n2*_C:(n2+1)*_C]
                        mx2 = max(row2); exps2 = [math.exp(v - mx2) for v in row2]; s2 = sum(exps2)
                        probs2 = [e / s2 for e in exps2]
                        t2 = _tf[n2]
                        for c2 in range(_C):
                            if t2 == _ig: d.append(0.0); continue
                            grad_v = probs2[c2] - ((1 - _ls) if c2 == t2 else 0.0) - _ls / _C
                            if _red != 'none': grad_v /= v2
                            d.append(grad_v * g_scale)
                    _dt_ce = str(logits._data.dtype) if hasattr(logits._data, 'dtype') else 'float32'
                    return (Tensor(_nd(d, dtype=_dt_ce).reshape(_orig_shape)),)
            result.grad_fn = _CEBackwardPy(); result._is_leaf = False
        return result


class BCELoss(Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        # Use Tensor ops to keep autograd graph intact
        import vectorphile as _vp
        # clamp pred to [eps, 1-eps]
        eps = 1e-7
        pf = _get_flat(pred._data)
        pf_clamped = [max(eps, min(1 - eps, float(v))) for v in pf]
        from vectorphile._core.ndarray import array as _nd
        pred_shape = pred._data.shape if hasattr(pred._data, 'shape') else (len(pf),)
        pred_clamped = Tensor(_nd(pf_clamped, pred_shape))
        # pred_clamped must share grad_fn with pred
        # Instead, build loss via differentiable ops
        # loss = -(t * log(p) + (1-t) * log(1-p))
        log_p = _vp.log(pred_clamped)
        # rebuild with grad from pred
        # We need actual clamp op for autograd; use manual backward instead
        from vectorphile._core.ndarray import array as _nd2
        tf = _get_flat(target._data)
        # Compute gradient manually: d_loss/d_p = -(t/p - (1-t)/(1-p))
        out_vals = [-(tf[i] * math.log(pf_clamped[i]) + (1-tf[i]) * math.log(max(eps, 1-pf_clamped[i])))
                    for i in range(len(pf))]
        n = len(out_vals)
        if self.reduction == 'mean':
            loss_val = sum(out_vals) / n
        elif self.reduction == 'sum':
            loss_val = sum(out_vals)
        else:
            loss_val = None

        result = Tensor(loss_val if loss_val is not None else _nd2(out_vals, pred_shape))

        if pred.requires_grad:
            _pred = pred; _target = target; _red = self.reduction; _eps = eps
            class _BCEBackward:
                def __init__(s): s.inputs = (_pred,)
                def backward(s, grad):
                    _pf = _get_flat(_pred._data)
                    _tf = _get_flat(_target._data)
                    n2 = len(_pf)
                    _gf = grad._data.to_flat_list() if hasattr(grad._data, 'to_flat_list') else [float(grad._data)]
                    gval = _gf[0] if len(_gf) == 1 else 1.0
                    if _red == 'mean': gval /= n2
                    d = [-(_tf[i] / max(_eps, min(1-_eps, _pf[i])) - (1-_tf[i]) / max(_eps, 1-min(1-_eps, _pf[i]))) * gval
                         for i in range(n2)]
                    _sh = _pred._data.shape if hasattr(_pred._data, 'shape') else (n2,)
                    return (Tensor(_nd2(d, _sh)),)
            result.grad_fn = _BCEBackward()
            result._is_leaf = False
        return result


class BCEWithLogitsLoss(Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        from vectorphile._core.ndarray import array as _nd
        eps = 1e-7
        lf = _get_flat(logits._data); tf = _get_flat(target._data)
        out = [max(a, 0) - a * t + math.log1p(math.exp(-abs(a))) for a, t in zip(lf, tf)]
        n = len(out)
        if self.reduction == 'mean':
            loss_val = sum(out) / n
        elif self.reduction == 'sum':
            loss_val = sum(out)
        else:
            loss_val = None
        _sh = logits._data.shape if hasattr(logits._data, 'shape') else (n,)
        _dt_bce = str(logits._data.dtype) if hasattr(logits._data, 'dtype') else 'float32'
        if loss_val is not None:
            result = Tensor(_nd([loss_val], dtype=_dt_bce))
        else:
            result = Tensor(_nd(out, dtype=_dt_bce).reshape(_sh))
        if logits.requires_grad:
            _logits = logits; _target = target; _red = self.reduction
            class _BCEWLBackward:
                def __init__(s): s.inputs = (_logits,)
                def backward(s, grad):
                    _lf2 = _get_flat(_logits._data)
                    _tf2 = _get_flat(_target._data)
                    n2 = len(_lf2)
                    _gf = grad._data.to_flat_list() if hasattr(grad._data, 'to_flat_list') else ([float(x) for x in grad._data.flatten()] if hasattr(grad._data, 'flatten') else [float(grad._data)])
                    gval = _gf[0] if len(_gf) == 1 else 1.0
                    if _red == 'mean': gval /= n2
                    def _sig(x): return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))
                    d = [(_sig(_lf2[i]) - _tf2[i]) * gval for i in range(n2)]
                    _sh2 = _logits._data.shape if hasattr(_logits._data, 'shape') else (n2,)
                    _dt2 = str(_logits._data.dtype) if hasattr(_logits._data, 'dtype') else 'float32'
                    return (Tensor(_nd(d, dtype=_dt2).reshape(_sh2)),)
            result.grad_fn = _BCEWLBackward()
            result._is_leaf = False
        return result


class NLLLoss(Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, log_probs: Tensor, targets: Tensor) -> Tensor:
        lf = _get_flat(log_probs._data)
        lsh = log_probs._data.shape if hasattr(log_probs._data, 'shape') else (len(lf),)
        N = lsh[0]; C = lsh[1] if len(lsh) > 1 else 1
        tf = [int(v) for v in _get_flat(targets._data)]
        rg = log_probs.requires_grad
        losses = [-lf[n*C + tf[n]] for n in range(N)]
        if self.reduction == 'mean': loss_val = sum(losses) / N
        elif self.reduction == 'sum': loss_val = sum(losses)
        else: loss_val = losses
        from vectorphile._core.ndarray import array as _nd
        result = Tensor(_nd([loss_val]) if isinstance(loss_val, float) else _nd(loss_val, (N,)),
                        device=log_probs.device, requires_grad=rg)
        if rg:
            _lf2, _tf2, _N2, _C2, _red2 = lf, tf, N, C, self.reduction
            class _NLLBwd:
                def __init__(s): s.inputs = (log_probs,)
                def backward(s, grad):
                    gf2 = _get_flat(grad._data); g_scale = gf2[0] if len(gf2) == 1 else 1.0
                    d = [0.0] * (_N2 * _C2)
                    for n3 in range(_N2):
                        d[n3*_C2 + _tf2[n3]] = -1.0 / (_N2 if _red2 == 'mean' else 1) * g_scale
                    return (Tensor(_nd(d, (_N2, _C2))),)
            result.grad_fn = _NLLBwd(); result._is_leaf = False
        return result


class _HuberBackward:
    def __init__(self, pred, target, delta, reduction, n):
        self.inputs = (pred,)
        self.target = target; self.delta = delta; self.reduction = reduction; self.n = n
    def backward(self, grad):
        from vectorphile._core.ndarray import array as _nd_hub
        pf = _get_flat(self.target._data if isinstance(self.target, Tensor) else self.target)
        qf = _get_flat(grad._data if isinstance(grad, Tensor) else grad)
        pred_d = self.target  # unused — we stored pred separately
        g_up = qf[0] if len(qf) == 1 else qf
        pf_pred = _get_flat(self.inputs[0]._data)
        pf_tgt = _get_flat(self.target._data if isinstance(self.target, Tensor) else self.target)
        d = self.delta; n = self.n
        g_up_val = g_up[0] if hasattr(g_up, '__len__') else float(g_up)
        grads = []
        for a, b in zip(pf_pred, pf_tgt):
            diff = a - b
            if abs(diff) < d:
                grads.append(diff / d)
            else:
                grads.append(1.0 if diff > 0 else -1.0)
        if self.reduction == 'mean':
            grads = [g * g_up_val / n for g in grads]
        else:
            grads = [g * g_up_val for g in grads]
        _dt_hub = str(self.inputs[0]._data.dtype) if hasattr(self.inputs[0]._data, 'dtype') else 'float32'
        sh = self.inputs[0]._data.shape if hasattr(self.inputs[0]._data, 'shape') else (len(grads),)
        return (Tensor(_nd_hub(grads, dtype=_dt_hub).reshape(sh)),)


class HuberLoss(Module):
    def __init__(self, reduction='mean', delta=1.0):
        super().__init__()
        self.reduction = reduction
        self.delta = delta

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        from vectorphile._core.ndarray import array as _nd_hub2
        pf = _get_flat(pred._data); tf = _get_flat(target._data)
        d = self.delta
        out = [0.5*(a-b)**2/d if abs(a-b) < d else abs(a-b) - 0.5*d for a, b in zip(pf, tf)]
        n = len(out)
        rg = pred.requires_grad
        _dt = str(pred._data.dtype) if hasattr(pred._data, 'dtype') else 'float32'
        if self.reduction == 'mean':
            val = sum(out) / n
            result = Tensor(_nd_hub2([val], dtype=_dt))
        elif self.reduction == 'sum':
            val = sum(out)
            result = Tensor(_nd_hub2([val], dtype=_dt))
        else:
            result = Tensor(_nd_hub2(out, dtype=_dt).reshape(pred._data.shape if hasattr(pred._data, 'shape') else (n,)))
        if rg:
            result.requires_grad = True
            result.grad_fn = _HuberBackward(pred, target, d, self.reduction, n)
            result._is_leaf = False
        return result


class KLDivLoss(Module):
    def __init__(self, reduction='mean', log_target=False):
        super().__init__()
        self.reduction = reduction
        self.log_target = log_target

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if self.log_target:
            loss = target.exp() * (target - input)
        else:
            tf = _get_flat(target._data); inf = _get_flat(input._data)
            from vectorphile._core.ndarray import array as _nd
            sh = input._data.shape if hasattr(input._data, 'shape') else (len(inf),)
            _dt_kl = str(input._data.dtype) if hasattr(input._data, 'dtype') else 'float32'
            log_t = _nd([math.log(t + 1e-12) for t in tf], dtype=_dt_kl).reshape(sh)
            loss = target * (Tensor(log_t) - input)
        if self.reduction == 'mean': return loss.mean()
        if self.reduction == 'sum': return loss.sum()
        return loss


class CosineSimilarityLoss(Module):
    def __init__(self, dim=1, eps=1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x1f = _get_flat(x1._data); x2f = _get_flat(x2._data)
        sh = x1._data.shape if hasattr(x1._data, 'shape') else (len(x1f),)
        dim_size = sh[self.dim] if self.dim < len(sh) else sh[-1]
        n = len(x1f) // dim_size
        sims = []
        for i in range(n):
            a = x1f[i*dim_size:(i+1)*dim_size]
            b = x2f[i*dim_size:(i+1)*dim_size]
            dot = sum(ai*bi for ai, bi in zip(a, b))
            na = math.sqrt(sum(v**2 for v in a)); nb = math.sqrt(sum(v**2 for v in b))
            sims.append(1.0 - dot / max(na * nb, self.eps))
        return Tensor(sum(sims) / len(sims))


class TripletMarginLoss(Module):
    """max(d(a,p) - d(a,n) + margin, 0) — Euclidean distance by default."""
    def __init__(self, margin=1.0, p=2, eps=1e-6, reduction='mean'):
        super().__init__()
        self.margin = margin
        self.p = p
        self.eps = eps
        self.reduction = reduction

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        af = _get_flat(anchor._data)
        pf = _get_flat(positive._data)
        nf = _get_flat(negative._data)
        sh = anchor._data.shape if hasattr(anchor._data, 'shape') else (len(af),)
        dim = sh[-1] if len(sh) > 1 else len(af)
        n = len(af) // dim
        losses = []
        for i in range(n):
            a = af[i*dim:(i+1)*dim]
            pos = pf[i*dim:(i+1)*dim]
            neg = nf[i*dim:(i+1)*dim]
            if self.p == 2:
                dp = math.sqrt(sum((x-y)**2 for x,y in zip(a,pos)) + self.eps)
                dn = math.sqrt(sum((x-y)**2 for x,y in zip(a,neg)) + self.eps)
            else:
                dp = sum(abs(x-y)**self.p for x,y in zip(a,pos)) ** (1/self.p)
                dn = sum(abs(x-y)**self.p for x,y in zip(a,neg)) ** (1/self.p)
            losses.append(max(dp - dn + self.margin, 0.0))
        if self.reduction == 'none':
            return Tensor(losses)
        val = sum(losses) / len(losses) if self.reduction == 'mean' else sum(losses)
        return Tensor(val)


class MultiLabelMarginLoss(Module):
    """Multi-label margin ranking loss (Crammer & Singer style)."""
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        xf = _get_flat(x._data)
        tf = [int(v) for v in _get_flat(target._data)]
        sh = x._data.shape if hasattr(x._data, 'shape') else (len(xf),)
        C = sh[-1] if len(sh) > 1 else len(xf)
        n = len(xf) // C
        losses = []
        for i in range(n):
            scores = xf[i*C:(i+1)*C]
            tgt = tf[i*C:(i+1)*C]
            pos_idx = [j for j in range(C) if tgt[j] >= 0]
            neg_idx = [j for j in range(C) if j not in pos_idx]
            loss = 0.0
            count = 0
            for p in pos_idx:
                for q in neg_idx:
                    loss += max(1.0 - scores[p] + scores[q], 0.0)
                    count += 1
            losses.append(loss / max(count, 1))
        if self.reduction == 'none':
            return Tensor(losses)
        val = sum(losses) / len(losses) if self.reduction == 'mean' else sum(losses)
        return Tensor(val)


class PoissonNLLLoss(Module):
    """Poisson negative log-likelihood: loss = exp(input) - target * input."""
    def __init__(self, log_input=True, full=False, eps=1e-8, reduction='mean'):
        super().__init__()
        self.log_input = log_input
        self.full = full
        self.eps = eps
        self.reduction = reduction

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        xf = _get_flat(x._data)
        tf = _get_flat(target._data)
        losses = []
        for xi, ti in zip(xf, tf):
            if self.log_input:
                loss = math.exp(xi) - ti * xi
            else:
                loss = xi - ti * math.log(max(xi, self.eps))
            if self.full:
                # Stirling approximation term
                if ti > 1:
                    loss += ti * math.log(ti) - ti + 0.5 * math.log(2 * math.pi * ti)
            losses.append(loss)
        if self.reduction == 'none':
            return Tensor(losses)
        val = sum(losses) / len(losses) if self.reduction == 'mean' else sum(losses)
        return Tensor(val)


class FocalLoss(Module):
    """Focal loss: (1 - pt)^gamma * CE. Lin et al. 2017 for class imbalance."""
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # per-class weight list or scalar
        self.reduction = reduction

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        xf = _get_flat(x._data)
        tf = [int(v) for v in _get_flat(target._data)]
        sh = x._data.shape if hasattr(x._data, 'shape') else (len(xf),)
        C = sh[-1] if len(sh) > 1 else len(xf)
        n = len(xf) // C
        losses = []
        for i in range(n):
            logits = xf[i*C:(i+1)*C]
            t = tf[i] if i < len(tf) else 0
            # softmax
            m = max(logits)
            exps = [math.exp(v - m) for v in logits]
            s = sum(exps)
            probs = [e / s for e in exps]
            pt = max(probs[t], 1e-8)
            ce = -math.log(pt)
            fl = (1 - pt) ** self.gamma * ce
            if self.alpha is not None:
                a = self.alpha[t] if hasattr(self.alpha, '__getitem__') else self.alpha
                fl = a * fl
            losses.append(fl)
        if self.reduction == 'none':
            return Tensor(losses)
        val = sum(losses) / len(losses) if self.reduction == 'mean' else sum(losses)
        return Tensor(val)


class CTCLoss(Module):
    """Connectionist Temporal Classification loss (Graves et al. 2006).
    Uses log-space forward-backward algorithm. Blank label = 0 by default."""
    def __init__(self, blank=0, reduction='mean', zero_infinity=False):
        super().__init__()
        self.blank = blank
        self.reduction = reduction
        self.zero_infinity = zero_infinity

    @staticmethod
    def _log_sum_exp(a, b):
        if a == float('-inf'): return b
        if b == float('-inf'): return a
        m = max(a, b)
        return m + math.log(math.exp(a - m) + math.exp(b - m))

    def forward(self, log_probs: Tensor, targets: Tensor,
                input_lengths: list, target_lengths: list) -> Tensor:
        lpf = _get_flat(log_probs._data)
        tf = [int(v) for v in _get_flat(targets._data)]
        lp_sh = log_probs._data.shape if hasattr(log_probs._data, 'shape') else (len(lpf),)
        T, N, C = lp_sh[0], lp_sh[1] if len(lp_sh) > 1 else 1, lp_sh[2] if len(lp_sh) > 2 else len(lpf)
        losses = []
        t_offset = 0
        NEG_INF = float('-inf')
        lse = self._log_sum_exp
        for b in range(N):
            L = target_lengths[b]
            Tb = input_lengths[b]
            lab = tf[t_offset:t_offset + L]
            t_offset += L
            # Build extended label sequence with blanks: b l1 b l2 b ... lL b
            ext = [self.blank]
            for c in lab:
                ext.append(c)
                ext.append(self.blank)
            S = len(ext)
            # Init alpha (log-space)
            alpha = [[NEG_INF] * S for _ in range(Tb)]
            alpha[0][0] = lpf[b * C + self.blank] if len(lp_sh) > 2 else NEG_INF
            if S > 1:
                alpha[0][1] = lpf[b * C + ext[1]] if len(lp_sh) > 2 else NEG_INF
            for t in range(1, Tb):
                row_offset = (t * N + b) * C if len(lp_sh) > 2 else 0
                for s in range(S):
                    a = alpha[t-1][s]
                    if s > 0:
                        a = lse(a, alpha[t-1][s-1])
                    if s > 1 and ext[s] != self.blank and ext[s] != ext[s-2]:
                        a = lse(a, alpha[t-1][s-2])
                    lp = lpf[row_offset + ext[s]] if row_offset + ext[s] < len(lpf) else NEG_INF
                    alpha[t][s] = a + lp
            log_p = lse(alpha[Tb-1][S-1], alpha[Tb-1][S-2]) if S >= 2 else alpha[Tb-1][S-1]
            loss = -log_p
            if self.zero_infinity and (math.isinf(loss) or math.isnan(loss)):
                loss = 0.0
            losses.append(loss)
        if self.reduction == 'none':
            return Tensor(losses)
        val = sum(losses) / len(losses) if self.reduction == 'mean' else sum(losses)
        return Tensor(val)
