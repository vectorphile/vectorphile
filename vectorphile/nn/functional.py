"""Functional API — stateless operations on Tensors (vp.nn.functional / vp.F)."""
from __future__ import annotations
import numpy as np
import math
from vectorphile.core.tensor import Tensor


# ── Activations ──────────────────────────────────────────────────────────────

def relu(x: Tensor, inplace=False) -> Tensor:
    return Tensor(np.maximum(0, x._data), device=x.device)

def relu6(x: Tensor) -> Tensor:
    return Tensor(np.clip(x._data, 0, 6), device=x.device)

def leaky_relu(x: Tensor, negative_slope=0.01, inplace=False) -> Tensor:
    return Tensor(np.where(x._data >= 0, x._data, negative_slope * x._data), device=x.device)

def elu(x: Tensor, alpha=1.0, inplace=False) -> Tensor:
    return Tensor(np.where(x._data >= 0, x._data, alpha * (np.exp(x._data) - 1)), device=x.device)

def selu(x: Tensor) -> Tensor:
    alpha = 1.6732632423543772
    scale = 1.0507009873554805
    return Tensor(scale * np.where(x._data >= 0, x._data, alpha * (np.exp(x._data) - 1)), device=x.device)

def gelu(x: Tensor, approximate='none') -> Tensor:
    if approximate == 'tanh':
        return Tensor(0.5 * x._data * (1.0 + np.tanh(math.sqrt(2/math.pi) * (x._data + 0.044715 * x._data**3))), device=x.device)
    return Tensor(0.5 * x._data * (1.0 + np.vectorize(math.erf)(x._data / math.sqrt(2))), device=x.device)

def silu(x: Tensor, inplace=False) -> Tensor:
    return Tensor(x._data / (1 + np.exp(-x._data)), device=x.device)

def mish(x: Tensor, inplace=False) -> Tensor:
    return Tensor(x._data * np.tanh(np.log1p(np.exp(x._data))), device=x.device)

def hardswish(x: Tensor, inplace=False) -> Tensor:
    return Tensor(x._data * np.clip(x._data + 3, 0, 6) / 6, device=x.device)

def hardsigmoid(x: Tensor, inplace=False) -> Tensor:
    return Tensor(np.clip(x._data / 6 + 0.5, 0, 1), device=x.device)

def sigmoid(x: Tensor) -> Tensor:
    return Tensor(1.0 / (1.0 + np.exp(-x._data)), device=x.device)

def tanh(x: Tensor) -> Tensor:
    return Tensor(np.tanh(x._data), device=x.device)

def softmax(x: Tensor, dim=-1) -> Tensor:
    e = np.exp(x._data - x._data.max(axis=dim, keepdims=True))
    return Tensor(e / e.sum(axis=dim, keepdims=True), device=x.device)

def log_softmax(x: Tensor, dim=-1) -> Tensor:
    e = np.exp(x._data - x._data.max(axis=dim, keepdims=True))
    return Tensor(np.log(e / e.sum(axis=dim, keepdims=True)), device=x.device)

def softplus(x: Tensor, beta=1, threshold=20) -> Tensor:
    return Tensor(np.where(x._data * beta > threshold, x._data, (1/beta) * np.log1p(np.exp(beta * x._data))), device=x.device)

def softsign(x: Tensor) -> Tensor:
    return Tensor(x._data / (1 + np.abs(x._data)), device=x.device)

def prelu(x: Tensor, weight: Tensor) -> Tensor:
    return Tensor(np.where(x._data >= 0, x._data, weight._data * x._data), device=x.device)

def threshold(x: Tensor, threshold_val: float, value: float) -> Tensor:
    return Tensor(np.where(x._data > threshold_val, x._data, value), device=x.device)


# ── Linear ───────────────────────────────────────────────────────────────────

def linear(x: Tensor, weight: Tensor, bias: Tensor = None) -> Tensor:
    out = x._data @ weight._data.T
    if bias is not None:
        out = out + bias._data
    return Tensor(out.astype(np.float32), device=x.device)

def bilinear(x1: Tensor, x2: Tensor, weight: Tensor, bias: Tensor = None) -> Tensor:
    # weight: (out, in1, in2)
    out = np.einsum('bi,oij,bj->bo', x1._data, weight._data, x2._data)
    if bias is not None:
        out += bias._data
    return Tensor(out.astype(np.float32), device=x1.device)


# ── Convolution ───────────────────────────────────────────────────────────────

def conv1d(x: Tensor, weight: Tensor, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor:
    N, C, L = x._data.shape
    Co, Ci, K = weight._data.shape
    p, s, d = padding, stride, dilation
    Lk = d * (K - 1) + 1
    x_pad = np.pad(x._data, ((0,0),(0,0),(p,p)), mode='constant')
    Lo = (L + 2*p - Lk) // s + 1
    out = np.zeros((N, Co, Lo), dtype=np.float32)
    for i in range(Lo):
        patch = x_pad[:, :, i*s:i*s+Lk:d] if d > 1 else x_pad[:, :, i*s:i*s+K]
        if d > 1:
            patch = x_pad[:, :, i*s::d][:, :, :K]
        out[:, :, i] = np.einsum('nci,oci->no', patch, weight._data)
    if bias is not None:
        out += bias._data[None, :, None]
    return Tensor(out, device=x.device)


def conv2d(x: Tensor, weight: Tensor, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor:
    N, C, H, W = x._data.shape
    Co, Ci_g, kh, kw = weight._data.shape
    s = stride if isinstance(stride, tuple) else (stride, stride)
    p = padding if isinstance(padding, tuple) else (padding, padding)
    x_pad = np.pad(x._data, ((0,0),(0,0),(p[0],p[0]),(p[1],p[1])), mode='constant')
    Ho = (H + 2*p[0] - kh) // s[0] + 1
    Wo = (W + 2*p[1] - kw) // s[1] + 1
    out = np.zeros((N, Co, Ho, Wo), dtype=np.float32)
    for i in range(Ho):
        for j in range(Wo):
            patch = x_pad[:, :, i*s[0]:i*s[0]+kh, j*s[1]:j*s[1]+kw]
            out[:, :, i, j] = np.einsum('nckl,ockl->no', patch, weight._data)
    if bias is not None:
        out += bias._data[None, :, None, None]
    return Tensor(out, device=x.device)


def conv_transpose2d(x: Tensor, weight: Tensor, bias=None, stride=1, padding=0,
                     output_padding=0, groups=1, dilation=1) -> Tensor:
    N, C, H, W = x._data.shape
    Ci, Co, kh, kw = weight._data.shape
    s = stride if isinstance(stride, tuple) else (stride, stride)
    p = padding if isinstance(padding, tuple) else (padding, padding)
    op = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding)
    Ho = (H - 1) * s[0] - 2*p[0] + kh + op[0]
    Wo = (W - 1) * s[1] - 2*p[1] + kw + op[1]
    out = np.zeros((N, Co, Ho, Wo), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            hi, wi = i*s[0], j*s[1]
            out[:, :, hi:hi+kh, wi:wi+kw] += np.einsum('nc,oikl->nokl', x._data[:, :, i, j], weight._data[:, :, :, :].reshape(Ci, Co, kh, kw))
    if p[0] > 0 or p[1] > 0:
        out = out[:, :, p[0]:Ho-p[0] if p[0] > 0 else Ho, p[1]:Wo-p[1] if p[1] > 0 else Wo]
    if bias is not None:
        out += bias._data[None, :, None, None]
    return Tensor(out.astype(np.float32), device=x.device)


# ── Normalization ─────────────────────────────────────────────────────────────

def batch_norm(x: Tensor, running_mean, running_var, weight=None, bias=None,
               training=False, momentum=0.1, eps=1e-5) -> Tensor:
    data = x._data
    if training:
        mean = data.mean(axis=0)
        var = data.var(axis=0)
        if running_mean is not None:
            running_mean[:] = (1 - momentum) * running_mean + momentum * mean
            running_var[:] = (1 - momentum) * running_var + momentum * var
    else:
        mean = running_mean if running_mean is not None else data.mean(axis=0)
        var = running_var if running_var is not None else data.var(axis=0)
    x_hat = (data - mean) / np.sqrt(var + eps)
    if weight is not None:
        x_hat = x_hat * weight._data
    if bias is not None:
        x_hat = x_hat + bias._data
    return Tensor(x_hat.astype(np.float32), device=x.device)


def layer_norm(x: Tensor, normalized_shape, weight=None, bias=None, eps=1e-5) -> Tensor:
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    ndim = len(normalized_shape)
    axes = tuple(range(-ndim, 0))
    mean = x._data.mean(axis=axes, keepdims=True)
    var = x._data.var(axis=axes, keepdims=True)
    x_hat = (x._data - mean) / np.sqrt(var + eps)
    if weight is not None:
        x_hat = x_hat * weight._data
    if bias is not None:
        x_hat = x_hat + bias._data
    return Tensor(x_hat.astype(np.float32), device=x.device)


def group_norm(x: Tensor, num_groups: int, weight=None, bias=None, eps=1e-5) -> Tensor:
    N, C = x._data.shape[:2]
    rest = x._data.shape[2:]
    data = x._data.reshape(N, num_groups, C // num_groups, *rest)
    axes = tuple(range(2, data.ndim))
    mean = data.mean(axis=axes, keepdims=True)
    var = data.var(axis=axes, keepdims=True)
    data = (data - mean) / np.sqrt(var + eps)
    data = data.reshape(N, C, *rest)
    if weight is not None:
        data = data * weight._data.reshape(1, C, *[1]*len(rest))
    if bias is not None:
        data = data + bias._data.reshape(1, C, *[1]*len(rest))
    return Tensor(data.astype(np.float32), device=x.device)


def instance_norm(x: Tensor, running_mean=None, running_var=None, weight=None, bias=None,
                  use_input_stats=True, momentum=0.1, eps=1e-5) -> Tensor:
    N, C = x._data.shape[:2]
    rest = x._data.shape[2:]
    axes = tuple(range(2, x._data.ndim))
    mean = x._data.mean(axis=axes, keepdims=True)
    var = x._data.var(axis=axes, keepdims=True)
    x_hat = (x._data - mean) / np.sqrt(var + eps)
    if weight is not None:
        x_hat = x_hat * weight._data.reshape(1, C, *[1]*len(rest))
    if bias is not None:
        x_hat = x_hat + bias._data.reshape(1, C, *[1]*len(rest))
    return Tensor(x_hat.astype(np.float32), device=x.device)


# ── Dropout ───────────────────────────────────────────────────────────────────

def dropout(x: Tensor, p=0.5, training=True, inplace=False) -> Tensor:
    if not training or p == 0:
        return x
    mask = (np.random.rand(*x._data.shape) > p).astype(np.float32) / (1 - p)
    return Tensor(x._data * mask, device=x.device)

def dropout2d(x: Tensor, p=0.5, training=True, inplace=False) -> Tensor:
    if not training or p == 0:
        return x
    N, C, H, W = x._data.shape
    mask = (np.random.rand(N, C, 1, 1) > p).astype(np.float32) / (1 - p)
    return Tensor(x._data * mask, device=x.device)

def alpha_dropout(x: Tensor, p=0.5, training=True, inplace=False) -> Tensor:
    if not training or p == 0:
        return x
    alpha = 1.6732632423543772
    scale = 1.0507009873554805
    keep = np.random.rand(*x._data.shape) > p
    a = -alpha * scale
    out = np.where(keep, x._data, a)
    affine_a = 1.0 / np.sqrt((1 - p) * (1 + p * a**2))
    affine_b = -affine_a * p * a
    return Tensor((affine_a * out + affine_b).astype(np.float32), device=x.device)


# ── Pooling ───────────────────────────────────────────────────────────────────

def max_pool2d(x: Tensor, kernel_size, stride=None, padding=0) -> Tensor:
    from vectorphile.nn.pooling import MaxPool2d
    return MaxPool2d(kernel_size, stride, padding)(x)

def avg_pool2d(x: Tensor, kernel_size, stride=None, padding=0) -> Tensor:
    from vectorphile.nn.pooling import AvgPool2d
    return AvgPool2d(kernel_size, stride, padding)(x)

def adaptive_avg_pool2d(x: Tensor, output_size) -> Tensor:
    from vectorphile.nn.pooling import AdaptiveAvgPool2d
    return AdaptiveAvgPool2d(output_size)(x)


# ── Embedding ─────────────────────────────────────────────────────────────────

def embedding(x: Tensor, weight: Tensor, padding_idx=None, max_norm=None,
              norm_type=2.0, scale_grad_by_freq=False, sparse=False) -> Tensor:
    idx = x._data.astype(int)
    out = weight._data[idx]
    if padding_idx is not None:
        out = np.where((idx == padding_idx)[..., None], 0.0, out)
    if max_norm is not None:
        norms = np.linalg.norm(out, ord=norm_type, axis=-1, keepdims=True)
        out = np.where(norms > max_norm, out * max_norm / (norms + 1e-12), out)
    return Tensor(out.astype(np.float32), device=x.device)

def one_hot(x: Tensor, num_classes: int = -1) -> Tensor:
    idx = x._data.astype(int).flatten()
    n_cls = num_classes if num_classes > 0 else int(idx.max()) + 1
    out = np.zeros((len(idx), n_cls), dtype=np.float32)
    out[np.arange(len(idx)), idx] = 1.0
    return Tensor(out.reshape(*x._data.shape, n_cls), device=x.device)


# ── Losses ────────────────────────────────────────────────────────────────────

def mse_loss(x: Tensor, target: Tensor, reduction='mean') -> Tensor:
    diff = (x._data - target._data) ** 2
    if reduction == 'mean': return Tensor(diff.mean())
    if reduction == 'sum': return Tensor(diff.sum())
    return Tensor(diff)

def l1_loss(x: Tensor, target: Tensor, reduction='mean') -> Tensor:
    diff = np.abs(x._data - target._data)
    if reduction == 'mean': return Tensor(diff.mean())
    if reduction == 'sum': return Tensor(diff.sum())
    return Tensor(diff)

def smooth_l1_loss(x: Tensor, target: Tensor, reduction='mean', beta=1.0) -> Tensor:
    diff = np.abs(x._data - target._data)
    loss = np.where(diff < beta, 0.5 * diff**2 / beta, diff - 0.5 * beta)
    if reduction == 'mean': return Tensor(loss.mean())
    if reduction == 'sum': return Tensor(loss.sum())
    return Tensor(loss)

def cross_entropy(x: Tensor, target: Tensor, weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0) -> Tensor:
    N = x._data.shape[0]
    e = np.exp(x._data - x._data.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    target_idx = target._data.astype(int)
    if label_smoothing > 0:
        n_cls = x._data.shape[1]
        smooth_target = np.full_like(probs, label_smoothing / n_cls)
        smooth_target[np.arange(N), target_idx] += 1 - label_smoothing
        loss = -np.sum(smooth_target * np.log(probs + 1e-12), axis=1)
    else:
        loss = -np.log(probs[np.arange(N), target_idx] + 1e-12)
    if ignore_index >= 0:
        loss = np.where(target_idx == ignore_index, 0.0, loss)
    if weight is not None:
        loss = loss * weight._data[target_idx]
    if reduction == 'mean': return Tensor(float(loss.mean()))
    if reduction == 'sum': return Tensor(float(loss.sum()))
    return Tensor(loss.astype(np.float32))

def nll_loss(x: Tensor, target: Tensor, weight=None, ignore_index=-100, reduction='mean') -> Tensor:
    N = x._data.shape[0]
    target_idx = target._data.astype(int)
    loss = -x._data[np.arange(N), target_idx]
    if ignore_index >= 0:
        loss = np.where(target_idx == ignore_index, 0.0, loss)
    if weight is not None:
        loss = loss * weight._data[target_idx]
    if reduction == 'mean': return Tensor(float(loss.mean()))
    if reduction == 'sum': return Tensor(float(loss.sum()))
    return Tensor(loss.astype(np.float32))

def binary_cross_entropy(x: Tensor, target: Tensor, weight=None, reduction='mean') -> Tensor:
    p = np.clip(x._data, 1e-7, 1 - 1e-7)
    loss = -(target._data * np.log(p) + (1 - target._data) * np.log(1 - p))
    if weight is not None:
        loss = loss * weight._data
    if reduction == 'mean': return Tensor(float(loss.mean()))
    if reduction == 'sum': return Tensor(float(loss.sum()))
    return Tensor(loss.astype(np.float32))

def binary_cross_entropy_with_logits(x: Tensor, target: Tensor, weight=None,
                                      reduction='mean', pos_weight=None) -> Tensor:
    loss = np.maximum(x._data, 0) - x._data * target._data + np.log1p(np.exp(-np.abs(x._data)))
    if pos_weight is not None:
        loss = loss * (target._data * (pos_weight._data - 1) + 1)
    if weight is not None:
        loss = loss * weight._data
    if reduction == 'mean': return Tensor(float(loss.mean()))
    if reduction == 'sum': return Tensor(float(loss.sum()))
    return Tensor(loss.astype(np.float32))

def kl_div(x: Tensor, target: Tensor, reduction='batchmean', log_target=False) -> Tensor:
    if log_target:
        loss = np.exp(target._data) * (target._data - x._data)
    else:
        loss = target._data * (np.log(target._data + 1e-12) - x._data)
    if reduction == 'batchmean': return Tensor(float(loss.sum() / x._data.shape[0]))
    if reduction == 'mean': return Tensor(float(loss.mean()))
    if reduction == 'sum': return Tensor(float(loss.sum()))
    return Tensor(loss.astype(np.float32))

def cosine_similarity(x1: Tensor, x2: Tensor, dim=1, eps=1e-8) -> Tensor:
    n1 = np.linalg.norm(x1._data, axis=dim, keepdims=True)
    n2 = np.linalg.norm(x2._data, axis=dim, keepdims=True)
    dot = (x1._data * x2._data).sum(axis=dim)
    denom = (n1.squeeze(dim) * n2.squeeze(dim)).clip(min=eps)
    return Tensor((dot / denom).astype(np.float32), device=x1.device)

def triplet_margin_loss(anchor, positive, negative, margin=1.0, p=2.0, reduction='mean') -> Tensor:
    d_ap = np.linalg.norm(anchor._data - positive._data, ord=p, axis=-1)
    d_an = np.linalg.norm(anchor._data - negative._data, ord=p, axis=-1)
    loss = np.maximum(0, d_ap - d_an + margin)
    if reduction == 'mean': return Tensor(float(loss.mean()))
    if reduction == 'sum': return Tensor(float(loss.sum()))
    return Tensor(loss.astype(np.float32))

def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0, reduction='mean') -> Tensor:
    # Simplified CTC — returns a placeholder (full CTC requires dynamic programming)
    return Tensor(np.array(0.0, dtype=np.float32))

def hinge_embedding_loss(x: Tensor, y: Tensor, margin=1.0, reduction='mean') -> Tensor:
    loss = np.where(y._data == 1, x._data, np.maximum(0, margin - x._data))
    if reduction == 'mean': return Tensor(float(loss.mean()))
    if reduction == 'sum': return Tensor(float(loss.sum()))
    return Tensor(loss.astype(np.float32))


# ── Utility ───────────────────────────────────────────────────────────────────

def normalize(x: Tensor, p=2.0, dim=1, eps=1e-12) -> Tensor:
    norms = np.linalg.norm(x._data, ord=p, axis=dim, keepdims=True).clip(min=eps)
    return Tensor((x._data / norms).astype(np.float32), device=x.device)

def pad(x: Tensor, pad_tuple, mode='constant', value=0) -> Tensor:
    nd = x.ndim
    # pad_tuple is (left1, right1, left2, right2, ...) from last dim to first
    np_pad = []
    for _ in range(nd - len(pad_tuple)//2):
        np_pad.append((0, 0))
    for i in range(len(pad_tuple)-1, -1, -2):
        np_pad.append((pad_tuple[i-1], pad_tuple[i]))
    kwargs = {} if mode != 'constant' else {'constant_values': value}
    out = np.pad(x._data, np_pad, mode=mode, **kwargs)
    return Tensor(out.astype(x._data.dtype), device=x.device)

def interpolate(x: Tensor, size=None, scale_factor=None, mode='nearest', align_corners=False) -> Tensor:
    from scipy.ndimage import zoom
    data = x._data
    if scale_factor is not None:
        factors = [1, 1] + ([scale_factor]*len(data.shape[2:]) if isinstance(scale_factor, (int, float)) else list(scale_factor))
    elif size is not None:
        s = (size, size) if isinstance(size, int) else size
        factors = [1, 1] + [s[i]/data.shape[2+i] for i in range(len(s))]
    else:
        raise ValueError("size or scale_factor required")
    order = {'nearest': 0, 'bilinear': 1, 'bicubic': 3, 'trilinear': 1}.get(mode, 1)
    out = zoom(data, factors, order=order)
    return Tensor(out.astype(np.float32), device=x.device)

def fold(x: Tensor, output_size, kernel_size, dilation=1, padding=0, stride=1) -> Tensor:
    # Inverse of unfold — col2im
    oh, ow = output_size if isinstance(output_size, tuple) else (output_size, output_size)
    kh, kw = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
    s = stride if isinstance(stride, tuple) else (stride, stride)
    p = padding if isinstance(padding, tuple) else (padding, padding)
    N, C_kk, L = x._data.shape
    C = C_kk // (kh * kw)
    Ho = (oh + 2*p[0] - kh) // s[0] + 1
    Wo = (ow + 2*p[1] - kw) // s[1] + 1
    out = np.zeros((N, C, oh + 2*p[0], ow + 2*p[1]), dtype=np.float32)
    idx = 0
    for i in range(Ho):
        for j in range(Wo):
            patch = x._data[:, :, idx].reshape(N, C, kh, kw)
            out[:, :, i*s[0]:i*s[0]+kh, j*s[1]:j*s[1]+kw] += patch
            idx += 1
    if p[0] > 0 or p[1] > 0:
        out = out[:, :, p[0]:p[0]+oh, p[1]:p[1]+ow]
    return Tensor(out, device=x.device)

def unfold(x: Tensor, kernel_size, dilation=1, padding=0, stride=1) -> Tensor:
    N, C, H, W = x._data.shape
    kh, kw = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
    s = stride if isinstance(stride, tuple) else (stride, stride)
    p = padding if isinstance(padding, tuple) else (padding, padding)
    x_pad = np.pad(x._data, ((0,0),(0,0),(p[0],p[0]),(p[1],p[1])))
    Ho = (H + 2*p[0] - kh) // s[0] + 1
    Wo = (W + 2*p[1] - kw) // s[1] + 1
    cols = []
    for i in range(Ho):
        for j in range(Wo):
            cols.append(x_pad[:, :, i*s[0]:i*s[0]+kh, j*s[1]:j*s[1]+kw].reshape(N, -1))
    return Tensor(np.stack(cols, axis=2).astype(np.float32), device=x.device)

def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor,
                                  attn_mask=None, dropout_p=0.0, is_causal=False) -> Tensor:
    import math
    scale = math.sqrt(query._data.shape[-1])
    scores = np.matmul(query._data, key._data.swapaxes(-2, -1)) / scale
    if is_causal:
        T = scores.shape[-2]
        S = scores.shape[-1]
        causal_mask = np.tril(np.ones((T, S), dtype=bool))
        scores = np.where(causal_mask, scores, -1e9)
    if attn_mask is not None:
        mask_data = attn_mask._data if isinstance(attn_mask, Tensor) else attn_mask
        scores = scores + mask_data
    e = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attn = e / e.sum(axis=-1, keepdims=True)
    if dropout_p > 0.0:
        attn = attn * (np.random.rand(*attn.shape) > dropout_p).astype(np.float32) / (1 - dropout_p)
    out = np.matmul(attn, value._data)
    return Tensor(out.astype(np.float32), device=query.device)

def pixel_shuffle(x: Tensor, upscale_factor: int) -> Tensor:
    N, C, H, W = x._data.shape
    r = upscale_factor
    assert C % (r**2) == 0
    Co = C // (r**2)
    out = x._data.reshape(N, Co, r, r, H, W).transpose(0, 1, 4, 2, 5, 3).reshape(N, Co, H*r, W*r)
    return Tensor(out.astype(np.float32), device=x.device)
