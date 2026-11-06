"""Loss functions."""
import numpy as np
from vectorphile.nn.module import Module
from vectorphile.core.tensor import Tensor


class MSELoss(Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        diff = pred - target
        sq = diff * diff
        if self.reduction == 'mean':
            return sq.mean()
        elif self.reduction == 'sum':
            return sq.sum()
        return sq


class MAELoss(Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        diff = np.abs(pred._data - target._data)
        if self.reduction == 'mean':
            return Tensor(diff.mean(), device=pred.device, requires_grad=True)
        elif self.reduction == 'sum':
            return Tensor(diff.sum(), device=pred.device, requires_grad=True)
        return Tensor(diff, device=pred.device, requires_grad=True)


class CrossEntropyLoss(Module):
    def __init__(self, reduction='mean', ignore_index=-100, label_smoothing=0.0):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        log_probs = logits._data - np.log(np.exp(logits._data).sum(axis=-1, keepdims=True) + 1e-12)
        t = targets._data.astype(int)
        n = log_probs.shape[0]
        nll = -log_probs[np.arange(n), t]
        if self.label_smoothing > 0:
            smooth_loss = -log_probs.mean(axis=-1)
            nll = (1 - self.label_smoothing) * nll + self.label_smoothing * smooth_loss
        if self.reduction == 'mean':
            return Tensor(nll.mean(), device=logits.device, requires_grad=True)
        elif self.reduction == 'sum':
            return Tensor(nll.sum(), device=logits.device, requires_grad=True)
        return Tensor(nll, device=logits.device, requires_grad=True)


class _BCEBackward:
    """Custom backward for BCE loss that keeps the computation in the autograd graph."""
    def __init__(self, pred, target, reduction):
        self.inputs = (pred,)
        self.pred = pred
        self.target = target
        self.reduction = reduction

    def backward(self, grad):
        p = np.clip(self.pred._data, 1e-7, 1 - 1e-7)
        t = self.target._data
        n = p.size if self.reduction == 'mean' else 1
        dp = (-(t / p) + (1 - t) / (1 - p)) / n
        return (Tensor(grad._data * dp),)


class BCELoss(Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        p = np.clip(pred._data, 1e-7, 1 - 1e-7)
        t = target._data
        loss_data = -(t * np.log(p) + (1 - t) * np.log(1 - p))
        if self.reduction == 'mean':
            val = loss_data.mean()
        elif self.reduction == 'sum':
            val = loss_data.sum()
        else:
            val = loss_data
        result = Tensor(val, device=pred.device, requires_grad=pred.requires_grad)
        if pred.requires_grad:
            result.grad_fn = _BCEBackward(pred, target, self.reduction)
            result._is_leaf = False
        return result


class BCEWithLogitsLoss(Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        d = logits._data
        t = target._data
        loss = np.maximum(d, 0) - d * t + np.log(1 + np.exp(-np.abs(d)))
        if self.reduction == 'mean':
            return Tensor(loss.mean(), device=logits.device, requires_grad=True)
        elif self.reduction == 'sum':
            return Tensor(loss.sum(), device=logits.device, requires_grad=True)
        return Tensor(loss, device=logits.device, requires_grad=True)


class NLLLoss(Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, log_probs: Tensor, targets: Tensor) -> Tensor:
        t = targets._data.astype(int)
        n = log_probs._data.shape[0]
        nll = -log_probs._data[np.arange(n), t]
        if self.reduction == 'mean':
            return Tensor(nll.mean(), device=log_probs.device, requires_grad=True)
        elif self.reduction == 'sum':
            return Tensor(nll.sum(), device=log_probs.device, requires_grad=True)
        return Tensor(nll, device=log_probs.device, requires_grad=True)


class HuberLoss(Module):
    def __init__(self, reduction='mean', delta=1.0):
        super().__init__()
        self.reduction = reduction
        self.delta = delta

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        diff = np.abs(pred._data - target._data)
        loss = np.where(diff < self.delta, 0.5 * diff**2, self.delta * (diff - 0.5 * self.delta))
        if self.reduction == 'mean':
            return Tensor(loss.mean(), device=pred.device, requires_grad=True)
        elif self.reduction == 'sum':
            return Tensor(loss.sum(), device=pred.device, requires_grad=True)
        return Tensor(loss, device=pred.device, requires_grad=True)


class KLDivLoss(Module):
    def __init__(self, reduction='mean', log_target=False):
        super().__init__()
        self.reduction = reduction
        self.log_target = log_target

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if self.log_target:
            loss = np.exp(target._data) * (target._data - input._data)
        else:
            loss = target._data * (np.log(target._data + 1e-12) - input._data)
        if self.reduction == 'mean':
            return Tensor(loss.mean(), device=input.device, requires_grad=True)
        elif self.reduction == 'sum':
            return Tensor(loss.sum(), device=input.device, requires_grad=True)
        return Tensor(loss, device=input.device, requires_grad=True)


class CosineSimilarityLoss(Module):
    def __init__(self, dim=1, eps=1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        n1 = np.linalg.norm(x1._data, axis=self.dim, keepdims=True)
        n2 = np.linalg.norm(x2._data, axis=self.dim, keepdims=True)
        cos_sim = (x1._data * x2._data).sum(axis=self.dim) / (n1.squeeze(self.dim) * n2.squeeze(self.dim) + self.eps)
        return Tensor((1 - cos_sim).mean(), device=x1.device, requires_grad=True)
