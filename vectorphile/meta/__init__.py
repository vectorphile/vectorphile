"""vectorphile.meta — Meta-Learning algorithms.

Provides:
- MAML (Model-Agnostic Meta-Learning, Finn et al. 2017)
- First-Order MAML (FOMAML)
- Reptile (OpenAI, 2018)
- Prototypical Networks (Snell et al. 2017)
- Matching Networks stub
- Few-shot learning utilities

No external dependencies.

Usage::

    from vectorphile.meta import MAML, ProtoNet

    maml = MAML(model, inner_lr=0.01, inner_steps=5)
    # Meta-train
    for tasks in task_sampler:
        maml.meta_train_step(tasks, meta_lr=0.001)

    # Prototypical Networks
    proto = ProtoNet(encoder)
    proto.fit(support_X, support_y)
    logits = proto.predict(query_X)
"""
from __future__ import annotations
import copy
import math
import random as _random
from typing import Any, Callable, Dict, List, Optional, Tuple

from vectorphile.core.tensor import Tensor
from vectorphile.nn.module import Module
from vectorphile._core.ndarray import NDArray, array as nd_array, zeros


def _to_flat(d):
    if isinstance(d, NDArray): return d.to_flat_list()
    if isinstance(d, (list, tuple)): return list(d)
    return [float(d)]

def _get_shape(d):
    if isinstance(d, NDArray): return d.shape
    if isinstance(d, (list, tuple)): return (len(d),)
    return (1,)


# ---------------------------------------------------------------------------
# MAML
# ---------------------------------------------------------------------------

class MAML:
    """Model-Agnostic Meta-Learning (Finn et al., 2017)."""

    def __init__(self, model: Module, inner_lr: float = 0.01,
                 inner_steps: int = 5, first_order: bool = False):
        self.model = model
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.first_order = first_order

    def _clone_model(self) -> Module:
        return copy.deepcopy(self.model)

    def inner_adapt(self, model: Module, support_X, support_y, loss_fn) -> Module:
        """Run inner-loop gradient steps on support set."""
        from vectorphile.optim import SGD
        cloned = copy.deepcopy(model)
        opt = SGD(cloned.parameters(), lr=self.inner_lr)
        X = Tensor(support_X) if not isinstance(support_X, Tensor) else support_X
        y = Tensor(support_y) if not isinstance(support_y, Tensor) else support_y
        for _ in range(self.inner_steps):
            opt.zero_grad()
            pred = cloned(X)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
        return cloned

    def meta_train_step(self, tasks: List[Tuple], meta_lr: float = 1e-3,
                        loss_fn=None) -> float:
        """Compute meta-gradient and update model."""
        if loss_fn is None:
            from vectorphile.nn.losses import CrossEntropyLoss
            loss_fn = CrossEntropyLoss()
        from vectorphile.optim import Adam
        meta_opt = Adam(self.model.parameters(), lr=meta_lr)
        meta_opt.zero_grad()
        total_loss = 0.0
        for sup_X, sup_y, qry_X, qry_y in tasks:
            adapted = self.inner_adapt(self.model, sup_X, sup_y, loss_fn)
            qX = Tensor(qry_X) if not isinstance(qry_X, Tensor) else qry_X
            qy = Tensor(qry_y) if not isinstance(qry_y, Tensor) else qry_y
            pred = adapted(qX)
            query_loss = loss_fn(pred, qy)
            query_loss.backward()
            total_loss += float(query_loss._data if hasattr(query_loss, '_data') else query_loss)
            # Copy grads to original model
            for (name, p), (_, ap) in zip(self.model.named_parameters(),
                                           adapted.named_parameters()):
                if ap.grad is not None:
                    if p.grad is None:
                        sh = _get_shape(p._data)
                        p.grad = Tensor(zeros(sh, dtype='float32'))
                    ag = _to_flat(ap.grad._data)
                    pg = _to_flat(p.grad._data)
                    n = len(tasks)
                    merged = [pg[i] + ag[i] / n for i in range(len(pg))]
                    p.grad._data = nd_array(merged, dtype='float32').reshape(_get_shape(p._data))
        meta_opt.step()
        return total_loss / max(len(tasks), 1)

    def evaluate(self, task_X, task_y, loss_fn=None, n_adapt_steps=None) -> float:
        """Evaluate adapted model on a task."""
        if loss_fn is None:
            from vectorphile.nn.losses import CrossEntropyLoss
            loss_fn = CrossEntropyLoss()
        n_adapt_steps = n_adapt_steps or self.inner_steps
        split = max(1, len(task_X) // 2)
        sup_X, qry_X = task_X[:split], task_X[split:]
        sup_y, qry_y = task_y[:split], task_y[split:]
        adapted = self.inner_adapt(self.model, sup_X, sup_y, loss_fn)
        pred = adapted(Tensor(qry_X) if not isinstance(qry_X, Tensor) else qry_X)
        loss = loss_fn(pred, Tensor(qry_y) if not isinstance(qry_y, Tensor) else qry_y)
        return float(loss._data if hasattr(loss, '_data') else loss)


class FOMAML(MAML):
    """First-Order MAML: skip second-order gradients."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, first_order=True, **kwargs)


# ---------------------------------------------------------------------------
# Reptile
# ---------------------------------------------------------------------------

class Reptile:
    """Reptile meta-learning (Nichol et al., 2018)."""

    def __init__(self, model: Module, inner_lr: float = 0.02, inner_steps: int = 8):
        self.model = model
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps

    def _get_params(self, model: Module) -> Dict[str, list]:
        return {n: _to_flat(p._data)[:] for n, p in model.named_parameters()}

    def _set_params(self, model: Module, params: Dict[str, list]) -> None:
        for n, p in model.named_parameters():
            if n in params:
                sh = _get_shape(p._data)
                p._data = nd_array(params[n], dtype='float32').reshape(sh)

    def meta_update(self, tasks: List[Tuple], meta_lr: float = 0.5, loss_fn=None) -> float:
        if loss_fn is None:
            from vectorphile.nn.losses import CrossEntropyLoss
            loss_fn = CrossEntropyLoss()
        from vectorphile.optim import SGD
        meta_params = self._get_params(self.model)
        total_loss = 0.0
        for sup_X, sup_y, qry_X, qry_y in tasks:
            self._set_params(self.model, meta_params)
            opt = SGD(self.model.parameters(), lr=self.inner_lr)
            X = Tensor(sup_X) if not isinstance(sup_X, Tensor) else sup_X
            y = Tensor(sup_y) if not isinstance(sup_y, Tensor) else sup_y
            for _ in range(self.inner_steps):
                opt.zero_grad()
                pred = self.model(X)
                loss = loss_fn(pred, y)
                loss.backward()
                opt.step()
                total_loss += float(loss._data if hasattr(loss, '_data') else loss)
            task_params = self._get_params(self.model)
            n = max(len(tasks), 1)
            for name in meta_params:
                mp = meta_params[name]; tp = task_params[name]
                meta_params[name] = [mp[i] + meta_lr * (tp[i] - mp[i]) / n for i in range(len(mp))]
        self._set_params(self.model, meta_params)
        return total_loss / max(len(tasks) * self.inner_steps, 1)


# ---------------------------------------------------------------------------
# Prototypical Networks
# ---------------------------------------------------------------------------

class ProtoNet:
    """Prototypical Networks for few-shot classification (Snell et al., 2017)."""

    def __init__(self, encoder: Module, distance: str = 'euclidean'):
        self.encoder = encoder
        self.distance = distance
        self._prototypes: Optional[list] = None  # list of lists (n_classes, embed_dim)
        self._classes: Optional[list] = None

    def _embed(self, x) -> list:
        import vectorphile as vp
        with vp.autograd.no_grad():
            emb = self.encoder(Tensor(x) if not isinstance(x, Tensor) else x)
        d = emb._data if hasattr(emb, '_data') else emb
        if isinstance(d, NDArray):
            flat = d.to_flat_list()
            sh = d.shape
            # Return as list of rows
            n = sh[0] if len(sh) > 1 else 1
            dim = sh[1] if len(sh) > 1 else sh[0]
            return [flat[i*dim:(i+1)*dim] for i in range(n)]
        return list(d)

    def fit(self, support_X, support_y) -> 'ProtoNet':
        """Compute class prototypes from support set."""
        embs = self._embed(support_X)  # list of (embed_dim,) rows
        y_list = list(support_y) if not isinstance(support_y, list) else support_y
        classes = list(dict.fromkeys(y_list))
        self._classes = classes
        prototypes = []
        for c in classes:
            indices = [i for i, lbl in enumerate(y_list) if lbl == c]
            class_embs = [embs[i] for i in indices]
            dim = len(class_embs[0])
            proto = [sum(e[d] for e in class_embs) / len(class_embs) for d in range(dim)]
            prototypes.append(proto)
        self._prototypes = prototypes  # (n_classes, embed_dim)
        return self

    def predict(self, query_X, temperature: float = 1.0) -> list:
        """Return logits (batch, n_classes) for each query."""
        embs = self._embed(query_X)  # (batch, embed_dim)
        n_classes = len(self._prototypes)
        result = []
        for emb in embs:
            row = []
            for proto in self._prototypes:
                if self.distance == 'cosine':
                    dot = sum(emb[i] * proto[i] for i in range(len(emb)))
                    ne = math.sqrt(sum(v*v for v in emb)) + 1e-8
                    np_ = math.sqrt(sum(v*v for v in proto)) + 1e-8
                    row.append(dot / (ne * np_) / temperature)
                else:  # euclidean
                    dist2 = sum((emb[i] - proto[i])**2 for i in range(len(emb)))
                    row.append(-dist2 / temperature)
            result.append(row)
        return result

    def predict_classes(self, query_X) -> list:
        logits = self.predict(query_X)
        return [self._classes[max(range(len(row)), key=lambda i: row[i])] for row in logits]


# ---------------------------------------------------------------------------
# Task samplers for N-way K-shot episodes
# ---------------------------------------------------------------------------

class FewShotSampler:
    """Sample N-way K-shot episodes from a dataset."""

    def __init__(self, X, y, n_way: int = 5, k_shot: int = 1, q_query: int = 15):
        self.X = list(X)
        self.y = list(y)
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self._classes = list(dict.fromkeys(self.y))

    def sample_episode(self):
        """Sample one N-way K-shot episode."""
        selected_classes = _random.sample(self._classes, min(self.n_way, len(self._classes)))
        sup_X, sup_y, qry_X, qry_y = [], [], [], []
        for new_label, cls in enumerate(selected_classes):
            idx = [i for i, lbl in enumerate(self.y) if lbl == cls]
            chosen = _random.sample(idx, min(self.k_shot + self.q_query, len(idx)))
            sup_idx = chosen[:self.k_shot]
            qry_idx = chosen[self.k_shot:]
            for i in sup_idx: sup_X.append(self.X[i]); sup_y.append(new_label)
            for i in qry_idx: qry_X.append(self.X[i]); qry_y.append(new_label)
        return sup_X, sup_y, qry_X, qry_y

    def episodes(self, n: int):
        for _ in range(n):
            yield self.sample_episode()


__all__ = [
    'MAML', 'FOMAML', 'Reptile',
    'ProtoNet',
    'FewShotSampler',
]
