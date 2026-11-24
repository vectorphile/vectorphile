"""vectorphile.federated — Federated Learning framework.

Provides:
- FedAvg (McMahan et al., 2017) — classic federated averaging
- FedProx — with proximal term for heterogeneous clients
- Scaffold — variance-reduced federated optimization
- Client / Server abstraction
- Differential privacy noise injection (Gaussian mechanism)
- Gradient compression (TopK sparsification)

No external dependencies.

Usage::

    from vectorphile.federated import FederatedServer, FederatedClient, FedAvg

    # Create server and clients
    server = FederatedServer(global_model)
    clients = [FederatedClient(local_data_i, model_cls) for i in range(10)]

    # Run FedAvg for 10 rounds
    aggregator = FedAvg(server, clients, rounds=10, fraction=0.3)
    aggregator.run()
    print('Final accuracy:', aggregator.evaluate(test_data))
"""
from __future__ import annotations
import copy
import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple
import math as _math_fed


# ---------------------------------------------------------------------------
# Differential privacy utilities
# ---------------------------------------------------------------------------

def _flat_fed(x):
    if hasattr(x, 'to_flat_list'):
        return [float(v) for v in x.to_flat_list()]
    if hasattr(x, 'tolist'):
        v = x.tolist()
        if isinstance(v, list) and v and isinstance(v[0], list):
            return [float(item) for row in v for item in row]
        return [float(item) for item in v]
    if isinstance(x, list):
        return [float(item) for item in x]
    return [float(x)]

def _bm_noise(sigma: float) -> float:
    import math as _m
    u1 = max(random.random(), 1e-15)
    u2 = random.random()
    return sigma * _m.sqrt(-2*_m.log(u1)) * _m.cos(2*_m.pi*u2)


def gaussian_noise(tensor, sensitivity: float = 1.0,
                   epsilon: float = 1.0, delta: float = 1e-5) -> Any:
    """Add calibrated Gaussian noise for (ε, δ)-DP."""
    sigma = math.sqrt(2 * math.log(1.25 / delta)) * sensitivity / epsilon
    flat = _flat_fed(tensor)
    return [v + _bm_noise(sigma) for v in flat]


def clip_gradients(grads: dict, max_norm: float = 1.0) -> dict:
    """Clip gradient dict by global L2 norm."""
    total_norm = math.sqrt(sum(sum(v*v for v in _flat_fed(g)) for g in grads.values()))
    scale = min(1.0, max_norm / (total_norm + 1e-8))
    return {k: [v * scale for v in _flat_fed(val)] for k, val in grads.items()}


# ---------------------------------------------------------------------------
# Gradient compression
# ---------------------------------------------------------------------------

def topk_sparsify(grad, ratio: float = 0.1):
    """TopK gradient sparsification. Returns (indices, values)."""
    flat = _flat_fed(grad)
    k = max(1, int(len(flat) * ratio))
    indexed = sorted(range(len(flat)), key=lambda i: abs(flat[i]), reverse=True)[:k]
    return indexed, [flat[i] for i in indexed]


def topk_restore(indices, values, shape: tuple):
    n = 1
    for s in shape:
        n *= s
    flat = [0.0] * n
    for i, v in zip(indices, values):
        flat[i] = float(v)
    return flat


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class FederatedClient:
    """Single federated learning client holding local data.

    Args:
        data: (X, y) tuple or any iterable of batches.
        model_fn: Callable that returns a fresh model (or None to share).
        optimizer_fn: Callable(params) → optimizer.
        local_epochs: Number of local training epochs.
        batch_size: Local minibatch size.
        client_id: Identifier string.
    """

    def __init__(self, data, model_fn: Optional[Callable] = None,
                 optimizer_fn: Optional[Callable] = None,
                 local_epochs: int = 1, batch_size: int = 32,
                 client_id: Optional[str] = None,
                 dp_epsilon: Optional[float] = None):
        self.data = data
        self.model_fn = model_fn
        self.optimizer_fn = optimizer_fn
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.client_id = client_id or str(id(self))
        self.dp_epsilon = dp_epsilon
        self.model = None
        self._local_steps = 0

    def set_model(self, global_params: dict) -> None:
        """Load global parameters into local model."""
        if self.model is not None and hasattr(self.model, 'load_state_dict'):
            self.model.load_state_dict(copy.deepcopy(global_params))

    def get_state_dict(self) -> dict:
        if self.model is not None and hasattr(self.model, 'state_dict'):
            return copy.deepcopy(self.model.state_dict())
        return {}

    def local_update(self, global_params: dict, loss_fn: Optional[Callable] = None,
                     mu: float = 0.0) -> Tuple[dict, int]:
        """Train locally and return updated params + number of samples.

        Args:
            global_params: Current global model state dict.
            loss_fn: Loss function(output, target) → scalar Tensor.
            mu: Proximal term coefficient (FedProx). 0 = FedAvg.

        Returns:
            (updated_state_dict, n_samples)
        """
        if self.model is None:
            raise RuntimeError('Model not initialized. Call set_model() first.')
        self.set_model(global_params)
        X, y = self.data
        X_list = [[float(v) for v in row] if isinstance(row,(list,tuple)) else [float(row)] for row in X]
        y_list = [float(v) for v in y]
        n = len(X_list)
        if self.optimizer_fn is None:
            raise RuntimeError('optimizer_fn not set')
        opt = self.optimizer_fn(self.model.parameters())
        global_snapshot = copy.deepcopy(global_params)
        for _ in range(self.local_epochs):
            idx = list(range(n)); random.shuffle(idx)
            for start in range(0, n, self.batch_size):
                batch_idx = idx[start:start + self.batch_size]
                from vectorphile.core.tensor import Tensor
                from vectorphile._core.ndarray import array as _nd
                bX_flat = [v for i in batch_idx for v in X_list[i]]
                by_flat = [y_list[i] for i in batch_idx]
                bX = Tensor(_nd(bX_flat))
                by = Tensor(_nd(by_flat))
                opt.zero_grad()
                pred = self.model(bX)
                loss = loss_fn(pred, by)
                if mu > 0:
                    for name, param in self.model.named_parameters():
                        if name in global_snapshot:
                            pd = _flat_fed(param._data)
                            gd = _flat_fed(global_snapshot[name]._data if hasattr(global_snapshot[name],'_data') else global_snapshot[name])
                            prox_val = mu/2 * sum((a-b)**2 for a,b in zip(pd,gd))
                            prox = Tensor(_nd([prox_val]))
                            loss = loss + prox
                loss.backward()
                opt.step()
                self._local_steps += 1
        updated = self.get_state_dict()
        if self.dp_epsilon is not None:
            for k in updated:
                arr = _flat_fed(updated[k]._data if hasattr(updated[k],'_data') else updated[k])
                updated[k] = gaussian_noise(arr, epsilon=self.dp_epsilon)
        return updated, n


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class FederatedServer:
    """Central server holding the global model.

    Usage::

        server = FederatedServer(model)
        server.broadcast()  # → copies params to all clients
        server.aggregate(client_updates)  # → FedAvg
    """

    def __init__(self, model):
        self.model = model
        self.round = 0
        self._history: List[dict] = []

    def get_global_params(self) -> dict:
        return copy.deepcopy(self.model.state_dict())

    def aggregate(self, updates: List[Tuple[dict, int]]) -> None:
        """FedAvg aggregation: weighted average by number of samples."""
        if not updates:
            return
        total_n = sum(n for _, n in updates)
        global_sd = self.get_global_params()
        avg_sd = {}
        from vectorphile._core.ndarray import array as _nd
        for key in global_sd:
            weighted_sum = None
            for params, n in updates:
                val = params.get(key)
                if val is None:
                    continue
                arr = _flat_fed(val._data if hasattr(val, '_data') else val)
                w = n / total_n
                weighted = [v * w for v in arr]
                weighted_sum = weighted if weighted_sum is None else [a+b for a,b in zip(weighted_sum, weighted)]
            if weighted_sum is not None:
                avg_sd[key] = weighted_sum
        cur_sd = global_sd
        for k, v in avg_sd.items():
            if k in cur_sd and hasattr(cur_sd[k], '_data'):
                nd = _nd(v)
                if hasattr(nd, 'reshape') and hasattr(cur_sd[k]._data, 'shape'):
                    nd = nd.reshape(cur_sd[k]._data.shape)
                cur_sd[k]._data = nd
        self.model.load_state_dict(cur_sd)
        self.round += 1

    def evaluate(self, X, y, metric: str = 'accuracy') -> float:
        """Evaluate global model on given data."""
        from vectorphile.core.tensor import Tensor
        from vectorphile._core.ndarray import array as _nd
        import vectorphile as vp
        X_list = [[float(v) for v in row] if isinstance(row,(list,tuple)) else [float(row)] for row in X]
        y_list = [float(v) for v in y]
        flat_X = [v for row in X_list for v in row]
        with vp.autograd.no_grad():
            pred = self.model(Tensor(_nd(flat_X)))
        preds = _flat_fed(pred._data if hasattr(pred, '_data') else pred)
        n = len(y_list)
        if metric == 'accuracy':
            n_out = len(preds) // max(n, 1)
            if n_out > 1:
                pred_cls = [max(range(n_out), key=lambda j: preds[i*n_out+j]) for i in range(n)]
            else:
                pred_cls = [int(round(preds[i])) for i in range(n)]
            return sum(1 for p,t in zip(pred_cls, y_list) if p == int(t)) / max(n, 1)
        elif metric == 'mse':
            return sum((preds[i] - y_list[i])**2 for i in range(min(len(preds), n))) / max(n, 1)
        return 0.0


# ---------------------------------------------------------------------------
# FedAvg algorithm
# ---------------------------------------------------------------------------

class FedAvg:
    """Federated Averaging (McMahan et al., 2017).

    Usage::

        fed = FedAvg(server, clients, rounds=20, fraction=0.5,
                     loss_fn=CrossEntropyLoss())
        fed.run()
    """

    def __init__(self, server: FederatedServer, clients: List[FederatedClient],
                 rounds: int = 10, fraction: float = 0.5,
                 loss_fn=None, mu: float = 0.0, verbose: bool = True):
        self.server = server
        self.clients = clients
        self.rounds = rounds
        self.fraction = fraction
        self.loss_fn = loss_fn
        self.mu = mu  # 0 = FedAvg, >0 = FedProx
        self.verbose = verbose
        self.history: List[dict] = []

    def run(self) -> List[dict]:
        """Run all rounds and return history."""
        for rnd in range(self.rounds):
            # Select fraction of clients
            n_clients = max(1, int(len(self.clients) * self.fraction))
            selected = random.sample(self.clients, n_clients)
            global_params = self.server.get_global_params()
            updates = []
            for client in selected:
                # Initialize client model from server model
                if client.model is None and client.model_fn is not None:
                    client.model = client.model_fn()
                try:
                    upd, n = client.local_update(global_params, self.loss_fn, self.mu)
                    updates.append((upd, n))
                except Exception as e:
                    if self.verbose:
                        print(f'Client {client.client_id} failed: {e}')
            self.server.aggregate(updates)
            info = {'round': rnd + 1, 'n_clients': len(selected)}
            self.history.append(info)
            if self.verbose:
                print(f'[FedAvg] Round {rnd+1}/{self.rounds} — {len(selected)} clients')
        return self.history


class FedProx(FedAvg):
    """FedProx (Li et al., 2020) — FedAvg with proximal regularization."""

    def __init__(self, *args, mu: float = 0.01, **kwargs):
        super().__init__(*args, mu=mu, **kwargs)


class Scaffold(FedAvg):
    """SCAFFOLD (Karimireddy et al., 2020) — variance-reduced federated learning.

    Uses server and client control variates to correct client drift.
    Each client update: Δy = y - c + c_i, where c is server control variate,
    c_i is client control variate. After round: c += (c_i_new - c_i) / n_clients.
    """

    def __init__(self, server: FederatedServer, clients: list, rounds: int = 10,
                 fraction: float = 1.0, loss_fn=None, mu: float = 0.0,
                 lr: float = 0.01, local_steps: int = 5, verbose: bool = False):
        super().__init__(server, clients, rounds, fraction, loss_fn, mu, verbose)
        self.lr = lr
        self.local_steps = local_steps
        # Server control variate: same structure as model params, initialized to zero
        self._server_cv: dict = {}
        self._client_cvs: List[dict] = [{} for _ in range(len(clients))]
        self._init_control_variates()

    def _init_control_variates(self):
        sd = self.server.get_global_params()
        for k, v in sd.items():
            flat = _flat_fed(v._data if hasattr(v, '_data') else v)
            self._server_cv[k] = [0.0] * len(flat)
        for i in range(len(self.clients)):
            self._client_cvs[i] = {k: [0.0] * len(v) for k, v in self._server_cv.items()}

    def run(self):
        for rnd in range(self.rounds):
            n_sel = max(1, int(len(self.clients) * self.fraction))
            import random as _rnd
            selected_idxs = _rnd.sample(range(len(self.clients)), n_sel)
            selected = [self.clients[i] for i in selected_idxs]

            global_params = self.server.get_global_params()
            updates, cv_deltas = [], []

            for ci, client in zip(selected_idxs, selected):
                c_cv = self._client_cvs[ci]
                # Client local update with control variate correction
                import copy
                local_model = copy.deepcopy(self.server.model)
                opt = client.optimizer if hasattr(client, 'optimizer') else None

                from vectorphile.optim import SGD
                local_opt = SGD(local_model.parameters(), lr=self.lr)

                for _ in range(self.local_steps):
                    local_opt.zero_grad()
                    if client.X is not None and self.loss_fn is not None:
                        from vectorphile.core.tensor import Tensor
                        from vectorphile._core.ndarray import array as _nd
                        X_t = Tensor(_nd([float(v) for row in client.X for v in row])).reshape(len(client.X), -1) if hasattr(client.X[0], '__iter__') else Tensor(_nd([float(v) for v in client.X]))
                        y_t = Tensor(_nd([float(v) for v in client.y]))
                        pred = local_model(X_t)
                        loss = self.loss_fn(pred, y_t)
                        loss.backward()
                        # Apply SCAFFOLD correction: add (c - c_i) to gradient
                        for name, param in local_model.named_parameters():
                            if param.grad is not None and name in self._server_cv:
                                sc = self._server_cv[name]
                                cc = c_cv[name]
                                g = _flat_fed(param.grad._data if hasattr(param.grad, '_data') else param.grad)
                                corrected = [g[j] + sc[j] - cc[j] for j in range(len(g))]
                                from vectorphile._core.ndarray import array as _nd2
                                param.grad._data = _nd2(corrected).reshape(param.grad.shape)
                    local_opt.step()

                # Compute client update
                local_sd = local_model.state_dict()
                n_samples = len(client.X) if client.X is not None else 1

                # Update client control variate: c_i_new = c_i - c + (y - x) / (K * lr)
                K = max(1, self.local_steps)
                new_cv = {}
                for name in self._server_cv:
                    if name in global_params and name in local_sd:
                        gp_flat = _flat_fed(global_params[name]._data if hasattr(global_params[name], '_data') else global_params[name])
                        lp_flat = _flat_fed(local_sd[name]._data if hasattr(local_sd[name], '_data') else local_sd[name])
                        sc = self._server_cv[name]
                        cc = c_cv[name]
                        new_cv[name] = [cc[j] - sc[j] + (gp_flat[j] - lp_flat[j]) / (K * self.lr)
                                        for j in range(len(cc))]

                cv_delta = {}
                for name in self._server_cv:
                    if name in new_cv:
                        cv_delta[name] = [new_cv[name][j] - c_cv[name][j]
                                          for j in range(len(c_cv[name]))]
                        self._client_cvs[ci][name] = new_cv[name]
                cv_deltas.append(cv_delta)
                updates.append((local_sd, n_samples))

            self.server.aggregate(updates)
            # Update server control variate
            if cv_deltas:
                for name in self._server_cv:
                    total_delta = [0.0] * len(self._server_cv[name])
                    for cd in cv_deltas:
                        if name in cd:
                            for j in range(len(total_delta)):
                                total_delta[j] += cd[name][j]
                    n = max(len(cv_deltas), 1)
                    self._server_cv[name] = [self._server_cv[name][j] + total_delta[j] / n
                                             for j in range(len(total_delta))]

            info = {'round': rnd + 1, 'n_clients': n_sel}
            self.history.append(info)
            if self.verbose:
                print(f'[SCAFFOLD] Round {rnd+1}/{self.rounds} — {n_sel} clients')
        return self.history


__all__ = [
    'FederatedServer', 'FederatedClient',
    'FedAvg', 'FedProx', 'Scaffold',
    'gaussian_noise', 'clip_gradients',
    'topk_sparsify', 'topk_restore',
]
