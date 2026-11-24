"""Model interpretability — SaliencyMap, PermutationImportance, LIME."""
import math
import random
from vectorphile.core.tensor import Tensor
from vectorphile._core.ndarray import NDArray, zeros as _nd_zeros, _c_strides, _DType
import array as _arr


def _is_numpy(x):
    t = type(x)
    return t.__module__ == 'numpy' or (hasattr(x, 'flatten') and hasattr(x, 'dtype') and not hasattr(x, 'to_flat_list'))


def _make_nd(flat, shape, dtype='float32'):
    buf = _arr.array('f', [float(v) for v in flat])
    return NDArray(buf, shape, _c_strides(shape), _DType(dtype))


def _to_flat(x) -> list:
    if hasattr(x, 'to_flat_list'):
        return [float(v) for v in x.to_flat_list()]
    try:
        return [float(v) for v in x.ravel()]
    except Exception:
        return [float(v) for v in x]


def _argmax_1d(lst: list) -> int:
    return max(range(len(lst)), key=lambda i: lst[i])


def _softmax_1d(lst: list) -> list:
    m = max(lst)
    e = [math.exp(v - m) for v in lst]
    s = sum(e)
    return [v / s for v in e]


def _randn(n: int) -> list:
    out = []
    while len(out) < n:
        u1 = random.random() + 1e-15
        u2 = random.random()
        z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        z1 = math.sqrt(-2 * math.log(u1)) * math.sin(2 * math.pi * u2)
        out.extend([z0, z1])
    return out[:n]


class SaliencyMap:
    """Gradient-based saliency: |d_loss/d_x| for a target class."""

    def __init__(self, model):
        self.model = model

    def __call__(self, x, target: int = 0):
        flat = _to_flat(x)
        n = len(flat)
        nd = _make_nd(flat, (1, n))
        x_t = Tensor(nd, requires_grad=True)
        out = self.model(x_t)
        out_flat = _to_flat(out._data)
        num_classes = len(out_flat)
        # One-hot backward
        grad_data = [1.0 if i == target else 0.0 for i in range(num_classes)]
        grad_nd = _make_nd(grad_data, (1, num_classes))
        out.backward(Tensor(grad_nd))
        if x_t.grad is None:
            sal = [0.0] * n
        else:
            sal = [abs(v) for v in _to_flat(x_t.grad._data)]
        if _is_numpy(x):
            import numpy as _np
            orig_shape = x.shape if hasattr(x, 'shape') else (n,)
            return _np.array(sal, dtype='float32').reshape(orig_shape)
        return sal


class PermutationImportance:
    """Feature importance via permutation."""

    def __init__(self, model, n_repeats: int = 5, metric=None):
        self.model = model
        self.n_repeats = n_repeats
        self._metric = metric or self._accuracy

    def _accuracy(self, model, X, y) -> float:
        correct = 0
        for i, row in enumerate(X):
            flat = _to_flat(row) if not isinstance(row, list) else row
            nd = _make_nd(flat, (1, len(flat)))
            out = model(Tensor(nd))
            pred = _argmax_1d(_to_flat(out._data))
            if pred == int(y[i]):
                correct += 1
        return correct / max(1, len(X))

    def __call__(self, X, y) -> list:
        # X: list of rows or NDArray with shape (N, F)
        if hasattr(X, 'shape'):
            N, F = X.shape[0], X.shape[1]
            flat_X = _to_flat(X)
            rows = [[flat_X[i * F + j] for j in range(F)] for i in range(N)]
        else:
            rows = [[float(v) for v in row] for row in X]
            N = len(rows); F = len(rows[0]) if N > 0 else 0
        y_list = list(y) if not isinstance(y, list) else y

        was_numpy = _is_numpy(X)
        base = self._metric(self.model, rows, y_list)
        importances = [0.0] * F
        for f in range(F):
            scores = []
            for _ in range(self.n_repeats):
                col = [rows[i][f] for i in range(N)]
                random.shuffle(col)
                X_perm = [[rows[i][j] if j != f else col[i] for j in range(F)] for i in range(N)]
                scores.append(base - self._metric(self.model, X_perm, y_list))
            importances[f] = sum(scores) / max(1, len(scores))
        if was_numpy:
            import numpy as _np
            return _np.array(importances, dtype='float32')
        return importances


class LIME:
    """Local Interpretable Model-agnostic Explanations (tabular)."""

    def __init__(self, model, n_samples: int = 100, sigma: float = 0.1):
        self.model = model
        self.n_samples = n_samples
        self.sigma = sigma

    def explain_tabular(self, x, class_idx: int = 0) -> dict:
        x_flat = _to_flat(x) if not isinstance(x, list) else x
        n = len(x_flat)
        # Perturb
        X_perturb = [[x_flat[j] + _randn(1)[0] * self.sigma for j in range(n)]
                     for _ in range(self.n_samples)]
        # Weights by proximity
        kernel_width = 0.75 * math.sqrt(n)
        dists = [math.sqrt(sum((X_perturb[i][j] - x_flat[j]) ** 2 for j in range(n)))
                 for i in range(self.n_samples)]
        weights = [math.exp(-(d ** 2) / (2 * kernel_width ** 2)) for d in dists]

        # Model predictions
        y_pred = []
        for row in X_perturb:
            nd = _make_nd(row, (1, n))
            out = self.model(Tensor(nd))
            probs_flat = _to_flat(out._data)
            probs = _softmax_1d(probs_flat)
            y_pred.append(probs[class_idx] if class_idx < len(probs) else 0.0)

        # Weighted least squares via normal equations
        # X^T W X coef = X^T W y
        W = weights
        XtWX = [[sum(W[r] * X_perturb[r][i] * X_perturb[r][j] for r in range(self.n_samples))
                 for j in range(n)] for i in range(n)]
        XtWy = [sum(W[r] * X_perturb[r][i] * y_pred[r] for r in range(self.n_samples))
                for i in range(n)]
        # Gaussian elimination
        aug = [XtWX[i][:] + [XtWy[i]] for i in range(n)]
        for col in range(n):
            pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
            aug[col], aug[pivot] = aug[pivot], aug[col]
            pv = aug[col][col]
            if abs(pv) < 1e-14:
                continue
            aug[col] = [v / pv for v in aug[col]]
            for r in range(n):
                if r != col and abs(aug[r][col]) > 0:
                    f = aug[r][col]
                    aug[r] = [aug[r][j] - f * aug[col][j] for j in range(n + 1)]
        coef = [aug[i][n] for i in range(n)]
        return {'weights': coef, 'feature_importances': [abs(c) for c in coef]}


class IntegratedGradients:
    """Integrated Gradients attribution method (Sundararajan et al., 2017)."""

    def __init__(self, model, n_steps: int = 50):
        self.model = model
        self.n_steps = n_steps

    def attribute(self, inputs: Tensor, baseline: Tensor = None, target: int = 0):
        inp_flat = _to_flat(inputs)
        n = len(inp_flat)
        if baseline is None:
            base_flat = [0.0] * n
        else:
            base_flat = _to_flat(baseline)
        accumulated = [0.0] * n
        for step in range(1, self.n_steps + 1):
            alpha = step / self.n_steps
            interp_flat = [base_flat[i] + alpha * (inp_flat[i] - base_flat[i]) for i in range(n)]
            interp_nd = _make_nd(interp_flat, inputs.shape if hasattr(inputs, 'shape') else [n])
            interp_t = Tensor(interp_nd)
            interp_t.requires_grad = True
            if hasattr(interp_t, '_data'):
                interp_t._data.requires_grad = True
            try:
                out = self.model(interp_t)
                out_flat = _to_flat(out)
                score = out_flat[target] if target < len(out_flat) else out_flat[0]
                score_t = Tensor(_make_nd([score], [1]))
                score_t.backward() if hasattr(score_t, 'backward') else None
                if hasattr(interp_t, 'grad') and interp_t.grad is not None:
                    grad_flat = _to_flat(interp_t.grad)
                    for i in range(n):
                        accumulated[i] += grad_flat[i]
            except Exception:
                # Finite differences fallback
                eps = 1e-4
                for i in range(n):
                    plus = interp_flat[:]
                    minus = interp_flat[:]
                    plus[i] += eps
                    minus[i] -= eps
                    out_p = _to_flat(self.model(Tensor(_make_nd(plus, [n]))))
                    out_m = _to_flat(self.model(Tensor(_make_nd(minus, [n]))))
                    sp = out_p[target] if target < len(out_p) else out_p[0]
                    sm = out_m[target] if target < len(out_m) else out_m[0]
                    accumulated[i] += (sp - sm) / (2 * eps)
        avg_grads = [g / self.n_steps for g in accumulated]
        attributions = [(inp_flat[i] - base_flat[i]) * avg_grads[i] for i in range(n)]
        shape = inputs.shape if hasattr(inputs, 'shape') else [n]
        return Tensor(_make_nd(attributions, shape))


class GradCAM:
    """Gradient-weighted Class Activation Mapping (approximate, model-agnostic)."""

    def __init__(self, model, target_layer=None):
        self.model = model
        self.target_layer = target_layer
        self._activations = None
        self._gradients = None

    def attribute(self, inputs: Tensor, target: int = 0):
        inp_flat = _to_flat(inputs)
        n = len(inp_flat)
        eps = 1e-4
        grads = []
        base_out = _to_flat(self.model(inputs))
        base_score = base_out[target] if target < len(base_out) else base_out[0]
        for i in range(n):
            perturbed = inp_flat[:]
            perturbed[i] += eps
            out = _to_flat(self.model(Tensor(_make_nd(perturbed, inputs.shape if hasattr(inputs, 'shape') else [n]))))
            score = out[target] if target < len(out) else out[0]
            grads.append((score - base_score) / eps)
        cam = [max(0.0, g) for g in grads]
        total = sum(cam) + 1e-10
        cam = [v / total for v in cam]
        shape = inputs.shape if hasattr(inputs, 'shape') else [n]
        return Tensor(_make_nd(cam, shape))


class KernelSHAP:
    """Kernel SHAP: Shapley values via weighted linear regression on coalitions."""

    def __init__(self, model, background=None, n_samples: int = 100):
        self.model = model
        self.background = background
        self.n_samples = n_samples

    def _shap_kernel(self, z_size: int, subset_size: int) -> float:
        if subset_size == 0 or subset_size == z_size:
            return 1e6
        from math import comb
        return (z_size - 1) / (comb(z_size, subset_size) * subset_size * (z_size - subset_size))

    def attribute(self, inputs: Tensor, target: int = 0):
        inp_flat = _to_flat(inputs)
        n = len(inp_flat)
        bg_flat = _to_flat(self.background) if self.background is not None else [0.0] * n
        rng = random.Random(42)
        coalitions = []
        weights = []
        for _ in range(self.n_samples):
            mask = [rng.randint(0, 1) for _ in range(n)]
            coalitions.append(mask)
            k = sum(mask)
            weights.append(self._shap_kernel(n, k))
        # Evaluate model on coalitions
        y_vals = []
        for mask in coalitions:
            x = [inp_flat[i] if mask[i] else bg_flat[i] for i in range(n)]
            out = _to_flat(self.model(Tensor(_make_nd(x, [n]))))
            score = out[target] if target < len(out) else out[0]
            y_vals.append(score)
        # Null/full predictions
        null_out = _to_flat(self.model(Tensor(_make_nd(bg_flat, [n]))))
        null_score = null_out[target] if target < len(null_out) else null_out[0]
        y_centered = [v - null_score for v in y_vals]
        # Weighted least squares: phi = (Z^T W Z)^-1 Z^T W y
        # Using simple diagonal weighting
        ZtWZ = [[sum(weights[s] * coalitions[s][i] * coalitions[s][j]
                     for s in range(self.n_samples)) for j in range(n)] for i in range(n)]
        ZtWy = [sum(weights[s] * coalitions[s][i] * y_centered[s]
                    for s in range(self.n_samples)) for i in range(n)]
        # Solve with Gauss-Seidel (simple diagonal approximation)
        shap_vals = [ZtWy[i] / (ZtWZ[i][i] + 1e-10) for i in range(n)]
        shape = inputs.shape if hasattr(inputs, 'shape') else [n]
        return Tensor(_make_nd(shap_vals, shape))


__all__ = ['SaliencyMap', 'PermutationImportance', 'LIME',
           'IntegratedGradients', 'GradCAM', 'KernelSHAP']
