"""Evaluation metrics."""
import numpy as np
from typing import Optional


def accuracy_score(y_true, y_pred, normalize=True) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    correct = (y_true == y_pred).sum()
    return float(correct / len(y_true)) if normalize else int(correct)


def confusion_matrix(y_true, y_pred, labels=None) -> np.ndarray:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    classes = labels if labels is not None else np.unique(np.concatenate([y_true, y_pred]))
    K = len(classes)
    cm = np.zeros((K, K), dtype=int)
    class_map = {c: i for i, c in enumerate(classes)}
    for t, p in zip(y_true, y_pred):
        cm[class_map[t], class_map[p]] += 1
    return cm


def precision_score(y_true, y_pred, average='binary', labels=None, zero_division=0) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    classes = labels or np.unique(np.concatenate([y_true, y_pred]))
    precisions = []
    for c in classes:
        tp = ((y_pred == c) & (y_true == c)).sum()
        fp = ((y_pred == c) & (y_true != c)).sum()
        denom = tp + fp
        precisions.append(tp / denom if denom > 0 else zero_division)
    if average == 'micro':
        tps = sum((y_pred == c) & (y_true == c) for c in classes).sum()
        fps = sum((y_pred == c) & (y_true != c) for c in classes).sum()
        return float(tps / (tps + fps)) if (tps + fps) > 0 else zero_division
    if average == 'macro':
        return float(np.mean(precisions))
    if average == 'weighted':
        weights = [(y_true == c).sum() for c in classes]
        return float(np.average(precisions, weights=weights))
    return float(precisions[0]) if len(precisions) == 1 else float(precisions[-1])


def recall_score(y_true, y_pred, average='binary', labels=None, zero_division=0) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    classes = labels or np.unique(np.concatenate([y_true, y_pred]))
    recalls = []
    for c in classes:
        tp = ((y_pred == c) & (y_true == c)).sum()
        fn = ((y_pred != c) & (y_true == c)).sum()
        denom = tp + fn
        recalls.append(tp / denom if denom > 0 else zero_division)
    if average == 'macro':
        return float(np.mean(recalls))
    if average == 'weighted':
        weights = [(y_true == c).sum() for c in classes]
        return float(np.average(recalls, weights=weights))
    return float(recalls[-1])


def f1_score(y_true, y_pred, average='binary', labels=None, zero_division=0) -> float:
    p = precision_score(y_true, y_pred, average=average, labels=labels, zero_division=zero_division)
    r = recall_score(y_true, y_pred, average=average, labels=labels, zero_division=zero_division)
    return float(2 * p * r / (p + r)) if (p + r) > 0 else zero_division


def roc_auc_score(y_true, y_score, average='macro') -> float:
    y_true, y_score = np.array(y_true), np.array(y_score)
    if y_score.ndim == 1:
        return _roc_auc_binary(y_true, y_score)
    classes = np.unique(y_true)
    aucs = [_roc_auc_binary((y_true == c).astype(int), y_score[:, i]) for i, c in enumerate(classes)]
    return float(np.mean(aucs))


def _roc_auc_binary(y_true, y_score) -> float:
    pos = y_true.sum()
    neg = len(y_true) - pos
    if pos == 0 or neg == 0:
        return 0.5
    order = np.argsort(y_score)[::-1]
    y_sorted = y_true[order]
    tpr = np.cumsum(y_sorted) / pos
    fpr = np.cumsum(1 - y_sorted) / neg
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])
    return float(np.trapz(tpr, fpr))


def mse(y_true, y_pred) -> float:
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred)**2))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true, y_pred) -> float:
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    return float(1 - ss_res / (ss_tot + 1e-12))


def mean_absolute_percentage_error(y_true, y_pred) -> float:
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def silhouette_score(X, labels) -> float:
    X, labels = np.array(X, dtype=float), np.array(labels)
    n = len(X)
    unique_labels = np.unique(labels)
    scores = []
    for i in range(n):
        same = X[labels == labels[i]]
        a = np.mean(np.linalg.norm(same - X[i], axis=1)) if len(same) > 1 else 0.0
        b_vals = []
        for c in unique_labels:
            if c == labels[i]:
                continue
            other = X[labels == c]
            if len(other) > 0:
                b_vals.append(np.mean(np.linalg.norm(other - X[i], axis=1)))
        b = min(b_vals) if b_vals else 0.0
        s = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
        scores.append(s)
    return float(np.mean(scores))


def log_loss(y_true, y_pred, eps=1e-15) -> float:
    y_true = np.array(y_true)
    y_pred = np.clip(np.array(y_pred, dtype=float), eps, 1 - eps)
    if y_pred.ndim == 1:
        return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))
    n = len(y_true)
    classes = np.unique(y_true)
    y_ohe = np.zeros((n, len(classes)))
    for i, c in enumerate(classes):
        y_ohe[:, i] = (y_true == c)
    return float(-np.mean(np.sum(y_ohe * np.log(y_pred), axis=1)))
