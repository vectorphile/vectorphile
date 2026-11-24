"""Evaluation metrics — pure Python."""
import math
from typing import List, Optional
from vectorphile.ml._np_compat import to_array as _ta


def _uniq_sorted(seq):
    return sorted(set(seq))


def accuracy_score(y_true, y_pred, normalize=True) -> float:
    yt, yp = list(y_true), list(y_pred)
    correct = sum(1 for a, b in zip(yt, yp) if a == b)
    return float(correct / len(yt)) if normalize else int(correct)


def confusion_matrix(y_true, y_pred, labels=None):
    yt, yp = list(y_true), list(y_pred)
    classes = labels if labels is not None else _uniq_sorted(yt + yp)
    K = len(classes)
    cm = [[0]*K for _ in range(K)]
    class_map = {c: i for i, c in enumerate(classes)}
    for t, p in zip(yt, yp):
        cm[class_map[t]][class_map[p]] += 1
    return _ta(cm)


def precision_score(y_true, y_pred, average='binary', labels=None, zero_division=0) -> float:
    yt, yp = list(y_true), list(y_pred)
    classes = labels or _uniq_sorted(yt + yp)
    precisions = []
    for c in classes:
        tp = sum(1 for a, b in zip(yp, yt) if b == c and a == c)
        fp = sum(1 for a, b in zip(yp, yt) if b != c and a == c)
        denom = tp + fp
        precisions.append(tp / denom if denom > 0 else zero_division)
    if average == 'micro':
        tps = sum(1 for a, b in zip(yp, yt) if a == b)
        fps = len(yt) - tps
        return float(tps / (tps + fps)) if (tps + fps) > 0 else zero_division
    if average == 'macro':
        return float(sum(precisions) / len(precisions))
    if average == 'weighted':
        weights = [sum(1 for y in yt if y == c) for c in classes]
        total = sum(weights)
        return float(sum(p*w for p,w in zip(precisions, weights)) / total) if total else zero_division
    return float(precisions[-1]) if precisions else zero_division


def recall_score(y_true, y_pred, average='binary', labels=None, zero_division=0) -> float:
    yt, yp = list(y_true), list(y_pred)
    classes = labels or _uniq_sorted(yt + yp)
    recalls = []
    for c in classes:
        tp = sum(1 for a, b in zip(yp, yt) if b == c and a == c)
        fn = sum(1 for a, b in zip(yp, yt) if b == c and a != c)
        denom = tp + fn
        recalls.append(tp / denom if denom > 0 else zero_division)
    if average == 'macro':
        return float(sum(recalls) / len(recalls))
    if average == 'weighted':
        weights = [sum(1 for y in yt if y == c) for c in classes]
        total = sum(weights)
        return float(sum(r*w for r,w in zip(recalls, weights)) / total) if total else zero_division
    return float(recalls[-1]) if recalls else zero_division


def f1_score(y_true, y_pred, average='binary', labels=None, zero_division=0) -> float:
    p = precision_score(y_true, y_pred, average=average, labels=labels, zero_division=zero_division)
    r = recall_score(y_true, y_pred, average=average, labels=labels, zero_division=zero_division)
    return float(2 * p * r / (p + r)) if (p + r) > 0 else zero_division


def _roc_auc_binary(y_true, y_score) -> float:
    yt = list(y_true)
    pairs = sorted(zip(y_score, yt), key=lambda x: -x[0])
    pos = sum(yt)
    neg = len(yt) - pos
    if pos == 0 or neg == 0:
        return 0.5
    tpr_pts, fpr_pts = [0.0], [0.0]
    tp = fp = 0
    for _, label in pairs:
        if label:
            tp += 1
        else:
            fp += 1
        tpr_pts.append(tp / pos)
        fpr_pts.append(fp / neg)
    auc = 0.0
    for i in range(1, len(tpr_pts)):
        auc += (fpr_pts[i] - fpr_pts[i-1]) * (tpr_pts[i] + tpr_pts[i-1]) / 2
    return float(auc)


def roc_auc_score(y_true, y_score, average='macro') -> float:
    yt = list(y_true)
    if not isinstance(y_score[0], (list, tuple)):
        return _roc_auc_binary(yt, list(y_score))
    classes = _uniq_sorted(yt)
    aucs = [_roc_auc_binary([1 if v == c else 0 for v in yt],
                             [row[i] for row in y_score])
            for i, c in enumerate(classes)]
    return float(sum(aucs) / len(aucs))


def mse(y_true, y_pred) -> float:
    yt = [float(v) for v in y_true]
    yp = [float(v) for v in y_pred]
    n = len(yt)
    return sum((a-b)**2 for a,b in zip(yt,yp)) / n


def rmse(y_true, y_pred) -> float:
    return math.sqrt(mse(y_true, y_pred))


def mae(y_true, y_pred) -> float:
    yt = [float(v) for v in y_true]
    yp = [float(v) for v in y_pred]
    return sum(abs(a-b) for a,b in zip(yt,yp)) / len(yt)


def r2_score(y_true, y_pred) -> float:
    yt = [float(v) for v in y_true]
    yp = [float(v) for v in y_pred]
    mu = sum(yt) / len(yt)
    ss_res = sum((a-b)**2 for a,b in zip(yt,yp))
    ss_tot = sum((a-mu)**2 for a in yt)
    return float(1 - ss_res / (ss_tot + 1e-12))


def mean_absolute_percentage_error(y_true, y_pred) -> float:
    yt = [float(v) for v in y_true]
    yp = [float(v) for v in y_pred]
    pairs = [(a, b) for a, b in zip(yt, yp) if a != 0]
    if not pairs:
        return 0.0
    return float(sum(abs((a-b)/a) for a,b in pairs) / len(pairs))


def silhouette_score(X, labels) -> float:
    X2 = [[float(v) for v in row] for row in X]
    lbls = list(labels)
    n = len(X2)
    p = len(X2[0]) if X2 else 0
    unique_lbls = _uniq_sorted(lbls)

    def dist(a, b):
        return math.sqrt(sum((a[k]-b[k])**2 for k in range(p)))

    scores = []
    for i in range(n):
        same_idxs = [j for j in range(n) if j != i and lbls[j] == lbls[i]]
        a = (sum(dist(X2[i], X2[j]) for j in same_idxs) / len(same_idxs)) if same_idxs else 0.0
        b_vals = []
        for c in unique_lbls:
            if c == lbls[i]:
                continue
            other = [j for j in range(n) if lbls[j] == c]
            if other:
                b_vals.append(sum(dist(X2[i], X2[j]) for j in other) / len(other))
        b = min(b_vals) if b_vals else 0.0
        mx = max(a, b)
        scores.append((b - a) / mx if mx > 0 else 0.0)
    return float(sum(scores) / n) if scores else 0.0


def log_loss(y_true, y_pred, eps=1e-15) -> float:
    yt = list(y_true)
    yp = list(y_pred)
    if not isinstance(yp[0], (list, tuple)):
        total = sum(-(t * math.log(max(min(float(p), 1-eps), eps)) +
                      (1-t) * math.log(max(min(1-float(p), 1-eps), eps)))
                    for t, p in zip(yt, yp))
        return float(total / len(yt))
    classes = _uniq_sorted(yt)
    n = len(yt)
    total = 0.0
    for i in range(n):
        ci = classes.index(yt[i])
        total -= math.log(max(float(yp[i][ci]), eps))
    return float(total / n)
