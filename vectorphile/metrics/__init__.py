"""vectorphile.metrics — Evaluation metrics for ML/DL/NLP tasks.

Classification, regression, NLP generation, ranking, clustering, calibration.
All implemented from scratch — no sklearn/nltk dependency.
"""
from __future__ import annotations
import numpy as np
import math
from collections import Counter
from typing import List, Optional, Union


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert y_true.shape == y_pred.shape, f'Shape mismatch: {y_true.shape} vs {y_pred.shape}'
    return y_true, y_pred


def _edit_distance(a: list, b: list) -> int:
    """Dynamic programming edit distance."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[j] = prev[j-1]
            else:
                dp[j] = 1 + min(prev[j], dp[j-1], prev[j-1])
    return dp[n]


# ── Classification ────────────────────────────────────────────────────────────

def accuracy(y_true, y_pred) -> float:
    """Fraction of correct predictions."""
    y_true, y_pred = _check(y_true, y_pred)
    return float((y_true == y_pred).mean())


def top_k_accuracy(y_true, y_pred_proba, k: int = 5) -> float:
    """Fraction where true label is in top-k predicted classes."""
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    top_k = np.argsort(y_pred_proba, axis=1)[:, -k:]
    correct = sum(y_true[i] in top_k[i] for i in range(len(y_true)))
    return correct / len(y_true)


def confusion_matrix(y_true, y_pred) -> np.ndarray:
    """Return (n_classes, n_classes) confusion matrix."""
    y_true, y_pred = _check(y_true, y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n = len(classes)
    idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx[t]][idx[p]] += 1
    return cm


def precision(y_true, y_pred, average: str = 'macro') -> float:
    y_true, y_pred = _check(y_true, y_pred)
    classes = np.unique(y_true)
    scores = []
    for c in classes:
        tp = ((y_pred == c) & (y_true == c)).sum()
        fp = ((y_pred == c) & (y_true != c)).sum()
        scores.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
    if average == 'macro':
        return float(np.mean(scores))
    if average == 'micro':
        tp_all = sum((y_pred == c) & (y_true == c) for c in classes).sum()
        fp_all = sum((y_pred == c) & (y_true != c) for c in classes).sum()
        return float(tp_all / (tp_all + fp_all)) if (tp_all + fp_all) > 0 else 0.0
    return float(np.mean(scores))


def recall(y_true, y_pred, average: str = 'macro') -> float:
    y_true, y_pred = _check(y_true, y_pred)
    classes = np.unique(y_true)
    scores = []
    for c in classes:
        tp = ((y_pred == c) & (y_true == c)).sum()
        fn = ((y_pred != c) & (y_true == c)).sum()
        scores.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    if average == 'macro':
        return float(np.mean(scores))
    if average == 'micro':
        tp_all = sum(((y_pred == c) & (y_true == c)).sum() for c in classes)
        fn_all = sum(((y_pred != c) & (y_true == c)).sum() for c in classes)
        return float(tp_all / (tp_all + fn_all)) if (tp_all + fn_all) > 0 else 0.0
    return float(np.mean(scores))


def f1_score(y_true, y_pred, average: str = 'macro') -> float:
    p = precision(y_true, y_pred, average)
    r = recall(y_true, y_pred, average)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def classification_report(y_true, y_pred, target_names: Optional[List[str]] = None) -> str:
    y_true, y_pred = _check(y_true, y_pred)
    classes = np.unique(y_true)
    if target_names is None:
        target_names = [str(c) for c in classes]
    lines = [f"{'':>20} {'precision':>9} {'recall':>9} {'f1-score':>9} {'support':>9}\n"]
    for c, name in zip(classes, target_names):
        tp = ((y_pred == c) & (y_true == c)).sum()
        fp = ((y_pred == c) & (y_true != c)).sum()
        fn = ((y_pred != c) & (y_true == c)).sum()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        sup = (y_true == c).sum()
        lines.append(f'{name:>20} {p:9.2f} {r:9.2f} {f:9.2f} {sup:9d}')
    lines.append(f'\n{"accuracy":>20} {"":9} {"":9} {accuracy(y_true, y_pred):9.2f} {len(y_true):9d}')
    return '\n'.join(lines)


def roc_auc_score(y_true, y_score, multi_class: str = 'ovr') -> float:
    """Area under the ROC curve (binary or multiclass OvR)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_score.ndim == 1:
        # Binary
        return _binary_roc_auc(y_true, y_score)
    classes = np.unique(y_true)
    aucs = []
    for i, c in enumerate(classes):
        binary = (y_true == c).astype(int)
        aucs.append(_binary_roc_auc(binary, y_score[:, i]))
    return float(np.mean(aucs))


def _binary_roc_auc(y_true, y_score) -> float:
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]
    tp = np.cumsum(y_true_sorted)
    fp = np.cumsum(1 - y_true_sorted)
    tp_rate = tp / (tp[-1] + 1e-10)
    fp_rate = fp / (fp[-1] + 1e-10)
    # Trapezoidal AUC
    return float(np.trapz(tp_rate, fp_rate))


def average_precision_score(y_true, y_score) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    order = np.argsort(-y_score)
    y_true_s = y_true[order]
    precisions, recalls = [], []
    tp = fp = 0
    for yt in y_true_s:
        if yt == 1:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / (y_true.sum() + 1e-10))
    return float(np.trapz(precisions, recalls))


def matthews_corrcoef(y_true, y_pred) -> float:
    y_true, y_pred = _check(y_true, y_pred)
    classes = np.unique(y_true)
    if len(classes) == 2:
        c = classes
        tp = ((y_pred == c[1]) & (y_true == c[1])).sum()
        tn = ((y_pred == c[0]) & (y_true == c[0])).sum()
        fp = ((y_pred == c[1]) & (y_true == c[0])).sum()
        fn = ((y_pred == c[0]) & (y_true == c[1])).sum()
        denom = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        return float((tp*tn - fp*fn) / denom) if denom else 0.0
    cm = confusion_matrix(y_true, y_pred).astype(float)
    n = cm.sum()
    s = cm.sum(axis=1)
    p = cm.sum(axis=0)
    cov_xx = n**2 - (s**2).sum()
    cov_yy = n**2 - (p**2).sum()
    cov_xy = n * np.trace(cm) - (s * p).sum()
    denom = math.sqrt(cov_xx * cov_yy)
    return float(cov_xy / denom) if denom else 0.0


def cohen_kappa_score(y_true, y_pred) -> float:
    y_true, y_pred = _check(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred).astype(float)
    n = cm.sum()
    po = np.trace(cm) / n
    pe = (cm.sum(axis=0) * cm.sum(axis=1)).sum() / n**2
    return float((po - pe) / (1 - pe)) if (1 - pe) != 0 else 1.0


# ── Regression ────────────────────────────────────────────────────────────────

def mse(y_true, y_pred) -> float:
    y_true, y_pred = _check(y_true, y_pred)
    return float(((y_true - y_pred) ** 2).mean())


def rmse(y_true, y_pred) -> float:
    return math.sqrt(mse(y_true, y_pred))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = _check(y_true, y_pred)
    return float(np.abs(y_true - y_pred).mean())


def r2_score(y_true, y_pred) -> float:
    y_true, y_pred = _check(y_true, y_pred)
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0


def explained_variance_score(y_true, y_pred) -> float:
    y_true, y_pred = _check(y_true, y_pred)
    var_res = np.var(y_true - y_pred)
    var_true = np.var(y_true)
    return float(1 - var_res / var_true) if var_true != 0 else 0.0


def huber_loss(y_true, y_pred, delta: float = 1.0) -> float:
    y_true, y_pred = _check(y_true, y_pred)
    err = np.abs(y_true - y_pred)
    loss = np.where(err <= delta, 0.5 * err**2, delta * (err - 0.5 * delta))
    return float(loss.mean())


# ── NLP / Generation ─────────────────────────────────────────────────────────

def perplexity(logits: np.ndarray, targets: np.ndarray) -> float:
    """Perplexity = exp(cross-entropy loss).

    Args:
        logits: (N, vocab_size)
        targets: (N,) int token ids
    """
    logits = np.asarray(logits)
    targets = np.asarray(targets, dtype=int)
    # log-softmax
    log_probs = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True) + 1e-10)
    nll = -log_probs[np.arange(len(targets)), targets].mean()
    return float(math.exp(min(nll, 500)))


def bleu_score(references: List[List[str]], hypotheses: List[str], max_n: int = 4) -> float:
    """Corpus BLEU score — Papineni et al. 2002.

    Args:
        references: list of reference token lists (one per hypothesis)
        hypotheses: list of hypothesis strings (space-tokenized)
    Returns:
        BLEU score in [0, 1]
    """
    if isinstance(hypotheses[0], str):
        hyp_tokens = [h.split() for h in hypotheses]
    else:
        hyp_tokens = hypotheses
    if isinstance(references[0], str):
        ref_tokens = [r.split() for r in references]
    else:
        ref_tokens = references

    clip_counts = [0] * max_n
    total_counts = [0] * max_n
    hyp_len = 0
    ref_len = 0

    for hyp, ref in zip(hyp_tokens, ref_tokens):
        hyp_len += len(hyp)
        ref_len += len(ref)
        ref_ngram_counts = {}
        for n in range(1, max_n + 1):
            ref_ngrams = Counter(tuple(ref[i:i+n]) for i in range(len(ref)-n+1))
            hyp_ngrams = Counter(tuple(hyp[i:i+n]) for i in range(len(hyp)-n+1))
            clipped = {ng: min(cnt, ref_ngrams.get(ng, 0)) for ng, cnt in hyp_ngrams.items()}
            clip_counts[n-1] += sum(clipped.values())
            total_counts[n-1] += sum(hyp_ngrams.values())

    # Geometric mean of precisions
    log_avg = 0.0
    for n in range(max_n):
        if clip_counts[n] == 0:
            return 0.0
        log_avg += math.log(clip_counts[n] / total_counts[n]) if total_counts[n] > 0 else -float('inf')
    log_avg /= max_n

    # Brevity penalty
    bp = 1.0 if hyp_len >= ref_len else math.exp(1 - ref_len / hyp_len)
    return float(bp * math.exp(log_avg))


def rouge_score(reference: str, hypothesis: str, rouge_type: str = 'rouge-l') -> dict:
    """ROUGE score — Lin 2004.

    Args:
        rouge_type: 'rouge-1', 'rouge-2', 'rouge-l'
    Returns:
        dict with 'precision', 'recall', 'f1'
    """
    ref_toks = reference.split()
    hyp_toks = hypothesis.split()

    if rouge_type == 'rouge-l':
        # LCS-based
        lcs = _lcs_length(ref_toks, hyp_toks)
        p = lcs / len(hyp_toks) if hyp_toks else 0.0
        r = lcs / len(ref_toks) if ref_toks else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return {'precision': p, 'recall': r, 'f1': f}

    n = int(rouge_type.split('-')[1])
    ref_ng = Counter(tuple(ref_toks[i:i+n]) for i in range(len(ref_toks)-n+1))
    hyp_ng = Counter(tuple(hyp_toks[i:i+n]) for i in range(len(hyp_toks)-n+1))
    overlap = sum(min(cnt, hyp_ng.get(ng, 0)) for ng, cnt in ref_ng.items())
    p = overlap / sum(hyp_ng.values()) if hyp_ng else 0.0
    r = overlap / sum(ref_ng.values()) if ref_ng else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {'precision': p, 'recall': r, 'f1': f}


def _lcs_length(a: list, b: list) -> int:
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    for i in range(m):
        curr = [0] * (n + 1)
        for j in range(n):
            curr[j+1] = prev[j] + 1 if a[i] == b[j] else max(prev[j+1], curr[j])
        prev = curr
    return prev[n]


def meteor_score(reference: str, hypothesis: str) -> float:
    """METEOR score (simplified) — Banerjee & Lavie 2005."""
    ref_toks = reference.lower().split()
    hyp_toks = hypothesis.lower().split()
    ref_set = set(ref_toks)
    matches = sum(1 for t in hyp_toks if t in ref_set)
    if matches == 0:
        return 0.0
    p = matches / len(hyp_toks)
    r = matches / len(ref_toks)
    f = 10 * p * r / (9 * p + r) if (9 * p + r) > 0 else 0.0
    # Penalty for fragmentation (chunks)
    chunks = 1
    prev_match = False
    for t in hyp_toks:
        if t in ref_set:
            if not prev_match:
                chunks += 1
            prev_match = True
        else:
            prev_match = False
    penalty = 0.5 * (chunks / matches) ** 3 if matches > 0 else 0
    return float(f * (1 - penalty))


def character_error_rate(reference: str, hypothesis: str) -> float:
    """CER = edit_distance(chars) / len(reference_chars)."""
    ref = list(reference)
    hyp = list(hypothesis)
    if not ref:
        return float(len(hyp))
    return _edit_distance(ref, hyp) / len(ref)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """WER = (S + D + I) / N using word-level edit distance."""
    ref = reference.split()
    hyp = hypothesis.split()
    if not ref:
        return float(len(hyp))
    return _edit_distance(ref, hyp) / len(ref)


# ── Ranking ───────────────────────────────────────────────────────────────────

def ndcg_score(y_true, y_score, k: Optional[int] = None) -> float:
    """Normalised Discounted Cumulative Gain."""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    order = np.argsort(-y_score)
    if k:
        order = order[:k]
    gains = y_true[order]
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = (gains * discounts).sum()
    ideal = np.sort(y_true)[::-1][:len(gains)]
    idcg = (ideal * discounts[:len(ideal)]).sum()
    return float(dcg / idcg) if idcg > 0 else 0.0


def mean_reciprocal_rank(y_true, y_score) -> float:
    """MRR — mean of 1/rank of first relevant item."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.ndim == 1:
        y_true = y_true[None, :]
        y_score = y_score[None, :]
    rrs = []
    for yt, ys in zip(y_true, y_score):
        order = np.argsort(-ys)
        for rank, idx in enumerate(order, 1):
            if yt[idx] > 0:
                rrs.append(1.0 / rank)
                break
        else:
            rrs.append(0.0)
    return float(np.mean(rrs))


# ── Clustering ────────────────────────────────────────────────────────────────

def silhouette_score(X, labels) -> float:
    """Mean silhouette coefficient."""
    X = np.asarray(X)
    labels = np.asarray(labels)
    unique = np.unique(labels)
    if len(unique) < 2:
        return 0.0
    scores = []
    for i in range(len(X)):
        same = X[labels == labels[i]]
        if len(same) > 1:
            a = np.linalg.norm(X[i] - same, axis=1).mean()
        else:
            a = 0.0
        b_vals = []
        for c in unique:
            if c == labels[i]:
                continue
            other = X[labels == c]
            b_vals.append(np.linalg.norm(X[i] - other, axis=1).mean())
        b = min(b_vals) if b_vals else 0.0
        s = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
        scores.append(s)
    return float(np.mean(scores))


def adjusted_rand_score(y_true, y_pred) -> float:
    """Adjusted Rand Index — corrected-for-chance measure of cluster similarity."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    classes_true = np.unique(y_true)
    classes_pred = np.unique(y_pred)
    contingency = np.zeros((len(classes_true), len(classes_pred)), dtype=int)
    t_idx = {c: i for i, c in enumerate(classes_true)}
    p_idx = {c: i for i, c in enumerate(classes_pred)}
    for t, p in zip(y_true, y_pred):
        contingency[t_idx[t]][p_idx[p]] += 1
    n = len(y_true)
    a = sum(math.comb(int(c), 2) for c in contingency.sum(axis=1))
    b = sum(math.comb(int(c), 2) for c in contingency.sum(axis=0))
    ab = math.comb(n, 2)
    c = sum(math.comb(int(v), 2) for row in contingency for v in row)
    expected = a * b / ab if ab > 0 else 0
    max_ab = (a + b) / 2
    denom = max_ab - expected
    return float((c - expected) / denom) if denom != 0 else 1.0


def normalized_mutual_info_score(y_true, y_pred) -> float:
    """NMI between two cluster assignments."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    classes_t = np.unique(y_true)
    classes_p = np.unique(y_pred)
    mi = 0.0
    for ct in classes_t:
        for cp in classes_p:
            joint = ((y_true == ct) & (y_pred == cp)).sum() / n
            pt = (y_true == ct).sum() / n
            pp = (y_pred == cp).sum() / n
            if joint > 0 and pt > 0 and pp > 0:
                mi += joint * math.log(joint / (pt * pp))
    ht = -sum((y_true == c).sum() / n * math.log((y_true == c).sum() / n + 1e-10)
              for c in classes_t)
    hp = -sum((y_pred == c).sum() / n * math.log((y_pred == c).sum() / n + 1e-10)
              for c in classes_p)
    return float(2 * mi / (ht + hp)) if (ht + hp) > 0 else 0.0


# ── Calibration ───────────────────────────────────────────────────────────────

def expected_calibration_error(y_true, y_proba, n_bins: int = 10) -> float:
    """ECE — mean |accuracy - confidence| weighted by bin size."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    if y_proba.ndim == 2:
        confidence = y_proba.max(axis=1)
        predicted = y_proba.argmax(axis=1)
        correct = (predicted == y_true).astype(float)
    else:
        confidence = y_proba
        predicted = (y_proba >= 0.5).astype(int)
        correct = (predicted == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (confidence > bins[i]) & (confidence <= bins[i+1])
        if mask.sum() == 0:
            continue
        acc = correct[mask].mean()
        conf = confidence[mask].mean()
        ece += mask.sum() / n * abs(acc - conf)
    return float(ece)


def reliability_diagram(y_true, y_proba, n_bins: int = 10) -> dict:
    """Data for plotting reliability diagram."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    if y_proba.ndim == 2:
        confidence = y_proba.max(axis=1)
        predicted = y_proba.argmax(axis=1)
        correct = (predicted == y_true).astype(float)
    else:
        confidence = y_proba
        correct = (((y_proba >= 0.5).astype(int)) == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs, bin_counts = [], [], []
    for i in range(n_bins):
        mask = (confidence > bins[i]) & (confidence <= bins[i+1])
        if mask.sum() == 0:
            bin_accs.append(0.0); bin_confs.append(bins[i] + 0.5/n_bins); bin_counts.append(0)
        else:
            bin_accs.append(float(correct[mask].mean()))
            bin_confs.append(float(confidence[mask].mean()))
            bin_counts.append(int(mask.sum()))
    return {'bin_accuracies': bin_accs, 'bin_confidences': bin_confs,
            'bin_counts': bin_counts, 'ece': expected_calibration_error(y_true, y_proba, n_bins)}
