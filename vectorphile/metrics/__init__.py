"""vectorphile.metrics — Evaluation metrics for ML/DL/NLP tasks.

Classification, regression, NLP generation, ranking, clustering, calibration.
All implemented from scratch — no external dependencies.
"""
from __future__ import annotations
import math
from collections import Counter
from typing import List, Optional, Union


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_flat(v) -> list:
    if hasattr(v, 'to_flat_list'):
        return v.to_flat_list()
    if hasattr(v, 'tolist'):
        data = v.tolist()
    else:
        data = list(v)
    flat = []
    for x in data:
        if isinstance(x, (list, tuple)):
            flat.extend(x)
        else:
            flat.append(x)
    return flat


def _to_list2d(v) -> List[list]:
    if hasattr(v, 'to_flat_list'):
        flat = v.to_flat_list()
        if hasattr(v, 'shape') and len(v.shape) == 2:
            n, d = v.shape
            return [[flat[i * d + j] for j in range(d)] for i in range(n)]
        return [[x] for x in flat]
    if hasattr(v, 'tolist'):
        data = v.tolist()
    else:
        data = list(v)
    if data and not isinstance(data[0], (list, tuple)):
        return [[x] for x in data]
    return [list(row) for row in data]


def _check(y_true, y_pred):
    yt = _to_flat(y_true)
    yp = _to_flat(y_pred)
    assert len(yt) == len(yp), f'Length mismatch: {len(yt)} vs {len(yp)}'
    return yt, yp


def _unique_sorted(lst: list) -> list:
    return sorted(set(lst))


def _edit_distance(a: list, b: list) -> int:
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
    yt, yp = _check(y_true, y_pred)
    return sum(1 for a, b in zip(yt, yp) if a == b) / max(len(yt), 1)


def top_k_accuracy(y_true, y_pred_proba, k: int = 5) -> float:
    yt = _to_flat(y_true)
    yp2d = _to_list2d(y_pred_proba)
    correct = 0
    for i, row in enumerate(yp2d):
        top_k = sorted(range(len(row)), key=lambda j: -row[j])[:k]
        if int(yt[i]) in top_k:
            correct += 1
    return correct / max(len(yt), 1)


def confusion_matrix(y_true, y_pred) -> List[List[int]]:
    yt, yp = _check(y_true, y_pred)
    classes = _unique_sorted(yt + yp)
    n = len(classes)
    idx = {c: i for i, c in enumerate(classes)}
    cm = [[0] * n for _ in range(n)]
    for t, p in zip(yt, yp):
        cm[idx[t]][idx[p]] += 1
    return cm


def precision(y_true, y_pred, average: str = 'macro') -> float:
    yt, yp = _check(y_true, y_pred)
    classes = _unique_sorted(yt)
    scores = []
    tp_all = fp_all = 0
    for c in classes:
        tp = sum(1 for a, b in zip(yt, yp) if b == c and a == c)
        fp = sum(1 for a, b in zip(yt, yp) if b == c and a != c)
        scores.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        tp_all += tp
        fp_all += fp
    if average == 'micro':
        return tp_all / (tp_all + fp_all) if (tp_all + fp_all) > 0 else 0.0
    return sum(scores) / max(len(scores), 1)


def recall(y_true, y_pred, average: str = 'macro') -> float:
    yt, yp = _check(y_true, y_pred)
    classes = _unique_sorted(yt)
    scores = []
    tp_all = fn_all = 0
    for c in classes:
        tp = sum(1 for a, b in zip(yt, yp) if b == c and a == c)
        fn = sum(1 for a, b in zip(yt, yp) if b != c and a == c)
        scores.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        tp_all += tp
        fn_all += fn
    if average == 'micro':
        return tp_all / (tp_all + fn_all) if (tp_all + fn_all) > 0 else 0.0
    return sum(scores) / max(len(scores), 1)


def f1_score(y_true, y_pred, average: str = 'macro') -> float:
    p = precision(y_true, y_pred, average)
    r = recall(y_true, y_pred, average)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def classification_report(y_true, y_pred, target_names: Optional[List[str]] = None) -> str:
    yt, yp = _check(y_true, y_pred)
    classes = _unique_sorted(yt)
    if target_names is None:
        target_names = [str(c) for c in classes]
    lines = [f"{'':>20} {'precision':>9} {'recall':>9} {'f1-score':>9} {'support':>9}\n"]
    for c, name in zip(classes, target_names):
        tp = sum(1 for a, b in zip(yt, yp) if b == c and a == c)
        fp = sum(1 for a, b in zip(yt, yp) if b == c and a != c)
        fn = sum(1 for a, b in zip(yt, yp) if b != c and a == c)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        sup = sum(1 for a in yt if a == c)
        lines.append(f'{name:>20} {p:9.2f} {r:9.2f} {f:9.2f} {sup:9d}')
    acc = accuracy(yt, yp)
    lines.append(f'\n{"accuracy":>20} {"":9} {"":9} {acc:9.2f} {len(yt):9d}')
    return '\n'.join(lines)


def _binary_roc_auc(y_true: list, y_score: list) -> float:
    pairs = sorted(zip(y_score, y_true), key=lambda x: -x[0])
    tp = fp = 0
    total_pos = sum(y_true)
    total_neg = len(y_true) - total_pos
    if total_pos == 0 or total_neg == 0:
        return 0.5
    tpr_pts = [0.0]
    fpr_pts = [0.0]
    for _, yt in pairs:
        if yt:
            tp += 1
        else:
            fp += 1
        tpr_pts.append(tp / total_pos)
        fpr_pts.append(fp / total_neg)
    # trapezoidal AUC
    auc = 0.0
    for i in range(1, len(tpr_pts)):
        auc += (fpr_pts[i] - fpr_pts[i-1]) * (tpr_pts[i] + tpr_pts[i-1]) / 2.0
    return abs(auc)


def roc_auc_score(y_true, y_score, multi_class: str = 'ovr') -> float:
    yt = _to_flat(y_true)
    # Check if y_score is 2D
    score_data = y_score
    if hasattr(score_data, 'tolist'):
        score_data = score_data.tolist()
    elif hasattr(score_data, 'to_flat_list'):
        pass
    if isinstance(score_data, list) and score_data and isinstance(score_data[0], (list, tuple)):
        yp2d = [list(row) for row in score_data]
        classes = _unique_sorted(yt)
        aucs = []
        for i, c in enumerate(classes):
            binary = [1 if v == c else 0 for v in yt]
            col = [row[i] for row in yp2d]
            aucs.append(_binary_roc_auc(binary, col))
        return sum(aucs) / max(len(aucs), 1)
    else:
        ys = _to_flat(y_score)
        return _binary_roc_auc(yt, ys)


def average_precision_score(y_true, y_score) -> float:
    yt = _to_flat(y_true)
    ys = _to_flat(y_score)
    pairs = sorted(zip(ys, yt), key=lambda x: -x[0])
    total_pos = sum(yt)
    tp = fp = 0
    precisions, recalls = [], []
    for score, label in pairs:
        if label:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / max(total_pos, 1))
    # integrate over recall
    ap = 0.0
    for i in range(1, len(precisions)):
        ap += (recalls[i] - recalls[i-1]) * precisions[i]
    return abs(ap)


def matthews_corrcoef(y_true, y_pred) -> float:
    yt, yp = _check(y_true, y_pred)
    classes = _unique_sorted(yt)
    if len(classes) == 2:
        c0, c1 = classes[0], classes[1]
        tp = sum(1 for a, b in zip(yt, yp) if b == c1 and a == c1)
        tn = sum(1 for a, b in zip(yt, yp) if b == c0 and a == c0)
        fp = sum(1 for a, b in zip(yt, yp) if b == c1 and a == c0)
        fn = sum(1 for a, b in zip(yt, yp) if b == c0 and a == c1)
        denom = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        return float((tp*tn - fp*fn) / denom) if denom else 0.0
    cm = confusion_matrix(yt, yp)
    n = sum(sum(row) for row in cm)
    K = len(cm)
    row_sums = [sum(cm[i]) for i in range(K)]
    col_sums = [sum(cm[i][j] for i in range(K)) for j in range(K)]
    trace = sum(cm[i][i] for i in range(K))
    cov_xx = n*n - sum(s*s for s in row_sums)
    cov_yy = n*n - sum(s*s for s in col_sums)
    cov_xy = n * trace - sum(row_sums[i] * col_sums[i] for i in range(K))
    denom = math.sqrt(cov_xx * cov_yy)
    return float(cov_xy / denom) if denom else 0.0


def cohen_kappa_score(y_true, y_pred) -> float:
    yt, yp = _check(y_true, y_pred)
    cm = confusion_matrix(yt, yp)
    n = sum(sum(row) for row in cm)
    K = len(cm)
    trace = sum(cm[i][i] for i in range(K))
    po = trace / n
    row_sums = [sum(cm[i]) for i in range(K)]
    col_sums = [sum(cm[i][j] for i in range(K)) for j in range(K)]
    pe = sum(row_sums[i] * col_sums[i] for i in range(K)) / (n * n)
    return float((po - pe) / (1 - pe)) if (1 - pe) != 0 else 1.0


# ── Regression ────────────────────────────────────────────────────────────────

def mse(y_true, y_pred) -> float:
    yt, yp = _check(y_true, y_pred)
    yf = [float(v) for v in yt]
    pf = [float(v) for v in yp]
    return sum((a - b) ** 2 for a, b in zip(yf, pf)) / max(len(yf), 1)


def rmse(y_true, y_pred) -> float:
    return math.sqrt(mse(y_true, y_pred))


def mae(y_true, y_pred) -> float:
    yt, yp = _check(y_true, y_pred)
    yf = [float(v) for v in yt]
    pf = [float(v) for v in yp]
    return sum(abs(a - b) for a, b in zip(yf, pf)) / max(len(yf), 1)


def r2_score(y_true, y_pred) -> float:
    yt, yp = _check(y_true, y_pred)
    yf = [float(v) for v in yt]
    pf = [float(v) for v in yp]
    mean_y = sum(yf) / max(len(yf), 1)
    ss_res = sum((a - b) ** 2 for a, b in zip(yf, pf))
    ss_tot = sum((v - mean_y) ** 2 for v in yf)
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0


def explained_variance_score(y_true, y_pred) -> float:
    yt, yp = _check(y_true, y_pred)
    yf = [float(v) for v in yt]
    pf = [float(v) for v in yp]
    residuals = [a - b for a, b in zip(yf, pf)]
    mean_res = sum(residuals) / max(len(residuals), 1)
    var_res = sum((v - mean_res) ** 2 for v in residuals) / max(len(residuals), 1)
    mean_y = sum(yf) / max(len(yf), 1)
    var_true = sum((v - mean_y) ** 2 for v in yf) / max(len(yf), 1)
    return float(1 - var_res / var_true) if var_true != 0 else 0.0


def huber_loss(y_true, y_pred, delta: float = 1.0) -> float:
    yt, yp = _check(y_true, y_pred)
    yf = [float(v) for v in yt]
    pf = [float(v) for v in yp]
    total = 0.0
    for a, b in zip(yf, pf):
        err = abs(a - b)
        total += 0.5 * err**2 if err <= delta else delta * (err - 0.5 * delta)
    return total / max(len(yf), 1)


# ── NLP / Generation ─────────────────────────────────────────────────────────

def perplexity(logits, targets) -> float:
    """Perplexity = exp(cross-entropy loss)."""
    logits2d = _to_list2d(logits)
    tgts = [int(v) for v in _to_flat(targets)]
    total_nll = 0.0
    for i, row in enumerate(logits2d):
        max_v = max(row)
        exp_row = [math.exp(v - max_v) for v in row]
        sum_exp = sum(exp_row)
        log_prob = math.log(exp_row[tgts[i]] / (sum_exp + 1e-10) + 1e-10)
        total_nll -= log_prob
    nll = total_nll / max(len(tgts), 1)
    return float(math.exp(min(nll, 500)))


def bleu_score(references: List, hypotheses: List, max_n: int = 4) -> float:
    """Corpus BLEU score — Papineni et al. 2002."""
    if isinstance(hypotheses[0], str):
        hyp_tokens = [h.split() for h in hypotheses]
    else:
        hyp_tokens = list(hypotheses)
    if isinstance(references[0], str):
        ref_tokens = [r.split() for r in references]
    else:
        ref_tokens = list(references)

    clip_counts = [0] * max_n
    total_counts = [0] * max_n
    hyp_len = 0
    ref_len = 0

    for hyp, ref in zip(hyp_tokens, ref_tokens):
        hyp_len += len(hyp)
        ref_len += len(ref)
        for n in range(1, max_n + 1):
            ref_ngrams = Counter(tuple(ref[i:i+n]) for i in range(len(ref)-n+1))
            hyp_ngrams = Counter(tuple(hyp[i:i+n]) for i in range(len(hyp)-n+1))
            clipped = {ng: min(cnt, ref_ngrams.get(ng, 0)) for ng, cnt in hyp_ngrams.items()}
            clip_counts[n-1] += sum(clipped.values())
            total_counts[n-1] += sum(hyp_ngrams.values())

    log_avg = 0.0
    for n in range(max_n):
        if clip_counts[n] == 0:
            return 0.0
        log_avg += math.log(clip_counts[n] / total_counts[n]) if total_counts[n] > 0 else -1e18
    log_avg /= max_n

    bp = 1.0 if hyp_len >= ref_len else math.exp(1 - ref_len / hyp_len)
    return float(bp * math.exp(log_avg))


def rouge_score(reference: str, hypothesis: str, rouge_type: str = 'rouge-l') -> dict:
    """ROUGE score — Lin 2004."""
    ref_toks = reference.split()
    hyp_toks = hypothesis.split()

    if rouge_type == 'rouge-l':
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
    ref = list(reference)
    hyp = list(hypothesis)
    if not ref:
        return float(len(hyp))
    return _edit_distance(ref, hyp) / len(ref)


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref = reference.split()
    hyp = hypothesis.split()
    if not ref:
        return float(len(hyp))
    return _edit_distance(ref, hyp) / len(ref)


# ── Ranking ───────────────────────────────────────────────────────────────────

def ndcg_score(y_true, y_score, k: Optional[int] = None) -> float:
    yt = [float(v) for v in _to_flat(y_true)]
    ys = [float(v) for v in _to_flat(y_score)]
    order = sorted(range(len(ys)), key=lambda i: -ys[i])
    if k:
        order = order[:k]
    gains = [yt[i] for i in order]
    discounts = [1.0 / math.log2(i + 2) for i in range(len(gains))]
    dcg = sum(g * d for g, d in zip(gains, discounts))
    ideal = sorted(yt, reverse=True)[:len(gains)]
    idcg = sum(g * d for g, d in zip(ideal, discounts[:len(ideal)]))
    return float(dcg / idcg) if idcg > 0 else 0.0


def mean_reciprocal_rank(y_true, y_score) -> float:
    # Try to handle 2D case
    score_data = y_score
    if hasattr(score_data, 'tolist'):
        score_data = score_data.tolist()
    true_data = y_true
    if hasattr(true_data, 'tolist'):
        true_data = true_data.tolist()

    if isinstance(score_data, list) and score_data and isinstance(score_data[0], (list, tuple)):
        rows_s = [list(r) for r in score_data]
        rows_t = [list(r) if isinstance(r, (list, tuple)) else r for r in true_data]
    else:
        rows_s = [_to_flat(y_score)]
        rows_t = [_to_flat(y_true)]

    rrs = []
    for yt_row, ys_row in zip(rows_t, rows_s):
        if not isinstance(yt_row, list):
            yt_row = [yt_row]
        order = sorted(range(len(ys_row)), key=lambda i: -ys_row[i])
        for rank, idx in enumerate(order, 1):
            if float(yt_row[idx]) > 0:
                rrs.append(1.0 / rank)
                break
        else:
            rrs.append(0.0)
    return sum(rrs) / max(len(rrs), 1)


# ── Clustering ────────────────────────────────────────────────────────────────

def _euclid(a: list, b: list) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def silhouette_score(X, labels) -> float:
    X2d = _to_list2d(X)
    labs = _to_flat(labels)
    unique = _unique_sorted(labs)
    if len(unique) < 2:
        return 0.0
    scores = []
    for i in range(len(X2d)):
        same = [X2d[k] for k in range(len(X2d)) if labs[k] == labs[i] and k != i]
        a = sum(_euclid(X2d[i], x) for x in same) / max(len(same), 1) if same else 0.0
        b_vals = []
        for c in unique:
            if c == labs[i]:
                continue
            other = [X2d[k] for k in range(len(X2d)) if labs[k] == c]
            if other:
                b_vals.append(sum(_euclid(X2d[i], x) for x in other) / len(other))
        b = min(b_vals) if b_vals else 0.0
        s = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
        scores.append(s)
    return sum(scores) / max(len(scores), 1)


def adjusted_rand_score(y_true, y_pred) -> float:
    yt = _to_flat(y_true)
    yp = _to_flat(y_pred)
    classes_t = _unique_sorted(yt)
    classes_p = _unique_sorted(yp)
    t_idx = {c: i for i, c in enumerate(classes_t)}
    p_idx = {c: i for i, c in enumerate(classes_p)}
    n = len(yt)
    contingency = [[0] * len(classes_p) for _ in range(len(classes_t))]
    for t, p in zip(yt, yp):
        contingency[t_idx[t]][p_idx[p]] += 1
    row_sums = [sum(contingency[i]) for i in range(len(classes_t))]
    col_sums = [sum(contingency[i][j] for i in range(len(classes_t))) for j in range(len(classes_p))]
    a = sum(math.comb(int(c), 2) for c in row_sums)
    b = sum(math.comb(int(c), 2) for c in col_sums)
    ab = math.comb(n, 2)
    c = sum(math.comb(int(contingency[i][j]), 2)
            for i in range(len(classes_t)) for j in range(len(classes_p)))
    expected = a * b / ab if ab > 0 else 0
    max_ab = (a + b) / 2
    denom = max_ab - expected
    return float((c - expected) / denom) if denom != 0 else 1.0


def normalized_mutual_info_score(y_true, y_pred) -> float:
    yt = _to_flat(y_true)
    yp = _to_flat(y_pred)
    n = len(yt)
    classes_t = _unique_sorted(yt)
    classes_p = _unique_sorted(yp)
    mi = 0.0
    for ct in classes_t:
        for cp in classes_p:
            joint = sum(1 for a, b in zip(yt, yp) if a == ct and b == cp) / n
            pt = sum(1 for a in yt if a == ct) / n
            pp = sum(1 for b in yp if b == cp) / n
            if joint > 0 and pt > 0 and pp > 0:
                mi += joint * math.log(joint / (pt * pp))
    ht = -sum((sum(1 for a in yt if a == c) / n) * math.log(sum(1 for a in yt if a == c) / n + 1e-10)
              for c in classes_t)
    hp = -sum((sum(1 for b in yp if b == c) / n) * math.log(sum(1 for b in yp if b == c) / n + 1e-10)
              for c in classes_p)
    return float(2 * mi / (ht + hp)) if (ht + hp) > 0 else 0.0


# ── Calibration ───────────────────────────────────────────────────────────────

def expected_calibration_error(y_true, y_proba, n_bins: int = 10) -> float:
    yt = _to_flat(y_true)
    # Check if 2D proba
    proba_data = y_proba
    if hasattr(proba_data, 'tolist'):
        proba_data = proba_data.tolist()
    if isinstance(proba_data, list) and proba_data and isinstance(proba_data[0], (list, tuple)):
        yp2d = [list(row) for row in proba_data]
        confidence = [max(row) for row in yp2d]
        predicted = [max(range(len(row)), key=lambda j: row[j]) for row in yp2d]
        correct = [1.0 if p == int(yt[i]) else 0.0 for i, p in enumerate(predicted)]
    else:
        confidence = [float(v) for v in _to_flat(y_proba)]
        predicted = [1 if v >= 0.5 else 0 for v in confidence]
        correct = [1.0 if p == int(yt[i]) else 0.0 for i, p in enumerate(predicted)]
    n = len(yt)
    bin_size = 1.0 / n_bins
    ece = 0.0
    for b in range(n_bins):
        lo, hi = b * bin_size, (b + 1) * bin_size
        mask = [i for i, c in enumerate(confidence) if lo < c <= hi]
        if not mask:
            continue
        acc = sum(correct[i] for i in mask) / len(mask)
        conf = sum(confidence[i] for i in mask) / len(mask)
        ece += len(mask) / n * abs(acc - conf)
    return float(ece)


def reliability_diagram(y_true, y_proba, n_bins: int = 10) -> dict:
    yt = _to_flat(y_true)
    proba_data = y_proba
    if hasattr(proba_data, 'tolist'):
        proba_data = proba_data.tolist()
    if isinstance(proba_data, list) and proba_data and isinstance(proba_data[0], (list, tuple)):
        yp2d = [list(row) for row in proba_data]
        confidence = [max(row) for row in yp2d]
        predicted = [max(range(len(row)), key=lambda j: row[j]) for row in yp2d]
        correct = [1.0 if p == int(yt[i]) else 0.0 for i, p in enumerate(predicted)]
    else:
        confidence = [float(v) for v in _to_flat(y_proba)]
        predicted = [1 if v >= 0.5 else 0 for v in confidence]
        correct = [1.0 if p == int(yt[i]) else 0.0 for i, p in enumerate(predicted)]
    n = len(yt)
    bin_size = 1.0 / n_bins
    bin_accs, bin_confs, bin_counts = [], [], []
    for b in range(n_bins):
        lo, hi = b * bin_size, (b + 1) * bin_size
        mask = [i for i, c in enumerate(confidence) if lo < c <= hi]
        if not mask:
            bin_accs.append(0.0)
            bin_confs.append(lo + bin_size / 2)
            bin_counts.append(0)
        else:
            bin_accs.append(sum(correct[i] for i in mask) / len(mask))
            bin_confs.append(sum(confidence[i] for i in mask) / len(mask))
            bin_counts.append(len(mask))
    return {'bin_accuracies': bin_accs, 'bin_confidences': bin_confs,
            'bin_counts': bin_counts, 'ece': expected_calibration_error(y_true, y_proba, n_bins)}


# ---------------------------------------------------------------------------
# Class-based stateful metrics (Keras-compatible)
# ---------------------------------------------------------------------------

class _StatefulMetric:
    def __init__(self, name: str):
        self.name = name
        self._total = 0.0
        self._count = 0

    def update_state(self, y_true, y_pred, sample_weight=None):
        raise NotImplementedError

    def result(self) -> float:
        return self._total / max(1, self._count)

    def reset_state(self):
        self._total = 0.0
        self._count = 0

    def __call__(self, y_true, y_pred):
        self.update_state(y_true, y_pred)
        return self.result()


class Accuracy(_StatefulMetric):
    def __init__(self):
        super().__init__('accuracy')

    def update_state(self, y_true, y_pred, sample_weight=None):
        yt = _to_flat(y_true)
        proba_data = y_pred
        if hasattr(proba_data, 'tolist'):
            proba_data = proba_data.tolist()
        if isinstance(proba_data, list) and proba_data and isinstance(proba_data[0], (list, tuple)):
            yp2d = [list(row) for row in proba_data]
            yp = [max(range(len(row)), key=lambda j: row[j]) for row in yp2d]
        else:
            yp = _to_flat(y_pred)
        self._total += sum(1 for a, b in zip(yt, yp) if a == b)
        self._count += len(yt)

    def result(self) -> float:
        return self._total / max(1, self._count)


class Precision(_StatefulMetric):
    def __init__(self, average='macro'):
        super().__init__('precision')
        self.average = average

    def update_state(self, y_true, y_pred, sample_weight=None):
        yt = _to_flat(y_true)
        yp_data = y_pred
        if hasattr(yp_data, 'tolist'):
            yp_data = yp_data.tolist()
        if isinstance(yp_data, list) and yp_data and isinstance(yp_data[0], (list, tuple)):
            yp2d = [list(row) for row in yp_data]
            yp = [max(range(len(row)), key=lambda j: row[j]) for row in yp2d]
        else:
            yp = _to_flat(y_pred)
        self._total += precision(yt, yp, self.average)
        self._count += 1


class Recall(_StatefulMetric):
    def __init__(self, average='macro'):
        super().__init__('recall')
        self.average = average

    def update_state(self, y_true, y_pred, sample_weight=None):
        yt = _to_flat(y_true)
        yp_data = y_pred
        if hasattr(yp_data, 'tolist'):
            yp_data = yp_data.tolist()
        if isinstance(yp_data, list) and yp_data and isinstance(yp_data[0], (list, tuple)):
            yp2d = [list(row) for row in yp_data]
            yp = [max(range(len(row)), key=lambda j: row[j]) for row in yp2d]
        else:
            yp = _to_flat(y_pred)
        self._total += recall(yt, yp, self.average)
        self._count += 1


class F1Score(_StatefulMetric):
    def __init__(self, average='macro'):
        super().__init__('f1_score')
        self.average = average

    def update_state(self, y_true, y_pred, sample_weight=None):
        yt = _to_flat(y_true)
        yp_data = y_pred
        if hasattr(yp_data, 'tolist'):
            yp_data = yp_data.tolist()
        if isinstance(yp_data, list) and yp_data and isinstance(yp_data[0], (list, tuple)):
            yp2d = [list(row) for row in yp_data]
            yp = [max(range(len(row)), key=lambda j: row[j]) for row in yp2d]
        else:
            yp = _to_flat(y_pred)
        self._total += f1_score(yt, yp, self.average)
        self._count += 1


class MeanSquaredError(_StatefulMetric):
    def __init__(self):
        super().__init__('mse')

    def update_state(self, y_true, y_pred, sample_weight=None):
        self._total += mse(y_true, y_pred)
        self._count += 1


class MeanAbsoluteError(_StatefulMetric):
    def __init__(self):
        super().__init__('mae')

    def update_state(self, y_true, y_pred, sample_weight=None):
        self._total += mae(y_true, y_pred)
        self._count += 1


class ROCAUC(_StatefulMetric):
    def __init__(self):
        super().__init__('roc_auc')

    def update_state(self, y_true, y_pred_proba, sample_weight=None):
        self._total += roc_auc_score(y_true, y_pred_proba)
        self._count += 1


__all__ = [
    'accuracy', 'top_k_accuracy', 'confusion_matrix',
    'precision', 'recall', 'f1_score', 'classification_report',
    'roc_auc_score', 'average_precision_score',
    'matthews_corrcoef', 'cohen_kappa_score',
    'mse', 'rmse', 'mae', 'r2_score', 'explained_variance_score', 'huber_loss',
    'perplexity', 'bleu_score', 'rouge_score', 'meteor_score',
    'character_error_rate', 'word_error_rate',
    'ndcg_score', 'mean_reciprocal_rank',
    'silhouette_score', 'adjusted_rand_score', 'normalized_mutual_info_score',
    'expected_calibration_error', 'reliability_diagram',
    'Accuracy', 'Precision', 'Recall', 'F1Score',
    'MeanSquaredError', 'MeanAbsoluteError', 'ROCAUC',
]
