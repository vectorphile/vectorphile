"""vectorphile.feature_extraction — Feature extraction utilities.

Provides:
Text:
- CountVectorizer: bag-of-words term frequencies
- TfidfVectorizer: TF-IDF weighted bag-of-words
- TfidfTransformer: transform count matrix to TF-IDF
- HashingVectorizer: hashing trick vectorizer

Utility:
- DictVectorizer: vectorize dictionaries of features
- FeatureHasher: hashing trick for feature dicts

No external dependencies.
"""
from __future__ import annotations
import re
import math
import hashlib
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Tokenizer helper
# ---------------------------------------------------------------------------

def _tokenize(text: str, analyzer: str = 'word',
              ngram_range: Tuple[int, int] = (1, 1),
              stop_words: Optional[List[str]] = None) -> List[str]:
    text = text.lower()
    if analyzer == 'word':
        tokens = re.findall(r'\b\w+\b', text)
    elif analyzer == 'char':
        tokens = list(text)
    else:
        tokens = re.findall(r'\b\w+\b', text)
    if stop_words:
        tokens = [t for t in tokens if t not in stop_words]
    if ngram_range == (1, 1):
        return tokens
    result = []
    for n in range(ngram_range[0], ngram_range[1] + 1):
        for i in range(len(tokens) - n + 1):
            result.append(' '.join(tokens[i:i + n]))
    return result


def _l2_normalize(rows: List[List[float]]) -> List[List[float]]:
    out = []
    for row in rows:
        norm = math.sqrt(sum(v * v for v in row))
        if norm < 1e-10:
            out.append(row[:])
        else:
            out.append([v / norm for v in row])
    return out


def _l1_normalize(rows: List[List[float]]) -> List[List[float]]:
    out = []
    for row in rows:
        norm = sum(abs(v) for v in row)
        if norm < 1e-10:
            out.append(row[:])
        else:
            out.append([v / norm for v in row])
    return out


# ---------------------------------------------------------------------------
# CountVectorizer
# ---------------------------------------------------------------------------

class CountVectorizer:
    """Convert text corpus to token count matrix."""

    def __init__(self, max_features: Optional[int] = None,
                 analyzer: str = 'word',
                 ngram_range: Tuple[int, int] = (1, 1),
                 stop_words: Optional[List[str]] = None,
                 min_df: int = 1, max_df: float = 1.0,
                 binary: bool = False, lowercase: bool = True):
        self.max_features = max_features
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.stop_words = stop_words or []
        self.min_df = min_df
        self.max_df = max_df
        self.binary = binary
        self.lowercase = lowercase
        self.vocabulary_: Optional[Dict[str, int]] = None
        self._feature_names: Optional[List[str]] = None

    def _get_tokens(self, text: str) -> List[str]:
        return _tokenize(text, self.analyzer, self.ngram_range, self.stop_words)

    def fit(self, corpus: Iterable[str]) -> 'CountVectorizer':
        corpus = list(corpus)
        n = len(corpus)
        df: Dict[str, int] = defaultdict(int)
        for doc in corpus:
            for token in set(self._get_tokens(doc)):
                df[token] += 1
        min_df = self.min_df
        max_df_abs = int(self.max_df * n) if isinstance(self.max_df, float) else self.max_df
        vocab = {t: c for t, c in df.items() if min_df <= c <= max_df_abs}
        sorted_vocab = sorted(vocab.keys(), key=lambda t: -df[t])
        if self.max_features:
            sorted_vocab = sorted_vocab[:self.max_features]
        self._feature_names = sorted_vocab
        self.vocabulary_ = {t: i for i, t in enumerate(sorted_vocab)}
        return self

    def transform(self, corpus: Iterable[str]) -> List[List[float]]:
        corpus = list(corpus)
        p = len(self.vocabulary_)
        result = []
        for doc in corpus:
            row = [0.0] * p
            for token in self._get_tokens(doc):
                j = self.vocabulary_.get(token)
                if j is not None:
                    row[j] += 1.0
            if self.binary:
                row = [1.0 if v > 0 else 0.0 for v in row]
            result.append(row)
        return result

    def fit_transform(self, corpus: Iterable[str]) -> List[List[float]]:
        corpus = list(corpus)
        return self.fit(corpus).transform(corpus)

    def get_feature_names_out(self) -> List[str]:
        return list(self._feature_names or [])

    def inverse_transform(self, X: List[List[float]]) -> List[List[str]]:
        inv = {v: k for k, v in self.vocabulary_.items()}
        return [[inv[j] for j, val in enumerate(row) if val > 0] for row in X]


# ---------------------------------------------------------------------------
# TfidfTransformer
# ---------------------------------------------------------------------------

class TfidfTransformer:
    """Transform count matrix to TF-IDF representation."""

    def __init__(self, norm: Optional[str] = 'l2', use_idf: bool = True,
                 smooth_idf: bool = True, sublinear_tf: bool = False):
        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf
        self.idf_: Optional[List[float]] = None

    def fit(self, X: List[List[float]]) -> 'TfidfTransformer':
        n = len(X)
        p = len(X[0]) if X else 0
        idf = []
        for j in range(p):
            df = sum(1 for row in X if row[j] > 0)
            if self.smooth_idf:
                idf.append(math.log((n + 1) / (df + 1)) + 1.0)
            else:
                idf.append(math.log(n / max(df, 1)) + 1.0)
        self.idf_ = idf
        return self

    def transform(self, X: List[List[float]]) -> List[List[float]]:
        p = len(self.idf_) if self.idf_ else (len(X[0]) if X else 0)
        result = []
        for row in X:
            new_row = [math.log1p(row[j]) if self.sublinear_tf else row[j] for j in range(p)]
            if self.use_idf:
                new_row = [new_row[j] * self.idf_[j] for j in range(p)]
            result.append(new_row)
        if self.norm == 'l2':
            result = _l2_normalize(result)
        elif self.norm == 'l1':
            result = _l1_normalize(result)
        return result

    def fit_transform(self, X: List[List[float]]) -> List[List[float]]:
        return self.fit(X).transform(X)


# ---------------------------------------------------------------------------
# TfidfVectorizer
# ---------------------------------------------------------------------------

class TfidfVectorizer(CountVectorizer):
    """Combine CountVectorizer + TfidfTransformer."""

    def __init__(self, norm: Optional[str] = 'l2', use_idf: bool = True,
                 smooth_idf: bool = True, sublinear_tf: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._tfidf = TfidfTransformer(norm=norm, use_idf=use_idf,
                                       smooth_idf=smooth_idf, sublinear_tf=sublinear_tf)

    def fit(self, corpus: Iterable[str]) -> 'TfidfVectorizer':
        corpus = list(corpus)
        super().fit(corpus)
        counts = super().transform(corpus)
        self._tfidf.fit(counts)
        return self

    def transform(self, corpus: Iterable[str]) -> List[List[float]]:
        counts = super().transform(corpus)
        return self._tfidf.transform(counts)

    def fit_transform(self, corpus: Iterable[str]) -> List[List[float]]:
        corpus = list(corpus)
        return self.fit(corpus).transform(corpus)


# ---------------------------------------------------------------------------
# HashingVectorizer
# ---------------------------------------------------------------------------

class HashingVectorizer:
    """Stateless hash-trick vectorizer (no vocabulary fitting)."""

    def __init__(self, n_features: int = 2 ** 20, analyzer: str = 'word',
                 ngram_range: Tuple[int, int] = (1, 1),
                 norm: Optional[str] = 'l2', alternate_sign: bool = True):
        self.n_features = n_features
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.norm = norm
        self.alternate_sign = alternate_sign

    def _hash(self, token: str) -> int:
        return int(hashlib.md5(token.encode()).hexdigest(), 16) % self.n_features

    def transform(self, corpus: Iterable[str]) -> List[List[float]]:
        corpus = list(corpus)
        result = []
        for doc in corpus:
            row = [0.0] * self.n_features
            tokens = _tokenize(doc, self.analyzer, self.ngram_range)
            for token in tokens:
                j = self._hash(token)
                sign = 1 if (not self.alternate_sign or
                             int(hashlib.md5((token + '_s').encode()).hexdigest(), 16) % 2 == 0) else -1
                row[j] += sign
            result.append(row)
        if self.norm == 'l2':
            result = _l2_normalize(result)
        elif self.norm == 'l1':
            result = _l1_normalize(result)
        return result

    def fit(self, corpus) -> 'HashingVectorizer':
        return self

    def fit_transform(self, corpus: Iterable[str]) -> List[List[float]]:
        return self.transform(corpus)


# ---------------------------------------------------------------------------
# DictVectorizer
# ---------------------------------------------------------------------------

class DictVectorizer:
    """Vectorize feature dictionaries into a dense matrix."""

    def __init__(self, sparse: bool = False, separator: str = '=',
                 sort: bool = True):
        self.sparse = sparse
        self.separator = separator
        self.sort = sort
        self.vocabulary_: Optional[Dict[str, int]] = None
        self.feature_names_: Optional[List[str]] = None

    def fit(self, dicts: List[Dict]) -> 'DictVectorizer':
        feature_set: set = set()
        for d in dicts:
            for k, v in d.items():
                if isinstance(v, (int, float)):
                    feature_set.add(str(k))
                else:
                    feature_set.add(f'{k}{self.separator}{v}')
        features = sorted(feature_set) if self.sort else list(feature_set)
        self.feature_names_ = features
        self.vocabulary_ = {f: i for i, f in enumerate(features)}
        return self

    def transform(self, dicts: List[Dict]) -> List[List[float]]:
        p = len(self.vocabulary_)
        result = []
        for d in dicts:
            row = [0.0] * p
            for k, v in d.items():
                if isinstance(v, (int, float)):
                    key = str(k)
                    if key in self.vocabulary_:
                        row[self.vocabulary_[key]] = float(v)
                else:
                    key = f'{k}{self.separator}{v}'
                    if key in self.vocabulary_:
                        row[self.vocabulary_[key]] = 1.0
            result.append(row)
        return result

    def fit_transform(self, dicts: List[Dict]) -> List[List[float]]:
        return self.fit(dicts).transform(dicts)

    def get_feature_names_out(self) -> List[str]:
        return list(self.feature_names_ or [])

    def inverse_transform(self, X: List[List[float]]) -> List[Dict]:
        inv = {i: f for f, i in self.vocabulary_.items()}
        result = []
        for row in X:
            d: Dict = {}
            for j, val in enumerate(row):
                if val != 0:
                    key = inv[j]
                    if self.separator in key:
                        k, v = key.split(self.separator, 1)
                        d[k] = v
                    else:
                        d[key] = val
            result.append(d)
        return result


__all__ = [
    'CountVectorizer', 'TfidfTransformer', 'TfidfVectorizer',
    'HashingVectorizer', 'DictVectorizer',
]
