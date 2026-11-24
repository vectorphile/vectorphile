"""NLP utilities — tokenizer, vocabulary, TF-IDF, n-gram LM, edit distance, BLEU, ROUGE."""
import math
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
class Tokenizer:
    """Simple whitespace + punctuation tokenizer."""
    _PUNCT = re.compile(r"([,.!?;:\"'()\[\]{}/\\])")

    def tokenize(self, text: str) -> List[str]:
        text = self._PUNCT.sub(r' \1 ', text)
        return [t.lower() for t in text.split() if t]


class Vocabulary:
    PAD, UNK = '<pad>', '<unk>'

    def __init__(self):
        self._tok2id: Dict[str, int] = {}
        self._id2tok: List[str] = []

    def build(self, corpus: List[List[str]], min_freq: int = 1):
        counts: Counter = Counter()
        for sent in corpus:
            counts.update(sent)
        self._tok2id = {self.PAD: 0, self.UNK: 1}
        self._id2tok = [self.PAD, self.UNK]
        for tok, cnt in counts.most_common():
            if cnt >= min_freq:
                self._tok2id[tok] = len(self._id2tok)
                self._id2tok.append(tok)

    def encode(self, tokens: List[str]) -> List[int]:
        unk = self._tok2id[self.UNK]
        return [self._tok2id.get(t, unk) for t in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        return [self._id2tok[i] if i < len(self._id2tok) else self.UNK for i in ids]

    def __len__(self):
        return len(self._id2tok)


class TFIDFVectorizer:
    def __init__(self, max_features: int = None, min_df: int = 1):
        self.max_features = max_features
        self.min_df = min_df
        self._vocab: Dict[str, int] = {}
        self._idf: List[float] = []

    def fit_transform(self, corpus: List[str]):
        tok = Tokenizer()
        docs = [tok.tokenize(d) for d in corpus]
        # Build vocabulary
        df: Counter = Counter()
        for doc in docs:
            df.update(set(doc))
        vocab_terms = [t for t, c in df.most_common() if c >= self.min_df]
        if self.max_features:
            vocab_terms = vocab_terms[:self.max_features]
        self._vocab = {t: i for i, t in enumerate(vocab_terms)}
        V = len(self._vocab)
        N = len(docs)
        X = [[0.0] * V for _ in range(N)]
        for i, doc in enumerate(docs):
            cnt = Counter(doc)
            for tok_, freq in cnt.items():
                if tok_ in self._vocab:
                    X[i][self._vocab[tok_]] = freq / max(1, len(doc))
        # IDF
        self._idf = [math.log((N + 1) / (1 + df.get(t, 0))) + 1 for t in vocab_terms]
        result = [[X[i][j] * self._idf[j] for j in range(V)] for i in range(N)]
        try:
            import numpy as _np
            return _np.array(result, dtype='float32')
        except ImportError:
            return result


class NGramModel:
    def __init__(self, n: int = 2):
        self.n = n
        self._counts: Dict = defaultdict(Counter)
        self._vocab: set = set()

    def fit(self, corpus: List[List[str]]):
        for sent in corpus:
            self._vocab.update(sent)
            for i in range(len(sent) - self.n + 1):
                ctx = tuple(sent[i:i + self.n - 1])
                word = sent[i + self.n - 1]
                self._counts[ctx][word] += 1

    def probability(self, tokens: List[str]) -> float:
        if len(tokens) < self.n:
            return 1.0 / max(len(self._vocab), 1)
        ctx = tuple(tokens[-(self.n - 1):]) if self.n > 1 else ()
        ctx_counts = self._counts.get(ctx, Counter())
        total = sum(ctx_counts.values())
        if total == 0:
            return 1.0 / max(len(self._vocab), 1)
        word = tokens[-1]
        return ctx_counts.get(word, 0) / total

    def perplexity(self, corpus: List[List[str]]) -> float:
        log_prob = 0.0; N = 0
        for sent in corpus:
            for i in range(self.n - 1, len(sent)):
                p = self.probability(sent[max(0, i - self.n + 1):i + 1])
                log_prob += math.log(max(p, 1e-10))
                N += 1
        if N == 0:
            return float('inf')
        return math.exp(-log_prob / N)


def edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n]


def bleu_score(references, hypothesis: List[str], max_n: int = 4) -> float:
    """Corpus BLEU score.

    references: List[List[str]] (single sentence, one or more refs)
                OR List[List[List[str]]] (multi-sentence corpus).
    hypothesis: List[str] (token list for one sentence) or List[List[str]].
    """
    def ngrams(tokens, n):
        return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))

    # Normalise to multi-sentence format: List[List[List[str]]]
    # Detect: if references[0] is a list of strings → single sentence, single ref
    # if references[0][0] is a string → single sentence, multiple refs
    # if references[0][0] is a list → multi-sentence corpus
    if not references:
        return 0.0
    r0 = references[0]
    if isinstance(r0, str):
        # references is a flat token list — treat as single ref for one sentence
        refs_corpus = [[references]]
        hyp_corpus = [hypothesis]
    elif isinstance(r0, list) and (len(r0) == 0 or isinstance(r0[0], str)):
        # references is List[List[str]]: one sentence with multiple refs
        refs_corpus = [references]
        hyp_corpus = [hypothesis]
    else:
        # Already List[List[List[str]]]
        refs_corpus = references
        hyp_corpus = hypothesis if isinstance(hypothesis[0], list) else [hypothesis]

    bp_ref_len = sum(min(len(r) for r in refs) for refs in refs_corpus)
    bp_hyp_len = sum(len(h) for h in hyp_corpus)
    bp = math.exp(min(0.0, 1.0 - bp_ref_len / max(bp_hyp_len, 1)))

    precisions = []
    for n in range(1, max_n + 1):
        total = 0; matched = 0
        for refs, hyp in zip(refs_corpus, hyp_corpus):
            hyp_ng = ngrams(hyp, n)
            clipped = Counter()
            for ng, cnt in hyp_ng.items():
                max_ref = max(ngrams(ref, n).get(ng, 0) for ref in refs)
                clipped[ng] = min(cnt, max_ref)
            total += sum(hyp_ng.values())
            matched += sum(clipped.values())
        if total == 0:
            precisions.append(0.0)
        else:
            precisions.append(matched / total)

    if any(p == 0 for p in precisions):
        return 0.0
    return bp * math.exp(sum(math.log(p) for p in precisions) / max_n)


def rouge_score(reference: List[str], hypothesis: List[str], n: int = 2) -> dict:
    """ROUGE-N F1."""
    def ngrams(tokens, k):
        return Counter(tuple(tokens[i:i + k]) for i in range(len(tokens) - k + 1))

    ref_ng = ngrams(reference, n)
    hyp_ng = ngrams(hypothesis, n)
    overlap = sum((ref_ng & hyp_ng).values())
    recall = overlap / max(sum(ref_ng.values()), 1)
    precision = overlap / max(sum(hyp_ng.values()), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    return {f'rouge-{n}': {'precision': precision, 'recall': recall, 'f1': f1}}



class BM25:
    """BM25 (Okapi) retrieval model."""
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1; self.b = b
        self.docs: List[List[str]] = []
        self.df: Counter = Counter()
        self.avgdl: float = 0.0
        self.N: int = 0

    def fit(self, corpus: List[List[str]]):
        self.docs = corpus; self.N = len(corpus)
        self.df = Counter()
        total_len = 0
        for doc in corpus:
            total_len += len(doc)
            for t in set(doc): self.df[t] += 1
        self.avgdl = total_len / max(self.N, 1)
        return self

    def score(self, query: List[str], doc_idx: int) -> float:
        doc = self.docs[doc_idx]; dl = len(doc)
        tf = Counter(doc); s = 0.0
        for t in query:
            f = tf.get(t, 0)
            idf = math.log((self.N - self.df.get(t, 0) + 0.5) / (self.df.get(t, 0) + 0.5) + 1)
            s += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    def get_scores(self, query: List[str]) -> List[float]:
        return [self.score(query, i) for i in range(self.N)]

    def get_top_n(self, query: List[str], n: int = 5) -> List[int]:
        scores = self.get_scores(query)
        return sorted(range(self.N), key=lambda i: -scores[i])[:n]


class SkipGram:
    """Word2Vec Skip-Gram (pure Python, no autograd)."""
    def __init__(self, vocab_size: int, embed_dim: int = 50, window: int = 2, neg_samples: int = 5):
        import random as _r
        self.vocab_size = vocab_size; self.embed_dim = embed_dim
        self.window = window; self.neg = neg_samples
        sc = 1.0 / embed_dim ** 0.5
        self.W = [[(_r.random()*2-1)*sc for _ in range(embed_dim)] for _ in range(vocab_size)]
        self.C = [[(_r.random()*2-1)*sc for _ in range(embed_dim)] for _ in range(vocab_size)]

    def _dot(self, a, b): return sum(x*y for x,y in zip(a,b))
    def _sig(self, x): return 1.0/(1+math.exp(-max(-500.0,min(500.0,x))))

    def train_pair(self, center: int, ctx: int, lr: float = 0.025) -> float:
        import random as _r
        loss = 0.0
        pos_score = self._sig(self._dot(self.W[center], self.C[ctx]))
        loss -= math.log(max(pos_score, 1e-10))
        grad_w = [self.C[ctx][k]*(pos_score-1) for k in range(self.embed_dim)]
        grad_c_pos = [self.W[center][k]*(pos_score-1) for k in range(self.embed_dim)]
        negs = [_r.randint(0, self.vocab_size-1) for _ in range(self.neg)]
        for neg in negs:
            neg_score = self._sig(self._dot(self.W[center], self.C[neg]))
            loss -= math.log(max(1-neg_score, 1e-10))
            for k in range(self.embed_dim):
                grad_w[k] += self.C[neg][k]*neg_score
                self.C[neg][k] -= lr*self.W[center][k]*neg_score
        for k in range(self.embed_dim):
            self.W[center][k] -= lr*grad_w[k]
            self.C[ctx][k] -= lr*grad_c_pos[k]
        return loss

    def embed(self, idx: int) -> List[float]: return self.W[idx]

    def most_similar(self, idx: int, top_n: int = 5) -> List[Tuple[int, float]]:
        v = self.W[idx]; scores = []
        for i, w in enumerate(self.W):
            if i == idx: continue
            denom = (sum(x**2 for x in v)**0.5)*(sum(x**2 for x in w)**0.5)
            scores.append((i, self._dot(v, w)/max(denom, 1e-10)))
        return sorted(scores, key=lambda x:-x[1])[:top_n]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x*y for x,y in zip(a,b))
    na = sum(x**2 for x in a)**0.5; nb = sum(x**2 for x in b)**0.5
    return dot / max(na*nb, 1e-10)


def sentence_similarity(sent1: str, sent2: str, vectorizer=None) -> float:
    """Cosine similarity between TF-IDF vectors of two sentences."""
    tv = vectorizer or TFIDFVectorizer()
    tv.fit([sent1, sent2])
    v1 = tv.transform([sent1])[0]; v2 = tv.transform([sent2])[0]
    return cosine_similarity(v1, v2)


def summarize(text: str, n_sentences: int = 3) -> str:
    """Extractive summarization via TF-IDF sentence scoring."""
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    if len(sentences) <= n_sentences: return text
    tv = TFIDFVectorizer()
    tv.fit(sentences)
    vecs = tv.transform(sentences)
    scores = [sum(v) for v in vecs]
    top = sorted(range(len(sentences)), key=lambda i:-scores[i])[:n_sentences]
    top.sort()
    return '. '.join(sentences[i] for i in top) + '.'


__all__ = [
    'Tokenizer', 'Vocabulary', 'TFIDFVectorizer', 'NGramModel',
    'BM25', 'SkipGram',
    'edit_distance', 'bleu_score', 'rouge_score',
    'cosine_similarity', 'sentence_similarity', 'summarize',
]
