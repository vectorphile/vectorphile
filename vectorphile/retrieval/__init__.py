"""vectorphile.retrieval — Retrieval-Augmented Generation (RAG) utilities.

Provides:
- VectorStore: in-memory vector store with cosine/dot/euclidean search
- BM25Retriever: keyword-based BM25 retrieval
- HybridRetriever: combine dense + sparse retrieval
- DocumentStore: store documents with metadata
- Chunker: split documents into chunks
- EmbeddingCache: cache embeddings

No external dependencies.
"""
from __future__ import annotations
import math
import re
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------

class Document:
    """A text document with optional metadata."""

    def __init__(self, text: str, doc_id: Optional[str] = None,
                 metadata: Optional[Dict] = None, embedding=None):
        self.text = text
        self.doc_id = doc_id or str(id(self))
        self.metadata = metadata or {}
        self.embedding = embedding

    def __repr__(self) -> str:
        return f'Document(id={self.doc_id!r}, text={self.text[:60]!r})'


# ---------------------------------------------------------------------------
# Pure-Python vector helpers
# ---------------------------------------------------------------------------

def _to_vec(x) -> List[float]:
    if hasattr(x, 'to_flat_list'):
        return [float(v) for v in x.to_flat_list()]
    if hasattr(x, 'tolist'):
        v = x.tolist()
        if isinstance(v, list) and v and isinstance(v[0], list):
            return [float(item) for row in v for item in row]
        return [float(item) for item in v]
    if isinstance(x, list):
        return [float(v) for v in x]
    return [float(x)]

def _dot(a: List[float], b: List[float]) -> float:
    return sum(x*y for x,y in zip(a, b))

def _norm(a: List[float]) -> float:
    return math.sqrt(sum(v*v for v in a))

def _cosine(a: List[float], b: List[float]) -> float:
    na = _norm(a); nb = _norm(b)
    return _dot(a, b) / (na * nb + 1e-10)

def _euclidean_neg(a: List[float], b: List[float]) -> float:
    return -math.sqrt(sum((x-y)**2 for x,y in zip(a, b)))


# ---------------------------------------------------------------------------
# VectorStore (dense retrieval)
# ---------------------------------------------------------------------------

class VectorStore:
    """In-memory vector store with approximate nearest-neighbor search."""

    def __init__(self, dim: int, metric: str = 'cosine'):
        self.dim = dim
        self.metric = metric
        self._ids: List[str] = []
        self._embeddings: List[List[float]] = []
        self._metadata: List[Dict] = []

    def add(self, doc_id: str, embedding, metadata: Optional[Dict] = None) -> None:
        self._ids.append(doc_id)
        self._embeddings.append(_to_vec(embedding))
        self._metadata.append(metadata or {})

    def add_documents(self, docs: List[Document]) -> None:
        for doc in docs:
            if doc.embedding is not None:
                self.add(doc.doc_id, doc.embedding, doc.metadata)

    def _score(self, q: List[float], emb: List[float]) -> float:
        if self.metric == 'cosine':
            return _cosine(q, emb)
        elif self.metric == 'dot':
            return _dot(q, emb)
        else:
            return _euclidean_neg(q, emb)

    def search(self, query_embedding, top_k: int = 5,
               filter_fn: Optional[Any] = None) -> List[Tuple[str, float, Dict]]:
        if not self._ids:
            return []
        q = _to_vec(query_embedding)
        scored = []
        for i, emb in enumerate(self._embeddings):
            if filter_fn and not filter_fn(self._metadata[i]):
                continue
            scored.append((i, self._score(q, emb)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(self._ids[i], score, self._metadata[i]) for i, score in scored[:top_k]]

    def delete(self, doc_id: str) -> bool:
        if doc_id in self._ids:
            idx = self._ids.index(doc_id)
            self._ids.pop(idx)
            self._embeddings.pop(idx)
            self._metadata.pop(idx)
            return True
        return False

    def __len__(self) -> int:
        return len(self._ids)

    def __repr__(self) -> str:
        return f'VectorStore(size={len(self)}, dim={self.dim}, metric={self.metric!r})'


# ---------------------------------------------------------------------------
# BM25 Retriever (sparse/keyword)
# ---------------------------------------------------------------------------

class BM25Retriever:
    """BM25 (Best Match 25) retrieval for keyword search."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._docs: List[str] = []
        self._doc_ids: List[str] = []
        self._tf: List[Dict[str, int]] = []
        self._df: Dict[str, int] = defaultdict(int)
        self._avgdl: float = 0.0
        self._n: int = 0

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())

    def index(self, documents: List[str], doc_ids: Optional[List[str]] = None) -> None:
        self._docs = documents
        self._doc_ids = doc_ids or [str(i) for i in range(len(documents))]
        self._tf = []
        self._df = defaultdict(int)
        total_len = 0
        for doc in documents:
            tokens = self._tokenize(doc)
            total_len += len(tokens)
            tf: Dict[str, int] = defaultdict(int)
            for t in tokens:
                tf[t] += 1
            self._tf.append(dict(tf))
            for t in set(tokens):
                self._df[t] += 1
        self._n = len(documents)
        self._avgdl = total_len / max(1, self._n)

    def _score_doc(self, tokens: List[str], doc_idx: int) -> float:
        tf = self._tf[doc_idx]
        dl = sum(tf.values())
        score = 0.0
        for t in tokens:
            if t not in tf:
                continue
            idf = math.log(1 + (self._n - self._df.get(t, 0) + 0.5) /
                           (self._df.get(t, 0) + 0.5))
            tf_norm = (tf[t] * (self.k1 + 1)) / (
                tf[t] + self.k1 * (1 - self.b + self.b * dl / self._avgdl))
            score += idf * tf_norm
        return score

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
        tokens = self._tokenize(query)
        scores = [(i, self._score_doc(tokens, i)) for i in range(self._n)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [(self._doc_ids[i], score, self._docs[i]) for i, score in scores[:top_k]]


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

class Chunker:
    """Split documents into overlapping chunks."""

    def __init__(self, chunk_size: int = 512, overlap: int = 64, separator: str = ' '):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separator = separator

    def split(self, text: str) -> List[str]:
        words = text.split(self.separator)
        chunks = []
        i = 0
        while i < len(words):
            chunk_words = words[i:i + self.chunk_size]
            chunks.append(self.separator.join(chunk_words))
            i += self.chunk_size - self.overlap
            if i >= len(words):
                break
        return chunks

    def split_documents(self, documents: List[str]) -> List[Document]:
        result = []
        for doc_idx, text in enumerate(documents):
            for chunk_idx, chunk in enumerate(self.split(text)):
                result.append(Document(
                    text=chunk,
                    doc_id=f'doc{doc_idx}_chunk{chunk_idx}',
                    metadata={'source_doc': doc_idx, 'chunk_idx': chunk_idx},
                ))
        return result


# ---------------------------------------------------------------------------
# Hybrid Retriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """Combine dense (vector) and sparse (BM25) retrieval via RRF."""

    def __init__(self, vector_store: Optional[VectorStore] = None,
                 bm25: Optional[BM25Retriever] = None,
                 embed_fn: Optional[Any] = None,
                 rrf_k: int = 60, alpha: float = 0.5):
        self.vector_store = vector_store
        self.bm25 = bm25
        self.embed_fn = embed_fn
        self.rrf_k = rrf_k
        self.alpha = alpha

    def search(self, query: str, query_embedding=None,
               top_k: int = 5) -> List[Tuple[str, float]]:
        rrf_scores: Dict[str, float] = defaultdict(float)
        if self.vector_store and query_embedding is not None:
            dense_results = self.vector_store.search(query_embedding, top_k=top_k * 2)
            for rank, (doc_id, score, _) in enumerate(dense_results):
                rrf_scores[doc_id] += 1.0 / (self.rrf_k + rank + 1)
        if self.bm25:
            sparse_results = self.bm25.search(query, top_k=top_k * 2)
            for rank, (doc_id, score, _) in enumerate(sparse_results):
                rrf_scores[doc_id] += 1.0 / (self.rrf_k + rank + 1)
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]


# ---------------------------------------------------------------------------
# EmbeddingCache
# ---------------------------------------------------------------------------

class EmbeddingCache:
    """Cache embeddings to avoid redundant computation."""

    def __init__(self, embed_fn: Callable = None, max_size: int = 10000):
        self._embed_fn = embed_fn
        self._cache: Dict[str, List[float]] = {}
        self.max_size = max_size

    def embed(self, text: str) -> List[float]:
        if text in self._cache:
            return self._cache[text]
        if self._embed_fn is None:
            raise ValueError('embed_fn not set')
        emb = self._embed_fn(text)
        if len(self._cache) < self.max_size:
            self._cache[text] = emb
        return emb

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


class HNSWIndex:
    """Hierarchical Navigable Small World approximate nearest neighbor index.

    Simplified HNSW: greedy graph-based search at multiple layers.
    """

    def __init__(self, dim: int, M: int = 16, ef_construction: int = 200, metric: str = 'cosine'):
        self.dim = dim
        self.M = M
        self.ef_construction = ef_construction
        self.metric = metric
        self._vectors: List[List[float]] = []
        self._ids: List[str] = []
        self._graph: List[List[int]] = []  # neighbors per node
        self._level_graph: List[dict] = []  # {node: [neighbors]} per layer
        self._levels: List[int] = []
        self._entry_point: int = -1
        import math
        self._ml = 1.0 / math.log(max(M, 2))

    def _dist(self, a: List[float], b: List[float]) -> float:
        if self.metric == 'cosine':
            dot = sum(x * y for x, y in zip(a, b))
            na = sum(x * x for x in a) ** 0.5
            nb = sum(x * x for x in b) ** 0.5
            return 1.0 - dot / (na * nb + 1e-10)
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

    def _random_level(self) -> int:
        import math, random
        return int(-math.log(random.random()) * self._ml)

    def _search_layer(self, query: List[float], entry: int, ef: int, level: int) -> List[int]:
        import heapq
        visited = {entry}
        candidates = [(self._dist(query, self._vectors[entry]), entry)]
        dynamic_list = list(candidates)
        while candidates:
            d, u = heapq.heappop(candidates)
            worst_in_dlist = max(dynamic_list, key=lambda x: x[0])[0]
            if d > worst_in_dlist:
                break
            nbrs = self._level_graph[level].get(u, []) if level < len(self._level_graph) else []
            for v in nbrs:
                if v not in visited:
                    visited.add(v)
                    dv = self._dist(query, self._vectors[v])
                    if len(dynamic_list) < ef or dv < worst_in_dlist:
                        heapq.heappush(candidates, (dv, v))
                        dynamic_list.append((dv, v))
                        if len(dynamic_list) > ef:
                            dynamic_list.sort(key=lambda x: x[0])
                            dynamic_list = dynamic_list[:ef]
        dynamic_list.sort(key=lambda x: x[0])
        return [v for _, v in dynamic_list]

    def add(self, vector: List[float], id: str = None):
        idx = len(self._vectors)
        self._vectors.append(list(vector))
        self._ids.append(id or str(idx))
        level = self._random_level()
        self._levels.append(level)
        while len(self._level_graph) <= level:
            self._level_graph.append({})
        if self._entry_point == -1:
            self._entry_point = idx
            return
        # Insert node
        ep = self._entry_point
        max_level = self._levels[ep]
        for lc in range(min(max_level, level), -1, -1):
            candidates = self._search_layer(vector, ep, self.ef_construction, lc)[:self.M]
            if lc not in range(len(self._level_graph)):
                self._level_graph.append({})
            if idx not in self._level_graph[lc]:
                self._level_graph[lc][idx] = []
            for c in candidates:
                self._level_graph[lc][idx].append(c)
                if c not in self._level_graph[lc]:
                    self._level_graph[lc][c] = []
                self._level_graph[lc][c].append(idx)
            ep = candidates[0] if candidates else ep
        if level > self._levels[self._entry_point]:
            self._entry_point = idx

    def search(self, query: List[float], k: int = 10) -> List[tuple]:
        if self._entry_point == -1:
            return []
        ep = self._entry_point
        max_level = self._levels[ep]
        for lc in range(max_level, 0, -1):
            candidates = self._search_layer(query, ep, 1, lc)
            ep = candidates[0] if candidates else ep
        candidates = self._search_layer(query, ep, max(k, 10), 0)
        results = [(self._ids[v], self._dist(query, self._vectors[v])) for v in candidates[:k]]
        results.sort(key=lambda x: x[1])
        return results

    def __len__(self) -> int:
        return len(self._vectors)


__all__ = [
    'Document', 'VectorStore', 'BM25Retriever',
    'Chunker', 'HybridRetriever', 'EmbeddingCache', 'HNSWIndex',
]
