"""Spatial algorithms — KDTree, distances, ConvexHull stubs, Voronoi stubs — pure Python."""
from __future__ import annotations
import math
from typing import Any, List, Optional, Tuple

try:
    import scipy.spatial as _ss
    KDTree = _ss.KDTree
    cKDTree = _ss.cKDTree
    def ConvexHull(points, incremental=False, qhull_options=''):
        return _ss.ConvexHull(points, incremental=incremental, qhull_options=qhull_options)
    def Delaunay(points, furthest_site=False, incremental=False, qhull_options=''):
        return _ss.Delaunay(points, furthest_site=furthest_site, incremental=incremental, qhull_options=qhull_options)
    def Voronoi(points, furthest_site=False, incremental=False, qhull_options=''):
        return _ss.Voronoi(points, furthest_site=furthest_site, incremental=incremental, qhull_options=qhull_options)
    def distance_matrix(x, y, p=2, threshold=1000000):
        return _ss.distance_matrix(x, y, p=p, threshold=threshold)
    def cdist(XA, XB, metric='euclidean', **kwargs):
        return _ss.distance.cdist(XA, XB, metric=metric, **kwargs)
    def pdist(X, metric='euclidean', **kwargs):
        return _ss.distance.pdist(X, metric=metric, **kwargs)
    def squareform(X, force='no', checks=True):
        return _ss.distance.squareform(X, force=force, checks=checks)
    def procrustes(data1, data2):
        return _ss.procrustes(data1, data2)
    _SCIPY = True
except ImportError:
    _SCIPY = False


# ── Pure Python vector helpers ────────────────────────────────────────────────

def _vec(x) -> List[float]:
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


def euclidean(u, v) -> float:
    u, v = _vec(u), _vec(v)
    return math.sqrt(sum((a-b)**2 for a,b in zip(u,v)))

def cityblock(u, v) -> float:
    u, v = _vec(u), _vec(v)
    return sum(abs(a-b) for a,b in zip(u,v))

def cosine(u, v) -> float:
    u, v = _vec(u), _vec(v)
    dot = sum(a*b for a,b in zip(u,v))
    nu = math.sqrt(sum(a*a for a in u))
    nv = math.sqrt(sum(b*b for b in v))
    return 1.0 - dot / (nu * nv + 1e-12)

def hamming(u, v) -> float:
    u, v = _vec(u), _vec(v)
    return sum(1 for a,b in zip(u,v) if a != b) / max(len(u), 1)

def jaccard(u, v) -> float:
    u, v = _vec(u), _vec(v)
    s = sum(1 for a,b in zip(u,v) if a or b)
    d = sum(1 for a,b in zip(u,v) if a != b)
    return d / max(s, 1)

def chebyshev(u, v) -> float:
    u, v = _vec(u), _vec(v)
    return max(abs(a-b) for a,b in zip(u,v))

def minkowski(u, v, p=2) -> float:
    u, v = _vec(u), _vec(v)
    return sum(abs(a-b)**p for a,b in zip(u,v)) ** (1.0/p)

def mahalanobis(u, v, VI) -> float:
    u_v, v_v = _vec(u), _vec(v)
    delta = [a - b for a, b in zip(u_v, v_v)]
    n = len(delta)
    # VI is the inverse covariance matrix (n x n list-of-lists or flat)
    if hasattr(VI, 'tolist'):
        vi = VI.tolist()
        if not (vi and isinstance(vi[0], list)):
            vi = [[vi[i * n + j] for j in range(n)] for i in range(n)]
    elif isinstance(VI, list):
        if VI and isinstance(VI[0], list):
            vi = VI
        else:
            vi = [[VI[i * n + j] for j in range(n)] for i in range(n)]
    else:
        vi = [[float(VI[i][j]) for j in range(n)] for i in range(n)]
    # Compute delta @ VI @ delta^T
    vi_delta = [sum(vi[i][j] * delta[j] for j in range(n)) for i in range(n)]
    return math.sqrt(max(0.0, sum(delta[i] * vi_delta[i] for i in range(n))))

def braycurtis(u, v) -> float:
    u, v = _vec(u), _vec(v)
    return sum(abs(a-b) for a,b in zip(u,v)) / (sum(abs(a+b) for a,b in zip(u,v)) + 1e-12)

def canberra(u, v) -> float:
    u, v = _vec(u), _vec(v)
    return sum(abs(a-b) / (abs(a) + abs(b) + 1e-12) for a,b in zip(u,v))

def correlation(u, v) -> float:
    u, v = _vec(u), _vec(v)
    mu = sum(u)/len(u); mv = sum(v)/len(v)
    uc = [a-mu for a in u]; vc = [b-mv for b in v]
    dot = sum(a*b for a,b in zip(uc,vc))
    nu = math.sqrt(sum(a*a for a in uc))
    nv = math.sqrt(sum(b*b for b in vc))
    return 1.0 - dot / (nu * nv + 1e-12)


# ── KDTree (pure Python) ──────────────────────────────────────────────────────

if not _SCIPY:
    class KDTree:
        """Pure Python k-d tree (brute force fallback)."""
        def __init__(self, data, leafsize=10):
            self.data = [[float(v) for v in _vec(row)] for row in data]

        def query(self, x, k=1, workers=1):
            pts = [[float(v) for v in _vec(row)] for row in (x if isinstance(x[0], (list, tuple)) else [x])]
            dists_all = []; idxs_all = []
            for q in pts:
                scored = sorted(enumerate(self.data), key=lambda iv: sum((a-b)**2 for a,b in zip(q,iv[1])))
                top = scored[:k]
                dists_all.append([math.sqrt(sum((a-b)**2 for a,b in zip(q,p))) for _,p in top])
                idxs_all.append([i for i,_ in top])
            if len(pts) == 1:
                return dists_all[0], idxs_all[0]
            return dists_all, idxs_all

        def query_ball_point(self, x, r):
            q = [float(v) for v in _vec(x)]
            return [i for i, p in enumerate(self.data) if math.sqrt(sum((a-b)**2 for a,b in zip(q,p))) <= r]

    cKDTree = KDTree

    def ConvexHull(points, **kwargs):
        raise ImportError("scipy required for ConvexHull")
    def Delaunay(points, **kwargs):
        raise ImportError("scipy required for Delaunay")
    def Voronoi(points, **kwargs):
        raise ImportError("scipy required for Voronoi")
    def SphericalVoronoi(points, **kwargs):
        raise ImportError("scipy required for SphericalVoronoi")
    def HalfspaceIntersection(halfspaces, interior_point, **kwargs):
        raise ImportError("scipy required for HalfspaceIntersection")
    def procrustes(data1, data2):
        raise ImportError("scipy required for procrustes")

    def cdist(XA, XB, metric='euclidean', **kwargs) -> List[List[float]]:
        rows_a = [[float(v) for v in _vec(row)] for row in XA]
        rows_b = [[float(v) for v in _vec(row)] for row in XB]
        fn = {'euclidean': euclidean, 'cityblock': cityblock, 'cosine': cosine}.get(metric, euclidean)
        return [[fn(a, b) for b in rows_b] for a in rows_a]

    def pdist(X, metric='euclidean', **kwargs) -> List[float]:
        rows = [[float(v) for v in _vec(row)] for row in X]
        fn = {'euclidean': euclidean, 'cityblock': cityblock, 'cosine': cosine}.get(metric, euclidean)
        result = []
        for i in range(len(rows)):
            for j in range(i+1, len(rows)):
                result.append(fn(rows[i], rows[j]))
        return result

    def squareform(X, **kwargs) -> List[List[float]]:
        n_flat = len(X)
        # infer n from n*(n-1)/2 = n_flat => n ~ (1+sqrt(1+8*n_flat))/2
        n = int((1 + math.sqrt(1 + 8*n_flat)) / 2)
        result = [[0.0]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                result[i][j] = result[j][i] = X[idx]
                idx += 1
        return result

    def distance_matrix(x, y, p=2, **kwargs) -> List[List[float]]:
        rows_x = [[float(v) for v in _vec(row)] for row in x]
        rows_y = [[float(v) for v in _vec(row)] for row in y]
        return [[minkowski(a, b, p) for b in rows_y] for a in rows_x]


# ── BallTree ──────────────────────────────────────────────────────────────────

class BallTree:
    """Simple ball tree (brute force) — pure Python."""

    def __init__(self, X, leaf_size=40, metric='euclidean'):
        self.X = [[float(v) for v in _vec(row)] for row in X]
        self.metric = metric

    def _dist(self, a, b):
        return euclidean(a, b) if self.metric == 'euclidean' else minkowski(a, b)

    def query(self, X, k=1, return_distance=True):
        pts = [[float(v) for v in _vec(row)] for row in X]
        all_dists = []; all_idxs = []
        for q in pts:
            scored = sorted(enumerate(self.X), key=lambda iv: self._dist(q, iv[1]))
            top = scored[:k]
            all_dists.append([self._dist(q, p) for _, p in top])
            all_idxs.append([i for i, _ in top])
        if return_distance:
            return all_dists, all_idxs
        return all_idxs

    def query_radius(self, X, r, return_distance=False, count_only=False):
        pts = [[float(v) for v in _vec(row)] for row in X]
        result = []
        for q in pts:
            idxs = [i for i, p in enumerate(self.X) if self._dist(q, p) <= r]
            if count_only:
                result.append(len(idxs))
            elif return_distance:
                result.append((idxs, [self._dist(q, self.X[i]) for i in idxs]))
            else:
                result.append(idxs)
        return result
