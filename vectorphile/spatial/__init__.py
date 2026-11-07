"""Spatial algorithms — KDTree, distances, ConvexHull, Delaunay, Voronoi."""
from __future__ import annotations
import numpy as np

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

    # Individual distance functions
    def euclidean(u, v): return _ss.distance.euclidean(u, v)
    def cityblock(u, v): return _ss.distance.cityblock(u, v)
    def cosine(u, v): return _ss.distance.cosine(u, v)
    def hamming(u, v): return _ss.distance.hamming(u, v)
    def jaccard(u, v): return _ss.distance.jaccard(u, v)
    def chebyshev(u, v): return _ss.distance.chebyshev(u, v)
    def minkowski(u, v, p=2): return _ss.distance.minkowski(u, v, p=p)
    def mahalanobis(u, v, VI): return _ss.distance.mahalanobis(u, v, VI)
    def braycurtis(u, v): return _ss.distance.braycurtis(u, v)
    def canberra(u, v): return _ss.distance.canberra(u, v)
    def correlation(u, v): return _ss.distance.correlation(u, v)

    def procrustes(data1, data2):
        return _ss.procrustes(data1, data2)

    def SphericalVoronoi(points, radius=1, center=None, threshold=1e-6):
        return _ss.SphericalVoronoi(points, radius=radius,
                                    center=np.zeros(points.shape[1]) if center is None else center,
                                    threshold=threshold)

    def HalfspaceIntersection(halfspaces, interior_point, incremental=False, qhull_options=''):
        return _ss.HalfspaceIntersection(halfspaces, interior_point, incremental=incremental, qhull_options=qhull_options)

except ImportError:
    def _no_scipy(*args, **kwargs):
        raise ImportError("scipy is required for vp.spatial. Install with: pip install scipy")
    KDTree = cKDTree = ConvexHull = Delaunay = Voronoi = _no_scipy
    distance_matrix = cdist = pdist = squareform = _no_scipy
    euclidean = cityblock = cosine = hamming = jaccard = chebyshev = minkowski = _no_scipy
    mahalanobis = braycurtis = canberra = correlation = procrustes = _no_scipy
    SphericalVoronoi = HalfspaceIntersection = _no_scipy

    # Fallback: basic distance functions in pure NumPy
    def euclidean(u, v):
        return float(np.sqrt(np.sum((np.array(u) - np.array(v))**2)))

    def cityblock(u, v):
        return float(np.sum(np.abs(np.array(u) - np.array(v))))

    def cosine(u, v):
        u, v = np.array(u, dtype=float), np.array(v, dtype=float)
        return float(1 - u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))

    def chebyshev(u, v):
        return float(np.max(np.abs(np.array(u) - np.array(v))))

    def minkowski(u, v, p=2):
        return float(np.sum(np.abs(np.array(u) - np.array(v))**p)**(1/p))


# Pure-NumPy nearest-neighbor search
class BallTree:
    """Simple ball tree implemented in NumPy (fallback when scipy unavailable)."""
    def __init__(self, X, leaf_size=40, metric='euclidean'):
        self.X = np.array(X, dtype=np.float64)
        self.metric = metric

    def query(self, X, k=1, return_distance=True):
        X = np.array(X, dtype=np.float64)
        D = np.sqrt(((X[:, None, :] - self.X[None, :, :])**2).sum(axis=2))
        idx = np.argsort(D, axis=1)[:, :k]
        dists = D[np.arange(len(X))[:, None], idx]
        if return_distance:
            return dists, idx
        return idx

    def query_radius(self, X, r, return_distance=False, count_only=False):
        X = np.array(X, dtype=np.float64)
        D = np.sqrt(((X[:, None, :] - self.X[None, :, :])**2).sum(axis=2))
        result = [np.where(D[i] <= r)[0] for i in range(len(X))]
        if count_only:
            return np.array([len(r_) for r_ in result])
        if return_distance:
            return result, [D[i, r_] for i, r_ in enumerate(result)]
        return result
