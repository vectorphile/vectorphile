"""vectorphile.graph — Graph algorithms and graph neural network utilities.

Provides:
- Graph data structures (adjacency matrix, edge list, COO sparse)
- Classic graph algorithms (BFS, DFS, Dijkstra, Bellman-Ford, PageRank, etc.)
- Graph signal processing (normalized Laplacian, spectral embeddings)
- GNN data utilities (batching, neighborhood sampling, negative sampling)

No external dependencies — pure Python.

Usage::

    from vectorphile.graph import Graph, GraphData, graph_to_adj

    g = Graph(n_nodes=5)
    g.add_edge(0, 1, weight=1.5)
    g.add_edge(1, 2)
    # Shortest path
    dist = g.dijkstra(source=0)
    # PageRank
    pr = g.pagerank()
"""
from __future__ import annotations
import math
import heapq
import random as _random
from typing import List, Dict, Optional, Tuple, Set
from collections import deque, defaultdict


def _zeros2d(n: int, m: int) -> List[List[float]]:
    return [[0.0] * m for _ in range(n)]

def _eye(n: int) -> List[List[float]]:
    m = _zeros2d(n, n)
    for i in range(n): m[i][i] = 1.0
    return m

def _mat_mul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    n, k, m = len(A), len(B), len(B[0])
    C = _zeros2d(n, m)
    for i in range(n):
        for j in range(m):
            s = 0.0
            for p in range(k):
                s += A[i][p] * B[p][j]
            C[i][j] = s
    return C

def _mat_T(A: List[List[float]]) -> List[List[float]]:
    n, m = len(A), len(A[0])
    return [[A[i][j] for i in range(n)] for j in range(m)]

def _mat_vec(A: List[List[float]], v: List[float]) -> List[float]:
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]

def _vec_max_abs_diff(a: List[float], b: List[float]) -> float:
    return max(abs(x - y) for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# Graph data structure
# ---------------------------------------------------------------------------

class Graph:
    """Weighted directed/undirected graph with adjacency list representation."""

    def __init__(self, n_nodes: int = 0, directed: bool = False):
        self.n_nodes = n_nodes
        self.directed = directed
        self._adj: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        self._edges: List[Tuple[int, int, float]] = []

    def add_node(self) -> int:
        idx = self.n_nodes
        self.n_nodes += 1
        return idx

    def add_edge(self, u: int, v: int, weight: float = 1.0) -> None:
        self._adj[u].append((v, weight))
        self._edges.append((u, v, weight))
        if not self.directed:
            self._adj[v].append((u, weight))

    def neighbors(self, u: int) -> List[Tuple[int, float]]:
        return self._adj[u]

    def num_edges(self) -> int:
        return len(self._edges)

    def to_adj_matrix(self) -> List[List[float]]:
        n = self.n_nodes
        A = _zeros2d(n, n)
        for u, v, w in self._edges:
            A[u][v] = w
            if not self.directed:
                A[v][u] = w
        return A

    def to_edge_index(self) -> Tuple[List[int], List[int]]:
        """Return COO edge index (src, dst) lists."""
        if not self._edges:
            return [], []
        srcs = [u for u, v, _ in self._edges]
        dsts = [v for u, v, _ in self._edges]
        return srcs, dsts

    def edge_weights(self) -> List[float]:
        return [w for _, _, w in self._edges]

    # ---------------------------------------------------------------------------
    # Traversal
    # ---------------------------------------------------------------------------

    def bfs(self, source: int) -> List[int]:
        visited = set()
        order = []
        q = deque([source])
        visited.add(source)
        while q:
            u = q.popleft()
            order.append(u)
            for v, _ in self.neighbors(u):
                if v not in visited:
                    visited.add(v)
                    q.append(v)
        return order

    def dfs(self, source: int) -> List[int]:
        visited = set()
        order = []

        def _dfs(u):
            visited.add(u)
            order.append(u)
            for v, _ in self.neighbors(u):
                if v not in visited:
                    _dfs(v)

        _dfs(source)
        return order

    def connected_components(self) -> List[List[int]]:
        visited = set()
        components = []
        for node in range(self.n_nodes):
            if node not in visited:
                comp = self.bfs(node)
                components.append(comp)
                visited.update(comp)
        return components

    def is_bipartite(self) -> bool:
        color = {}
        for start in range(self.n_nodes):
            if start in color:
                continue
            color[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                for v, _ in self.neighbors(u):
                    if v not in color:
                        color[v] = 1 - color[u]
                        q.append(v)
                    elif color[v] == color[u]:
                        return False
        return True

    def topological_sort(self) -> List[int]:
        """Kahn's algorithm (directed graphs only)."""
        in_deg = [0] * self.n_nodes
        for u in range(self.n_nodes):
            for v, _ in self.neighbors(u):
                in_deg[v] += 1
        q = deque([u for u in range(self.n_nodes) if in_deg[u] == 0])
        order = []
        while q:
            u = q.popleft()
            order.append(u)
            for v, _ in self.neighbors(u):
                in_deg[v] -= 1
                if in_deg[v] == 0:
                    q.append(v)
        return order if len(order) == self.n_nodes else []

    # ---------------------------------------------------------------------------
    # Shortest paths
    # ---------------------------------------------------------------------------

    def dijkstra(self, source: int) -> Dict[int, float]:
        dist = {i: float('inf') for i in range(self.n_nodes)}
        dist[source] = 0.0
        pq = [(0.0, source)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v, w in self.neighbors(u):
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return dist

    def bellman_ford(self, source: int) -> Tuple[Dict[int, float], bool]:
        dist = {i: float('inf') for i in range(self.n_nodes)}
        dist[source] = 0.0
        for _ in range(self.n_nodes - 1):
            for u, v, w in self._edges:
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
        neg_cycle = any(dist[u] + w < dist[v] for u, v, w in self._edges)
        return dist, neg_cycle

    def floyd_warshall(self) -> List[List[float]]:
        """All-pairs shortest paths — O(n³) — returns 2D list."""
        INF = float('inf')
        n = self.n_nodes
        dist = [[INF] * n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0
        for u, v, w in self._edges:
            dist[u][v] = min(dist[u][v], w)
            if not self.directed:
                dist[v][u] = min(dist[v][u], w)
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        return dist

    # ---------------------------------------------------------------------------
    # Centrality / ranking
    # ---------------------------------------------------------------------------

    def pagerank(self, damping: float = 0.85, max_iter: int = 100,
                 tol: float = 1e-6) -> List[float]:
        """PageRank (Page et al., 1998)."""
        n = self.n_nodes
        pr = [1.0 / n] * n
        A = self.to_adj_matrix()
        # Row-normalize
        for i in range(n):
            row_sum = sum(A[i])
            if row_sum > 0:
                A[i] = [v / row_sum for v in A[i]]
        AT = _mat_T(A)
        for _ in range(max_iter):
            pr_new = [(1 - damping) / n + damping * sum(AT[i][j] * pr[j] for j in range(n))
                      for i in range(n)]
            if _vec_max_abs_diff(pr_new, pr) < tol:
                break
            pr = pr_new
        return pr

    def degree_centrality(self) -> List[float]:
        n = self.n_nodes
        A = self.to_adj_matrix()
        return [sum(A[i]) / max(n - 1, 1) for i in range(n)]

    def betweenness_centrality(self) -> List[float]:
        """Brandes algorithm for betweenness centrality."""
        n = self.n_nodes
        cb = [0.0] * n
        for s in range(n):
            stack = []
            pred = [[] for _ in range(n)]
            sigma = [0.0] * n
            sigma[s] = 1.0
            dist_arr = [-1] * n
            dist_arr[s] = 0
            q = deque([s])
            while q:
                v = q.popleft()
                stack.append(v)
                for w, _ in self.neighbors(v):
                    if dist_arr[w] < 0:
                        q.append(w)
                        dist_arr[w] = dist_arr[v] + 1
                    if dist_arr[w] == dist_arr[v] + 1:
                        sigma[w] += sigma[v]
                        pred[w].append(v)
            delta = [0.0] * n
            while stack:
                w = stack.pop()
                for v in pred[w]:
                    delta[v] += (sigma[v] / (sigma[w] + 1e-10)) * (1 + delta[w])
                if w != s:
                    cb[w] += delta[w]
        scale = 1.0 / ((n - 1) * (n - 2)) if n > 2 else 1.0
        return [v * scale for v in cb]

    def random_walk(self, source: int, length: int, restart_prob: float = 0.0) -> List[int]:
        """Random walk of given length from source node."""
        import random as _rng
        walk = [source]
        current = source
        for _ in range(length - 1):
            if restart_prob > 0 and _rng.random() < restart_prob:
                current = source
            nbrs = self.neighbors(current)
            if not nbrs:
                break
            # Weighted selection
            total = sum(w for _, w in nbrs)
            r = _rng.random() * total
            cum = 0.0
            for v, w in nbrs:
                cum += w
                if cum >= r:
                    current = v
                    break
            walk.append(current)
        return walk

    def a_star(self, source: int, target: int,
               heuristic=None) -> Tuple[List[int], float]:
        """A* shortest path. heuristic(u, target) defaults to 0."""
        if heuristic is None:
            heuristic = lambda u, t: 0.0
        g_score = {source: 0.0}
        f_score = {source: heuristic(source, target)}
        came_from: Dict[int, int] = {}
        open_set = [(f_score[source], source)]
        closed_set: set = set()
        while open_set:
            _, current = heapq.heappop(open_set)
            if current == target:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(source)
                return list(reversed(path)), g_score[target]
            if current in closed_set:
                continue
            closed_set.add(current)
            for neighbor, w in self.neighbors(current):
                if neighbor in closed_set:
                    continue
                tentative = g_score.get(current, float('inf')) + w
                if tentative < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative
                    f_score[neighbor] = tentative + heuristic(neighbor, target)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return [], float('inf')

    # ---------------------------------------------------------------------------
    # Minimum spanning tree
    # ---------------------------------------------------------------------------

    def kruskal_mst(self) -> List[Tuple[int, int, float]]:
        """Kruskal's MST via Union-Find."""
        parent = list(range(self.n_nodes))
        rank = [0] * self.n_nodes

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1
            return True

        edges_sorted = sorted(self._edges, key=lambda e: e[2])
        mst = []
        for u, v, w in edges_sorted:
            if union(u, v):
                mst.append((u, v, w))
        return mst

    # ---------------------------------------------------------------------------
    # Spectral / GNN utilities
    # ---------------------------------------------------------------------------

    def normalized_laplacian(self) -> List[List[float]]:
        """Symmetric normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}."""
        n = self.n_nodes
        A = self.to_adj_matrix()
        d = [sum(A[i]) for i in range(n)]
        d_inv_sqrt = [1.0 / (math.sqrt(d[i]) + 1e-10) for i in range(n)]
        # D^{-1/2} A D^{-1/2}
        norm_A = [[A[i][j] * d_inv_sqrt[i] * d_inv_sqrt[j] for j in range(n)] for i in range(n)]
        I = _eye(n)
        return [[I[i][j] - norm_A[i][j] for j in range(n)] for i in range(n)]

    def spectral_embedding(self, k: int = 2) -> List[List[float]]:
        """Return first k non-trivial eigenvectors of normalized Laplacian (power iteration)."""
        L = self.normalized_laplacian()
        n = self.n_nodes
        # Simple power iteration for smallest eigenvectors via deflation
        vecs = []
        for _ in range(min(k + 1, n)):
            # Random init
            v = [_random.gauss(0, 1) for _ in range(n)]
            # Orthogonalize against found vectors
            for u in vecs:
                dot = sum(v[i] * u[i] for i in range(n))
                v = [v[i] - dot * u[i] for i in range(n)]
            norm = math.sqrt(sum(x*x for x in v)) or 1.0
            v = [x / norm for x in v]
            # Power iteration on (max_eig * I - L) to find smallest eigenvectors
            # First estimate max eigenvalue
            for _it in range(50):
                v_new = _mat_vec(L, v)
                norm = math.sqrt(sum(x*x for x in v_new)) or 1.0
                v = [x / norm for x in v_new]
            vecs.append(v)
        # Skip first trivial eigenvector, return next k
        result_vecs = vecs[1:k+1] if len(vecs) > 1 else vecs[:k]
        # Return as n x k (rows = nodes)
        actual_k = len(result_vecs)
        return [[result_vecs[j][i] for j in range(actual_k)] for i in range(n)]


# ---------------------------------------------------------------------------
# GraphData — PyG-compatible data container
# ---------------------------------------------------------------------------

class GraphData:
    """Graph data container for GNN training."""

    def __init__(self, x=None, edge_index=None, edge_attr=None, y=None,
                 n_nodes: Optional[int] = None):
        def _to_float2d(arr):
            if arr is None: return None
            if isinstance(arr, list):
                if arr and isinstance(arr[0], list):
                    return [[float(v) for v in row] for row in arr]
                return [[float(v)] for v in arr]
            if hasattr(arr, 'tolist'):
                v = arr.tolist()
                if v and isinstance(v[0], list):
                    return [[float(x) for x in row] for row in v]
                return [[float(x)] for x in v]
            return arr

        def _to_int2d(arr):
            if arr is None: return None
            if isinstance(arr, list):
                if arr and isinstance(arr[0], list):
                    return [[int(v) for v in row] for row in arr]
                return [int(v) for v in arr]
            if hasattr(arr, 'tolist'):
                v = arr.tolist()
                if v and isinstance(v[0], list):
                    return [[int(x) for x in row] for row in v]
                return [int(x) for x in v]
            return arr

        self.x = _to_float2d(x)
        self.edge_index = _to_int2d(edge_index)
        self.edge_attr = _to_float2d(edge_attr)
        self.y = _to_int2d(y)
        if n_nodes is not None:
            self.n_nodes = n_nodes
        elif self.x is not None:
            self.n_nodes = len(self.x)
        else:
            self.n_nodes = 0

    def to_adj(self, add_self_loops: bool = False) -> List[List[float]]:
        n = self.n_nodes
        A = _zeros2d(n, n)
        if self.edge_index is not None:
            if self.edge_index and isinstance(self.edge_index[0], list):
                # shape (2, n_edges)
                srcs, dsts = self.edge_index[0], self.edge_index[1]
            else:
                srcs = [e[0] for e in self.edge_index]
                dsts = [e[1] for e in self.edge_index]
            for i, (s, d) in enumerate(zip(srcs, dsts)):
                w = 1.0
                if self.edge_attr is not None and i < len(self.edge_attr):
                    ea = self.edge_attr[i]
                    w = ea[0] if isinstance(ea, list) else float(ea)
                A[s][d] = w
        if add_self_loops:
            for i in range(n):
                A[i][i] = 1.0
        return A

    def gcn_normalize(self) -> List[List[float]]:
        """GCN-normalized adjacency: D^{-1/2} (A + I) D^{-1/2}."""
        A = self.to_adj(add_self_loops=True)
        n = self.n_nodes
        d = [sum(A[i]) for i in range(n)]
        d_inv_sqrt = [1.0 / (math.sqrt(d[i]) + 1e-10) for i in range(n)]
        return [[A[i][j] * d_inv_sqrt[i] * d_inv_sqrt[j] for j in range(n)] for i in range(n)]


def batch_graphs(graphs: List[GraphData]) -> GraphData:
    """Batch multiple GraphData objects into one large graph (PyG-style)."""
    all_x, all_ei, all_y = [], [], []
    offset = 0
    for g in graphs:
        if g.x is not None:
            all_x.extend(g.x)
        if g.edge_index is not None:
            if g.edge_index and isinstance(g.edge_index[0], list):
                srcs = [s + offset for s in g.edge_index[0]]
                dsts = [d + offset for d in g.edge_index[1]]
                all_ei.append([srcs, dsts])
            else:
                shifted = [[e[0] + offset, e[1] + offset] for e in g.edge_index]
                all_ei.extend(shifted)
        if g.y is not None:
            all_y.extend(g.y if isinstance(g.y, list) else [g.y])
        offset += g.n_nodes
    x = all_x if all_x else None
    if all_ei:
        if all_ei and isinstance(all_ei[0], list) and len(all_ei[0]) == 2 and isinstance(all_ei[0][0], list):
            srcs = [s for ei in all_ei for s in ei[0]]
            dsts = [d for ei in all_ei for d in ei[1]]
            ei = [srcs, dsts]
        else:
            ei = all_ei
    else:
        ei = None
    y = all_y if all_y else None
    return GraphData(x=x, edge_index=ei, y=y, n_nodes=offset)


def negative_sampling(edge_index, n_nodes: int,
                      n_negative: Optional[int] = None):
    """Sample random negative (non-existing) edges."""
    if edge_index and isinstance(edge_index[0], list):
        srcs, dsts = edge_index[0], edge_index[1]
        existing = set(zip(srcs, dsts))
        n_neg = n_negative or len(srcs)
    else:
        existing = set(tuple(e) for e in edge_index) if edge_index else set()
        n_neg = n_negative or len(existing)
    neg = []
    while len(neg) < n_neg:
        u = _random.randrange(n_nodes)
        v = _random.randrange(n_nodes)
        if u != v and (u, v) not in existing:
            neg.append((u, v))
    return [[u for u, v in neg], [v for u, v in neg]]


def add_self_loops(edge_index, n_nodes: int):
    self_loops = list(range(n_nodes))
    if edge_index and isinstance(edge_index[0], list):
        return [edge_index[0] + self_loops, edge_index[1] + self_loops]
    loops = [[i, i] for i in range(n_nodes)]
    return (edge_index or []) + loops


__all__ = [
    'Graph', 'GraphData',
    'batch_graphs', 'negative_sampling', 'add_self_loops',
]
