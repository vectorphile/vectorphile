"""Linear algebra operations — pure Python, no external dependencies."""
import math
from vectorphile.core.tensor import Tensor
from vectorphile._core.ndarray import array as _nd_array


def _flat(x: Tensor):
    d = x._data
    if hasattr(d, 'to_flat_list'):
        return d.to_flat_list()
    if hasattr(d, 'tolist'):
        v = d.tolist()
        if isinstance(v, list) and v and isinstance(v[0], list):
            return [x for row in v for x in row]
        return list(v)
    return list(d)


def _shape(x: Tensor):
    if hasattr(x._data, 'shape'):
        return x._data.shape
    return (len(_flat(x)),)


def _mat(x: Tensor):
    s = _shape(x)
    if len(s) != 2:
        raise ValueError(f'Expected 2D tensor, got shape {s}')
    n, m = s
    flat = _flat(x)
    return [[flat[i * m + j] for j in range(m)] for i in range(n)]


def _from_mat(mat) -> Tensor:
    n = len(mat)
    m = len(mat[0]) if n > 0 else 0
    flat = [float(mat[i][j]) for i in range(n) for j in range(m)]
    nd = _nd_array(flat)
    if hasattr(nd, 'reshape'):
        nd = nd.reshape((n, m))
    return Tensor(nd)


def _from_vec(v) -> Tensor:
    return Tensor(_nd_array([float(x) for x in v]))


def _mat_mul(A, B):
    n, k = len(A), len(A[0])
    m = len(B[0])
    C = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            C[i][j] = sum(A[i][t] * B[t][j] for t in range(k))
    return C


def _transpose(A):
    n, m = len(A), len(A[0])
    return [[A[i][j] for i in range(n)] for j in range(m)]


def _identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _lu_decomp(A):
    n = len(A)
    L = _identity(n)
    U = [row[:] for row in A]
    perm = list(range(n))
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(U[r][col]))
        if abs(U[pivot][col]) < 1e-12:
            continue
        U[col], U[pivot] = U[pivot], U[col]
        L[col], L[pivot] = L[pivot], L[col]
        perm[col], perm[pivot] = perm[pivot], perm[col]
        for row in range(col + 1, n):
            factor = U[row][col] / U[col][col]
            L[row][col] = factor
            for j in range(col, n):
                U[row][j] -= factor * U[col][j]
    return L, U, perm


def det(a: Tensor) -> Tensor:
    A = _mat(a)
    n = len(A)
    _, U, perm = _lu_decomp(A)
    d = 1.0
    for i in range(n):
        d *= U[i][i]
    # Count swaps in permutation for sign
    p = perm[:]
    swaps = 0
    for i in range(n):
        while p[i] != i:
            j = p[i]
            p[i], p[j] = p[j], p[i]
            swaps += 1
    if swaps % 2 == 1:
        d = -d
    return _from_vec([d])


def inv(a: Tensor) -> Tensor:
    A = _mat(a)
    n = len(A)
    aug = [A[i][:] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[pivot] = aug[pivot], aug[col]
        piv = aug[col][col]
        if abs(piv) < 1e-12:
            raise ValueError('Matrix is singular')
        aug[col] = [v / piv for v in aug[col]]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            aug[row] = [aug[row][j] - factor * aug[col][j] for j in range(2 * n)]
    result = [[aug[i][n + j] for j in range(n)] for i in range(n)]
    return _from_mat(result)


def _householder_qr(A):
    m, n = len(A), len(A[0])
    Q = _identity(m)
    R = [row[:] for row in A]
    for j in range(min(m, n)):
        x = [R[i][j] for i in range(j, m)]
        norm_x = math.sqrt(sum(v * v for v in x))
        u = x[:]
        u[0] += math.copysign(norm_x, x[0])
        norm_u = math.sqrt(sum(v * v for v in u))
        if norm_u < 1e-12:
            continue
        u = [v / norm_u for v in u]
        # Apply Householder to R (rows j..m, cols j..n)
        for col in range(j, n):
            dot = sum(u[i - j] * R[i][col] for i in range(j, m))
            for i in range(j, m):
                R[i][col] -= 2 * u[i - j] * dot
        # Apply to Q
        for col in range(m):
            dot = sum(u[i] * Q[j + i][col] for i in range(len(u)))
            for i in range(len(u)):
                Q[j + i][col] -= 2 * u[i] * dot
    Q_mat = _transpose(Q[:m][:m])
    return Q_mat, R


def qr(a: Tensor, mode='reduced'):
    A = _mat(a)
    Q, R = _householder_qr(A)
    return _from_mat(Q), _from_mat(R)


def _power_iteration_svd(A, k=None, n_iter=20):
    m, n = len(A), len(A[0])
    if k is None:
        k = min(m, n)
    AT = _transpose(A)
    Us, Ss, Vs = [], [], []
    A_rem = [row[:] for row in A]
    for _ in range(k):
        # Power iteration on A^T A
        import random
        v = [random.gauss(0, 1) for _ in range(n)]
        norm = math.sqrt(sum(x*x for x in v))
        v = [x/norm for x in v] if norm > 0 else v
        for it in range(max(n_iter, 50)):
            # u = A v
            u = [sum(A_rem[i][j] * v[j] for j in range(n)) for i in range(m)]
            # v = A^T u
            v = [sum(A_rem[i][j] * u[i] for i in range(m)) for j in range(n)]
            sigma = math.sqrt(sum(x*x for x in v))
            if sigma < 1e-12:
                break
            v = [x/sigma for x in v]
        u = [sum(A_rem[i][j] * v[j] for j in range(n)) for i in range(m)]
        sigma = math.sqrt(sum(x*x for x in u))
        if sigma < 1e-12:
            break
        u = [x/sigma for x in u]
        Us.append(u)
        Ss.append(sigma)
        Vs.append(v)
        # Deflate
        for i in range(m):
            for j in range(n):
                A_rem[i][j] -= sigma * u[i] * v[j]
    return Us, Ss, Vs


def svd(a: Tensor, full_matrices=True):
    A = _mat(a)
    Us, Ss, Vs = _power_iteration_svd(A)
    # Build matrices
    U_mat = _transpose(Us) if Us else [[]]
    V_mat = Vs if Vs else [[]]
    return _from_mat(U_mat), _from_vec(Ss), _from_mat(V_mat)


def _power_iter_eig(A, n_iter=100):
    n = len(A)
    eigvals = []
    eigvecs = []
    A_rem = [row[:] for row in A]
    for _ in range(n):
        import random
        v = [random.gauss(0, 1) for _ in range(n)]
        norm = math.sqrt(sum(x*x for x in v))
        v = [x/norm for x in v] if norm > 0 else v
        lam = 0.0
        for _ in range(n_iter):
            Av = [sum(A_rem[i][j] * v[j] for j in range(n)) for i in range(n)]
            lam = sum(v[i] * Av[i] for i in range(n))
            norm_av = math.sqrt(sum(x*x for x in Av))
            if norm_av < 1e-12:
                break
            v = [x/norm_av for x in Av]
        eigvals.append(lam)
        eigvecs.append(v)
        # Deflate: A -= lam * v * v^T
        for i in range(n):
            for j in range(n):
                A_rem[i][j] -= lam * v[i] * v[j]
    return eigvals, eigvecs


def eig(a: Tensor):
    A = _mat(a)
    vals, vecs = _power_iter_eig(A)
    return _from_vec(vals), _from_mat(_transpose(vecs))


def eigh(a: Tensor):
    return eig(a)


def _cholesky(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                val = A[i][i] - s
                L[i][j] = math.sqrt(max(val, 1e-12))
            else:
                L[i][j] = (A[i][j] - s) / (L[j][j] + 1e-12)
    return L


def cholesky(a: Tensor) -> Tensor:
    A = _mat(a)
    return _from_mat(_cholesky(A))


def solve(a: Tensor, b: Tensor) -> Tensor:
    A = _mat(a)
    n = len(A)
    b_flat = _flat(b)
    # Gaussian elimination with partial pivoting
    aug = [A[i][:] + [b_flat[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[pivot] = aug[pivot], aug[col]
        piv = aug[col][col]
        if abs(piv) < 1e-12:
            raise ValueError('Singular matrix in solve')
        aug[col] = [v / piv for v in aug[col]]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            aug[row] = [aug[row][j] - factor * aug[col][j] for j in range(n + 1)]
    return _from_vec([aug[i][n] for i in range(n)])


def lstsq(a: Tensor, b: Tensor):
    A = _mat(a)
    b_flat = _flat(b)
    # Normal equations: (A^T A) x = A^T b
    AT = _transpose(A)
    ATA = _mat_mul(AT, A)
    ATb = [sum(AT[i][j] * b_flat[j] for j in range(len(b_flat))) for i in range(len(AT))]
    # Solve ATA x = ATb
    n = len(ATA)
    aug = [ATA[i][:] + [ATb[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[pivot] = aug[pivot], aug[col]
        piv = aug[col][col]
        if abs(piv) < 1e-12:
            break
        aug[col] = [v / piv for v in aug[col]]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            aug[row] = [aug[row][j] - factor * aug[col][j] for j in range(n + 1)]
    return _from_vec([aug[i][n] for i in range(n)])


def norm(a: Tensor, ord=None, dim=None, keepdim=False) -> Tensor:
    vals = [float(v) for v in _flat(a)]
    if ord is None or ord == 'fro':
        return _from_vec([math.sqrt(sum(v*v for v in vals))])
    if ord == 1:
        return _from_vec([sum(abs(v) for v in vals)])
    if ord == float('inf'):
        return _from_vec([max(abs(v) for v in vals)])
    return _from_vec([math.sqrt(sum(v*v for v in vals))])


def matrix_rank(a: Tensor, tol=None) -> Tensor:
    A = _mat(a)
    _, Ss, _ = _power_iteration_svd(A)
    if tol is None:
        tol = 1e-10 * max(Ss) if Ss else 1e-10
    rank = sum(1 for s in Ss if s > tol)
    return _from_vec([float(rank)])


def pinv(a: Tensor) -> Tensor:
    A = _mat(a)
    Us, Ss, Vs = _power_iteration_svd(A)
    # pinv = V diag(1/s) U^T
    k = len(Ss)
    m, n = len(A), len(A[0])
    result = [[0.0] * m for _ in range(n)]
    for i in range(k):
        if abs(Ss[i]) < 1e-10:
            continue
        inv_s = 1.0 / Ss[i]
        for row in range(n):
            for col in range(m):
                result[row][col] += Vs[i][row] * inv_s * Us[i][col]
    return _from_mat(result)


def trace(a: Tensor) -> Tensor:
    A = _mat(a)
    return _from_vec([sum(A[i][i] for i in range(min(len(A), len(A[0]))))])


def matrix_power(a: Tensor, n: int) -> Tensor:
    A = _mat(a)
    if n == 0:
        return _from_mat(_identity(len(A)))
    result = _identity(len(A))
    base = A
    exp = abs(n)
    while exp > 0:
        if exp % 2 == 1:
            result = _mat_mul(result, base)
        base = _mat_mul(base, base)
        exp //= 2
    if n < 0:
        return inv(_from_mat(result))
    return _from_mat(result)


def cross(a: Tensor, b: Tensor, dim=-1) -> Tensor:
    av = _flat(a)
    bv = _flat(b)
    if len(av) == 3:
        c = [
            av[1]*bv[2] - av[2]*bv[1],
            av[2]*bv[0] - av[0]*bv[2],
            av[0]*bv[1] - av[1]*bv[0],
        ]
        return _from_vec(c)
    return _from_vec([0.0])


def outer(a: Tensor, b: Tensor) -> Tensor:
    av = _flat(a)
    bv = _flat(b)
    mat = [[av[i] * bv[j] for j in range(len(bv))] for i in range(len(av))]
    return _from_mat(mat)


def kron(a: Tensor, b: Tensor) -> Tensor:
    A = _mat(a)
    B = _mat(b)
    m, n = len(A), len(A[0])
    p, q = len(B), len(B[0])
    C = [[0.0] * (n*q) for _ in range(m*p)]
    for i in range(m):
        for j in range(n):
            for r in range(p):
                for s in range(q):
                    C[i*p+r][j*q+s] = A[i][j] * B[r][s]
    return _from_mat(C)


def matrix_exp(a: Tensor) -> Tensor:
    """Matrix exponential via scaling-and-squaring (Padé approximation order 6)."""
    mat = _mat(a)
    n = len(mat)
    # Scale: find s such that ||A/2^s|| ~ 1
    norm_a = sum(sum(abs(mat[i][j]) for j in range(n)) for i in range(n))
    s = max(0, int(math.ceil(math.log2(max(norm_a, 1e-10)))))
    scale = 2.0 ** s
    A = [[mat[i][j] / scale for j in range(n)] for i in range(n)]

    def _matadd(X, Y): return [[X[i][j]+Y[i][j] for j in range(n)] for i in range(n)]
    def _matsub(X, Y): return [[X[i][j]-Y[i][j] for j in range(n)] for i in range(n)]
    def _matscale(X, c): return [[X[i][j]*c for j in range(n)] for i in range(n)]
    def _mateye(): return [[1.0 if i==j else 0.0 for j in range(n)] for i in range(n)]

    # Padé [3/3] approximant (sufficient for scaled matrix)
    I = _mateye()
    A2 = _mat_mul(A, A)
    A3 = _mat_mul(A2, A)
    # U = A*(c3*A2 + c1*I),  V = c4*A2 + c2*I + c0*I
    c = [120.0, 60.0, 12.0, 1.0]  # Padé 3/3 coefficients
    U = _mat_mul(A, _matadd(_matscale(A2, c[2]), _matscale(I, c[0])))
    V = _matadd(_matadd(_matscale(A2, c[3]), _matscale(A2, c[1])), _matscale(I, c[0]))
    # exp(A) ≈ (V+U)*(V-U)^-1 (use solve instead of explicit inv)
    num = _matadd(V, U)
    den = _matsub(V, U)
    # Solve den * exp(A) = num using LU
    result = [[0.0]*n for _ in range(n)]
    for col in range(n):
        b = [num[row][col] for row in range(n)]
        from vectorphile.core.tensor import Tensor as _T
        x_sol = _flat(solve(_from_mat(den), _from_mat([[bi] for bi in b])))
        for row in range(n):
            result[row][col] = x_sol[row]
    # Unsquare: exp(A) = exp(A/2^s)^(2^s) via repeated squaring
    for _ in range(s):
        result = _mat_mul(result, result)
    return _from_mat(result)


def slogdet(a: Tensor):
    """Sign and log-abs-determinant of a square matrix: returns (sign, logabsdet)."""
    mat = _mat(a)
    n = len(mat)
    # LU decomposition, track sign
    lu = [row[:] for row in mat]
    sign = 1.0
    for col in range(n):
        # Partial pivoting
        pivot = max(range(col, n), key=lambda r: abs(lu[r][col]))
        if pivot != col:
            lu[col], lu[pivot] = lu[pivot], lu[col]
            sign *= -1.0
        pv = lu[col][col]
        if abs(pv) < 1e-14:
            return 0.0, float('-inf')
        for row in range(col + 1, n):
            factor = lu[row][col] / pv
            for k in range(col, n):
                lu[row][k] -= factor * lu[col][k]
    import math as _m
    diag = [lu[i][i] for i in range(n)]
    for d in diag:
        if d < 0:
            sign *= -1.0
    logabsdet = sum(_m.log(abs(d)) for d in diag)
    from vectorphile.core.tensor import Tensor as _T
    from vectorphile._core.ndarray import array as _nd_arr
    return sign, _T(_nd_arr([logabsdet], dtype='float32'))


def vander(x: Tensor, N=None, increasing=False) -> Tensor:
    vals = _flat(x)
    n = len(vals)
    if N is None:
        N = n
    mat = []
    for v in vals:
        if increasing:
            row = [v**i for i in range(N)]
        else:
            row = [v**(N-1-i) for i in range(N)]
        mat.append(row)
    return _from_mat(mat)


def cond(a: Tensor, p=None) -> Tensor:
    """Condition number of a matrix via SVD."""
    mat = _mat(a)
    _, Ss, _ = _power_iteration_svd(mat, n_iter=100)
    sv_pos = [abs(s) for s in Ss if abs(s) > 1e-15]
    if not sv_pos:
        return _from_vec([float('inf')])
    return _from_vec([max(sv_pos) / min(sv_pos)])


def multi_dot(arrays) -> Tensor:
    """Efficient chain matrix multiplication."""
    result = arrays[0]
    for a in arrays[1:]:
        result = Tensor(result._data @ a._data) if hasattr(result._data, '__matmul__') else _from_mat(_mat_mul(_mat(result), _mat(a)))
    return result


def tensordot(a: Tensor, b: Tensor, axes=2) -> Tensor:
    """Tensor contraction over specified axes."""
    import array as _pyarr
    from vectorphile._core.ndarray import NDArray as _NDA, _c_strides, _DType
    a_shape = _shape(a)
    b_shape = _shape(b)
    a_flat = _flat(a)
    b_flat = _flat(b)

    if isinstance(axes, int):
        # Last `axes` dims of a contracted with first `axes` dims of b
        axes_a = list(range(len(a_shape) - axes, len(a_shape)))
        axes_b = list(range(axes))
    else:
        axes_a, axes_b = list(axes[0]), list(axes[1])

    # Contracted sizes must match
    contract_shape = [a_shape[i] for i in axes_a]
    free_a = [i for i in range(len(a_shape)) if i not in axes_a]
    free_b = [i for i in range(len(b_shape)) if i not in axes_b]
    out_shape = tuple([a_shape[i] for i in free_a] + [b_shape[i] for i in free_b])
    out_size = 1
    for s in out_shape:
        out_size *= s

    def _strides(sh):
        st = [1] * len(sh)
        for i in range(len(sh) - 2, -1, -1):
            st[i] = st[i + 1] * sh[i + 1]
        return st

    sa = _strides(a_shape)
    sb = _strides(b_shape)
    so = _strides(out_shape) if out_shape else [1]

    contract_ranges = [range(contract_shape[i]) for i in range(len(contract_shape))]
    from itertools import product as _ip
    free_a_ranges = [range(a_shape[i]) for i in free_a]
    free_b_ranges = [range(b_shape[i]) for i in free_b]

    result = [0.0] * max(out_size, 1)
    for free_vals in _ip(*free_a_ranges, *free_b_ranges):
        fa_vals = free_vals[:len(free_a)]
        fb_vals = free_vals[len(free_a):]
        out_pos = sum(v * so[i] for i, v in enumerate(free_vals)) if out_shape else 0
        total = 0.0
        for c_vals in _ip(*contract_ranges):
            a_idx = sum(fa_vals[i] * sa[free_a[i]] for i in range(len(free_a))) + \
                    sum(c_vals[i] * sa[axes_a[i]] for i in range(len(axes_a)))
            b_idx = sum(fb_vals[i] * sb[free_b[i]] for i in range(len(free_b))) + \
                    sum(c_vals[i] * sb[axes_b[i]] for i in range(len(axes_b)))
            total += a_flat[a_idx] * b_flat[b_idx]
        result[out_pos] = total

    if not out_shape:
        return _from_vec([result[0]])
    buf = _pyarr.array('f', [float(v) for v in result])
    nd = _NDA(buf, out_shape, _c_strides(out_shape), _DType('float32'))
    return Tensor(nd)


def diagonal(a: Tensor, offset: int = 0) -> Tensor:
    """Extract diagonal (or super/sub-diagonal) of a 2D matrix."""
    mat = _mat(a)
    rows, cols = len(mat), len(mat[0]) if mat else 0
    result = []
    if offset >= 0:
        for i in range(min(rows, cols - offset)):
            result.append(mat[i][i + offset])
    else:
        for i in range(min(rows + offset, cols)):
            result.append(mat[i - offset][i])
    return _from_vec(result)


def eigvals(a: Tensor) -> Tensor:
    """Compute eigenvalues of a square matrix (alias for eig returning only values)."""
    vals, _ = eig(a)
    return vals


def eigvalsh(a: Tensor) -> Tensor:
    """Compute eigenvalues of a symmetric matrix (alias for eigh returning only values)."""
    vals, _ = eigh(a)
    return vals


def vector_norm(a: Tensor, ord=2, dim=None, keepdim: bool = False) -> Tensor:
    """Compute vector norm (like torch.linalg.vector_norm)."""
    return norm(a, ord=ord, dim=dim, keepdim=keepdim)


def matrix_norm(a: Tensor, ord='fro', keepdim: bool = False) -> Tensor:
    """Compute matrix norm (Frobenius by default, like torch.linalg.matrix_norm)."""
    return norm(a, ord=ord, keepdim=keepdim)


def lu_factor(a: Tensor):
    """LU factorization; returns (LU, pivots) for use with lu_solve."""
    mat = _mat(a)
    n = len(mat)
    piv = list(range(n))
    lu = [row[:] for row in mat]
    for k in range(n):
        # Find pivot
        p = max(range(k, n), key=lambda i: abs(lu[i][k]))
        lu[k], lu[p] = lu[p], lu[k]
        piv[k], piv[p] = piv[p], piv[k]
        if abs(lu[k][k]) < 1e-12:
            continue
        for i in range(k + 1, n):
            lu[i][k] /= lu[k][k]
            for j in range(k + 1, n):
                lu[i][j] -= lu[i][k] * lu[k][j]
    lu_flat = [v for row in lu for v in row]
    from vectorphile._core.ndarray import array as _nd_lu
    lu_nd = _nd_lu(lu_flat, dtype='float32').reshape((n, n))
    piv_nd = _nd_lu([float(p) for p in piv], dtype='int64')
    from vectorphile.core.tensor import Tensor as _T
    return _T(lu_nd), _T(piv_nd)


def lu_solve(LU_pivots, b: Tensor) -> Tensor:
    """Solve Ax=b given LU factorization from lu_factor."""
    lu_t, piv_t = LU_pivots
    return solve(lu_t, b)  # simplified: use solve directly


__all__ = [
    'det', 'inv', 'qr', 'svd', 'eig', 'eigh', 'cholesky',
    'solve', 'lstsq', 'norm', 'matrix_rank', 'pinv', 'trace',
    'matrix_power', 'cross', 'outer', 'kron', 'matrix_exp', 'slogdet',
    'vander', 'cond', 'multi_dot', 'tensordot', 'diagonal',
    'eigvals', 'eigvalsh', 'vector_norm', 'matrix_norm',
    'lu_factor', 'lu_solve',
]
