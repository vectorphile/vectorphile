"""
vectorphile lazy evaluation + kernel fusion engine.

Records element-wise operations into a DAG and materializes them as a
single fused kernel, eliminating intermediate memory allocations and
reducing memory-bandwidth pressure.

Usage (transparent — users see this via NDArray's deferred operations):

    x = vp.randn(1024, 1024)
    # These don't execute immediately:
    y = x.sigmoid()   # Op(sigmoid, x)
    z = y.tanh()      # Op(tanh, Op(sigmoid, x))
    out = z.relu()    # Op(relu, Op(tanh, Op(sigmoid, x)))
    # materialized as one loop: for i: out[i] = relu(tanh(sigmoid(x[i])))

Fusion threshold: chains of ≥ 3 element-wise ops trigger fusion.
Fusion is currently CPU-only; GPU paths emit CUDA C code (Phase 5).
"""
from __future__ import annotations
import array
from typing import Optional, Callable, Tuple, List

# ── Op types ──────────────────────────────────────────────────────────────────
# Fused ops are strictly element-wise unary/binary with no shape change.
_FUSEABLE_UNARY = frozenset([
    "relu", "sigmoid", "tanh", "exp", "log", "sqrt", "abs", "neg",
    "sign", "erf", "gelu", "leaky_relu",
])

# Op code integers — must match VP_CHAIN_* constants in vp_core.c
_CHAIN_OPCODES: dict = {
    "relu":    0,
    "sigmoid": 1,
    "tanh":    2,
    "exp":     3,
    "log":     4,
    "sqrt":    5,
    "abs":     6,
    "neg":     7,
    "gelu":    8,
    "silu":    9,
    "sign":    10,
    "erf":     11,
}
_FUSEABLE_BINARY = frozenset([
    "add", "sub", "mul", "div", "maximum", "minimum",
])

class _LazyOp:
    """A deferred operation node in the fusion DAG."""
    __slots__ = ("op", "args", "kwargs", "shape", "dtype")

    def __init__(self, op: str, args: tuple, kwargs: dict,
                 shape: tuple, dtype: str = "f"):
        self.op     = op
        self.args   = args   # each arg is either a _LazyOp or an array.array
        self.kwargs = kwargs
        self.shape  = shape
        self.dtype  = dtype

    @property
    def n(self):
        n = 1
        for s in self.shape: n *= s
        return n

    def depth(self) -> int:
        d = 0
        for a in self.args:
            if isinstance(a, _LazyOp):
                d = max(d, a.depth() + 1)
        return d

    def should_fuse(self) -> bool:
        return self.op in _FUSEABLE_UNARY and self.depth() >= 2

# ── Python-level fused executor ───────────────────────────────────────────────
# For now we emit Python loops; Phase 4 will LLVM-JIT this.

def _scalar_op(op: str, x: float, y: float = 0.0) -> float:
    import math
    if op == "relu":     return max(0.0, x)
    if op == "sigmoid":  return 1.0 / (1.0 + math.exp(-x))
    if op == "tanh":     return math.tanh(x)
    if op == "exp":      return math.exp(x)
    if op == "log":      return math.log(x)
    if op == "sqrt":     return math.sqrt(x)
    if op == "abs":      return abs(x)
    if op == "neg":      return -x
    if op == "erf":      return math.erf(x)
    if op == "gelu":     s = 1.0 / (1.0 + math.exp(-1.702 * x)); return x * s
    if op == "add":      return x + y
    if op == "sub":      return x - y
    if op == "mul":      return x * y
    if op == "div":      return x / y if y != 0.0 else 0.0
    if op == "maximum":  return max(x, y)
    if op == "minimum":  return min(x, y)
    return x

def execute_lazy(node: _LazyOp) -> array.array:
    """
    Evaluate a lazy op tree.
    If the chain is deep enough, fuse element-wise ops into a single C pass.
    Otherwise fall through to the C extension kernels (one op at a time).
    """
    if not node.should_fuse():
        return _materialize(node)

    # Walk the unary chain to collect op names and reach the base array.
    chain: list = []
    cur = node
    while isinstance(cur, _LazyOp) and cur.op in _FUSEABLE_UNARY:
        chain.append(cur.op)
        cur = cur.args[0]
    # cur is now the innermost operand (array.array or non-fuseable node)
    base_arr = _materialize(cur) if isinstance(cur, _LazyOp) else cur
    chain.reverse()  # innermost op first

    # Map op names to integer codes for the C kernel
    op_codes = [_CHAIN_OPCODES.get(op, -1) for op in chain]
    if any(c < 0 for c in op_codes):
        # Fallback: some op in the chain has no C opcode — use scalar loop
        n = len(base_arr)
        dst = array.array("f", bytes(n * 4))
        for i in range(n):
            v = base_arr[i]
            for op in chain:
                v = _scalar_op(op, v)
            dst[i] = v
        return dst

    # Fast path: single C AVX2 pass (no intermediate allocations)
    from vectorphile._core import _vp_core as C
    n = len(base_arr)
    dst = array.array("f", bytes(n * 4))
    ops_arr = array.array("i", op_codes)
    C.fused_chain_f32(base_arr, ops_arr, dst)
    _stats.fused_chains += 1
    _stats.saved_allocs += len(chain) - 1  # each intermediate that was skipped
    return dst

def _materialize(node: _LazyOp) -> array.array:
    """Eager materialization via C extension."""
    from vectorphile._core import _vp_core as C
    from vectorphile._core.arr_pool import alloc

    resolved = [
        _materialize(a) if isinstance(a, _LazyOp) else a
        for a in node.args
    ]
    n = node.n
    dst = alloc(n)

    op = node.op
    if op in _FUSEABLE_UNARY:
        fn = getattr(C, f"{op}_f32", None) or getattr(C, f"{op}_f_f32", None)
        if fn:
            fn(resolved[0], dst)
            return dst
    if op in _FUSEABLE_BINARY:
        fn = getattr(C, f"{op}_f32", None)
        if fn:
            fn(resolved[0], resolved[1], dst)
            return dst

    raise NotImplementedError(f"Cannot materialize op: {op!r}")


# ── Fusion statistics ─────────────────────────────────────────────────────────
class FusionStats:
    fused_chains: int = 0
    saved_allocs: int = 0

_stats = FusionStats()

def reset_stats():
    _stats.fused_chains = 0
    _stats.saved_allocs = 0

def get_stats() -> dict:
    return {"fused_chains": _stats.fused_chains, "saved_allocs": _stats.saved_allocs}
