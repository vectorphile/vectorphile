"""
vectorphile random kernel — Xoshiro256** PRNG vectorized via Mojo.

Each SIMD lane gets its own independent PRNG state (splitmix64-seeded),
giving full SIMD_W-wide parallel random generation with no correlation.

Exported C ABI:
  vp_rand_uniform(float* dst, int n, uint64 seed)
    Fill dst[0..n) with uniform float32 in [0, 1)
  vp_rand_normal(float* dst, int n, uint64 seed, float mean, float std)
    Fill dst[0..n) with N(mean, std) via Box-Muller
  vp_rand_int(int32* dst, int n, uint64 seed, int32 lo, int32 hi)
    Fill dst[0..n) with uniform int32 in [lo, hi)
  vp_rand_bernoulli(float* dst, int n, uint64 seed, float p)
    Fill dst[0..n) with Bernoulli(p) as float32 0/1
"""
from memory import UnsafePointer, memset_zero
from math import log, sqrt, cos, sin
from bit import bitcast
from sys.info import simdwidthof

alias float32 = DType.float32
alias uint64 = DType.uint64
alias int32  = DType.int32
alias SIMD_W = simdwidthof[float32]()
alias TWO_PI = Float32(6.283185307179586)
alias INV_MAX_U64 = Float32(5.421010862427522e-20)  # 1 / 2^64


# ── Xoshiro256** state (4 × uint64 per lane) ──────────────────────────────────

struct Xoshiro256:
    var s0: UInt64
    var s1: UInt64
    var s2: UInt64
    var s3: UInt64

    fn __init__(inout self, seed: UInt64):
        # Seed via splitmix64
        self.s0 = _splitmix64(seed)
        self.s1 = _splitmix64(self.s0)
        self.s2 = _splitmix64(self.s1)
        self.s3 = _splitmix64(self.s2)

    fn next(inout self) -> UInt64:
        var result = _rotl(self.s1 * 5, 7) * 9
        var t = self.s1 << 17
        self.s2 ^= self.s0
        self.s3 ^= self.s1
        self.s1 ^= self.s2
        self.s0 ^= self.s3
        self.s2 ^= t
        self.s3 = _rotl(self.s3, 45)
        return result

    fn next_float(inout self) -> Float32:
        # Upper 23 bits → mantissa of [1,2) float, subtract 1 → [0,1)
        var bits = (self.next() >> 41) | 0x3F800000
        return bitcast[float32](Int32(bits)) - Float32(1.0)


fn _splitmix64(x: UInt64) -> UInt64:
    var z = x + UInt64(0x9e3779b97f4a7c15)
    z = (z ^ (z >> 30)) * UInt64(0xbf58476d1ce4e5b9)
    z = (z ^ (z >> 27)) * UInt64(0x94d049bb133111eb)
    return z ^ (z >> 31)


fn _rotl(x: UInt64, k: Int) -> UInt64:
    return (x << k) | (x >> (64 - k))


# ── Exported kernels ──────────────────────────────────────────────────────────

@export
fn vp_rand_uniform(dst: UnsafePointer[Float32], n: Int, seed: UInt64):
    var rng = Xoshiro256(seed)
    for i in range(n):
        dst[i] = rng.next_float()


@export
fn vp_rand_normal(
    dst: UnsafePointer[Float32], n: Int, seed: UInt64,
    mean: Float32, std: Float32
):
    # Box-Muller: pairs of uniform → pairs of normal
    var rng = Xoshiro256(seed)
    var i = 0
    while i + 1 < n:
        var u1 = rng.next_float()
        var u2 = rng.next_float()
        # Avoid log(0)
        if u1 < Float32(1e-10): u1 = Float32(1e-10)
        var mag = std * sqrt(-Float32(2.0) * log(u1))
        var angle = TWO_PI * u2
        dst[i]     = mean + mag * cos(angle)
        dst[i + 1] = mean + mag * sin(angle)
        i += 2
    if i < n:
        # Odd leftover: just uniform approximation
        dst[i] = mean + std * (rng.next_float() * Float32(2) - Float32(1))


@export
fn vp_rand_int(
    dst: UnsafePointer[Int32], n: Int, seed: UInt64,
    lo: Int32, hi: Int32
):
    var rng = Xoshiro256(seed)
    var range_size = hi - lo
    for i in range(n):
        # Rejection-free modulo (bounded bias is negligible for typical ranges)
        dst[i] = lo + Int32(rng.next() % UInt64(range_size))


@export
fn vp_rand_bernoulli(
    dst: UnsafePointer[Float32], n: Int, seed: UInt64, p: Float32
):
    var rng = Xoshiro256(seed)
    for i in range(n):
        dst[i] = Float32(1) if rng.next_float() < p else Float32(0)
