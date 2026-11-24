# Building the Mojo kernels

## Prerequisites
- [Mojo SDK](https://docs.modular.com/mojo/manual/get-started/) ≥ 24.x
- AVX2-capable CPU (or ARM NEON — the kernels auto-detect SIMD width)

## Compile

```bash
# Linux / macOS
mojo build vectorphile/mojo/kernels/ops.mojo \
    -o vectorphile/mojo/libvp_mojo.so \
    --target native           # optimize for host CPU
```

```powershell
# Windows (PowerShell, requires Mojo for Windows preview)
mojo build vectorphile/mojo/kernels/ops.mojo `
    -o vectorphile/mojo/libvp_mojo.dll `
    --target native
```

## Verify

```python
import vectorphile.mojo as vm
print(vm.mojo_backend_info())
# "Mojo (libvp_mojo at ...)" if compiled, "NumPy fast path" otherwise
```

## What gets accelerated

| Operation           | Mojo speedup vs pure Python | Notes                              |
|---------------------|-----------------------------|------------------------------------|
| matmul(M,K,N)       | 20–80×                      | Tiled SIMD, AVX2/AVX-512          |
| flash_attention_fwd | 15–60×                      | O(N·d) tiled, SIMD inner loop     |
| layer_norm          | 10–30×                      | Fused mean+var+scale in one pass   |
| rms_norm            | 10–25×                      | Fused                              |
| silu / gelu         | 8–20×                       | Vectorised, no temporaries         |
| softmax             | 8–15×                       | Vectorised, numerically stable     |
| conv2d              | 10–40×                      | im2col + GEMM, tiled BLAS-style    |

Speedups measured on Intel i9-13900K vs CPython loop baseline.
NumPy fallback is still BLAS-accelerated and competitive with PyTorch CPU.
