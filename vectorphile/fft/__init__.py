"""FFT operations — Cooley-Tukey DFT backed by vectorphile's own ComplexNDArray.

Returns Tensor objects whose _data is a ComplexNDArray for complex outputs,
or a plain NDArray for real outputs (rfft magnitudes, fftfreq, etc.).
"""
import math
from vectorphile.core.tensor import Tensor
from vectorphile._core.ndarray import (
    array as _nd_array, zeros as _nd_zeros,
    ComplexNDArray, fft as _nd_fft, ifft as _nd_ifft, fft2 as _nd_fft2,
)


def _to_complex_nd(inp: Tensor):
    """Extract or promote the input to ComplexNDArray."""
    d = inp._data
    if isinstance(d, ComplexNDArray):
        return d
    return ComplexNDArray.from_real(d)


def fft(inp: Tensor, n=None, dim: int = -1, norm=None) -> Tensor:
    """1-D FFT. Returns Tensor backed by ComplexNDArray."""
    cnd = _to_complex_nd(inp)
    if n is not None:
        # Pad or truncate
        sz = cnd.shape[-1] if cnd.ndim > 0 else 1
        if n > sz:
            pad_re = _nd_zeros(cnd.real.shape[:-1] + (n - sz,) if cnd.ndim > 1 else (n - sz,), dtype='float32')
            from vectorphile._core.ndarray import concatenate as _cat
            new_re = _cat([cnd.real, pad_re], axis=-1)
            pad_im = _nd_zeros(pad_re.shape, dtype='float32')
            new_im = _cat([cnd.imag, pad_im], axis=-1)
            cnd = ComplexNDArray(new_re, new_im, cnd.dtype)
        elif n < sz:
            cnd = ComplexNDArray(cnd.real[..., :n], cnd.imag[..., :n], cnd.dtype)
    result = _nd_fft(cnd, axis=dim)
    t = Tensor.__new__(Tensor)
    t._data = result
    t.device = inp.device
    t.requires_grad = False
    t.grad = None; t.grad_fn = None; t._is_leaf = True
    return t


def ifft(inp: Tensor, n=None, dim: int = -1, norm=None) -> Tensor:
    """1-D inverse FFT."""
    cnd = _to_complex_nd(inp)
    result = _nd_ifft(cnd, axis=dim)
    t = Tensor.__new__(Tensor)
    t._data = result
    t.device = inp.device
    t.requires_grad = False
    t.grad = None; t.grad_fn = None; t._is_leaf = True
    return t


def rfft(inp: Tensor, n=None, dim: int = -1, norm=None) -> Tensor:
    """FFT of a real input, returning only the non-redundant half."""
    result_t = fft(inp, n=n, dim=dim, norm=norm)
    cnd = result_t._data  # ComplexNDArray
    if isinstance(cnd, ComplexNDArray):
        sz = cnd.shape[-1] if cnd.ndim > 0 else 1
        half = sz // 2 + 1
        new_re = cnd.real[..., :half] if cnd.ndim > 1 else cnd.real[:half]
        new_im = cnd.imag[..., :half] if cnd.ndim > 1 else cnd.imag[:half]
        result_t._data = ComplexNDArray(new_re, new_im, cnd.dtype)
    return result_t


def irfft(inp: Tensor, n=None, dim: int = -1, norm=None) -> Tensor:
    """Inverse of rfft: reconstructs the real signal."""
    cnd = _to_complex_nd(inp)
    # Mirror the spectrum (Hermitian symmetry)
    sz = cnd.shape[-1] if cnd.ndim > 0 else 1
    n_out = (sz - 1) * 2 if n is None else n

    re_flat = list(cnd.real.to_flat_list())
    im_flat = list(cnd.imag.to_flat_list())
    # Build full symmetric spectrum
    full_re = re_flat + [re_flat[i] for i in range(len(re_flat) - 2, 0, -1)]
    full_im = im_flat + [-im_flat[i] for i in range(len(im_flat) - 2, 0, -1)]
    full_re = full_re[:n_out]
    full_im = full_im[:n_out]
    full_cnd = ComplexNDArray(_nd_array(full_re, dtype='float32'), _nd_array(full_im, dtype='float32'))
    result = _nd_ifft(full_cnd)
    # Return only the real part
    t = Tensor(result.real)
    return t


def fft2(inp: Tensor, s=None, dim=(-2, -1), norm=None) -> Tensor:
    """2-D FFT."""
    cnd = _to_complex_nd(inp)
    result = _nd_fft2(cnd)
    t = Tensor.__new__(Tensor)
    t._data = result
    t.device = inp.device
    t.requires_grad = False
    t.grad = None; t.grad_fn = None; t._is_leaf = True
    return t


def ifft2(inp: Tensor, s=None, dim=(-2, -1), norm=None) -> Tensor:
    """2-D inverse FFT via conjugation trick: ifft2 = conj(fft2(conj(x))) / N."""
    cnd = _to_complex_nd(inp)
    conj_in = cnd.conj()
    out = _nd_fft2(conj_in).conj()
    n = out.size
    result = ComplexNDArray(out.real / float(n), out.imag / float(n), out.dtype)
    t = Tensor.__new__(Tensor)
    t._data = result
    t.device = inp.device
    t.requires_grad = False
    t.grad = None; t.grad_fn = None; t._is_leaf = True
    return t


def fftshift(inp: Tensor, dim=None) -> Tensor:
    """Shift the zero-frequency component to the center."""
    d = inp._data
    if isinstance(d, ComplexNDArray):
        n = d.shape[-1]
        half = n // 2
        new_re = _nd_array(list(d.real.to_flat_list())[half:] + list(d.real.to_flat_list())[:half], dtype='float32')
        new_im = _nd_array(list(d.imag.to_flat_list())[half:] + list(d.imag.to_flat_list())[:half], dtype='float32')
        t = Tensor.__new__(Tensor)
        t._data = ComplexNDArray(new_re, new_im, d.dtype)
        t.device = inp.device
        t.requires_grad = False
        t.grad = None; t.grad_fn = None; t._is_leaf = True
        return t
    vals = list(d.to_flat_list())
    half = len(vals) // 2
    return Tensor(_nd_array(vals[half:] + vals[:half]))


def ifftshift(inp: Tensor, dim=None) -> Tensor:
    """Inverse of fftshift."""
    d = inp._data
    if isinstance(d, ComplexNDArray):
        n = d.shape[-1]
        half = (n + 1) // 2
        new_re = _nd_array(list(d.real.to_flat_list())[half:] + list(d.real.to_flat_list())[:half], dtype='float32')
        new_im = _nd_array(list(d.imag.to_flat_list())[half:] + list(d.imag.to_flat_list())[:half], dtype='float32')
        t = Tensor.__new__(Tensor)
        t._data = ComplexNDArray(new_re, new_im, d.dtype)
        t.device = inp.device
        t.requires_grad = False
        t.grad = None; t.grad_fn = None; t._is_leaf = True
        return t
    vals = list(d.to_flat_list())
    n = len(vals); half = (n + 1) // 2
    return Tensor(_nd_array(vals[half:] + vals[:half]))


def fftfreq(n: int, d: float = 1.0) -> Tensor:
    """DFT sample frequencies."""
    freqs = []
    for i in range(n):
        freqs.append(i / (n * d) if i < (n + 1) // 2 else (i - n) / (n * d))
    return Tensor(_nd_array(freqs))


def rfftfreq(n: int, d: float = 1.0) -> Tensor:
    """DFT sample frequencies for rfft output."""
    half = n // 2 + 1
    return Tensor(_nd_array([i / (n * d) for i in range(half)]))


def fftn(inp: Tensor, s=None, dim=None, norm=None) -> Tensor:
    """N-D FFT — applied repeatedly along each specified dim."""
    # Apply 1-D fft along each dim in sequence
    result = inp
    ndim = len(inp._data.shape) if hasattr(inp._data, 'shape') else 1
    dims = dim if dim is not None else list(range(ndim))
    sizes = s if s is not None else [None] * len(dims)
    for d, n in zip(dims, sizes):
        result = fft(result, n=n, dim=d, norm=norm)
    return result


def ifftn(inp: Tensor, s=None, dim=None, norm=None) -> Tensor:
    """N-D inverse FFT."""
    result = inp
    ndim = len(inp._data.shape) if hasattr(inp._data, 'shape') else 1
    dims = dim if dim is not None else list(range(ndim))
    sizes = s if s is not None else [None] * len(dims)
    for d, n in zip(dims, sizes):
        result = ifft(result, n=n, dim=d, norm=norm)
    return result


def rfftn(inp: Tensor, s=None, dim=None, norm=None) -> Tensor:
    """N-D real FFT — rfft on last dim, fft on the rest."""
    ndim = len(inp._data.shape) if hasattr(inp._data, 'shape') else 1
    dims = dim if dim is not None else list(range(ndim))
    sizes = s if s is not None else [None] * len(dims)
    result = inp
    for i, (d, n) in enumerate(zip(dims, sizes)):
        if i == len(dims) - 1:
            result = rfft(result, n=n, dim=d, norm=norm)
        else:
            result = fft(result, n=n, dim=d, norm=norm)
    return result


def irfftn(inp: Tensor, s=None, dim=None, norm=None) -> Tensor:
    """N-D inverse real FFT."""
    ndim = len(inp._data.shape) if hasattr(inp._data, 'shape') else 1
    dims = dim if dim is not None else list(range(ndim))
    sizes = s if s is not None else [None] * len(dims)
    result = inp
    for i, (d, n) in enumerate(zip(dims, sizes)):
        if i == len(dims) - 1:
            result = irfft(result, n=n, dim=d, norm=norm)
        else:
            result = ifft(result, n=n, dim=d, norm=norm)
    return result


def hfft(inp: Tensor, n=None, dim: int = -1, norm=None) -> Tensor:
    """FFT of a Hermitian signal — input is complex, output is real-valued."""
    cnd = _to_complex_nd(inp)
    # hfft(x) = real(fft(conj(x)))
    conj_in = cnd.conj()
    raw = _nd_fft(conj_in)
    real_part = raw.real if hasattr(raw, 'real') else raw
    flat = list(real_part.to_flat_list() if hasattr(real_part, 'to_flat_list') else real_part)
    return Tensor(_nd_array(flat))


def ihfft(inp: Tensor, n=None, dim: int = -1, norm=None) -> Tensor:
    """Inverse FFT of a real-valued spectrum — returns conjugate-symmetric output."""
    # ihfft(x) = conj(rfft(x)) / N
    r = rfft(inp, n=n, dim=dim, norm=norm)
    cnd = r._data
    if isinstance(cnd, ComplexNDArray):
        n_pts = cnd.size
        result = ComplexNDArray(cnd.real / float(n_pts), (-1.0) * cnd.imag / float(n_pts), cnd.dtype)
        t = Tensor.__new__(Tensor)
        t._data = result; t.device = inp.device
        t.requires_grad = False; t.grad = None; t.grad_fn = None; t._is_leaf = True
        return t
    return r


__all__ = [
    'fft', 'ifft', 'rfft', 'irfft', 'fft2', 'ifft2',
    'fftn', 'ifftn', 'rfftn', 'irfftn',
    'hfft', 'ihfft',
    'fftshift', 'ifftshift', 'fftfreq', 'rfftfreq',
]
