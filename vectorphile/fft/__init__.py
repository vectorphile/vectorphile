"""FFT operations."""
import numpy as np
from vectorphile.core.tensor import Tensor


def fft(input: Tensor, n=None, dim=-1, norm=None) -> Tensor:
    return Tensor(np.fft.fft(input._data, n=n, axis=dim, norm=norm))

def ifft(input: Tensor, n=None, dim=-1, norm=None) -> Tensor:
    return Tensor(np.fft.ifft(input._data, n=n, axis=dim, norm=norm))

def fft2(input: Tensor, s=None, dim=(-2, -1), norm=None) -> Tensor:
    return Tensor(np.fft.fft2(input._data, s=s, axes=dim, norm=norm))

def ifft2(input: Tensor, s=None, dim=(-2, -1), norm=None) -> Tensor:
    return Tensor(np.fft.ifft2(input._data, s=s, axes=dim, norm=norm))

def rfft(input: Tensor, n=None, dim=-1, norm=None) -> Tensor:
    return Tensor(np.fft.rfft(input._data, n=n, axis=dim, norm=norm))

def irfft(input: Tensor, n=None, dim=-1, norm=None) -> Tensor:
    return Tensor(np.fft.irfft(input._data, n=n, axis=dim, norm=norm).astype(np.float32))

def fftshift(input: Tensor, dim=None) -> Tensor:
    return Tensor(np.fft.fftshift(input._data, axes=dim))

def ifftshift(input: Tensor, dim=None) -> Tensor:
    return Tensor(np.fft.ifftshift(input._data, axes=dim))

def fftfreq(n: int, d: float = 1.0) -> Tensor:
    return Tensor(np.fft.fftfreq(n, d).astype(np.float32))

def rfftfreq(n: int, d: float = 1.0) -> Tensor:
    return Tensor(np.fft.rfftfreq(n, d).astype(np.float32))
