"""Digital signal processing."""
import numpy as np
from vectorphile.core.tensor import Tensor


def fft(x: Tensor) -> Tensor:
    return Tensor(np.fft.fft(x._data))

def ifft(x: Tensor) -> Tensor:
    return Tensor(np.fft.ifft(x._data))

def fftfreq(n: int, d: float = 1.0) -> Tensor:
    return Tensor(np.fft.fftfreq(n, d).astype(np.float32))

def convolve(x: Tensor, h: Tensor, mode='full') -> Tensor:
    return Tensor(np.convolve(x._data.flatten(), h._data.flatten(), mode=mode).astype(np.float32))

def correlate(x: Tensor, y: Tensor, mode='full') -> Tensor:
    return Tensor(np.correlate(x._data.flatten(), y._data.flatten(), mode=mode).astype(np.float32))

def stft(x: Tensor, n_fft=512, hop_length=None, win_length=None, window='hann') -> Tensor:
    signal = x._data.flatten()
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft
    if window == 'hann':
        win = np.hanning(win_length)
    else:
        win = np.ones(win_length)
    n = len(signal)
    frames = []
    for i in range(0, n - win_length + 1, hop_length):
        frame = signal[i:i+win_length] * win
        frames.append(np.fft.rfft(frame, n=n_fft))
    return Tensor(np.array(frames))

def spectrogram(x: Tensor, n_fft=512, hop_length=None, win_length=None) -> Tensor:
    s = stft(x, n_fft, hop_length, win_length)
    return Tensor(np.abs(s._data)**2)

def butter(N: int, Wn, btype='low', output='ba'):
    try:
        from scipy.signal import butter as sp_butter
        return sp_butter(N, Wn, btype=btype, output=output)
    except ImportError:
        raise ImportError("butter requires scipy: pip install scipy")

def filtfilt(b, a, x: Tensor) -> Tensor:
    try:
        from scipy.signal import filtfilt as sp_filtfilt
        return Tensor(sp_filtfilt(b, a, x._data).astype(np.float32))
    except ImportError:
        raise ImportError("filtfilt requires scipy: pip install scipy")

def hilbert(x: Tensor) -> Tensor:
    try:
        from scipy.signal import hilbert as sp_hilbert
        return Tensor(np.abs(sp_hilbert(x._data)).astype(np.float32))
    except ImportError:
        return x

def resample(x: Tensor, num: int) -> Tensor:
    signal = x._data.flatten()
    n = len(signal)
    indices = np.linspace(0, n - 1, num)
    return Tensor(np.interp(indices, np.arange(n), signal).astype(np.float32))
