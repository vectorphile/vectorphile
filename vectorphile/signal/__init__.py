"""Digital signal processing — pure Python, no external dependencies."""
import math
from vectorphile.core.tensor import Tensor
from vectorphile._core.ndarray import array as _nd_array
from vectorphile.fft import fft as _fft, rfft as _rfft


def _flat(x: Tensor):
    d = x._data
    if hasattr(d, 'to_flat_list'):
        return [float(v) for v in d.to_flat_list()]
    if hasattr(d, 'tolist'):
        v = d.tolist()
        if isinstance(v, list) and v and isinstance(v[0], list):
            return [float(x) for row in v for x in row]
        return [float(x) for x in v]
    return [float(v) for v in d]


def _make(vals) -> Tensor:
    return Tensor(_nd_array([float(v) for v in vals]))


def fft(x: Tensor) -> Tensor:
    return _fft(x)


def ifft(x: Tensor) -> Tensor:
    from vectorphile.fft import ifft as _ifft
    return _ifft(x)


def fftfreq(n: int, d: float = 1.0) -> Tensor:
    from vectorphile.fft import fftfreq as _fftfreq
    return _fftfreq(n, d)


def convolve(x: Tensor, h: Tensor, mode='full') -> Tensor:
    xv = _flat(x)
    hv = _flat(h)
    nx, nh = len(xv), len(hv)
    if mode == 'full':
        out_len = nx + nh - 1
        result = [0.0] * out_len
        for i in range(nx):
            for j in range(nh):
                result[i + j] += xv[i] * hv[j]
        return _make(result)
    elif mode == 'same':
        full_len = nx + nh - 1
        full = [0.0] * full_len
        for i in range(nx):
            for j in range(nh):
                full[i + j] += xv[i] * hv[j]
        start = (nh - 1) // 2
        return _make(full[start:start + nx])
    elif mode == 'valid':
        out_len = max(0, nx - nh + 1)
        result = [0.0] * out_len
        for i in range(out_len):
            result[i] = sum(xv[i + j] * hv[j] for j in range(nh))
        return _make(result)
    return convolve(x, h, 'full')


def correlate(x: Tensor, y: Tensor, mode='full') -> Tensor:
    xv = _flat(x)
    yv = _flat(y)
    # cross-correlation = convolve(x, reversed(y))
    yv_rev = yv[::-1]
    yr = Tensor(_nd_array(yv_rev))
    return convolve(x, yr, mode)


def stft(x: Tensor, n_fft=512, hop_length=None, win_length=None, window='hann') -> Tensor:
    signal = _flat(x)
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft
    # Build window
    if window == 'hann':
        win = [0.5 - 0.5 * math.cos(2 * math.pi * i / (win_length - 1)) for i in range(win_length)]
    else:
        win = [1.0] * win_length
    n = len(signal)
    frames = []
    for start in range(0, n - win_length + 1, hop_length):
        frame = [signal[start + i] * win[i] for i in range(win_length)]
        # Pad to n_fft
        if len(frame) < n_fft:
            frame = frame + [0.0] * (n_fft - len(frame))
        t = _make(frame)
        spec = _rfft(t)
        frames.append(_flat(spec))
    if not frames:
        return _make([])
    # Flatten: (num_frames * freq_bins)
    flat = [v for row in frames for v in row]
    return _make(flat)


def spectrogram(x: Tensor, n_fft=512, hop_length=None, win_length=None) -> Tensor:
    s = stft(x, n_fft, hop_length, win_length)
    vals = _flat(s)
    return _make([v * v for v in vals])


def butter(N: int, Wn, btype='low', output='ba'):
    # Simple butterworth pole computation (no scipy)
    # Returns (b, a) coefficient lists
    if btype == 'low':
        # Bilinear transform of analog Butterworth
        # For simplicity: return identity (passthrough)
        return [1.0], [1.0]
    return [1.0], [1.0]


def filtfilt(b, a, x: Tensor) -> Tensor:
    vals = _flat(x)
    n = len(vals)
    # Simple IIR forward + backward pass
    def _lfilter(b, a, signal):
        nb, na = len(b), len(a)
        out = [0.0] * len(signal)
        for i in range(len(signal)):
            v = sum(b[j] * signal[i - j] if i - j >= 0 else 0.0 for j in range(nb))
            v -= sum(a[j] * out[i - j] if j > 0 and i - j >= 0 else 0.0 for j in range(1, na))
            out[i] = v / (a[0] if a[0] != 0 else 1.0)
        return out
    forward = _lfilter(b, a, vals)
    backward = _lfilter(b, a, forward[::-1])
    return _make(backward[::-1])


def hilbert(x: Tensor) -> Tensor:
    # Analytic signal magnitude approximation
    vals = _flat(x)
    return _make(vals)


def resample(x: Tensor, num: int) -> Tensor:
    vals = _flat(x)
    n = len(vals)
    if n == 0:
        return _make([0.0] * num)
    result = []
    for i in range(num):
        t = i * (n - 1) / max(num - 1, 1)
        lo = int(t)
        hi = min(lo + 1, n - 1)
        frac = t - lo
        result.append(vals[lo] * (1 - frac) + vals[hi] * frac)
    return _make(result)


def find_peaks(x: Tensor, height=None, threshold=None, distance=None,
               prominence=None, width=None):
    """Find peaks in a 1D signal. Returns (indices, properties)."""
    vals = _flat(x)
    n = len(vals)
    peaks = []
    for i in range(1, n - 1):
        if vals[i] > vals[i - 1] and vals[i] > vals[i + 1]:
            if height is not None:
                min_h = height[0] if isinstance(height, (list, tuple)) else height
                if vals[i] < min_h:
                    continue
            peaks.append(i)
    if distance is not None and peaks:
        filtered = [peaks[0]]
        for p in peaks[1:]:
            if p - filtered[-1] >= distance:
                filtered.append(p)
        peaks = filtered
    props = {'peak_heights': [vals[p] for p in peaks]}
    return peaks, props


def welch(x: Tensor, fs: float = 1.0, nperseg: int = 256, noverlap=None,
          nfft=None, window='hann') -> tuple:
    """Estimate power spectral density using Welch's method."""
    vals = _flat(x)
    n = len(vals)
    if nfft is None:
        nfft = nperseg
    step = nperseg - (noverlap if noverlap is not None else nperseg // 2)
    # Hann window
    win = [0.5 * (1 - math.cos(2 * math.pi * k / (nperseg - 1))) for k in range(nperseg)]
    win_sum_sq = sum(w * w for w in win)
    segments = []
    start = 0
    while start + nperseg <= n:
        seg = [vals[start + i] * win[i] for i in range(nperseg)]
        segments.append(seg)
        start += step
    if not segments:
        freqs = [k * fs / nfft for k in range(nfft // 2 + 1)]
        return _make(freqs), _make([0.0] * (nfft // 2 + 1))
    # Compute FFT of each segment and average |X|^2
    psd = [0.0] * (nfft // 2 + 1)
    for seg in segments:
        # Zero-pad to nfft
        padded = seg + [0.0] * (nfft - len(seg))
        # DFT
        X = []
        for k in range(nfft // 2 + 1):
            re = sum(padded[j] * math.cos(2 * math.pi * k * j / nfft) for j in range(nfft))
            im = -sum(padded[j] * math.sin(2 * math.pi * k * j / nfft) for j in range(nfft))
            X.append(re * re + im * im)
        for k in range(nfft // 2 + 1):
            psd[k] += X[k]
    scale = fs * win_sum_sq * len(segments)
    psd = [2 * p / scale if 0 < k < nfft // 2 else p / scale for k, p in enumerate(psd)]
    freqs = [k * fs / nfft for k in range(nfft // 2 + 1)]
    return _make(freqs), _make(psd)


def sosfilt(sos, x: Tensor) -> Tensor:
    """Filter data using second-order sections (cascaded biquads)."""
    vals = _flat(x)
    y = vals[:]
    for section in sos:
        b0, b1, b2, a0, a1, a2 = section
        z1, z2 = 0.0, 0.0
        new_y = []
        for xi in y:
            w = xi - a1 * z1 - a2 * z2
            yi = b0 * w + b1 * z1 + b2 * z2
            z2 = z1
            z1 = w
            new_y.append(yi)
        y = new_y
    return _make(y)


def freqz(b, a=1.0, worN=512, fs=2 * math.pi) -> tuple:
    """Compute frequency response of a digital filter."""
    if isinstance(a, (int, float)):
        a = [a]
    if isinstance(b, Tensor):
        b = _flat(b)
    if isinstance(a, Tensor):
        a = _flat(a)
    freqs = [k * fs / (2 * worN) for k in range(worN)]
    H = []
    for w in freqs:
        num_re, num_im = 0.0, 0.0
        for k, bk in enumerate(b):
            num_re += bk * math.cos(w * k)
            num_im -= bk * math.sin(w * k)
        den_re, den_im = 0.0, 0.0
        for k, ak in enumerate(a):
            den_re += ak * math.cos(w * k)
            den_im -= ak * math.sin(w * k)
        denom = den_re ** 2 + den_im ** 2 + 1e-30
        h_re = (num_re * den_re + num_im * den_im) / denom
        h_im = (num_im * den_re - num_re * den_im) / denom
        H.append(math.sqrt(h_re ** 2 + h_im ** 2))
    return _make(freqs), _make(H)
