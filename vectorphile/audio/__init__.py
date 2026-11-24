"""vectorphile.audio — Audio processing and feature extraction.

Implements spectrograms, MFCCs, mel-filterbanks, pitch detection, and
common audio transforms from scratch — pure Python, no external dependencies.

Usage::

    from vectorphile.audio import (
        stft, istft, mel_spectrogram, mfcc, MelSpectrogram,
        amplitude_to_db, db_to_amplitude, resample,
    )

    spec = mel_spectrogram(waveform, sample_rate=22050, n_mels=80)
    features = mfcc(waveform, sample_rate=22050, n_mfcc=13)
"""
from __future__ import annotations
import math
import cmath
from typing import List, Optional, Tuple


def _to_flat(x) -> List[float]:
    if hasattr(x, 'to_flat_list'):
        return [float(v) for v in x.to_flat_list()]
    if hasattr(x, 'tolist'):
        v = x.tolist()
        if isinstance(v, list) and v and isinstance(v[0], list):
            return [float(item) for row in v for item in row]
        return [float(item) for item in v]
    if isinstance(x, list):
        if x and isinstance(x[0], list):
            return [float(item) for row in x for item in row]
        return [float(v) for v in x]
    return [float(x)]


def _fft(x: List[complex]) -> List[complex]:
    """Cooley-Tukey radix-2 FFT (length must be power of 2)."""
    n = len(x)
    if n == 1:
        return [x[0]]
    if n & (n - 1):
        # DFT fallback for non-power-of-2
        return [sum(x[k] * cmath.exp(-2j * math.pi * k * m / n) for k in range(n)) for m in range(n)]
    even = _fft(x[0::2])
    odd = _fft(x[1::2])
    T = [cmath.exp(-2j * math.pi * k / n) * odd[k] for k in range(n // 2)]
    return [even[k] + T[k] for k in range(n // 2)] + [even[k] - T[k] for k in range(n // 2)]


def _rfft(x: List[float], n: Optional[int] = None) -> List[complex]:
    """Real FFT — returns n//2+1 complex values."""
    if n is None:
        n = len(x)
    xc = [complex(x[i] if i < len(x) else 0.0) for i in range(n)]
    full = _fft(xc)
    return full[:n // 2 + 1]


def _irfft(X: List[complex], n: Optional[int] = None) -> List[float]:
    """Inverse real FFT."""
    if n is None:
        n = (len(X) - 1) * 2
    full = list(X) + [X[n // 2 - k].conjugate() for k in range(1, n // 2)]
    if len(full) < n:
        full += [0.0] * (n - len(full))
    full = full[:n]
    inv = _fft([v.conjugate() for v in full])
    return [v.conjugate().real / n for v in inv]


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


# ---------------------------------------------------------------------------
# Window functions
# ---------------------------------------------------------------------------

def hann_window(n: int) -> List[float]:
    return [0.5 * (1 - math.cos(2 * math.pi * i / n)) for i in range(n)]


def hamming_window(n: int) -> List[float]:
    return [0.54 - 0.46 * math.cos(2 * math.pi * i / n) for i in range(n)]


def blackman_window(n: int) -> List[float]:
    alpha = 0.16
    a0, a1, a2 = (1 - alpha) / 2, 0.5, alpha / 2
    return [a0 - a1 * math.cos(2 * math.pi * i / n) + a2 * math.cos(4 * math.pi * i / n)
            for i in range(n)]


# ---------------------------------------------------------------------------
# STFT / iSTFT
# ---------------------------------------------------------------------------

def stft(signal, n_fft: int = 1024, hop_length: Optional[int] = None,
         win_length: Optional[int] = None, window: str = 'hann',
         center: bool = True) -> List[List[complex]]:
    """Short-Time Fourier Transform.

    Returns complex spectrogram of shape (n_fft//2+1, frames) as list of lists.
    """
    sig = _to_flat(signal)
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft
    if window == 'hann':
        win = hann_window(win_length)
    elif window == 'hamming':
        win = hamming_window(win_length)
    else:
        win = [1.0] * win_length
    # Pad window to n_fft
    win_pad = win + [0.0] * (n_fft - win_length)
    if center:
        pad = n_fft // 2
        sig = [sig[max(0, pad - i - 1)] if i < pad else
               sig[min(len(sig) - 1, i - pad)] for i in range(pad)] + sig + \
              [sig[max(0, len(sig) - 1 - i)] for i in range(pad)]
    n = len(sig)
    num_frames = max(1, 1 + (n - n_fft) // hop_length)
    n_freqs = n_fft // 2 + 1
    # Result: (n_freqs, num_frames)
    result = [[complex(0)] * num_frames for _ in range(n_freqs)]
    for fi in range(num_frames):
        start = fi * hop_length
        frame = [sig[start + k] * win_pad[k] if start + k < n else 0.0 for k in range(n_fft)]
        spec = _rfft(frame, n_fft)
        for freq_i in range(n_freqs):
            result[freq_i][fi] = spec[freq_i]
    return result


def istft(spec: List[List[complex]], hop_length: Optional[int] = None,
          win_length: Optional[int] = None, window: str = 'hann',
          center: bool = True, length: Optional[int] = None) -> List[float]:
    """Inverse STFT via overlap-add."""
    n_freqs = len(spec)
    n_fft = (n_freqs - 1) * 2
    num_frames = len(spec[0]) if spec else 0
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft
    if window == 'hann':
        win = hann_window(win_length)
    else:
        win = [1.0] * win_length
    win_pad = win + [0.0] * (n_fft - win_length)
    n_out = n_fft + (num_frames - 1) * hop_length
    out = [0.0] * n_out
    win_sum = [0.0] * n_out
    for fi in range(num_frames):
        col = [spec[freq_i][fi] for freq_i in range(n_freqs)]
        frame = _irfft(col, n_fft)
        start = fi * hop_length
        for k in range(n_fft):
            if start + k < n_out:
                out[start + k] += frame[k] * win_pad[k]
                win_sum[start + k] += win_pad[k] ** 2
    out = [out[i] / (win_sum[i] + 1e-8) for i in range(n_out)]
    if center:
        pad = n_fft // 2
        out = out[pad:]
    if length is not None:
        out = out[:length]
    return out


def spectrogram(signal, n_fft: int = 1024, hop_length: int = 256,
                power: float = 2.0) -> List[List[float]]:
    """Power/magnitude spectrogram (n_freqs, frames)."""
    spec = stft(signal, n_fft=n_fft, hop_length=hop_length)
    return [[abs(spec[i][j]) ** power for j in range(len(spec[i]))]
            for i in range(len(spec))]


# ---------------------------------------------------------------------------
# Mel filterbank / mel spectrogram
# ---------------------------------------------------------------------------

def hz_to_mel(hz: float) -> float:
    return 2595 * math.log10(1 + hz / 700)


def mel_to_hz(mel: float) -> float:
    return 700 * (10 ** (mel / 2595) - 1)


def mel_filterbank(n_mels: int = 80, n_fft: int = 1024,
                   sample_rate: int = 22050, fmin: float = 0.0,
                   fmax: Optional[float] = None) -> List[List[float]]:
    """Compute mel filterbank matrix of shape (n_mels, n_fft//2+1)."""
    if fmax is None:
        fmax = sample_rate / 2
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    n_freqs = n_fft // 2 + 1
    mel_points = [mel_min + i * (mel_max - mel_min) / (n_mels + 1) for i in range(n_mels + 2)]
    hz_points = [mel_to_hz(m) for m in mel_points]
    freqs = [k * (sample_rate / 2) / (n_freqs - 1) for k in range(n_freqs)]
    filters = [[0.0] * n_freqs for _ in range(n_mels)]
    for m in range(n_mels):
        f_m_minus = hz_points[m]
        f_m = hz_points[m + 1]
        f_m_plus = hz_points[m + 2]
        for k, f in enumerate(freqs):
            if f_m_minus <= f <= f_m:
                filters[m][k] = (f - f_m_minus) / (f_m - f_m_minus + 1e-10)
            elif f_m < f <= f_m_plus:
                filters[m][k] = (f_m_plus - f) / (f_m_plus - f_m + 1e-10)
    return filters


def _mat_vec_mul(M: List[List[float]], v: List[float]) -> List[float]:
    return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]


def _mat_mat_mul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    n, k, m = len(A), len(B), len(B[0]) if B else 0
    return [[sum(A[i][p] * B[p][j] for p in range(k)) for j in range(m)] for i in range(n)]


def mel_spectrogram(signal, sample_rate: int = 22050,
                    n_fft: int = 1024, hop_length: int = 256,
                    n_mels: int = 80, fmin: float = 0.0,
                    fmax: Optional[float] = None, power: float = 2.0) -> List[List[float]]:
    """Mel spectrogram of shape (n_mels, frames)."""
    spec = spectrogram(signal, n_fft=n_fft, hop_length=hop_length, power=power)
    fb = mel_filterbank(n_mels, n_fft, sample_rate, fmin, fmax)
    # fb: (n_mels, n_freqs), spec: (n_freqs, frames) → result: (n_mels, frames)
    n_frames = len(spec[0]) if spec else 0
    n_freqs = len(spec)
    result = [[0.0] * n_frames for _ in range(n_mels)]
    for m in range(n_mels):
        for f in range(n_frames):
            result[m][f] = sum(fb[m][k] * spec[k][f] for k in range(n_freqs))
    return result


def mfcc(signal, sample_rate: int = 22050, n_mfcc: int = 13,
         n_fft: int = 1024, hop_length: int = 256, n_mels: int = 80,
         dct_type: int = 2) -> List[List[float]]:
    """Mel-Frequency Cepstral Coefficients of shape (n_mfcc, frames)."""
    mel_spec = mel_spectrogram(signal, sample_rate, n_fft, hop_length, n_mels)
    n = len(mel_spec)
    n_frames = len(mel_spec[0]) if mel_spec else 0
    log_mel = [[math.log(mel_spec[i][j] + 1e-10) for j in range(n_frames)] for i in range(n)]
    coeffs = []
    for k in range(n_mfcc):
        weights = [math.cos(math.pi * k * (i + 0.5) / n) for i in range(n)]
        row = [sum(weights[i] * log_mel[i][j] for i in range(n)) for j in range(n_frames)]
        coeffs.append(row)
    return coeffs


def delta_features(features: List[List[float]], order: int = 1, width: int = 9) -> List[List[float]]:
    """Compute delta (velocity) or delta-delta (acceleration) features."""
    half = width // 2
    denom = 2 * sum(i ** 2 for i in range(1, half + 1)) + 1e-10
    n_coeff = len(features)
    n_frames = len(features[0]) if features else 0
    delta = [[0.0] * n_frames for _ in range(n_coeff)]
    for t in range(n_frames):
        for ni in range(1, half + 1):
            t1 = min(t + ni, n_frames - 1)
            t2 = max(t - ni, 0)
            for c in range(n_coeff):
                delta[c][t] += ni * (features[c][t1] - features[c][t2])
    delta = [[delta[c][t] / denom for t in range(n_frames)] for c in range(n_coeff)]
    if order == 1:
        return delta
    return delta_features(delta, order=1, width=width)


# ---------------------------------------------------------------------------
# Amplitude / dB conversion
# ---------------------------------------------------------------------------

def amplitude_to_db(S, ref: float = 1.0, amin: float = 1e-5,
                    top_db: Optional[float] = 80.0) -> List[List[float]]:
    """Convert amplitude spectrogram (2D list) to dB scale."""
    if isinstance(S, list) and S and isinstance(S[0], list):
        flat = [[max(amin, abs(v)) for v in row] for row in S]
        log_spec = [[10 * math.log10(v) for v in row] for row in flat]
        ref_db = 10 * math.log10(max(amin, abs(ref)))
        log_spec = [[v - ref_db for v in row] for row in log_spec]
        if top_db is not None:
            max_val = max(v for row in log_spec for v in row)
            log_spec = [[max(v, max_val - top_db) for v in row] for row in log_spec]
        return log_spec
    flat_s = [max(amin, abs(v)) for v in _to_flat(S)]
    log_spec = [10 * math.log10(v) for v in flat_s]
    ref_db = 10 * math.log10(max(amin, abs(ref)))
    log_spec = [v - ref_db for v in log_spec]
    if top_db is not None:
        mx = max(log_spec)
        log_spec = [max(v, mx - top_db) for v in log_spec]
    return log_spec


def db_to_amplitude(S_db, ref: float = 1.0):
    """Convert dB spectrogram to linear amplitude."""
    if isinstance(S_db, list) and S_db and isinstance(S_db[0], list):
        return [[ref * (10 ** (v / 10)) for v in row] for row in S_db]
    return [ref * (10 ** (v / 10)) for v in _to_flat(S_db)]


def power_to_db(S, ref: float = 1.0, amin: float = 1e-10,
                top_db: Optional[float] = 80.0):
    """Convert power spectrogram to dB."""
    return amplitude_to_db(S, ref=ref, amin=amin, top_db=top_db)


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def resample(signal, orig_sr: int, target_sr: int) -> List[float]:
    """Linear interpolation resampling."""
    sig = _to_flat(signal)
    if orig_sr == target_sr:
        return sig
    n_orig = len(sig)
    n_target = int(n_orig * target_sr / orig_sr)
    result = []
    for i in range(n_target):
        pos = i * (n_orig - 1) / max(n_target - 1, 1)
        lo = int(pos)
        hi = min(lo + 1, n_orig - 1)
        frac = pos - lo
        result.append(sig[lo] * (1 - frac) + sig[hi] * frac)
    return result


# ---------------------------------------------------------------------------
# Pitch estimation
# ---------------------------------------------------------------------------

def yin_pitch(signal, sample_rate: int = 22050, fmin: float = 50.0,
              fmax: float = 2000.0, threshold: float = 0.1) -> float:
    """YIN algorithm for pitch (F0) estimation.

    Returns fundamental frequency estimate in Hz, or 0.0 if unvoiced.
    """
    sig = _to_flat(signal)
    tau_min = int(sample_rate / fmax)
    tau_max = int(sample_rate / fmin)
    n = min(len(sig), tau_max * 2)
    # Difference function
    d = [0.0] * tau_max
    for tau in range(1, tau_max):
        d[tau] = sum((sig[i] - sig[i + tau]) ** 2 for i in range(n - tau))
    # Cumulative mean normalized difference
    cmnd = [1.0] * tau_max
    running_sum = 0.0
    for tau in range(1, tau_max):
        running_sum += d[tau]
        cmnd[tau] = d[tau] * tau / (running_sum + 1e-10)
    # Find first local minimum below threshold
    for tau in range(tau_min, tau_max):
        if cmnd[tau] < threshold:
            if 0 < tau < tau_max - 1:
                s0, s1, s2 = cmnd[tau - 1], cmnd[tau], cmnd[tau + 1]
                denom = 2 * (2 * s1 - s2 - s0)
                tau_refined = tau + (s2 - s0) / (denom + 1e-10) if abs(denom) > 1e-10 else tau
            else:
                tau_refined = tau
            return float(sample_rate / (tau_refined + 1e-10))
    return 0.0


# ---------------------------------------------------------------------------
# Torch-style transforms (class API)
# ---------------------------------------------------------------------------

class MelSpectrogram:
    """Mel spectrogram transform (class API, like torchaudio)."""
    def __init__(self, sample_rate: int = 22050, n_fft: int = 1024,
                 hop_length: int = 256, n_mels: int = 80,
                 fmin: float = 0.0, fmax: Optional[float] = None):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax

    def __call__(self, waveform):
        return mel_spectrogram(waveform, self.sample_rate, self.n_fft,
                               self.hop_length, self.n_mels, self.fmin, self.fmax)


class MFCC:
    """MFCC transform (class API)."""
    def __init__(self, sample_rate: int = 22050, n_mfcc: int = 13, **kwargs):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.kwargs = kwargs

    def __call__(self, waveform):
        return mfcc(waveform, self.sample_rate, self.n_mfcc, **self.kwargs)


class Spectrogram:
    """Power spectrogram transform."""
    def __init__(self, n_fft: int = 1024, hop_length: int = 256, power: float = 2.0):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.power = power

    def __call__(self, waveform):
        return spectrogram(waveform, self.n_fft, self.hop_length, self.power)


__all__ = [
    'hann_window', 'hamming_window', 'blackman_window',
    'stft', 'istft', 'spectrogram',
    'mel_filterbank', 'mel_spectrogram', 'mfcc', 'delta_features',
    'amplitude_to_db', 'db_to_amplitude', 'power_to_db',
    'resample', 'yin_pitch',
    'MelSpectrogram', 'MFCC', 'Spectrogram',
    'hz_to_mel', 'mel_to_hz',
]
