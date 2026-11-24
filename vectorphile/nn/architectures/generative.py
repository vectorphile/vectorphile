"""Generative architectures — VAE, DCGAN, WGAN-GP, DDPM."""
from __future__ import annotations
import math as _math
import random as _random
from typing import Optional, Tuple, List
from vectorphile.nn.module import Module
from vectorphile.core.tensor import Tensor
from vectorphile._core.ndarray import array as _nd_arr, NDArray


def _nd_randn(shape, scale=1.0):
    n = 1
    for s in shape: n *= s
    flat = [_math.sqrt(-2*_math.log(max(_random.random(),1e-10)))*_math.cos(2*_math.pi*_random.random())*scale
            for _ in range(n)]
    return _nd_arr(flat, dtype='float32').reshape(shape)


def _nd_zeros(shape):
    n = 1
    for s in shape: n *= s
    return _nd_arr([0.0]*n, dtype='float32').reshape(shape)


def _nd_ones(shape):
    n = 1
    for s in shape: n *= s
    return _nd_arr([1.0]*n, dtype='float32').reshape(shape)


def _relu(x: NDArray) -> NDArray:
    fl = x.to_flat_list()
    return _nd_arr([max(0.0, v) for v in fl], dtype='float32').reshape(x.shape)

def _silu(x: NDArray) -> NDArray:
    fl = x.to_flat_list()
    return _nd_arr([v / (1 + _math.exp(-v)) for v in fl], dtype='float32').reshape(x.shape)

def _sigmoid(x: NDArray) -> NDArray:
    fl = x.to_flat_list()
    return _nd_arr([1.0 / (1 + _math.exp(-v)) for v in fl], dtype='float32').reshape(x.shape)

def _tanh_nd(x: NDArray) -> NDArray:
    fl = x.to_flat_list()
    return _nd_arr([_math.tanh(v) for v in fl], dtype='float32').reshape(x.shape)

def _leaky_relu(x: NDArray, neg=0.2) -> NDArray:
    fl = x.to_flat_list()
    return _nd_arr([v if v > 0 else neg * v for v in fl], dtype='float32').reshape(x.shape)

def _nd_log(x: NDArray) -> NDArray:
    fl = x.to_flat_list()
    return _nd_arr([_math.log(max(v, 1e-30)) for v in fl], dtype='float32').reshape(x.shape)

def _nd_exp(x: NDArray) -> NDArray:
    fl = x.to_flat_list()
    return _nd_arr([_math.exp(v) for v in fl], dtype='float32').reshape(x.shape)

def _nd_sqrt_elem(x: NDArray) -> NDArray:
    fl = x.to_flat_list()
    return _nd_arr([_math.sqrt(max(v, 0.0)) for v in fl], dtype='float32').reshape(x.shape)

def _nd_clip(x: NDArray, lo: float, hi: float) -> NDArray:
    fl = x.to_flat_list()
    return _nd_arr([max(lo, min(hi, v)) for v in fl], dtype='float32').reshape(x.shape)

def _nd_mean(x: NDArray) -> float:
    fl = x.to_flat_list()
    return sum(fl) / max(len(fl), 1)

def _nd_scalar(v: float) -> NDArray:
    return _nd_arr([v], dtype='float32').reshape((1,))

def _nd_exp_half(x: NDArray) -> NDArray:
    fl = x.to_flat_list()
    return _nd_arr([_math.exp(0.5 * v) for v in fl], dtype='float32').reshape(x.shape)

def _nd_cumprod(lst: list) -> list:
    result = []
    acc = 1.0
    for v in lst:
        acc *= v
        result.append(acc)
    return result

def _nd_linspace(start: float, stop: float, n: int) -> list:
    if n == 1:
        return [start]
    step = (stop - start) / (n - 1)
    return [start + i * step for i in range(n)]


def _linear(x, w, b=None):
    out = x @ w
    return out + b if b is not None else out


def _layer_norm(x: NDArray, w: NDArray, b: NDArray, eps=1e-5) -> NDArray:
    sh = x.shape
    fl = x.to_flat_list()
    last = sh[-1]
    n_rows = len(fl) // last
    wf = w.to_flat_list()
    bf = b.to_flat_list()
    result = []
    for r in range(n_rows):
        row = fl[r*last:(r+1)*last]
        mu = sum(row) / last
        var = sum((v - mu)**2 for v in row) / last
        inv = 1.0 / _math.sqrt(var + eps)
        result.extend([(v - mu) * inv * wf[i] + bf[i] for i, v in enumerate(row)])
    return _nd_arr(result, dtype='float32').reshape(sh)


def _conv_transpose2d(x: NDArray, w: NDArray, b: Optional[NDArray],
                       stride: int, padding: int, output_padding: int = 0) -> NDArray:
    """Transposed conv2d (NCHW) via upsample + regular conv."""
    from vectorphile.core.tensor import _nd_pad
    N, C_in, H, W = x.shape
    C_out, _, kH, kW = w.shape
    H_up = H * stride + output_padding
    W_up = W * stride + output_padding
    # Build upsampled array (zero-insert)
    xf = x.to_flat_list()
    up_flat = [0.0] * (N * C_in * H_up * W_up)
    for n in range(N):
        for c in range(C_in):
            for h in range(H):
                for ww in range(W):
                    up_flat[n*C_in*H_up*W_up + c*H_up*W_up + h*stride*W_up + ww*stride] = \
                        xf[n*C_in*H*W + c*H*W + h*W + ww]
    x_up = _nd_arr(up_flat, dtype='float32').reshape((N, C_in, H_up, W_up))
    p = kH - 1 - padding
    if p > 0:
        x_up = _nd_pad(x_up, ((0,0),(0,0),(p,p),(p,p)))
    Hu, Wu = x_up.shape[2], x_up.shape[3]
    H_out = Hu - kH + 1
    W_out = Wu - kW + 1
    xuf = x_up.to_flat_list()
    # w shape (C_in, C_out, kH, kW); flip spatial dims for transpose conv
    wf = w.to_flat_list()
    wflip = []
    for ci in range(C_in):
        for co in range(C_out):
            for kh in range(kH):
                for kw_ in range(kW):
                    wflip.append(wf[ci*C_out*kH*kW + co*kH*kW + (kH-1-kh)*kW + (kW-1-kw_)])
    out_flat = [0.0] * (N * C_out * H_out * W_out)
    for n in range(N):
        for co in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    s = 0.0
                    for ci in range(C_in):
                        for kh in range(kH):
                            for kw_ in range(kW):
                                xi = xuf[n*C_in*Hu*Wu + ci*Hu*Wu + (i+kh)*Wu + (j+kw_)]
                                wi = wflip[ci*C_out*kH*kW + co*kH*kW + kh*kW + kw_]
                                s += xi * wi
                    out_flat[n*C_out*H_out*W_out + co*H_out*W_out + i*W_out + j] = s
    out = _nd_arr(out_flat, dtype='float32').reshape((N, C_out, H_out, W_out))
    if b is not None:
        bf = b.to_flat_list()
        of = out.to_flat_list()
        of2 = [of[n*C_out*H_out*W_out + co*H_out*W_out + i*W_out + j] + bf[co]
               for n in range(N) for co in range(C_out) for i in range(H_out) for j in range(W_out)]
        out = _nd_arr(of2, dtype='float32').reshape((N, C_out, H_out, W_out))
    return out


def _conv2d(x: NDArray, w: NDArray, b=None, stride=1, padding=1) -> NDArray:
    from vectorphile.core.tensor import _nd_pad
    N, C_in, H, W = x.shape
    C_out, _, kH, kW = w.shape
    H_out = (H + 2 * padding - kH) // stride + 1
    W_out = (W + 2 * padding - kW) // stride + 1
    if padding > 0:
        x = _nd_pad(x, ((0,0),(0,0),(padding,padding),(padding,padding)))
    Hp, Wp = x.shape[2], x.shape[3]
    xf = x.to_flat_list()
    wf = w.to_flat_list()
    out_flat = [0.0] * (N * C_out * H_out * W_out)
    for n in range(N):
        for co in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    s = 0.0
                    for ci in range(C_in):
                        for kh in range(kH):
                            for kw_ in range(kW):
                                s += xf[n*C_in*Hp*Wp + ci*Hp*Wp + (i*stride+kh)*Wp + (j*stride+kw_)] * \
                                     wf[co*C_in*kH*kW + ci*kH*kW + kh*kW + kw_]
                    out_flat[n*C_out*H_out*W_out + co*H_out*W_out + i*W_out + j] = s
    out = _nd_arr(out_flat, dtype='float32').reshape((N, C_out, H_out, W_out))
    if b is not None:
        bf = b.to_flat_list()
        of = out.to_flat_list()
        of2 = [of[n*C_out*H_out*W_out + co*H_out*W_out + i*W_out + j] + bf[co]
               for n in range(N) for co in range(C_out) for i in range(H_out) for j in range(W_out)]
        out = _nd_arr(of2, dtype='float32').reshape((N, C_out, H_out, W_out))
    return out


def _batch_norm2d(x: NDArray, w: NDArray, b: NDArray, rm: list, rv: list,
                   training: bool, momentum=0.1, eps=1e-5) -> NDArray:
    N, C, H, W = x.shape
    xf = x.to_flat_list()
    wf = w.to_flat_list()
    bf = b.to_flat_list()
    if training:
        # compute per-channel mean/var
        mean_c = []
        var_c  = []
        for c in range(C):
            vals = [xf[n*C*H*W + c*H*W + i*W + j] for n in range(N) for i in range(H) for j in range(W)]
            mu = sum(vals) / len(vals)
            va = sum((v-mu)**2 for v in vals) / len(vals)
            mean_c.append(mu)
            var_c.append(va)
            rm[c] = (1 - momentum) * rm[c] + momentum * mu
            rv[c] = (1 - momentum) * rv[c] + momentum * va
    else:
        mean_c = rm[:]
        var_c  = rv[:]
    out_flat = []
    for n in range(N):
        for c in range(C):
            inv = 1.0 / _math.sqrt(var_c[c] + eps)
            for i in range(H):
                for j in range(W):
                    v = xf[n*C*H*W + c*H*W + i*W + j]
                    out_flat.append((v - mean_c[c]) * inv * wf[c] + bf[c])
    return _nd_arr(out_flat, dtype='float32').reshape((N, C, H, W))


# ── VAE ───────────────────────────────────────────────────────────────────────

class VAE(Module):
    """Variational Autoencoder — Kingma & Welling 2013.

    Fully-connected VAE with configurable hidden layers.

    Args:
        input_dim: flattened input size (e.g. 784 for MNIST)
        latent_dim: latent space dimension
        hidden_dims: list of encoder hidden layer sizes
        activation: 'relu' or 'silu'
    """

    def __init__(self, input_dim: int = 784, latent_dim: int = 20,
                 hidden_dims: Optional[List[int]] = None, activation: str = 'relu'):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]
        self.latent_dim = latent_dim
        self.activation_fn = _relu if activation == 'relu' else _silu
        # Encoder
        enc_dims = [input_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            w = Tensor(_nd_randn((enc_dims[i], enc_dims[i+1]), _math.sqrt(2 / enc_dims[i])))
            b = Tensor(_nd_zeros((enc_dims[i+1],)))
            w.requires_grad = True; b.requires_grad = True
            setattr(self, f'enc_w{i}', w); setattr(self, f'enc_b{i}', b)
            self._parameters[f'enc_w{i}'] = w; self._parameters[f'enc_b{i}'] = b
        self.n_enc = len(hidden_dims)
        # μ and log σ² heads
        for head in ['mu', 'logvar']:
            w = Tensor(_nd_randn((hidden_dims[-1], latent_dim), 0.01))
            b = Tensor(_nd_zeros((latent_dim,)))
            w.requires_grad = True; b.requires_grad = True
            setattr(self, f'{head}_w', w); setattr(self, f'{head}_b', b)
            self._parameters[f'{head}_w'] = w; self._parameters[f'{head}_b'] = b
        # Decoder
        dec_dims = [latent_dim] + list(reversed(hidden_dims)) + [input_dim]
        self.n_dec = len(dec_dims) - 1
        for i in range(self.n_dec):
            w = Tensor(_nd_randn((dec_dims[i], dec_dims[i+1]), _math.sqrt(2 / dec_dims[i])))
            b = Tensor(_nd_zeros((dec_dims[i+1],)))
            w.requires_grad = True; b.requires_grad = True
            setattr(self, f'dec_w{i}', w); setattr(self, f'dec_b{i}', b)
            self._parameters[f'dec_w{i}'] = w; self._parameters[f'dec_b{i}'] = b

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns μ and log σ² tensors."""
        d = x._data
        for i in range(self.n_enc):
            d = self.activation_fn(_linear(d, getattr(self, f'enc_w{i}')._data, getattr(self, f'enc_b{i}')._data))
        mu     = _linear(d, self.mu_w._data,     self.mu_b._data)
        logvar = _linear(d, self.logvar_w._data, self.logvar_b._data)
        return Tensor(mu), Tensor(logvar)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """z = μ + ε · σ,  ε ~ N(0, I)."""
        eps = _nd_randn(mu._data.shape)
        std = _nd_exp_half(logvar._data)
        return Tensor(mu._data + eps * std)

    def decode(self, z: Tensor) -> Tensor:
        d = z._data
        for i in range(self.n_dec - 1):
            d = self.activation_fn(_linear(d, getattr(self, f'dec_w{i}')._data, getattr(self, f'dec_b{i}')._data))
        # Last layer: sigmoid for pixel values in [0, 1]
        d = _sigmoid(_linear(d, getattr(self, f'dec_w{self.n_dec-1}')._data,
                              getattr(self, f'dec_b{self.n_dec-1}')._data))
        return Tensor(d)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Returns (reconstruction, mu, logvar)."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def elbo_loss(self, x: Tensor, beta: float = 1.0) -> Tensor:
        """ELBO = reconstruction_loss + β · KL divergence."""
        recon, mu, logvar = self.forward(x)
        rf = recon._data.to_flat_list()
        xf = x._data.to_flat_list()
        recon_loss = -sum(xf[i] * _math.log(max(rf[i], 1e-8)) +
                          (1 - xf[i]) * _math.log(max(1 - rf[i], 1e-8))
                          for i in range(len(xf))) / max(len(xf), 1)
        mf = mu._data.to_flat_list()
        lf = logvar._data.to_flat_list()
        kl = -0.5 * sum(1 + lf[i] - mf[i]**2 - _math.exp(lf[i]) for i in range(len(mf))) / max(len(mf), 1)
        return Tensor(_nd_arr([recon_loss + beta * kl], dtype='float32').reshape((1,)))

    def sample(self, n: int) -> Tensor:
        """Sample n images from the prior N(0, I)."""
        z = _nd_randn((n, self.latent_dim))
        return self.decode(Tensor(z))


# ── DCGAN ─────────────────────────────────────────────────────────────────────

class DCGANGenerator(Module):
    """DCGAN Generator — Radford et al. 2015.

    Projects noise z → feature map, then upsamples via transposed convolutions.

    Args:
        latent_dim: noise vector size
        ngf: base feature map channels
        nc: output image channels
        image_size: output spatial size (must be power of 2, ≥ 16)
    """

    def __init__(self, latent_dim: int = 100, ngf: int = 64,
                 nc: int = 3, image_size: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        n_up = int(_math.log2(image_size)) - 3   # number of upsample blocks
        # Initial projection
        init_channels = ngf * (2 ** n_up)
        self.proj_w = Tensor(_nd_randn((latent_dim, init_channels * 4 * 4), 0.02))
        self.proj_b = Tensor(_nd_zeros((init_channels * 4 * 4,)))
        self.proj_w.requires_grad = True; self.proj_b.requires_grad = True
        self._parameters['proj_w'] = self.proj_w
        self._parameters['proj_b'] = self.proj_b
        self.init_channels = init_channels
        # Upsample blocks
        self.up_ws = []
        self.up_rm = []
        self.up_rv = []
        self.up_bn_w = []
        self.up_bn_b = []
        c_in = init_channels
        for i in range(n_up):
            c_out = c_in // 2
            w = Tensor(_nd_randn((c_in, c_out, 4, 4), 0.02))
            w.requires_grad = True
            self.up_ws.append(w)
            self._parameters[f'up_w{i}'] = w
            bn_w = Tensor(_nd_ones((c_out,))); bn_w.requires_grad = True
            bn_b = Tensor(_nd_zeros((c_out,))); bn_b.requires_grad = True
            self.up_bn_w.append(bn_w); self.up_bn_b.append(bn_b)
            self._parameters[f'up_bn_w{i}'] = bn_w; self._parameters[f'up_bn_b{i}'] = bn_b
            self.up_rm.append([0.0] * c_out)
            self.up_rv.append([1.0] * c_out)
            c_in = c_out
        # Final conv: → nc channels, tanh
        self.final_w = Tensor(_nd_randn((c_in, nc, 4, 4), 0.02))
        self.final_b = Tensor(_nd_zeros((nc,)))
        self.final_w.requires_grad = True; self.final_b.requires_grad = True
        self._parameters['final_w'] = self.final_w; self._parameters['final_b'] = self.final_b

    def forward(self, z: Tensor) -> Tensor:
        """z: (B, latent_dim) → (B, nc, H, W) in [-1, 1]."""
        B = z._data.shape[0]
        d = _linear(z._data, self.proj_w._data, self.proj_b._data).reshape((B, self.init_channels, 4, 4))
        d = _relu(d)
        for i, (w, bn_w, bn_b, rm, rv) in enumerate(
                zip(self.up_ws, self.up_bn_w, self.up_bn_b, self.up_rm, self.up_rv)):
            d = _conv_transpose2d(d, w._data, None, stride=2, padding=1)
            d = _batch_norm2d(d, bn_w._data, bn_b._data, rm, rv, self.training)
            d = _relu(d)
        d = _conv_transpose2d(d, self.final_w._data, self.final_b._data, stride=2, padding=1)
        return Tensor(_tanh_nd(d))


class DCGANDiscriminator(Module):
    """DCGAN Discriminator — conv → leaky ReLU → BN → sigmoid."""

    def __init__(self, ndf: int = 64, nc: int = 3, image_size: int = 64):
        super().__init__()
        n_down = int(_math.log2(image_size)) - 3
        self.down_ws = []
        self.down_rm = []
        self.down_rv = []
        self.down_bn_w = []
        self.down_bn_b = []
        c_in = nc
        c_out = ndf
        for i in range(n_down):
            w = Tensor(_nd_randn((c_out, c_in, 4, 4), 0.02))
            w.requires_grad = True
            self.down_ws.append(w)
            self._parameters[f'down_w{i}'] = w
            bn_w = Tensor(_nd_ones((c_out,))); bn_w.requires_grad = True
            bn_b = Tensor(_nd_zeros((c_out,))); bn_b.requires_grad = True
            self.down_bn_w.append(bn_w); self.down_bn_b.append(bn_b)
            self._parameters[f'down_bn_w{i}'] = bn_w; self._parameters[f'down_bn_b{i}'] = bn_b
            self.down_rm.append([0.0] * c_out)
            self.down_rv.append([1.0] * c_out)
            c_in = c_out; c_out *= 2
        # Final classifier
        self.final_w = Tensor(_nd_randn((c_in * 4 * 4, 1), 0.02))
        self.final_b = Tensor(_nd_zeros((1,)))
        self.final_w.requires_grad = True; self.final_b.requires_grad = True
        self._parameters['final_w'] = self.final_w; self._parameters['final_b'] = self.final_b

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, nc, H, W) → (B, 1) logit."""
        d = x._data
        for i, (w, bn_w, bn_b, rm, rv) in enumerate(
                zip(self.down_ws, self.down_bn_w, self.down_bn_b, self.down_rm, self.down_rv)):
            d = _conv2d(d, w, None, stride=2, padding=1)
            if i > 0:  # no BN on first layer (standard DCGAN)
                d = _batch_norm2d(d, bn_w._data, bn_b._data, rm, rv, self.training)
            d = _leaky_relu(d)
        flat_size = 1
        for s in d.shape[1:]: flat_size *= s
        d = d.reshape((d.shape[0], flat_size))
        return Tensor(_linear(d, self.final_w._data, self.final_b._data))


class DCGAN(Module):
    """Full DCGAN with generator and discriminator."""

    def __init__(self, latent_dim: int = 100, ngf: int = 64, ndf: int = 64,
                 nc: int = 3, image_size: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.generator     = DCGANGenerator(latent_dim, ngf, nc, image_size)
        self.discriminator = DCGANDiscriminator(ndf, nc, image_size)
        self._modules['generator']     = self.generator
        self._modules['discriminator'] = self.discriminator

    def generate(self, n: int) -> Tensor:
        z = Tensor(_nd_randn((n, self.latent_dim)))
        return self.generator(z)

    def discriminator_loss(self, real: Tensor, fake: Tensor) -> Tensor:
        """Non-saturating GAN loss for discriminator."""
        d_real = self.discriminator(real)._data
        d_fake = self.discriminator(fake)._data
        sr = _sigmoid(d_real).to_flat_list()
        sf = _sigmoid(d_fake).to_flat_list()
        loss = sum(-_math.log(max(sr[i], 1e-8)) - _math.log(max(1 - sf[i], 1e-8)) for i in range(len(sr))) / max(len(sr), 1)
        return Tensor(_nd_arr([loss], dtype='float32').reshape((1,)))

    def generator_loss(self, fake: Tensor) -> Tensor:
        d_fake = self.discriminator(fake)._data
        sf = _sigmoid(d_fake).to_flat_list()
        loss = sum(-_math.log(max(v, 1e-8)) for v in sf) / max(len(sf), 1)
        return Tensor(_nd_arr([loss], dtype='float32').reshape((1,)))

    def gradient_penalty(self, real: Tensor, fake: Tensor, lam: float = 10.0) -> float:
        """WGAN-GP gradient penalty (approximate — full autograd needed for exact)."""
        B = real._data.shape[0]
        rf = real._data.to_flat_list()
        ff = fake._data.to_flat_list()
        alpha = _random.random()
        interp_flat = [alpha * rf[i] + (1 - alpha) * ff[i] for i in range(len(rf))]
        interpolated = _nd_arr(interp_flat, dtype='float32').reshape(real._data.shape)
        d_interp_f = self.discriminator(Tensor(interpolated))._data.to_flat_list()
        eps = 1e-4
        gp = 0.0
        for _ in range(4):
            idx_flat = _random.randint(0, len(interp_flat) - 1)
            x_plus_flat = interp_flat[:]
            x_plus_flat[idx_flat] += eps
            x_plus = _nd_arr(x_plus_flat, dtype='float32').reshape(real._data.shape)
            d_plus_f = self.discriminator(Tensor(x_plus))._data.to_flat_list()
            gp += sum((d_plus_f[i] - d_interp_f[i])**2 / eps**2 for i in range(len(d_interp_f))) / max(len(d_interp_f), 1)
        return float(lam * gp / 4)

    def forward(self, z: Tensor) -> Tensor:
        return self.generator(z)


# ── DDPM ─────────────────────────────────────────────────────────────────────

class SinusoidalTimeEmbedding(Module):
    """Sinusoidal time step embedding (Ho et al. 2020 / Transformer positional enc)."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj_w = Tensor(_nd_randn((embed_dim, embed_dim * 4), 0.02))
        self.proj_b = Tensor(_nd_zeros((embed_dim * 4,)))
        self.out_w  = Tensor(_nd_randn((embed_dim * 4, embed_dim), 0.02))
        self.out_b  = Tensor(_nd_zeros((embed_dim,)))
        for n, p in [('proj_w',self.proj_w),('proj_b',self.proj_b),
                     ('out_w',self.out_w),('out_b',self.out_b)]:
            p.requires_grad = True; self._parameters[n] = p

    def forward(self, t) -> NDArray:
        """t: list/NDArray of (B,) int timesteps → (B, embed_dim) embedding."""
        if isinstance(t, NDArray):
            t_list = [int(v) for v in t.to_flat_list()]
        else:
            t_list = [int(v) for v in t]
        B = len(t_list)
        half = self.embed_dim // 2
        freq = [_math.exp(-_math.log(10000) * i / half) for i in range(half)]
        cos_flat, sin_flat = [], []
        for ti in t_list:
            for fi in freq:
                arg = float(ti) * fi
                cos_flat.append(_math.cos(arg))
                sin_flat.append(_math.sin(arg))
        emb_flat = []
        for i in range(B):
            emb_flat.extend(cos_flat[i*half:(i+1)*half])
            emb_flat.extend(sin_flat[i*half:(i+1)*half])
        emb = _nd_arr(emb_flat, dtype='float32').reshape((B, self.embed_dim))
        emb = _silu(_linear(emb, self.proj_w._data, self.proj_b._data))
        return _linear(emb, self.out_w._data, self.out_b._data)


class ResBlock(Module):
    """UNet-style residual block with time conditioning."""

    def __init__(self, in_c: int, out_c: int, time_dim: int, groups: int = 8):
        super().__init__()
        # Conv 1
        self.c1_w = Tensor(_nd_randn((out_c, in_c, 3, 3), 0.02))
        self.c1_w.requires_grad = True; self._parameters['c1_w'] = self.c1_w
        # Group norm params (approximate: use layer norm per channel)
        self.gn1_w = Tensor(_nd_ones((in_c,))); self.gn1_w.requires_grad = True
        self.gn1_b = Tensor(_nd_zeros((in_c,))); self.gn1_b.requires_grad = True
        self._parameters['gn1_w'] = self.gn1_w; self._parameters['gn1_b'] = self.gn1_b
        # Time projection
        self.t_w = Tensor(_nd_randn((time_dim, out_c), 0.02))
        self.t_w.requires_grad = True; self._parameters['t_w'] = self.t_w
        # Conv 2
        self.c2_w = Tensor(_nd_randn((out_c, out_c, 3, 3), 0.02))
        self.c2_w.requires_grad = True; self._parameters['c2_w'] = self.c2_w
        self.gn2_w = Tensor(_nd_ones((out_c,))); self.gn2_w.requires_grad = True
        self.gn2_b = Tensor(_nd_zeros((out_c,))); self.gn2_b.requires_grad = True
        self._parameters['gn2_w'] = self.gn2_w; self._parameters['gn2_b'] = self.gn2_b
        # Residual projection
        self.skip_w = None
        if in_c != out_c:
            self.skip_w = Tensor(_nd_randn((out_c, in_c, 1, 1), 0.02))
            self.skip_w.requires_grad = True; self._parameters['skip_w'] = self.skip_w

    def _gn(self, x: NDArray, w: NDArray, b: NDArray) -> NDArray:
        N, C, H, W = x.shape
        xf = x.to_flat_list()
        wf = w.to_flat_list()
        bf = b.to_flat_list()
        out_flat = []
        for n in range(N):
            for c in range(C):
                vals = [xf[n*C*H*W + c*H*W + i*W + j] for i in range(H) for j in range(W)]
                mu = sum(vals) / len(vals)
                va = sum((v-mu)**2 for v in vals) / len(vals)
                inv = 1.0 / _math.sqrt(va + 1e-5)
                for v in vals:
                    out_flat.append((v - mu) * inv * wf[c] + bf[c])
        return _nd_arr(out_flat, dtype='float32').reshape((N, C, H, W))

    def forward(self, x: Tensor, t_emb: NDArray) -> Tensor:
        d = _silu(self._gn(x._data, self.gn1_w._data, self.gn1_b._data))
        d = _conv2d(d, self.c1_w._data, None, stride=1, padding=1)
        # Add time embedding: t_emb (B, time_dim) @ t_w (time_dim, out_c) → (B, out_c)
        # broadcast to (B, out_c, 1, 1) then add to d (B, out_c, H, W)
        B, C, H, W = d.shape
        t_proj_2d = t_emb @ self.t_w._data  # (B, out_c)
        tf = t_proj_2d.to_flat_list()
        df = d.to_flat_list()
        d_flat = [df[n*C*H*W + c*H*W + i*W + j] + tf[n*C + c]
                  for n in range(B) for c in range(C) for i in range(H) for j in range(W)]
        d = _nd_arr(d_flat, dtype='float32').reshape((B, C, H, W))
        d = _silu(self._gn(d, self.gn2_w._data, self.gn2_b._data))
        d = _conv2d(d, self.c2_w._data, None, stride=1, padding=1)
        identity = x._data
        if self.skip_w is not None:
            identity = _conv2d(identity, self.skip_w._data, None, stride=1, padding=0)
        return Tensor(d + identity)


class DDPM(Module):
    """Denoising Diffusion Probabilistic Model — Ho et al. 2020.

    Implements:
    - Linear noise schedule (β_1 to β_T)
    - Forward process: q(x_t | x_0) = N(√ᾱ_t x_0, (1-ᾱ_t) I)
    - Backward process: p_θ(x_{t-1} | x_t) via ε-prediction UNet
    - Training objective: simple L2 on predicted noise
    - Sampling: DDPM ancestral sampler

    Args:
        image_channels: number of image channels
        image_size: spatial size
        base_channels: UNet base channel count
        channel_mults: channel multipliers per resolution
        num_res_blocks: residual blocks per resolution
        time_embed_dim: sinusoidal time embedding size
        T: total diffusion steps
        beta_start, beta_end: linear schedule endpoints
    """

    def __init__(self, image_channels: int = 3, image_size: int = 32,
                 base_channels: int = 64,
                 channel_mults: Tuple = (1, 2, 4, 8),
                 num_res_blocks: int = 2,
                 time_embed_dim: int = 128,
                 T: int = 1000,
                 beta_start: float = 1e-4, beta_end: float = 0.02):
        super().__init__()
        self.T = T
        self.image_channels = image_channels

        # Noise schedule
        betas_list = _nd_linspace(beta_start, beta_end, T)
        alphas_list = [1.0 - b for b in betas_list]
        alpha_bars_list = _nd_cumprod(alphas_list)
        self.betas      = betas_list
        self.alpha_bars = alpha_bars_list
        self.alphas     = alphas_list

        # Time embedding
        self.time_emb = SinusoidalTimeEmbedding(time_embed_dim)
        self._modules['time_emb'] = self.time_emb

        # UNet encoder/decoder with skip connections
        self.down_blocks = []
        self.up_blocks   = []
        self.res_blocks  = []

        channels = [base_channels * m for m in channel_mults]

        # Stem
        self.stem_w = Tensor(_nd_randn((channels[0], image_channels, 3, 3), 0.02))
        self.stem_w.requires_grad = True; self._parameters['stem_w'] = self.stem_w

        # Build encoder (downsampling path)
        in_c = channels[0]
        for level, out_c in enumerate(channels):
            for _ in range(num_res_blocks):
                blk = ResBlock(in_c, out_c, time_embed_dim)
                self.down_blocks.append(blk)
                self._modules[f'down_{len(self.down_blocks)}'] = blk
                in_c = out_c
            if level < len(channels) - 1:
                # Downsample (stride-2 conv)
                dw = Tensor(_nd_randn((in_c, in_c, 3, 3), 0.02))
                dw.requires_grad = True; self._parameters[f'down_stride_{level}'] = dw
                self.down_blocks.append(('downsample', dw, in_c))

        # Bottleneck
        self.mid_block1 = ResBlock(in_c, in_c, time_embed_dim)
        self.mid_block2 = ResBlock(in_c, in_c, time_embed_dim)
        self._modules['mid1'] = self.mid_block1
        self._modules['mid2'] = self.mid_block2

        # Build decoder (upsampling path)
        for level, out_c in reversed(list(enumerate(channels))):
            for _ in range(num_res_blocks + 1):
                blk = ResBlock(in_c + out_c, out_c, time_embed_dim)
                self.up_blocks.append(blk)
                self._modules[f'up_{len(self.up_blocks)}'] = blk
                in_c = out_c
            if level > 0:
                uw = Tensor(_nd_randn((in_c, in_c, 3, 3), 0.02))
                uw.requires_grad = True; self._parameters[f'up_stride_{level}'] = uw
                self.up_blocks.append(('upsample', uw, in_c))

        # Output head
        self.out_gn_w = Tensor(_nd_ones((channels[0],))); self.out_gn_w.requires_grad = True
        self.out_gn_b = Tensor(_nd_zeros((channels[0],))); self.out_gn_b.requires_grad = True
        self.out_w = Tensor(_nd_zeros((image_channels, channels[0], 3, 3)))
        self.out_w.requires_grad = True
        self._parameters['out_gn_w'] = self.out_gn_w; self._parameters['out_gn_b'] = self.out_gn_b
        self._parameters['out_w'] = self.out_w

    def q_sample(self, x0: NDArray, t: list) -> Tuple[NDArray, NDArray]:
        """Forward diffusion: add noise at step t."""
        eps = _nd_randn(x0.shape)
        x0f = x0.to_flat_list()
        ef = eps.to_flat_list()
        B, C, H, W = x0.shape
        x_t_flat = []
        for i, ti in enumerate(t):
            ab = self.alpha_bars[int(ti)]
            sq_ab = _math.sqrt(ab)
            sq_1mab = _math.sqrt(1 - ab)
            for idx in range(C * H * W):
                x_t_flat.append(sq_ab * x0f[i*C*H*W + idx] + sq_1mab * ef[i*C*H*W + idx])
        return _nd_arr(x_t_flat, dtype='float32').reshape(x0.shape), eps

    def _unet_forward(self, x: NDArray, t: list) -> NDArray:
        """UNet ε-prediction backbone."""
        t_emb = self.time_emb(t)   # (B, time_embed_dim)

        # Stem
        d = _conv2d(x, self.stem_w._data, None, stride=1, padding=1)

        # Encoder
        skips = []
        for blk in self.down_blocks:
            if isinstance(blk, tuple):
                name, w, c = blk
                if name == 'downsample':
                    skips.append(d)
                    d = _conv2d(d, w._data, None, stride=2, padding=1)
            else:
                d = blk(Tensor(d), t_emb)._data
                skips.append(d)

        # Bottleneck
        d = self.mid_block1(Tensor(d), t_emb)._data
        d = self.mid_block2(Tensor(d), t_emb)._data

        # Decoder
        for blk in self.up_blocks:
            if isinstance(blk, tuple):
                name, w, c = blk
                if name == 'upsample':
                    # Nearest-neighbor upsample ×2
                    B, C, H, W = d.shape
                    df = d.to_flat_list()
                    up_flat = []
                    for n_ in range(B):
                        for c_ in range(C):
                            for ih in range(H):
                                row = [df[n_*C*H*W + c_*H*W + ih*W + iw] for iw in range(W)]
                                up_flat.extend(row + row)  # repeat along W
                    # Now shape (B, C, H, 2W); repeat along H
                    W2 = W * 2
                    up2_flat = []
                    for n_ in range(B):
                        for c_ in range(C):
                            for ih in range(H):
                                row = up_flat[n_*C*H*W2 + c_*H*W2 + ih*W2:(n_*C*H*W2 + c_*H*W2 + ih*W2 + W2)]
                                up2_flat.extend(row + row)
                    d = _nd_arr(up2_flat, dtype='float32').reshape((B, C, H*2, W*2))
                    d = _conv2d(d, w._data, None, stride=1, padding=1)
            else:
                skip = skips.pop()
                if skip.shape != d.shape:
                    # Crop to min spatial size
                    hc = min(d.shape[2], skip.shape[2])
                    wc = min(d.shape[3], skip.shape[3])
                    Bd, Cd, Hd, Wd = d.shape
                    df2 = d.to_flat_list()
                    d_crop = [df2[n_*Cd*Hd*Wd + c_*Hd*Wd + ih*Wd + iw]
                              for n_ in range(Bd) for c_ in range(Cd) for ih in range(hc) for iw in range(wc)]
                    d = _nd_arr(d_crop, dtype='float32').reshape((Bd, Cd, hc, wc))
                    Bs, Cs, Hs, Ws = skip.shape
                    sf2 = skip.to_flat_list()
                    s_crop = [sf2[n_*Cs*Hs*Ws + c_*Hs*Ws + ih*Ws + iw]
                              for n_ in range(Bs) for c_ in range(Cs) for ih in range(hc) for iw in range(wc)]
                    skip = _nd_arr(s_crop, dtype='float32').reshape((Bs, Cs, hc, wc))
                from vectorphile.core.tensor import _nd_concatenate as _ddpm_cat
                d = _ddpm_cat([d, skip], axis=1)
                d = blk(Tensor(d), t_emb)._data

        # Output — group norm approximated as instance norm per channel
        N, C, H, W = d.shape
        df = d.to_flat_list()
        wf = self.out_gn_w._data.to_flat_list()
        bf = self.out_gn_b._data.to_flat_list()
        out_flat = []
        for n in range(N):
            for c in range(C):
                vals = [df[n*C*H*W + c*H*W + i*W + j] for i in range(H) for j in range(W)]
                mu = sum(vals) / len(vals)
                va = sum((v-mu)**2 for v in vals) / len(vals)
                inv = 1.0 / _math.sqrt(va + 1e-5)
                for v in vals:
                    out_flat.append((v - mu) * inv * wf[c] + bf[c])
        d = _nd_arr(out_flat, dtype='float32').reshape((N, C, H, W))
        d = _silu(d)
        return _conv2d(d, self.out_w._data, None, stride=1, padding=1)

    def forward(self, x0: Tensor, t=None) -> Tuple[Tensor, Tensor]:
        """Training step: sample random t, add noise, predict ε."""
        B = x0._data.shape[0]
        if t is None:
            t = [_random.randint(0, self.T - 1) for _ in range(B)]
        x_t, eps = self.q_sample(x0._data, t)
        eps_pred = self._unet_forward(x_t, t)
        return Tensor(eps_pred), Tensor(eps)

    @staticmethod
    def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
        pf = pred._data.to_flat_list()
        tf = target._data.to_flat_list()
        loss = sum((pf[i] - tf[i])**2 for i in range(len(pf))) / max(len(pf), 1)
        return Tensor(_nd_arr([loss], dtype='float32').reshape((1,)))

    def p_sample(self, x_t: NDArray, t: int) -> NDArray:
        """Single reverse diffusion step: sample x_{t-1} from x_t."""
        B = x_t.shape[0]
        t_arr = [t] * B
        eps_pred = self._unet_forward(x_t, t_arr)
        ab   = self.alpha_bars[t]
        ab_p = self.alpha_bars[t - 1] if t > 0 else 1.0
        b    = self.betas[t]
        a    = self.alphas[t]
        # Equation 11 Ho et al.
        xf = x_t.to_flat_list()
        ef = eps_pred.to_flat_list()
        sq_ab = _math.sqrt(ab)
        sq_1mab = _math.sqrt(1 - ab)
        x0f = [max(-1.0, min(1.0, (xf[i] - sq_1mab * ef[i]) / sq_ab)) for i in range(len(xf))]
        sq_ab_p = _math.sqrt(ab_p)
        sq_a = _math.sqrt(a)
        c1 = sq_ab_p * b / (1 - ab)
        c2 = sq_a * (1 - ab_p) / (1 - ab)
        mean_flat = [c1 * x0f[i] + c2 * xf[i] for i in range(len(xf))]
        if t == 0:
            return _nd_arr(mean_flat, dtype='float32').reshape(x_t.shape)
        variance = b * (1 - ab_p) / (1 - ab)
        noise = _nd_randn(x_t.shape)
        nf = noise.to_flat_list()
        sq_var = _math.sqrt(max(variance, 0.0))
        result = [mean_flat[i] + sq_var * nf[i] for i in range(len(mean_flat))]
        return _nd_arr(result, dtype='float32').reshape(x_t.shape)

    def sample(self, shape: Tuple, n: Optional[int] = None) -> NDArray:
        """Full DDPM reverse diffusion from pure noise."""
        if n is not None:
            shape = (n,) + shape
        x = _nd_randn(shape)
        for t in reversed(range(self.T)):
            x = self.p_sample(x, t)
        return x
