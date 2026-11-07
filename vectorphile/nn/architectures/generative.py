"""Generative architectures — VAE, DCGAN, WGAN-GP, DDPM."""
from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, List
from vectorphile.nn.module import Module
from vectorphile.core.tensor import Tensor


def _relu(x): return np.maximum(x, 0)
def _silu(x): return x * (1 / (1 + np.exp(-x)))
def _sigmoid(x): return 1 / (1 + np.exp(-x))
def _tanh(x): return np.tanh(x)
def _leaky_relu(x, neg=0.2): return np.where(x > 0, x, neg * x)


def _linear(x, w, b=None):
    out = x @ w
    return out + b if b is not None else out


def _layer_norm(x, w, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def _conv_transpose2d(x: np.ndarray, w: np.ndarray, b: Optional[np.ndarray],
                       stride: int, padding: int, output_padding: int = 0) -> np.ndarray:
    """Transposed conv2d (NCHW) via upsample + regular conv."""
    N, C_in, H, W = x.shape
    C_out, _, kH, kW = w.shape  # weight shape: (C_in, C_out, kH, kW)
    # Upsample
    H_up = H * stride
    W_up = W * stride
    x_up = np.zeros((N, C_in, H_up + output_padding, W_up + output_padding), dtype=np.float32)
    x_up[:, :, ::stride, ::stride] = x
    # Pad
    p = kH - 1 - padding
    if p > 0:
        x_up = np.pad(x_up, ((0,0),(0,0),(p,p),(p,p)))
    H_out = x_up.shape[2] - kH + 1
    W_out = x_up.shape[3] - kW + 1
    out = np.zeros((N, C_out, H_out, W_out), dtype=np.float32)
    # w is stored as (C_in, C_out, kH, kW); flip for transpose conv
    w_flip = w[:, :, ::-1, ::-1]
    for i in range(H_out):
        for j in range(W_out):
            patch = x_up[:, :, i:i+kH, j:j+kW]
            out[:, :, i, j] = np.einsum('nchw,cohw->no', patch, w_flip)
    if b is not None:
        out += b[None, :, None, None]
    return out


def _conv2d(x, w, b=None, stride=1, padding=1):
    N, C_in, H, W = x.shape
    C_out, _, kH, kW = w.shape
    H_out = (H + 2 * padding - kH) // stride + 1
    W_out = (W + 2 * padding - kW) // stride + 1
    if padding > 0:
        x = np.pad(x, ((0,0),(0,0),(padding,padding),(padding,padding)))
    out = np.zeros((N, C_out, H_out, W_out), dtype=np.float32)
    for i in range(H_out):
        for j in range(W_out):
            out[:, :, i, j] = np.einsum('nchw,ochw->no',
                x[:, :, i*stride:i*stride+kH, j*stride:j*stride+kW], w)
    if b is not None:
        out += b[None, :, None, None]
    return out


def _batch_norm2d(x, w, b, rm, rv, training, momentum=0.1, eps=1e-5):
    if training:
        mean = x.mean((0, 2, 3), keepdims=True)
        var  = x.var((0, 2, 3), keepdims=True)
        rm[:] = (1 - momentum) * rm + momentum * x.mean((0, 2, 3))
        rv[:] = (1 - momentum) * rv + momentum * x.var((0, 2, 3))
    else:
        mean = rm[None, :, None, None]
        var  = rv[None, :, None, None]
    return (x - mean) / np.sqrt(var + eps) * w[None, :, None, None] + b[None, :, None, None]


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
            w = Tensor(np.random.randn(enc_dims[i], enc_dims[i+1]).astype(np.float32) * np.sqrt(2 / enc_dims[i]))
            b = Tensor(np.zeros(enc_dims[i+1], dtype=np.float32))
            w.requires_grad = True; b.requires_grad = True
            setattr(self, f'enc_w{i}', w); setattr(self, f'enc_b{i}', b)
            self._parameters[f'enc_w{i}'] = w; self._parameters[f'enc_b{i}'] = b
        self.n_enc = len(hidden_dims)
        # μ and log σ² heads
        for head in ['mu', 'logvar']:
            w = Tensor(np.random.randn(hidden_dims[-1], latent_dim).astype(np.float32) * 0.01)
            b = Tensor(np.zeros(latent_dim, dtype=np.float32))
            w.requires_grad = True; b.requires_grad = True
            setattr(self, f'{head}_w', w); setattr(self, f'{head}_b', b)
            self._parameters[f'{head}_w'] = w; self._parameters[f'{head}_b'] = b
        # Decoder
        dec_dims = [latent_dim] + list(reversed(hidden_dims)) + [input_dim]
        self.n_dec = len(dec_dims) - 1
        for i in range(self.n_dec):
            w = Tensor(np.random.randn(dec_dims[i], dec_dims[i+1]).astype(np.float32) * np.sqrt(2 / dec_dims[i]))
            b = Tensor(np.zeros(dec_dims[i+1], dtype=np.float32))
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
        eps = np.random.randn(*mu._data.shape).astype(np.float32)
        std = np.exp(0.5 * logvar._data)
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
        # Binary cross-entropy reconstruction
        recon_loss = -(x._data * np.log(recon._data + 1e-8) +
                       (1 - x._data) * np.log(1 - recon._data + 1e-8)).mean()
        # KL divergence: -0.5 * sum(1 + logvar - μ² - exp(logvar))
        kl = -0.5 * (1 + logvar._data - mu._data ** 2 - np.exp(logvar._data)).mean()
        return Tensor(np.array(recon_loss + beta * kl, dtype=np.float32))

    def sample(self, n: int) -> Tensor:
        """Sample n images from the prior N(0, I)."""
        z = np.random.randn(n, self.latent_dim).astype(np.float32)
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
        n_up = int(np.log2(image_size)) - 3   # number of upsample blocks
        # Initial projection
        init_channels = ngf * (2 ** n_up)
        self.proj_w = Tensor(np.random.randn(latent_dim, init_channels * 4 * 4).astype(np.float32) * 0.02)
        self.proj_b = Tensor(np.zeros(init_channels * 4 * 4, dtype=np.float32))
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
            w = Tensor(np.random.randn(c_in, c_out, 4, 4).astype(np.float32) * 0.02)
            w.requires_grad = True
            self.up_ws.append(w)
            self._parameters[f'up_w{i}'] = w
            bn_w = Tensor(np.ones(c_out, dtype=np.float32)); bn_w.requires_grad = True
            bn_b = Tensor(np.zeros(c_out, dtype=np.float32)); bn_b.requires_grad = True
            self.up_bn_w.append(bn_w); self.up_bn_b.append(bn_b)
            self._parameters[f'up_bn_w{i}'] = bn_w; self._parameters[f'up_bn_b{i}'] = bn_b
            self.up_rm.append(np.zeros(c_out, dtype=np.float32))
            self.up_rv.append(np.ones(c_out, dtype=np.float32))
            c_in = c_out
        # Final conv: → nc channels, tanh
        self.final_w = Tensor(np.random.randn(c_in, nc, 4, 4).astype(np.float32) * 0.02)
        self.final_b = Tensor(np.zeros(nc, dtype=np.float32))
        self.final_w.requires_grad = True; self.final_b.requires_grad = True
        self._parameters['final_w'] = self.final_w; self._parameters['final_b'] = self.final_b

    def forward(self, z: Tensor) -> Tensor:
        """z: (B, latent_dim) → (B, nc, H, W) in [-1, 1]."""
        d = (_linear(z._data, self.proj_w._data, self.proj_b._data)
             .reshape(-1, self.init_channels, 4, 4))
        d = _relu(d)
        for i, (w, bn_w, bn_b, rm, rv) in enumerate(
                zip(self.up_ws, self.up_bn_w, self.up_bn_b, self.up_rm, self.up_rv)):
            d = _conv_transpose2d(d, w._data, None, stride=2, padding=1)
            d = _batch_norm2d(d, bn_w._data, bn_b._data, rm, rv, self.training)
            d = _relu(d)
        d = _conv_transpose2d(d, self.final_w._data, self.final_b._data, stride=2, padding=1)
        return Tensor(_tanh(d))


class DCGANDiscriminator(Module):
    """DCGAN Discriminator — conv → leaky ReLU → BN → sigmoid."""

    def __init__(self, ndf: int = 64, nc: int = 3, image_size: int = 64):
        super().__init__()
        n_down = int(np.log2(image_size)) - 3
        self.down_ws = []
        self.down_rm = []
        self.down_rv = []
        self.down_bn_w = []
        self.down_bn_b = []
        c_in = nc
        c_out = ndf
        for i in range(n_down):
            w = Tensor(np.random.randn(c_out, c_in, 4, 4).astype(np.float32) * 0.02)
            w.requires_grad = True
            self.down_ws.append(w)
            self._parameters[f'down_w{i}'] = w
            bn_w = Tensor(np.ones(c_out, dtype=np.float32)); bn_w.requires_grad = True
            bn_b = Tensor(np.zeros(c_out, dtype=np.float32)); bn_b.requires_grad = True
            self.down_bn_w.append(bn_w); self.down_bn_b.append(bn_b)
            self._parameters[f'down_bn_w{i}'] = bn_w; self._parameters[f'down_bn_b{i}'] = bn_b
            self.down_rm.append(np.zeros(c_out, dtype=np.float32))
            self.down_rv.append(np.ones(c_out, dtype=np.float32))
            c_in = c_out; c_out *= 2
        # Final classifier
        self.final_w = Tensor(np.random.randn(c_in * 4 * 4, 1).astype(np.float32) * 0.02)
        self.final_b = Tensor(np.zeros(1, dtype=np.float32))
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
        d = d.reshape(d.shape[0], -1)
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
        z = Tensor(np.random.randn(n, self.latent_dim).astype(np.float32))
        return self.generator(z)

    def discriminator_loss(self, real: Tensor, fake: Tensor) -> Tensor:
        """Non-saturating GAN loss for discriminator."""
        d_real = self.discriminator(real)._data
        d_fake = self.discriminator(fake)._data
        loss = (-np.log(_sigmoid(d_real) + 1e-8) - np.log(1 - _sigmoid(d_fake) + 1e-8)).mean()
        return Tensor(np.array(loss, dtype=np.float32))

    def generator_loss(self, fake: Tensor) -> Tensor:
        d_fake = self.discriminator(fake)._data
        loss = -np.log(_sigmoid(d_fake) + 1e-8).mean()
        return Tensor(np.array(loss, dtype=np.float32))

    def gradient_penalty(self, real: Tensor, fake: Tensor, lam: float = 10.0) -> float:
        """WGAN-GP gradient penalty (approximate — full autograd needed for exact)."""
        alpha = np.random.rand(real._data.shape[0], 1, 1, 1).astype(np.float32)
        interpolated = alpha * real._data + (1 - alpha) * fake._data
        d_interp = self.discriminator(Tensor(interpolated))._data
        # Numerical gradient estimate (finite difference)
        eps = 1e-4
        gp = 0.0
        for _ in range(4):  # sample a few dimensions
            idx = tuple([np.random.randint(s) for s in interpolated.shape[1:]])
            x_plus = interpolated.copy()
            x_plus[(slice(None),) + idx] += eps
            d_plus = self.discriminator(Tensor(x_plus))._data
            grad = (d_plus - d_interp) / eps
            gp += (grad ** 2).mean()
        return float(lam * gp / 4)

    def forward(self, z: Tensor) -> Tensor:
        return self.generator(z)


# ── DDPM ─────────────────────────────────────────────────────────────────────

class SinusoidalTimeEmbedding(Module):
    """Sinusoidal time step embedding (Ho et al. 2020 / Transformer positional enc)."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj_w = Tensor(np.random.randn(embed_dim, embed_dim * 4).astype(np.float32) * 0.02)
        self.proj_b = Tensor(np.zeros(embed_dim * 4, dtype=np.float32))
        self.out_w  = Tensor(np.random.randn(embed_dim * 4, embed_dim).astype(np.float32) * 0.02)
        self.out_b  = Tensor(np.zeros(embed_dim, dtype=np.float32))
        for n, p in [('proj_w',self.proj_w),('proj_b',self.proj_b),
                     ('out_w',self.out_w),('out_b',self.out_b)]:
            p.requires_grad = True; self._parameters[n] = p

    def forward(self, t: np.ndarray) -> np.ndarray:
        """t: (B,) int timesteps → (B, embed_dim) embedding."""
        half = self.embed_dim // 2
        freq = np.exp(-np.log(10000) * np.arange(half, dtype=np.float32) / half)
        args = t[:, None].astype(np.float32) * freq[None, :]
        emb = np.concatenate([np.cos(args), np.sin(args)], axis=-1)
        emb = _silu(_linear(emb, self.proj_w._data, self.proj_b._data))
        return _linear(emb, self.out_w._data, self.out_b._data)


class ResBlock(Module):
    """UNet-style residual block with time conditioning."""

    def __init__(self, in_c: int, out_c: int, time_dim: int, groups: int = 8):
        super().__init__()
        # Conv 1
        self.c1_w = Tensor(np.random.randn(out_c, in_c, 3, 3).astype(np.float32) * 0.02)
        self.c1_w.requires_grad = True; self._parameters['c1_w'] = self.c1_w
        # Group norm params (approximate: use layer norm per channel)
        self.gn1_w = Tensor(np.ones(in_c, dtype=np.float32)); self.gn1_w.requires_grad = True
        self.gn1_b = Tensor(np.zeros(in_c, dtype=np.float32)); self.gn1_b.requires_grad = True
        self._parameters['gn1_w'] = self.gn1_w; self._parameters['gn1_b'] = self.gn1_b
        # Time projection
        self.t_w = Tensor(np.random.randn(time_dim, out_c).astype(np.float32) * 0.02)
        self.t_w.requires_grad = True; self._parameters['t_w'] = self.t_w
        # Conv 2
        self.c2_w = Tensor(np.random.randn(out_c, out_c, 3, 3).astype(np.float32) * 0.02)
        self.c2_w.requires_grad = True; self._parameters['c2_w'] = self.c2_w
        self.gn2_w = Tensor(np.ones(out_c, dtype=np.float32)); self.gn2_w.requires_grad = True
        self.gn2_b = Tensor(np.zeros(out_c, dtype=np.float32)); self.gn2_b.requires_grad = True
        self._parameters['gn2_w'] = self.gn2_w; self._parameters['gn2_b'] = self.gn2_b
        # Residual projection
        self.skip_w = None
        if in_c != out_c:
            self.skip_w = Tensor(np.random.randn(out_c, in_c, 1, 1).astype(np.float32) * 0.02)
            self.skip_w.requires_grad = True; self._parameters['skip_w'] = self.skip_w

    def _gn(self, x, w, b):
        # Approximate group norm with instance norm per channel
        mean = x.mean((-2, -1), keepdims=True)
        var  = x.var((-2, -1), keepdims=True)
        return (x - mean) / np.sqrt(var + 1e-5) * w[None, :, None, None] + b[None, :, None, None]

    def forward(self, x: Tensor, t_emb: np.ndarray) -> Tensor:
        d = _silu(self._gn(x._data, self.gn1_w._data, self.gn1_b._data))
        d = _conv2d(d, self.c1_w._data, None, stride=1, padding=1)
        # Add time embedding
        t_proj = (t_emb @ self.t_w._data)[:, :, None, None]
        d = d + t_proj
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
        betas = np.linspace(beta_start, beta_end, T, dtype=np.float64)
        alphas = 1.0 - betas
        alpha_bars = np.cumprod(alphas)
        self.betas      = betas.astype(np.float32)
        self.alpha_bars = alpha_bars.astype(np.float32)
        self.alphas     = alphas.astype(np.float32)

        # Time embedding
        self.time_emb = SinusoidalTimeEmbedding(time_embed_dim)
        self._modules['time_emb'] = self.time_emb

        # UNet encoder/decoder with skip connections
        self.down_blocks = []
        self.up_blocks   = []
        self.res_blocks  = []

        channels = [base_channels * m for m in channel_mults]

        # Stem
        self.stem_w = Tensor(np.random.randn(channels[0], image_channels, 3, 3).astype(np.float32) * 0.02)
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
                dw = Tensor(np.random.randn(in_c, in_c, 3, 3).astype(np.float32) * 0.02)
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
                uw = Tensor(np.random.randn(in_c, in_c, 3, 3).astype(np.float32) * 0.02)
                uw.requires_grad = True; self._parameters[f'up_stride_{level}'] = uw
                self.up_blocks.append(('upsample', uw, in_c))

        # Output head
        self.out_gn_w = Tensor(np.ones(channels[0], dtype=np.float32)); self.out_gn_w.requires_grad = True
        self.out_gn_b = Tensor(np.zeros(channels[0], dtype=np.float32)); self.out_gn_b.requires_grad = True
        self.out_w = Tensor(np.zeros((image_channels, channels[0], 3, 3), dtype=np.float32))
        self.out_w.requires_grad = True
        self._parameters['out_gn_w'] = self.out_gn_w; self._parameters['out_gn_b'] = self.out_gn_b
        self._parameters['out_w'] = self.out_w

    def q_sample(self, x0: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward diffusion: add noise at step t.

        Returns (x_t, eps) where eps is the added noise.
        """
        eps = np.random.randn(*x0.shape).astype(np.float32)
        ab = self.alpha_bars[t][:, None, None, None]
        x_t = np.sqrt(ab) * x0 + np.sqrt(1 - ab) * eps
        return x_t, eps

    def _unet_forward(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
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
                    # Bilinear upsample (approximate)
                    B, C, H, W = d.shape
                    d = d.repeat(2, axis=2).repeat(2, axis=3)
                    d = _conv2d(d, w._data, None, stride=1, padding=1)
            else:
                skip = skips.pop()
                if skip.shape != d.shape:
                    # Pad or crop to match (simple crop)
                    h, w = min(d.shape[2], skip.shape[2]), min(d.shape[3], skip.shape[3])
                    d    = d[:, :, :h, :w]
                    skip = skip[:, :, :h, :w]
                d = np.concatenate([d, skip], axis=1)
                d = blk(Tensor(d), t_emb)._data

        # Output
        mean = d.mean((-2, -1), keepdims=True)
        var  = d.var((-2, -1), keepdims=True)
        d = (d - mean) / np.sqrt(var + 1e-5) * self.out_gn_w._data[None, :, None, None] + self.out_gn_b._data[None, :, None, None]
        d = _silu(d)
        return _conv2d(d, self.out_w._data, None, stride=1, padding=1)

    def forward(self, x0: Tensor, t: Optional[np.ndarray] = None) -> Tuple[Tensor, Tensor]:
        """Training step: sample random t, add noise, predict ε.

        Returns:
            (eps_pred, eps_true) for computing MSE loss
        """
        B = x0._data.shape[0]
        if t is None:
            t = np.random.randint(0, self.T, size=(B,))
        x_t, eps = self.q_sample(x0._data, t)
        eps_pred = self._unet_forward(x_t, t)
        return Tensor(eps_pred), Tensor(eps)

    @staticmethod
    def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
        return Tensor(np.array(((pred._data - target._data) ** 2).mean(), dtype=np.float32))

    def p_sample(self, x_t: np.ndarray, t: int) -> np.ndarray:
        """Single reverse diffusion step: sample x_{t-1} from x_t."""
        t_arr = np.full((x_t.shape[0],), t, dtype=int)
        eps_pred = self._unet_forward(x_t, t_arr)
        ab   = self.alpha_bars[t]
        ab_p = self.alpha_bars[t - 1] if t > 0 else 1.0
        b    = self.betas[t]
        a    = self.alphas[t]
        # Equation 11 Ho et al.
        x0_pred = (x_t - np.sqrt(1 - ab) * eps_pred) / np.sqrt(ab)
        x0_pred = np.clip(x0_pred, -1, 1)
        mean = (np.sqrt(ab_p) * b / (1 - ab)) * x0_pred + \
               (np.sqrt(a) * (1 - ab_p) / (1 - ab)) * x_t
        if t == 0:
            return mean
        variance = b * (1 - ab_p) / (1 - ab)
        noise = np.random.randn(*x_t.shape).astype(np.float32)
        return mean + np.sqrt(variance) * noise

    def sample(self, shape: Tuple, n: Optional[int] = None) -> np.ndarray:
        """Full DDPM reverse diffusion from pure noise.

        Args:
            shape: (B, C, H, W) or just (C, H, W) with n specified
        Returns:
            Generated images as numpy array in [-1, 1]
        """
        if n is not None:
            shape = (n,) + shape
        x = np.random.randn(*shape).astype(np.float32)
        for t in reversed(range(self.T)):
            x = self.p_sample(x, t)
        return x
