"""Pooling layers — MaxPool, AvgPool, AdaptiveAvgPool."""
from __future__ import annotations
import numpy as np
from vectorphile.nn.module import Module
from vectorphile.core.tensor import Tensor


class MaxPool1d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        N, C, L = x._data.shape
        p, k, s = self.padding, self.kernel_size, self.stride
        x_pad = np.pad(x._data, ((0,0),(0,0),(p,p)), constant_values=-np.inf)
        Lo = (L + 2*p - k) // s + 1
        out = np.full((N, C, Lo), -np.inf, dtype=np.float32)
        for i in range(Lo):
            out[:, :, i] = x_pad[:, :, i*s:i*s+k].max(axis=2)
        return Tensor(out, device=x.device)

    def __repr__(self):
        return f'MaxPool1d(kernel_size={self.kernel_size}, stride={self.stride})'


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = (stride if isinstance(stride, tuple) else (stride, stride)) if stride else self.kernel_size
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x._data.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        x_pad = np.pad(x._data, ((0,0),(0,0),(ph,ph),(pw,pw)), constant_values=-np.inf)
        Ho = (H + 2*ph - kh) // sh + 1
        Wo = (W + 2*pw - kw) // sw + 1
        out = np.full((N, C, Ho, Wo), -np.inf, dtype=np.float32)
        for i in range(Ho):
            for j in range(Wo):
                out[:, :, i, j] = x_pad[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw].max(axis=(2, 3))
        return Tensor(out, device=x.device)

    def __repr__(self):
        return f'MaxPool2d(kernel_size={self.kernel_size}, stride={self.stride})'


class MaxPool3d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,)*3
        self.stride = (stride if isinstance(stride, tuple) else (stride,)*3) if stride else self.kernel_size
        self.padding = padding if isinstance(padding, tuple) else (padding,)*3

    def forward(self, x: Tensor) -> Tensor:
        N, C, D, H, W = x._data.shape
        kd, kh, kw = self.kernel_size
        sd, sh, sw = self.stride
        pd, ph, pw = self.padding
        x_pad = np.pad(x._data, ((0,0),(0,0),(pd,pd),(ph,ph),(pw,pw)), constant_values=-np.inf)
        Do = (D + 2*pd - kd) // sd + 1
        Ho = (H + 2*ph - kh) // sh + 1
        Wo = (W + 2*pw - kw) // sw + 1
        out = np.full((N, C, Do, Ho, Wo), -np.inf, dtype=np.float32)
        for d in range(Do):
            for i in range(Ho):
                for j in range(Wo):
                    patch = x_pad[:, :, d*sd:d*sd+kd, i*sh:i*sh+kh, j*sw:j*sw+kw]
                    out[:, :, d, i, j] = patch.max(axis=(2, 3, 4))
        return Tensor(out, device=x.device)


class AvgPool1d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        N, C, L = x._data.shape
        p, k, s = self.padding, self.kernel_size, self.stride
        x_pad = np.pad(x._data, ((0,0),(0,0),(p,p)), constant_values=0)
        Lo = (L + 2*p - k) // s + 1
        out = np.zeros((N, C, Lo), dtype=np.float32)
        for i in range(Lo):
            out[:, :, i] = x_pad[:, :, i*s:i*s+k].mean(axis=2)
        return Tensor(out, device=x.device)


class AvgPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = (stride if isinstance(stride, tuple) else (stride, stride)) if stride else self.kernel_size
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x._data.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        x_pad = np.pad(x._data, ((0,0),(0,0),(ph,ph),(pw,pw)), constant_values=0)
        Ho = (H + 2*ph - kh) // sh + 1
        Wo = (W + 2*pw - kw) // sw + 1
        out = np.zeros((N, C, Ho, Wo), dtype=np.float32)
        for i in range(Ho):
            for j in range(Wo):
                out[:, :, i, j] = x_pad[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw].mean(axis=(2, 3))
        return Tensor(out, device=x.device)

    def __repr__(self):
        return f'AvgPool2d(kernel_size={self.kernel_size}, stride={self.stride})'


class AvgPool3d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,)*3
        self.stride = (stride if isinstance(stride, tuple) else (stride,)*3) if stride else self.kernel_size
        self.padding = padding if isinstance(padding, tuple) else (padding,)*3

    def forward(self, x: Tensor) -> Tensor:
        N, C, D, H, W = x._data.shape
        kd, kh, kw = self.kernel_size
        sd, sh, sw = self.stride
        pd, ph, pw = self.padding
        x_pad = np.pad(x._data, ((0,0),(0,0),(pd,pd),(ph,ph),(pw,pw)), constant_values=0)
        Do = (D + 2*pd - kd) // sd + 1
        Ho = (H + 2*ph - kh) // sh + 1
        Wo = (W + 2*pw - kw) // sw + 1
        out = np.zeros((N, C, Do, Ho, Wo), dtype=np.float32)
        for d in range(Do):
            for i in range(Ho):
                for j in range(Wo):
                    patch = x_pad[:, :, d*sd:d*sd+kd, i*sh:i*sh+kh, j*sw:j*sw+kw]
                    out[:, :, d, i, j] = patch.mean(axis=(2, 3, 4))
        return Tensor(out, device=x.device)


class AdaptiveAvgPool1d(Module):
    def __init__(self, output_size: int):
        super().__init__()
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        N, C, L = x._data.shape
        out = np.zeros((N, C, self.output_size), dtype=np.float32)
        for i in range(self.output_size):
            start = int(np.floor(i * L / self.output_size))
            end = int(np.ceil((i + 1) * L / self.output_size))
            out[:, :, i] = x._data[:, :, start:end].mean(axis=2)
        return Tensor(out, device=x.device)


class AdaptiveAvgPool2d(Module):
    def __init__(self, output_size):
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x._data.shape
        oh, ow = self.output_size
        out = np.zeros((N, C, oh, ow), dtype=np.float32)
        for i in range(oh):
            hs = int(np.floor(i * H / oh))
            he = int(np.ceil((i + 1) * H / oh))
            for j in range(ow):
                ws = int(np.floor(j * W / ow))
                we = int(np.ceil((j + 1) * W / ow))
                out[:, :, i, j] = x._data[:, :, hs:he, ws:we].mean(axis=(2, 3))
        return Tensor(out, device=x.device)

    def __repr__(self):
        return f'AdaptiveAvgPool2d(output_size={self.output_size})'


class AdaptiveMaxPool2d(Module):
    def __init__(self, output_size):
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x._data.shape
        oh, ow = self.output_size
        out = np.full((N, C, oh, ow), -np.inf, dtype=np.float32)
        for i in range(oh):
            hs = int(np.floor(i * H / oh))
            he = int(np.ceil((i + 1) * H / oh))
            for j in range(ow):
                ws = int(np.floor(j * W / ow))
                we = int(np.ceil((j + 1) * W / ow))
                out[:, :, i, j] = x._data[:, :, hs:he, ws:we].max(axis=(2, 3))
        return Tensor(out, device=x.device)


# Convenience alias
GlobalAvgPool2d = lambda: AdaptiveAvgPool2d(1)
