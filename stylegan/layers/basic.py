from math import sqrt
from numpy import float32
from numpy.random import Generator, PCG64
import torch
from torch import nn
from torch.nn import functional as F


class GaussianDistribution:

    def __init__(self, link, mean=0.0, sd=1.0):
        self.link = link
        self.mean = mean
        self.sd = sd

    def __call__(self, *shape):
        parameter = next(self.link.parameters())
        return torch.randn(shape, device=parameter.device, dtype=parameter.dtype) * self.sd + self.mean

    def deterministic(self, *shape, seed=0):
        # Keep the original CPU RNG so fixed noise survives the framework change.
        x = Generator(PCG64(seed)).normal(size=shape, loc=self.mean, scale=self.sd).astype(float32)
        parameter = next(self.link.parameters())
        return torch.as_tensor(x, device=parameter.device, dtype=parameter.dtype)


class LeakyRelu(nn.Module):

    def __init__(self, a=0.2):
        super().__init__()
        self.a = a

    def forward(self, x):
        return F.leaky_relu(x, self.a)


class EqualizedLinear(nn.Module):

    def __init__(self, in_size, out_size, nobias=False, initial_bias=0.0, gain=sqrt(2)):
        super().__init__()
        self.c = gain / sqrt(in_size)
        self.w = nn.Parameter(torch.randn(out_size, in_size))
        self.b = None if nobias else nn.Parameter(torch.full((out_size,), float(initial_bias)))

    def forward(self, x):
        return F.linear(self.c * x.flatten(1), self.w, self.b)


class EqualizedConvolution2D(nn.Module):

    def __init__(self, in_channels, out_channels, ksize=3, stride=1, pad=0, nobias=False, initial_bias=0.0, gain=sqrt(2)):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.pad = pad
        self.c = gain / sqrt(in_channels * ksize**2)
        self.w = nn.Parameter(torch.randn(out_channels, in_channels, ksize, ksize))
        self.b = None if nobias else nn.Parameter(torch.full((out_channels,), float(initial_bias)))

    def forward(self, x):
        return F.conv2d(self.c * x, self.w, self.b, self.stride, self.pad)


def symmetric_pad2d(x, padding):
    """Pad including the boundary pixels, matching NumPy/Chainer symmetric mode."""
    left, right, top, bottom = padding
    height, width = x.shape[-2:]
    h = torch.arange(-top, height + bottom, device=x.device).remainder(2 * height)
    w = torch.arange(-left, width + right, device=x.device).remainder(2 * width)
    h = torch.minimum(h, 2 * height - 1 - h)
    w = torch.minimum(w, 2 * width - 1 - w)
    return x.index_select(-2, h).index_select(-1, w)
