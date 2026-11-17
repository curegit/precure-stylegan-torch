from numpy import array, pad as padded, float32
import torch
from torch import nn
from torch.nn import functional as F
from stylegan.layers.basic import GaussianDistribution, LeakyRelu, EqualizedLinear, symmetric_pad2d


class LearnableConstant(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.c = nn.Parameter(torch.randn(channels, 4, 4))

    def forward(self, batch):
        return self.c.expand(batch, *self.c.shape)


class BicubicUpsampler(nn.Module):

    def __init__(self, b=0.0, c=0.5):
        super().__init__()
        self.b, self.c = b, c
        ys1 = array([self.kernel(i + 0.25) for i in range(-2, 2)])
        ys2 = array([self.kernel(i + 0.75) for i in range(-2, 2)])
        k1 = padded(ys1.reshape(1, 4) * ys1.reshape(4, 1), ((0, 1), (0, 1)))
        k2 = padded(ys2.reshape(1, 4) * ys1.reshape(4, 1), ((0, 1), (1, 0)))
        k3 = padded(ys1.reshape(1, 4) * ys2.reshape(4, 1), ((1, 0), (0, 1)))
        k4 = padded(ys2.reshape(1, 4) * ys2.reshape(4, 1), ((1, 0), (1, 0)))
        self.register_buffer("w", torch.from_numpy(array([[k1], [k2], [k3], [k4]], dtype=float32)), persistent=False)

    def forward(self, x):
        batch, channels, height, width = x.shape
        h1 = x.reshape(batch * channels, 1, height, width)
        h2 = symmetric_pad2d(h1, (2, 2, 2, 2))
        h3 = F.conv2d(h2, self.w)
        h4 = F.pixel_shuffle(h3, 2)
        return h4.reshape(batch, channels, height * 2, width * 2)

    def kernel(self, x):
        if abs(x) < 1:
            return 1 / 6 * ((12 - 9 * self.b - 6 * self.c) * abs(x) ** 3 + (-18 + 12 * self.b + 6 * self.c) * abs(x) ** 2 + (6 - 2 * self.b))
        elif 1 <= abs(x) < 2:
            return 1 / 6 * ((-self.b - 6 * self.c) * abs(x) ** 3 + (6 * self.b + 30 * self.c) * abs(x) ** 2 + (-12 * self.b - 48 * self.c) * abs(x) + (8 * self.b + 24 * self.c))
        else:
            return 0.0


class StyleAffineTransformation(nn.Module):

    def __init__(self, size, channels):
        super().__init__()
        self.linear = EqualizedLinear(size, channels, initial_bias=1.0, gain=1)

    def forward(self, w):
        return self.linear(w)


class WeightModulatedConvolution(nn.Module):

    def __init__(self, in_channels, out_channels, pointwise=False, demod=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.demod = demod
        self.ksize = 1 if pointwise else 3
        self.pad = 0 if pointwise else 1
        self.w = nn.Parameter(torch.randn(out_channels, in_channels, self.ksize, self.ksize))
        self.b = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x, y):
        out_channels = self.b.shape[0]
        batch, in_channels, height, width = x.shape
        modulated_w = self.w * y.reshape(batch, 1, in_channels, 1, 1)
        w = modulated_w / torch.sqrt(torch.sum(modulated_w**2, dim=(2, 3, 4), keepdim=True) + 1e-08) if self.demod else modulated_w
        grouped_w = w.reshape(batch * out_channels, in_channels, self.ksize, self.ksize)
        grouped_x = x.reshape(1, batch * in_channels, height, width)
        padded_grouped_x = F.pad(grouped_x, (self.pad,) * 4, mode="replicate") if self.pad else grouped_x
        h = F.conv2d(padded_grouped_x, grouped_w, stride=1, padding=0, groups=batch)
        return h.reshape(batch, out_channels, height, width) + self.b.reshape(1, out_channels, 1, 1)


class NoiseAdder(nn.Module):

    def __init__(self, id=0):
        super().__init__()
        self.id = id
        self.sampler = GaussianDistribution(self)
        self.s = nn.Parameter(torch.zeros(1))

    def forward(self, x, coefficient=1.0, fixed=None):
        if coefficient == 0.0:
            return x
        else:
            batch, channels, height, width = x.shape
            scale = self.s if coefficient == 1.0 else coefficient * self.s
            if fixed is None:
                return x + scale * self.sampler(batch, 1, height, width)
            else:
                return x + scale * self.sampler.deterministic(1, 1, height, width, seed=(fixed + self.id))


class ToRGB(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.wmconv = WeightModulatedConvolution(in_channels, 3, pointwise=True, demod=False)

    def forward(self, x, y):
        return self.wmconv(x, y)


class InitialSkipArchitecture(nn.Module):

    def __init__(self, size, in_channels, out_channels, level=1):
        super().__init__()
        self.const = LearnableConstant(in_channels)
        self.style1 = StyleAffineTransformation(size, in_channels)
        self.wmconv = WeightModulatedConvolution(in_channels, out_channels)
        self.noise = NoiseAdder(level * 10 + 1)
        self.act = LeakyRelu()
        self.style2 = StyleAffineTransformation(size, out_channels)
        self.torgb = ToRGB(out_channels)

    def forward(self, w, noise=1.0, fixed=None):
        h1 = self.const(w.shape[0])
        h2 = self.wmconv(h1, self.style1(w))
        h3 = self.noise(h2, coefficient=noise, fixed=fixed)
        h4 = self.act(h3)
        return h4, self.torgb(h4, self.style2(w))

    @property
    def channels(self):
        yield self.wmconv.in_channels
        yield self.wmconv.out_channels


class SkipArchitecture(nn.Module):

    def __init__(self, size, in_channels, out_channels, level=2):
        super().__init__()
        self.up = BicubicUpsampler()
        self.style1 = StyleAffineTransformation(size, in_channels)
        self.wmconv1 = WeightModulatedConvolution(in_channels, out_channels)
        self.noise1 = NoiseAdder(level * 10 + 1)
        self.act1 = LeakyRelu()
        self.style2 = StyleAffineTransformation(size, out_channels)
        self.wmconv2 = WeightModulatedConvolution(out_channels, out_channels)
        self.noise2 = NoiseAdder(level * 10 + 2)
        self.act2 = LeakyRelu()
        self.style3 = StyleAffineTransformation(size, out_channels)
        self.torgb = ToRGB(out_channels)
        self.skip = BicubicUpsampler(1, 0)

    def forward(self, x, y, w, noise=1.0, fixed=None):
        h1 = self.up(x)
        h2 = self.wmconv1(h1, self.style1(w))
        h3 = self.noise1(h2, coefficient=noise, fixed=fixed)
        h4 = self.act1(h3)
        h5 = self.wmconv2(h4, self.style2(w))
        h6 = self.noise2(h5, coefficient=noise, fixed=fixed)
        h7 = self.act2(h6)
        return h7, self.skip(y) + self.torgb(h7, self.style3(w))

    @property
    def channels(self):
        yield self.wmconv1.in_channels
        yield self.wmconv1.out_channels
        yield self.wmconv2.out_channels
