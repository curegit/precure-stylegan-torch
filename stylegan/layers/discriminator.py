from math import sqrt as root
from numpy import array, sinc, float32
import torch
from torch import nn
from torch.nn import functional as F
from stylegan.layers.basic import LeakyRelu, EqualizedLinear, EqualizedConvolution2D, symmetric_pad2d


class Downsampler(nn.Module):

    def __init__(self, lanczos=True, n=2):
        super().__init__()
        self.lanczos = lanczos
        if lanczos:
            self.n = n
            kernel = lambda x: sinc(x) * sinc(x / n)
            ys = array([kernel(i / 2 + 0.25) for i in range(-2 * n, 2 * n)])
            k = ys.reshape(1, n * 4) * ys.reshape(n * 4, 1)
            self.register_buffer("w", torch.from_numpy(array([[k / k.sum()]], dtype=float32)), persistent=False)

    def forward(self, x):
        if self.lanczos:
            p = 2 * self.n - 1
            batch, channels, height, width = x.shape
            h1 = x.reshape(batch * channels, 1, height, width)
            h2 = symmetric_pad2d(h1, (p, p, p, p))
            h3 = F.conv2d(h2, self.w, stride=2)
            return h3.reshape(batch, channels, height // 2, width // 2)
        else:
            return F.avg_pool2d(x, kernel_size=2, stride=2)


class MinibatchStandardDeviation(nn.Module):

    def __init__(self, group_size=None):
        super().__init__()
        self.group_size = group_size

    def forward(self, x):
        batch, channels, height, width = x.shape
        group_size = min(batch, self.group_size or batch)
        if group_size < 1 or batch % group_size:
            raise ValueError("Minibatch standard deviation group size must be positive and divide the batch size")
        groups = batch // group_size
        grouped_x = x.reshape(groups, group_size, channels, height, width)
        variance = torch.mean((grouped_x - torch.mean(grouped_x, dim=1, keepdim=True)) ** 2, dim=1, keepdim=True)
        deviation = torch.mean(torch.sqrt(variance + 1e-08), dim=(2, 3, 4), keepdim=True)
        deviation_map = deviation.expand(groups, group_size, 1, height, width).reshape(batch, 1, height, width)
        return torch.cat((x, deviation_map), dim=1)


class FromRGB(nn.Module):

    def __init__(self, out_channels):
        super().__init__()
        self.conv = EqualizedConvolution2D(3, out_channels, ksize=1, stride=1, pad=0)
        self.act = LeakyRelu()

    def forward(self, x):
        return self.act(self.conv(x))


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = EqualizedConvolution2D(in_channels, in_channels, ksize=3, stride=1, pad=1)
        self.act1 = LeakyRelu()
        self.conv2 = EqualizedConvolution2D(in_channels, out_channels, ksize=3, stride=1, pad=1)
        self.act2 = LeakyRelu()
        self.pool = Downsampler(lanczos=False)
        self.conv3 = EqualizedConvolution2D(in_channels, out_channels, ksize=1, stride=1, pad=0, nobias=True)
        self.down = Downsampler()

    def forward(self, x):
        h = self.pool(self.act2(self.conv2(self.act1(self.conv1(x)))))
        skip = self.down(self.conv3(x))
        return (h + skip) / root(2)

    @property
    def channels(self):
        yield self.conv1.in_channels
        yield self.conv1.out_channels
        yield self.conv2.out_channels


class OutputBlock(nn.Module):

    def __init__(self, in_channels, conditional=False, group_size=None):
        super().__init__()
        self.mbstd = MinibatchStandardDeviation(group_size)
        self.conv1 = EqualizedConvolution2D(in_channels + 1, in_channels, ksize=3, stride=1, pad=1)
        self.act1 = LeakyRelu()
        self.conv2 = EqualizedConvolution2D(in_channels, in_channels, ksize=4, stride=1, pad=0)
        self.act2 = LeakyRelu()
        self.linear = EqualizedLinear(in_channels, in_channels if conditional else 1, gain=1)

    def forward(self, x):
        h1 = self.act1(self.conv1(self.mbstd(x)))
        h2 = self.act2(self.conv2(h1))
        return self.linear(h2)

    @property
    def channels(self):
        yield self.conv1.in_channels
        yield self.conv1.out_channels
        yield self.conv2.out_channels
