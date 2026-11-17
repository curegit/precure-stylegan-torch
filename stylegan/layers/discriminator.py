from math import sqrt as root
from numpy import array, sinc, float32
from chainer import Chain, Sequential
from chainer.functions import sqrt, sum, mean, average_pooling_2d, convolution_2d, concat, broadcast_to, pad
from stylegan.layers.basic import LeakyRelu, EqualizedLinear, EqualizedConvolution2D
from utilities.stdio import eprint


class Downsampler:

    def __init__(self, lanczos=True, n=2):
        self.lanczos = lanczos
        if lanczos:
            self.n = n
            kernel = lambda x: sinc(x) * sinc(x / n)
            ys = array([kernel(i / 2 + 0.25) for i in range(-2 * n, 2 * n)])
            k = ys.reshape(1, n * 4) * ys.reshape(n * 4, 1)
            self.w = array([[k / k.sum()]], dtype=float32)

    def __call__(self, x):
        if self.lanczos:
            p = 2 * self.n - 1
            batch, channels, height, width = x.shape
            h1 = x.reshape(batch * channels, 1, height, width)
            h2 = pad(h1, ((0, 0), (0, 0), (p, p), (p, p)), mode="symmetric")
            h3 = convolution_2d(h2, x.xp.asarray(self.w), stride=2)
            return h3.reshape(batch, channels, height // 2, width // 2)
        else:
            return average_pooling_2d(x, ksize=2, stride=2)


class MinibatchStandardDeviation:

    def __init__(self, group_size=None):
        self.group_size = group_size

    def __call__(self, x):
        batch, channels, height, width = x.shape
        group_size = min(batch, self.group_size or batch)
        groups = batch // group_size
        grouped_x = x.reshape(groups, group_size, channels, height, width)
        variance = mean((grouped_x - mean(grouped_x, axis=1, keepdims=True)) ** 2, axis=1, keepdims=True)
        deviation = mean(sqrt(variance + 1e-08), axis=(2, 3, 4), keepdims=True)
        deviation_map = broadcast_to(deviation, (groups, group_size, 1, height, width)).reshape(batch, 1, height, width)
        return concat((x, deviation_map), axis=1)


class FromRGB(Chain):

    def __init__(self, out_channels):
        super().__init__()
        with self.init_scope():
            self.conv = EqualizedConvolution2D(3, out_channels, ksize=1, stride=1, pad=0)
            self.act = LeakyRelu()

    def __call__(self, x):
        return self.act(self.conv(x))


class ResidualBlock(Chain):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        with self.init_scope():
            self.conv1 = EqualizedConvolution2D(in_channels, in_channels, ksize=3, stride=1, pad=1)
            self.act1 = LeakyRelu()
            self.conv2 = EqualizedConvolution2D(in_channels, out_channels, ksize=3, stride=1, pad=1)
            self.act2 = LeakyRelu()
            self.pool = Downsampler(lanczos=False)
            self.conv3 = EqualizedConvolution2D(in_channels, out_channels, ksize=1, stride=1, pad=0, nobias=True)
            self.down = Downsampler()

    def __call__(self, x):
        h = self.pool(self.act2(self.conv2(self.act1(self.conv1(x)))))
        skip = self.down(self.conv3(x))
        return (h + skip) / root(2)

    @property
    def channels(self):
        yield self.conv1.in_channels
        yield self.conv1.out_channels
        yield self.conv2.out_channels


class OutputBlock(Chain):

    def __init__(self, in_channels, conditional=False, group_size=None):
        super().__init__()
        with self.init_scope():
            self.mbstd = MinibatchStandardDeviation(group_size)
            self.conv1 = EqualizedConvolution2D(in_channels + 1, in_channels, ksize=3, stride=1, pad=1)
            self.act1 = LeakyRelu()
            self.conv2 = EqualizedConvolution2D(in_channels, in_channels, ksize=4, stride=1, pad=0)
            self.act2 = LeakyRelu()
            self.linear = EqualizedLinear(in_channels, in_channels if conditional else 1, gain=1)

    def __call__(self, x):
        h1 = self.act1(self.conv1(self.mbstd(x)))
        h2 = self.act2(self.conv2(h1))
        return self.linear(h2)

    @property
    def channels(self):
        yield self.conv1.in_channels
        yield self.conv1.out_channels
        yield self.conv2.out_channels


class Discriminator(Chain):

    def __init__(self, levels=7, first_channels=16, last_channels=512, categories=1, depth=8, group_size=None):
        super().__init__()
        in_channels = [first_channels] * (levels - 1)
        out_channels = [last_channels] * (levels - 1)
        for i in range(1, levels - 1):
            channels = min(first_channels * 2**i, last_channels)
            in_channels[i] = channels
            out_channels[i - 1] = channels
        with self.init_scope():
            self.main = Sequential(
                FromRGB(first_channels),
                *[ResidualBlock(i, o) for i, o in zip(in_channels, out_channels)],
                OutputBlock(last_channels, categories > 1, group_size),
            )
            if categories > 1:
                self.embedder = EqualizedLinear(categories, last_channels, gain=1)
                self.mapper = Sequential(EqualizedLinear(last_channels, last_channels), LeakyRelu()).repeat(depth)

    def __call__(self, x, c=None):
        if c is not None:
            embedded = self.embedder(c)
            normalized = embedded / sqrt(mean(embedded**2, axis=1, keepdims=True) + 1e-08)
            c1 = self.mapper(normalized)
        h = self.main(x)
        batch, channels = h.shape
        return h.reshape(batch) if c is None else sum(h * c1, axis=1) / root(channels)

    def freeze(self, levels=[]):
        for i, b in self.blocks:
            if i in levels:
                if isinstance(b, FromRGB):
                    b.disable_update()
                elif isinstance(b, ResidualBlock):
                    b.disable_update()
                elif isinstance(b, OutputBlock):
                    b.conv1.disable_update()
                    b.conv2.disable_update()

    def transfer(self, source, levels=[]):
        for (i, dest), (j, src) in zip(self.blocks, source.blocks):
            if i in levels:
                if isinstance(dest, FromRGB):
                    if not isinstance(src, FromRGB):
                        eprint("Network architecture doesn't match!")
                        raise RuntimeError("Model error")
                    dest.copyparams(src)
                elif isinstance(dest, ResidualBlock):
                    if not isinstance(src, ResidualBlock):
                        eprint("Network architecture doesn't match!")
                        raise RuntimeError("Model error")
                    dest.copyparams(src)
                elif isinstance(dest, OutputBlock):
                    if not isinstance(src, OutputBlock):
                        eprint("Network architecture doesn't match!")
                        raise RuntimeError("Model error")
                    dest.conv1.copyparams(src.conv1)
                    dest.conv2.copyparams(src.conv2)
                else:
                    raise RuntimeError()

    @property
    def blocks(self):
        for i, s in enumerate(self.main):
            yield i, s
