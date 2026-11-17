from math import sqrt as root
from random import randint
import torch
from torch import nn
from torch.nn import functional as F
from stylegan.layers.basic import GaussianDistribution, LeakyRelu, EqualizedLinear
from stylegan.layers.generator import InitialSkipArchitecture, SkipArchitecture
from utilities.math import lerp
from utilities.stdio import eprint


class Mapper(nn.Module):

    def __init__(self, size, depth, conditional=False):
        super().__init__()
        self.mlp = nn.Sequential(
            EqualizedLinear(size * 2 if conditional else size, size),
            LeakyRelu(),
            *[l for b in [[EqualizedLinear(size, size), LeakyRelu()] for _ in range(depth - 1)] for l in b],
        )

    def forward(self, z, c=None):
        h1 = z / torch.sqrt(torch.mean(z**2, dim=1, keepdim=True) + 1e-08)
        h2 = h1 if c is None else torch.cat((h1, c / torch.sqrt(torch.mean(c**2, dim=1, keepdim=True) + 1e-08)), dim=1)
        return self.mlp(h2)


class Synthesizer(nn.Module):

    def __init__(self, size, levels, first_channels, last_channels):
        super().__init__()
        in_channels = [first_channels] * levels
        out_channels = [last_channels] * levels
        for i in range(1, levels):
            channels = min(first_channels, last_channels * 2**i)
            in_channels[-i] = channels
            out_channels[-i - 1] = channels
        self.init = InitialSkipArchitecture(size, in_channels[0], out_channels[0], level=1)
        self.skips = nn.ModuleList([SkipArchitecture(size, i, o, level=l) for l, (i, o) in enumerate(zip(in_channels[1:], out_channels[1:]), 2)])

    def forward(self, ws, noise=1.0, fixed=None):
        h, rgb = self.init(ws[0], noise=noise, fixed=fixed)
        for s, w in zip(self.skips, ws[1:]):
            h, rgb = s(h, rgb, w, noise=noise, fixed=fixed)
        return rgb

    @property
    def blocks(self):
        yield 1, self.init
        for i, s in enumerate(self.skips, 2):
            yield i, s


class Generator(nn.Module):

    def __init__(self, size=512, depth=8, levels=7, first_channels=512, last_channels=64, categories=1, labels=None):
        super().__init__()
        self.size = size
        self.depth = depth
        self.levels = levels
        self.first_channels = first_channels
        self.last_channels = last_channels
        self.categories = categories
        self.resolution = (2 * 2**levels, 2 * 2**levels)
        self.labels = [f"cat{i}" for i in range(categories)]
        if labels is not None:
            self.embed_labels(labels)
        self.sampler = GaussianDistribution(self)
        self.mapper = Mapper(size, depth, categories > 1)
        self.synthesizer = Synthesizer(size, levels, first_channels, last_channels)
        if categories > 1:
            self.embedder = EqualizedLinear(categories, size, gain=1)

    def forward(self, z, c=None, random_mix=None, psi=1.0, mean_w=None, categories=None, noise=1.0, fixed=None):
        z, *zs = z if isinstance(z, tuple) or isinstance(z, list) else [z]
        if self.conditional and c is None:
            _, c = self.generate_conditions(len(z), categories)
        if c is not None:
            c = self.embedder(c)
        w = self.truncation_trick(self.mapper(z, c), psi, mean_w, categories)
        ws = [w] * self.levels
        stop = self.levels
        if self.levels > 1 and random_mix is not None:
            mix_level = randint(1, self.levels - 1)
            mix_w = self.truncation_trick(self.mapper(random_mix, c), psi, mean_w, categories)
            ws[mix_level:stop] = [mix_w] * (stop - mix_level)
            stop = mix_level
        for i, z in zip(range(1, stop), zs):
            if z is not Ellipsis:
                ws[i:stop] = [self.truncation_trick(self.mapper(z, c), psi, mean_w, categories)] * (stop - i)
        return ws, self.synthesizer(ws, noise=noise, fixed=fixed)

    def generate_latents(self, batch, center=None, sd=1.0):
        return self.sampler(batch, self.size) * sd + (0.0 if center is None else center)

    def generate_conditions(self, batch, categories=None):
        parameter = next(self.parameters())
        if categories is None:
            ind = torch.randint(self.categories, (batch,), device=parameter.device)
        else:
            choices = torch.as_tensor(categories, device=parameter.device, dtype=torch.long)
            ind = choices[torch.randint(len(choices), (batch,), device=parameter.device)]
        return ind.tolist(), F.one_hot(ind, num_classes=self.categories).to(dtype=parameter.dtype)

    def generate_masks(self, batch):
        return self.sampler(batch, 3, self.height, self.width) / root(self.height * self.width)

    def truncation_trick(self, w, psi, mean_w=None, categories=None):
        if psi != 1.0:
            if mean_w is None:
                mean_w = self.calculate_mean_w(categories=categories)
            return lerp(mean_w, w, psi)
        return w

    def calculate_mean_w(self, n=50000, categories=None):
        c = self.embedder(self.generate_conditions(n, categories)[1]) if self.conditional else None
        return torch.mean(self.mapper(self.generate_latents(n), c), dim=0)

    def calculate_mean_ws_by_category(self, n=50000):
        return torch.stack([self.calculate_mean_w(n=n, categories=[i]) for i in range(self.categories)])

    @property
    def width(self):
        return self.resolution[0]

    @property
    def height(self):
        return self.resolution[1]

    @property
    def conditional(self):
        return self.categories > 1

    def embed_labels(self, labels):
        for i, l in enumerate(labels):
            self.labels[i] = str(l)

    def lookup_label(self, label):
        for i, l in enumerate(self.labels):
            if l == label:
                return i
        eprint(f"Invalid label: {label}")
        eprint("No such label in the model!")
        raise RuntimeError("Label error")

    def freeze(self, levels=[]):
        for i, b in self.synthesizer.blocks:
            if i in levels:
                if isinstance(b, InitialSkipArchitecture):
                    b.wmconv.requires_grad_(False)
                    b.torgb.requires_grad_(False)
                elif isinstance(b, SkipArchitecture):
                    b.wmconv1.requires_grad_(False)
                    b.wmconv2.requires_grad_(False)
                    b.torgb.requires_grad_(False)

    def transfer(self, source, levels=[]):
        for (i, dest), (j, src) in zip(reversed(list(self.synthesizer.blocks)), reversed(list(source.synthesizer.blocks))):
            if i in levels:
                if isinstance(dest, InitialSkipArchitecture):
                    if not isinstance(src, InitialSkipArchitecture):
                        eprint("Network architecture doesn't match!")
                        raise RuntimeError("Model error")
                    dest.wmconv.load_state_dict(src.wmconv.state_dict())
                    dest.torgb.load_state_dict(src.torgb.state_dict())
                elif isinstance(dest, SkipArchitecture):
                    if not isinstance(src, SkipArchitecture):
                        eprint("Network architecture doesn't match!")
                        raise RuntimeError("Model error")
                    dest.wmconv1.load_state_dict(src.wmconv1.state_dict())
                    dest.wmconv2.load_state_dict(src.wmconv2.state_dict())
                    dest.torgb.load_state_dict(src.torgb.state_dict())
                else:
                    raise RuntimeError()

    def save(self, filepath):
        torch.save({"format_version": 1, "params": self.model_params(), "state_dict": self.state_dict()}, filepath)

    def model_params(self):
        return {
            "size": self.size,
            "depth": self.depth,
            "levels": self.levels,
            "first_channels": self.first_channels,
            "last_channels": self.last_channels,
            "categories": self.categories,
            "labels": list(self.labels),
        }

    @staticmethod
    def load(filepath, device="cpu"):
        checkpoint = torch.load(filepath, map_location="cpu", weights_only=True)
        if checkpoint["format_version"] != 1:
            raise ValueError("Unsupported generator checkpoint format")
        generator = Generator(**checkpoint["params"])
        generator.load_state_dict(checkpoint["state_dict"])
        return generator.to(device).eval()
