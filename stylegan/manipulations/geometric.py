import torch
from math import sqrt, log, sin, cos, pi
from numpy import array, eye, stack, float32
from numpy.random import uniform, normal, lognormal
from stylegan.layers.basic import symmetric_pad2d
from stylegan.manipulations.base import Manipulation

identity = eye(3, dtype=float32)


def translation(x, y):
    return array([[1, 0, x], [0, 1, y], [0, 0, 1]], dtype=float32)


def rotation(theta):
    return array([[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]], dtype=float32)


def scaling(s, t):
    return array([[s, 0, 0], [0, t, 0], [0, 0, 1]], dtype=float32)


def inverse_translation(x, y):
    return translation(-x, -y)


def inverse_rotation(theta):
    return rotation(-theta)


def inverse_scaling(s, t):
    return scaling(1 / s, 1 / t)


class AffineTransformation(Manipulation):

    def __init__(self, translation=0.125, rotation=360, scale=0.2, probability_multiplier=1.0):
        super().__init__()
        self.translation = translation
        self.rotation = rotation / 360 * pi
        self.scale = scale * log(2)
        self.probability_multiplier = probability_multiplier

    def forward(self, x, p=1.0):
        p = min(1.0, p * self.probability_multiplier)
        if p <= 0:
            return x
        batch, _, height, width = x.shape
        centering = inverse_translation(-height / 2 + 0.5, -width / 2 + 0.5)
        affine = identity @ centering
        condition = uniform(size=batch) < p
        scale = lognormal(sigma=self.scale, size=batch)
        isotropic_scaling = stack([inverse_scaling(s, s) if c else identity for c, s in zip(condition, scale)])
        affine = affine @ isotropic_scaling
        condition = uniform(size=batch) < 1 - sqrt(1 - p)
        theta = uniform(low=-self.rotation, high=self.rotation, size=batch)
        pre_rotation = stack([inverse_rotation(t) if c else identity for c, t in zip(condition, theta)])
        affine = affine @ pre_rotation
        condition = uniform(size=batch) < p
        scale = lognormal(sigma=self.scale, size=batch)
        anisotropic_scaling = stack([inverse_scaling(1 / s, s) if c else identity for c, s in zip(condition, scale)])
        affine = affine @ anisotropic_scaling
        condition = uniform(size=batch) < 1 - sqrt(1 - p)
        theta = uniform(low=-self.rotation, high=self.rotation, size=batch)
        post_rotation = stack([inverse_rotation(t) if c else identity for c, t in zip(condition, theta)])
        affine = affine @ post_rotation
        condition = uniform(size=batch) < p
        th = height * normal(scale=self.translation, size=batch)
        tw = width * normal(scale=self.translation, size=batch)
        translation = stack([inverse_translation(h, w) if c else identity for c, h, w in zip(condition, th, tw)])
        affine = affine @ translation
        inverse_centering = inverse_translation(height / 2 - 0.5, width / 2 - 0.5)
        affine = affine @ inverse_centering
        affine = torch.as_tensor(affine, device=x.device, dtype=x.dtype)
        indices = torch.meshgrid(torch.arange(height, device=x.device, dtype=x.dtype), torch.arange(width, device=x.device, dtype=x.dtype), indexing="ij")
        coordinate = torch.stack((*indices, torch.ones_like(indices[0])), dim=-1)
        resampling_coordinate = affine.reshape(batch, 1, 3, 3) @ coordinate.reshape(1, height * width, 3, 1)
        resampling_grid = resampling_coordinate.reshape(batch, height, width, 3).permute(3, 0, 1, 2)[0:2]
        return self.lanczos_resampling(x, resampling_grid)

    def lanczos_resampling(self, x, grid):
        batch, channels, height, width = x.shape
        i2 = grid.floor().long()
        i1, i3, i4 = i2 - 1, i2 + 1, i2 + 2
        indices = torch.stack((i1, i2, i3, i4))
        t = (indices - grid).permute(1, 0, 2, 3, 4)
        dirty_weight = torch.sinc(t) * torch.sinc(t / 2)
        wh, ww = dirty_weight / dirty_weight.sum(dim=1, keepdim=True)
        weight = wh.reshape(4, 1, batch, 1, height, width) * ww.reshape(1, 4, batch, 1, height, width)
        lower = i1.amin(dim=(1, 2, 3))
        upper = i4.amax(dim=(1, 2, 3))
        top = abs(min(int(lower[0]), 0))
        left = abs(min(int(lower[1]), 0))
        bottom = max(int(upper[0]) - height + 1, 0)
        right = max(int(upper[1]) - width + 1, 0)
        ih, iw = indices.permute(1, 0, 2, 3, 4)
        h = ih.reshape(4, 1, batch, 1, height, width) + top
        w = iw.reshape(1, 4, batch, 1, height, width) + left
        b = torch.arange(batch, device=x.device).reshape(1, 1, batch, 1, 1, 1)
        c = torch.arange(channels, device=x.device).reshape(1, 1, 1, channels, 1, 1)
        padded_x = symmetric_pad2d(x, (left, right, top, bottom))
        neighbor = padded_x[b, c, h, w]
        return (neighbor * weight).sum(dim=(0, 1))
