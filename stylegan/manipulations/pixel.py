import torch
from stylegan.layers.basic import symmetric_pad2d
from stylegan.manipulations.base import Manipulation


class Mirror(Manipulation):

    def __init__(self, probability_multiplier=1.0):
        super().__init__()
        self.probability_multiplier = probability_multiplier

    def forward(self, x, p):
        p *= self.probability_multiplier
        if p <= 0:
            return x
        mirrored = self.random_where(0.5, x, x.flip(3))
        return self.random_where(p, mirrored, x)


class Rotation(Manipulation):

    def __init__(self, probability_multiplier=1.0):
        super().__init__()
        self.probability_multiplier = probability_multiplier

    def forward(self, x, p):
        p *= self.probability_multiplier
        if p <= 0:
            return x
        transposed = x.transpose(2, 3)
        rot90 = transposed.flip(2)
        rot180 = x.flip((2, 3))
        rot270 = transposed.flip(3)
        rotated = self.random_where(0.5, self.random_where(0.5, x, rot90), self.random_where(0.5, rot180, rot270))
        return self.random_where(p, rotated, x)


class Shift(Manipulation):

    def __init__(self, horizontal=0.125, vertical=0.125, probability_multiplier=1.0):
        super().__init__()
        self.horizontal = horizontal
        self.vertical = vertical
        self.probability_multiplier = probability_multiplier

    def forward(self, x, p):
        p *= self.probability_multiplier
        if p <= 0:
            return x
        batch, _, height, width = x.shape
        sh, sw = round(height * self.vertical), round(width * self.horizontal)
        padded = symmetric_pad2d(x, (sw, sw, sh, sh))
        h = torch.arange(height, device=x.device).reshape(1, height, 1) + torch.randint(2 * sh + 1, (batch, 1, 1), device=x.device)
        w = torch.arange(width, device=x.device).reshape(1, 1, width) + torch.randint(2 * sw + 1, (batch, 1, 1), device=x.device)
        b = torch.arange(batch, device=x.device).reshape(batch, 1, 1)
        shifted = padded.permute(0, 2, 3, 1)[b, h, w].permute(0, 3, 1, 2)
        return self.random_where(p, shifted, x)
