import torch
from stylegan.manipulations.base import Manipulation


class AdditiveNoise(Manipulation):

    def __init__(self, sd=0.05, probability_multiplier=1.0):
        super().__init__()
        self.sd = sd
        self.probability_multiplier = probability_multiplier

    def forward(self, x, p):
        p *= self.probability_multiplier
        if p <= 0:
            return x
        batch = x.shape[0]
        half_normal = torch.randn((batch, 1, 1, 1), device=x.device, dtype=x.dtype).abs() * self.sd
        noise_added = x + torch.randn_like(x) * half_normal
        return self.random_where(p, noise_added, x)


class Cutout(Manipulation):

    def __init__(self, width=0.5, height=0.5, fill=0.5, probability_multiplier=1.0):
        super().__init__()
        self.width = width
        self.height = height
        self.fill = fill
        self.probability_multiplier = probability_multiplier

    def forward(self, x, p):
        p *= self.probability_multiplier
        if p <= 0:
            return x
        batch, _, height, width = x.shape
        horizontal_center = torch.rand((batch, 1), device=x.device) * width
        vertical_center = torch.rand((batch, 1), device=x.device) * height
        w2, h2 = self.width * width / 2, self.height * height / 2
        left, right = horizontal_center - w2, horizontal_center + w2
        top, bottom = vertical_center - h2, vertical_center + h2
        horizontal_indices = torch.arange(0.5, width, device=x.device)
        vertical_indices = torch.arange(0.5, height, device=x.device)
        w = ((left <= horizontal_indices) & (horizontal_indices < right)).reshape(batch, 1, 1, width)
        h = ((top <= vertical_indices) & (vertical_indices < bottom)).reshape(batch, 1, height, 1)
        cutout = torch.where(w & h, self.fill, x)
        return self.random_where(p, cutout, x)
