import torch
from torch import nn


class Manipulation(nn.Module):

    def random_where(self, p, x, y, axis=0):
        axes = axis if isinstance(axis, tuple) else (axis,)
        shape = tuple([n if i in axes else 1 for i, n in enumerate(x.shape)])
        condition = torch.rand(shape, device=x.device) < p
        return torch.where(condition, x, y)
