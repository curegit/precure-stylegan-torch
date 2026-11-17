from sys import float_info
from math import isfinite
from torch import device as TorchDevice


def uint(string):
    value = int(string)
    if value >= 0:
        return value
    raise ValueError()


def natural(string):
    value = int(string)
    if value > 0:
        return value
    raise ValueError()


def real(string):
    value = float(string)
    if isfinite(value):
        return value
    raise ValueError()


def ufloat(string):
    value = real(string)
    if value >= 0:
        return value
    raise ValueError()


def positive(string):
    value = real(string)
    if value >= float_info.epsilon:
        return value
    raise ValueError()


def rate(string):
    value = real(string)
    if 0 <= value <= 1:
        return value
    raise ValueError()


def device(string):
    value = string.upper()
    if value in ("CPU", "-1"):
        return TorchDevice("cpu")
    if value == "MPS":
        return TorchDevice("mps")
    if value in ("GPU", "CUDA"):
        return TorchDevice("cuda", 0)
    if value.startswith("CUDA:"):
        value = value[5:]
    try:
        value = int(value)
    except Exception:
        raise ValueError() from None
    if value >= 0:
        return TorchDevice("cuda", value)
    raise ValueError()
