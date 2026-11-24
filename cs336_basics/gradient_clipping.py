"""Global L2 gradient clipping for CS336 assignment 1."""

from collections.abc import Iterable

from torch.nn import Parameter
import torch
import math


def gradient_clipping(parameters: Iterable[Parameter], max_l2_norm: float) -> None:
    """Clip the combined L2 norm by modifying existing gradients in place."""
    sum = 0.0
    for param in parameters:
        if param.grad is not None:
            sum += torch.sum(torch.square(param.grad))

    norm = math.sqrt(sum)

    if norm >= max_l2_norm:
        factor = max_l2_norm / (norm + 10e-6)
        for param in parameters:
            if param.grad is not None:
                param.grad.mul_(factor)
