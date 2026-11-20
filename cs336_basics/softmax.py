import torch
import torch.nn as nn


def softmax(x: torch.Tensor, dim: int):
    normalized_x = x - torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(normalized_x)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)