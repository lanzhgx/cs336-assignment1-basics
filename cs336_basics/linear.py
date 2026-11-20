import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device=None, dtype: torch.dtype=None) -> None:
        super().__init__()
        self.weights = torch.empty((out_features, in_features), device=device, dtype=dtype)
        std = (2/(in_features + out_features))**0.5
        torch.nn.init.trunc_normal_(self.weights, mean=0.0, std=std, a=-3*std, b=3*std)
        self.weights = nn.Parameter(self.weights)

    def forward(self, x):
        return x @ self.weights.T