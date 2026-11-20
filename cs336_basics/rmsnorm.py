import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain = nn.Parameter(torch.ones((d_model, ), device=device, dtype=dtype))
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)


        rms = (torch.sum(x**2, dim=-1, keepdim=True) / self.d_model + self.eps)**0.5

        result = x / rms * self.gain

        return result.to(in_dtype)