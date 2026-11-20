import torch
import torch.nn as nn
from cs336_basics.linear import Linear

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff = None, device=None, dtype=None):
        super().__init__()
        d_ff = int(8 / 3 * d_model / 64)*64 if d_ff else d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x):
        return self.w2(self.w1(x) * torch.sigmoid(self.w1(x)) * self.w3(x))
