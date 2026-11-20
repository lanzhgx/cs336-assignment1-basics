import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        if d_k <= 0 or d_k % 2:
            raise ValueError("d_k must be positive and even")

        exponents = torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = positions[:, None] / (theta ** exponents)[None, :]
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        positions = token_positions.to(device=self.cos.device, dtype=torch.long)
        cos, sin = self.cos[positions], self.sin[positions]
        even, odd = x[..., ::2], x[..., 1::2]
        rotated_even = even * cos - odd * sin
        rotated_odd = even * sin + odd * cos
        output = torch.empty_like(x)
        output[..., ::2] = rotated_even
        output[..., 1::2] = rotated_odd
        return output
