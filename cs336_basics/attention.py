import torch
import torch.nn as nn
from cs336_basics.softmax import softmax
from cs336_basics.linear import Linear
from cs336_basics.rope import RotaryPositionalEmbedding

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / d_k**0.5
    scores = scores.masked_fill(~mask, float("-inf"))
    return softmax(scores, dim=-1) @ V

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, use_rope=False, theta=None, max_seq_len=None, token_positions=None):
        super().__init__()
        assert(d_model % num_heads == 0)
        self.num_heads = num_heads
        self.d_model = d_model
        self.d = d_model // num_heads

        self.Wq = Linear(d_model, d_model)
        self.Wk = Linear(d_model, d_model)
        self.Wv = Linear(d_model, d_model)

        self.Wo = Linear(d_model, d_model)

        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryPositionalEmbedding(theta=theta, d_k=self.d, max_seq_len=max_seq_len)
            self.token_positions = token_positions

    def forward(self, x):
        q = self.Wq(x) # batch, seq_len, d_model
        k = self.Wk(x)
        v = self.Wv(x)

        # batch, num_heads, seq_len, d_kv
        q_reshaped = q.reshape(q.shape[0], -1, self.num_heads, self.d).transpose(-2, -3)
        k_reshaped = k.reshape(q.shape[0], -1, self.num_heads, self.d).transpose(-2, -3)
        if self.use_rope:
            self.token_positions = self.token_positions if self.token_positions else torch.arange(x.shape[-2], device=x.device)
            q_reshaped = self.rope(q_reshaped, self.token_positions)
            k_reshaped = self.rope(k_reshaped, self.token_positions)
        v_reshaped = v.reshape(q.shape[0], -1, self.num_heads, self.d).transpose(-2, -3)

        mask = ~torch.triu(torch.ones((q.shape[0], self.num_heads, q.shape[1], q.shape[1]), dtype=torch.bool), diagonal=1)
        attention = scaled_dot_product_attention(Q=q_reshaped, K=k_reshaped, V=v_reshaped, mask=mask)

        return self.Wo(attention.transpose(-2, -3).reshape(q.shape[0], -1, self.d_model))


        
