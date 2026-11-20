import torch
import torch.nn as nn
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.attention import MultiheadSelfAttention
from cs336_basics.swiglu import SwiGLU
from cs336_basics.embedding import Embedding
from cs336_basics.linear import Linear
from cs336_basics.softmax import softmax

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, theta, max_seq_len):
        super().__init__()
        self.rmsnorm1 = RMSNorm(d_model=d_model)
        self.mha = MultiheadSelfAttention(d_model=d_model, num_heads=num_heads, use_rope=True, theta=theta, max_seq_len=max_seq_len)
        self.rmsnorm2 = RMSNorm(d_model=d_model)
        self.swiglu = SwiGLU(d_model=d_model, d_ff=d_ff)

    def forward(self, x):
        attn = x + self.mha(self.rmsnorm1(x))
        y = attn + self.swiglu(self.rmsnorm2(attn))
        return y

class TransformerLM(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, theta, vocab_size, context_length, num_layers):
        super().__init__()
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, theta=theta, max_seq_len=context_length) for _ in range(num_layers)])
        self.norm = RMSNorm(d_model)
        self.linear = Linear(d_model, vocab_size)
        
    def forward(self, x):
        embedding = self.embedding(x)

        for transformer_block in self.transformer_blocks:
            embedding = transformer_block(embedding)

        return self.linear(self.norm(embedding))