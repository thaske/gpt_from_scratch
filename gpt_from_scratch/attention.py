import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys: torch.Tensor = self.W_key(x)
        queries: torch.Tensor = self.W_query(x)
        values: torch.Tensor = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # b, num_tokens, num_heads, head_dim -> b, num_heads, num_tokens, head_dim
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 0, 1,         2,          3
        # b, num_heads, num_tokens, head_dim
        attn_scores = queries @ keys.transpose(2, 3)

        # After matul:
        # b, num_heads, num_tokens, num_tokens

        # Apply causal mask
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]  # type: ignore
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Softmax, logits -> prob
        attn_weights: torch.Tensor = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )

        # Apply dropout
        attn_weights: torch.Tensor = self.dropout(attn_weights)

        # b, num_heads, num_tokens, num_tokens @ b, num_heads, num_tokens, head_dim ->
        # b, num_heads, num_tokens, head_dim
        context_vec = attn_weights @ values
        # print("context_vec.shape:", context_vec.shape)
        # print("context_vec:", context_vec)

        # 0, 1,         2,          3
        # b, num_heads, num_tokens, head_dim -> b, num_tokens, num_heads, head_dim
        context_vec = context_vec.transpose(1, 2)

        # b, num_tokens, num_heads, head_dim -> b, num_tokens, d_out
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        context_vec: torch.Tensor = self.out_proj(context_vec)

        return context_vec
