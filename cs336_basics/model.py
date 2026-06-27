import math
from typing import Optional

import torch
import torch.nn as nn
from cs336_basics.utils import scaled_dot_product_attention
from tests.conftest import vocab_size


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    row_max = torch.amax(x, dim=dim, keepdim=True)
    e = (x - row_max).exp()
    return e / e.sum(dim=dim, keepdim=True)


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        self._init_weights()

    def _init_weights(self) -> None:
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSnorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        res = x / rms * self.weight
        return res.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if d_ff is None:
            d_ff = ((d_model * 8 // 3 + 63) // 64) * 64
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res1 = self.w1(x)
        res3 = self.w3(x)

        res1 = res1 * torch.sigmoid(res1)
        intermediate = res1 * res3
        return self.w2(intermediate)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        d_k: int,
        theta: float,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even for RoPE"
        self.d_k = d_k

        # inv_freq[i] = theta^(-2i/d_k) for i = 0 .. d_k/2 - 1
        exponents = torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k
        inv_freq = theta**-exponents

        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = positions[:, None] * inv_freq[None, :]  # [max_seq_len, d_k/2]

        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[token_positions]  # [..., seq, d_k/2]
        sin = self.sin[token_positions]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos

        out = torch.empty_like(x)
        out[..., 0::2] = out_even
        out[..., 1::2] = out_odd
        return out


class CasualMHA(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = 10000,
        theta: float | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.n_h = num_heads
        self.theta = theta
        if theta:
            self.rope = RotaryPositionalEmbedding(
                d_k=d_model // num_heads, theta=theta, max_seq_len=max_seq_len
            )

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor = None,
        mask: torch.Tensor | None = None,
    ):
        b, s = x.size(0), x.size(1)
        query = (
            self.q_proj(x).reshape(b, s, self.n_h, -1).transpose(1, 2)
        )  # (b, h, s, d_head)
        key = (
            self.k_proj(x).reshape(b, s, self.n_h, -1).transpose(1, 2)
        )  # (b, h, s, d_head)
        value = (
            self.v_proj(x).reshape(b, s, self.n_h, -1).transpose(1, 2)
        )  # (b, h, s, d_head)

        if self.theta:
            query = self.rope(query, token_positions)
            key = self.rope(key, token_positions)

        if mask is None:
            mask = torch.tril(torch.ones(s, s, device=x.device, dtype=torch.bool))
        output = scaled_dot_product_attention(
            query, key, value, mask
        )  # (b, h, s, d_head)
        output = output.permute(0, 2, 1, 3).contiguous().view(b, s, -1)
        return self.output_proj(output)  # ()


class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: Optional[int] = 10000,
        theta: Optional[float] = None,
    ):
        super().__init__()
        self.ln1 = RMSnorm(d_model)
        self.ln2 = RMSnorm(d_model)
        self.attn = CasualMHA(d_model, num_heads, max_seq_len, theta)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x, mask: torch.Tensor | None = None):
        token_positions = torch.arange(x.size(1), device=x.device)
        x = x + self.attn(self.ln1(x), token_positions, mask)
        x = x + self.ffn(self.ln2(x))

        return x


class LLM(nn.Module):
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        num_heads: int,
        d_ff: int,
        context_lenght: int,
        num_layers: int,
        theta: Optional[float] = None,
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, context_lenght, theta)
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSnorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(context_lenght, context_lenght, dtype=torch.bool)),
            persistent=False,
        )

    def forward(self, x):
        x = self.token_embeddings(x)
        mask = self.mask[: x.size(1), : x.size(1)]
        for layer in self.layers:
            x = layer(x, mask)
        x = self.lm_head(self.ln_final(x))

        return x
