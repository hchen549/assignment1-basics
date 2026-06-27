import math

import torch


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    row_max = torch.amax(x, dim=dim, keepdim=True)
    e = (x - row_max).exp()
    return e / e.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    d_k = query.size(-1)
    logits = query @ key.transpose(-1, -2) / math.sqrt(d_k)

    if mask is not None:
        logits = torch.where(mask, logits, torch.tensor(float("-inf")))
    prob = softmax(logits, dim=-1)
    return prob @ value
