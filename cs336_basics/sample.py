import torch
from cs336_basics.utils import cross_entropy_loss, scaled_dot_product_attention, softmax


def top_k_sampling(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Top k sampling
    """
    topk_val, topk_idx = torch.topk(logits, k, dim=-1)  # (B, S, k)

    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(-1, topk_idx, True)

    logits = logits.masked_fill(~mask, -float("inf"))
    return softmax(logits, dim=-1)


def top_p_sampling(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Top-p (nucleus) sampling: keep the smallest set of tokens whose
    cumulative probability is >= p, including the token that crosses p.
    """
    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
    probs = softmax(sorted_logits, dim=-1)
    cum_probs = torch.cumsum(probs, dim=-1)

    # cum_probs - probs is the cumulative mass strictly *before* each token,
    # so keeping where that is < p includes the crossing token.
    sorted_mask = (cum_probs - probs) < p
    sorted_mask[..., 0] = True  # always keep the most likely token

    # map the sorted-order mask back to original vocab positions
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(-1, sorted_idx, sorted_mask)

    logits = logits.masked_fill(~mask, -float("inf"))
    return softmax(logits, dim=-1)
