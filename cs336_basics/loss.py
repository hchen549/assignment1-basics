import torch


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_z = torch.logsumexp(logits, dim=-1)
    target_logits = torch.gather(logits, dim=-1, index=targets.unsqueeze(-1)).squeeze(
        -1
    )
    return (log_z - target_logits).mean()
