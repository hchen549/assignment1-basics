import torch


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_z = torch.logsumexp(logits, dim=-1)
    target_logits = torch.gather(logits, dim=-1, index=targets.unsqueeze(-1)).squeeze(
        -1
    )
    return (log_z - target_logits).mean()


def scalable_cross_entropy(
    n_buckets: int,
    X: torch.Tensor,
    Y: torch.Tensor,
    labels: torch.Tensor,
    b_x: int,
    b_y: int,
) -> torch.Tensor:
    """Scalable Cross-Entropy loss (Algorithm 1, arXiv:2409.18721).

    X: (N, d) query embeddings, Y: (C, d) class embeddings,
    labels: (N,) true class index per query.
    """
    d = X.size(-1)

    # Lines 2-8: bucketing + top-k selection carry no gradient.
    with torch.no_grad():
        B = torch.randn(n_buckets, d, device=X.device, dtype=X.dtype)
        x_proj = B @ X.T  # (n_buckets, N)
        y_proj = B @ Y.T  # (n_buckets, C)

        top_x_idx = torch.argsort(x_proj, dim=-1, descending=True)[
            ..., :b_x
        ]  # (n_buckets, b_x)
        top_y_idx = torch.argsort(y_proj, dim=-1, descending=True)[
            ..., :b_y
        ]  # (n_buckets, b_y)

    # Lines 9-14: gather actual embeddings (gradients flow from here on).
    top_x = X[top_x_idx]  # (n_buckets, b_x, d)
    top_y = Y[top_y_idx]  # (n_buckets, b_y, d)

    targets = labels[top_x_idx]  # (n_buckets, b_x) true class per selected query
    pos_emb = Y[targets]  # (n_buckets, b_x, d)
    pos_logit = (top_x * pos_emb).sum(-1)  # (n_buckets, b_x) correct-class logit

    neg_logits = torch.matmul(
        top_x, top_y.transpose(-2, -1)
    )  # (n_buckets, b_x, b_y) wrong-class logits

    # Line 15: positive sits at index 0, so loss = logsumexp(all) - pos_logit.
    all_logits = torch.cat([pos_logit.unsqueeze(-1), neg_logits], dim=-1)
    loss_bi = torch.logsumexp(all_logits, dim=-1) - pos_logit  # (n_buckets, b_x)

    # Lines 16-17: per query take the MAX loss over buckets, then mean over
    # queries that landed in at least one bucket.
    per_query = X.new_full((X.size(0),), float("-inf"))
    per_query.scatter_reduce_(
        0, top_x_idx.reshape(-1), loss_bi.reshape(-1), reduce="amax", include_self=True
    )
    seen = per_query != float("-inf")
    return per_query[seen].mean()
