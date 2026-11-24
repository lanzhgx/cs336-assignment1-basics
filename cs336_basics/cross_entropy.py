import torch


def cross_entropy(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    logits = x.reshape(-1, x.shape[-1])
    targets = y.reshape(-1).long()

    row_max = logits.max(dim=1, keepdim=True).values
    shifted = logits - row_max

    rows = torch.arange(logits.shape[0], device=logits.device)
    target_logits = shifted[rows, targets]

    log_denominator = shifted.exp().sum(dim=1).log()
    return (log_denominator - target_logits).mean()
