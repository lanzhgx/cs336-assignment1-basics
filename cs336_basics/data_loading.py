"""Batch sampling for language-model training."""

import numpy as np
import numpy.typing as npt
import torch
import random

def get_batch(
    dataset: npt.NDArray[np.integer],
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample input sequences and their next-token targets.

    Both outputs are torch.long tensors with shape (batch_size, context_length)
    on the requested device.
    """
    dataset_size = len(dataset)

    x_result = torch.empty((batch_size, context_length), dtype=torch.long, device=device)
    y_result = torch.empty((batch_size, context_length), dtype=torch.long, device=device)
    for batch in range(batch_size):
        start = random.randint(0, dataset_size - context_length - 1)
        x = dataset[start:start+context_length]
        y = dataset[start+1:start+context_length+1]

        x_result[batch] = torch.as_tensor(x.copy(), dtype=torch.long, device=device)
        y_result[batch] = torch.as_tensor(y.copy(), dtype=torch.long, device=device)
    return x_result, y_result
