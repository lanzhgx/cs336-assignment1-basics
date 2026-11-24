"""Save and restore language-model training checkpoints."""

from os import PathLike
from typing import BinaryIO

import torch
from torch.nn import Module
from torch.optim import Optimizer


def save_checkpoint(
    model: Module,
    optimizer: Optimizer,
    iteration: int,
    out: str | PathLike[str] | BinaryIO,
) -> None:
    """Save model state, optimizer state, and the completed iteration count."""
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()

    saved_obj = {"model_state": model_state, "optimizer_state": optimizer_state, "iteration": iteration}

    torch.save(saved_obj, out)


def load_checkpoint(
    src: str | PathLike[str] | BinaryIO,
    model: Module,
    optimizer: Optimizer,
) -> int:
    """Restore model and optimizer state in place and return the iteration count."""
    obj = torch.load(src)

    model_state = obj["model_state"]
    optimizer_state = obj["optimizer_state"]
    iteration = obj["iteration"]

    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
    return iteration
