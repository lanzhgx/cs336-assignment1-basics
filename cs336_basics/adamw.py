"""AdamW optimizer interface for CS336 assignment 1."""

from collections.abc import Callable, Iterable
from typing import Any

import torch
import math


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        """Set up parameter groups and optimizer hyperparameters."""

        super().__init__(params, {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay})

        for group in self.param_groups:
            for param in group["params"]:
                state = self.state[param]
                state["step"] = 1
                state["m"] = torch.zeros_like(param)
                state["v"] = torch.zeros_like(param)
    
    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        """Perform one AdamW update and return the closure loss if provided."""
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                state = self.state[param]
                step = state["step"]

                param.add_(param, alpha=-lr*weight_decay)

                current_lr = lr * math.sqrt(1 - betas[1]**step) / (1 - betas[0]**step)
                state["m"] = betas[0]*state["m"] + (1-betas[0])*grad
                state["v"] = betas[1]*state["v"] + (1-betas[1])*(grad**2)
                param.add_(-current_lr*state["m"]/(torch.sqrt(state["v"])+eps))
                
                state["step"] += 1


