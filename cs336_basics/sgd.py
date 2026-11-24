"""Assignment SGD example and learning-rate tuning experiment."""

import math
from collections.abc import Callable

import torch


class SGD(torch.optim.Optimizer):
    """SGD with a per-parameter learning rate of lr / sqrt(t + 1)."""

    def __init__(self, params, lr: float = 1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        super().__init__(params, {"lr": lr})

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                t = state.get("t", 0)
                step_size = group["lr"] / math.sqrt(t + 1)
                param.add_(param.grad, alpha=-step_size)
                state["t"] = t + 1

        return loss


def run_example(initial_weights: torch.Tensor, lr: float, iterations: int) -> None:
    weights = torch.nn.Parameter(initial_weights.clone())
    optimizer = SGD([weights], lr=lr)

    print(f"lr={lr:g}, iterations={iterations}")
    for step in range(iterations):
        optimizer.zero_grad()
        loss = (weights ** 2).mean()
        print(f"step {step:3d}: loss={loss.item():.8g}")
        loss.backward()
        optimizer.step()


def main() -> None:
    generator = torch.Generator().manual_seed(336)
    initial_weights = 5 * torch.randn((10, 10), generator=generator)

    run_example(initial_weights, lr=1, iterations=100)
    for lr in (1e1, 1e2, 1e3):
        print()
        run_example(initial_weights, lr=lr, iterations=10)


if __name__ == "__main__":
    main()
