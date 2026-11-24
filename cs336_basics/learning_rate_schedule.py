"""Learning-rate schedules for CS336 assignment 1."""
import math

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Return the learning rate at iteration it.

    warmup_iters is the end of warmup. cosine_cycle_iters is the iteration
    at which cosine decay reaches min_learning_rate.
    """
    if it < warmup_iters:
        return 1.0 * it / warmup_iters * max_learning_rate
    elif warmup_iters <= it and it <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (1 + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)) * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate
