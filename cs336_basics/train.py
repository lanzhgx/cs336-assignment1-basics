"""Command-line entry point for Transformer LM training."""

import argparse
import json
import random
from math import isfinite
from pathlib import Path
from tempfile import mkdtemp
from time import perf_counter
from typing import TextIO

import numpy as np
import numpy.typing as npt
import torch

from .adamw import AdamW
from .transformer import TransformerLM
from .checkpointing import load_checkpoint, save_checkpoint
from .data_loading import get_batch
from .cross_entropy import cross_entropy
from .gradient_clipping import gradient_clipping
from .learning_rate_schedule import get_lr_cosine_schedule



def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    data = parser.add_argument_group("data")
    data.add_argument("--train-data", type=Path, required=True, help="Tokenized training .npy file.")
    data.add_argument("--val-data", type=Path, required=True, help="Tokenized validation .npy file.")
    data.add_argument("--device", default="cpu")

    model = parser.add_argument_group("model")
    model.add_argument("--vocab-size", type=int, default=10_000)
    model.add_argument("--context-length", type=int, default=128)
    model.add_argument("--d-model", type=int, default=128)
    model.add_argument("--num-heads", type=int, default=4)
    model.add_argument("--num-layers", type=int, default=4)
    model.add_argument("--d-ff", type=int, default=320)
    model.add_argument("--rope-theta", type=float, default=10_000.0)

    training = parser.add_argument_group("training")
    training.add_argument("--batch-size", type=int, default=16)
    training.add_argument("--max-steps", type=int, default=1_000)
    training.add_argument("--max-lr", type=float, default=3e-4)
    training.add_argument("--min-lr", type=float, default=3e-5)
    training.add_argument("--warmup-steps", type=int, default=100)
    training.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95), metavar=("BETA1", "BETA2"))
    training.add_argument("--eps", type=float, default=1e-8)
    training.add_argument("--weight-decay", type=float, default=0.01)
    training.add_argument("--max-grad-norm", type=float, default=1.0)
    training.add_argument("--seed", type=int, default=42)

    output = parser.add_argument_group("logging and checkpoints")
    output.add_argument(
        "--run-dir", type=Path, help="New experiment directory. Defaults to a unique directory under artifacts/runs."
    )
    output.add_argument("--log-every", type=int, default=10)
    output.add_argument("--eval-every", type=int, default=100)
    output.add_argument("--eval-batches", type=int, default=10)
    output.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    output.add_argument("--checkpoint-every", type=int, default=100)
    output.add_argument("--resume", type=Path, help="Path to a checkpoint to resume from.")

    return parser.parse_args(argv)


def log_loss(
    metrics: TextIO | None,
    split: str,
    step: int,
    elapsed_sec: float,
    loss: float,
    *,
    lr: float | None = None,
) -> None:
    print(f"step {step} | {elapsed_sec:.2f}s | {split} loss: {loss:.6f}", flush=True)
    if metrics is not None:
        record = {
            "split": split,
            "step": step,
            "elapsed_sec": elapsed_sec,
            "loss": loss if isfinite(loss) else None,
        }
        if lr is not None:
            record["lr"] = lr
        metrics.write(json.dumps(record, allow_nan=False) + "\n")


def train(
    model: TransformerLM,
    optimizer: AdamW,
    train_data: npt.NDArray[np.integer],
    val_data: npt.NDArray[np.integer],
    args: argparse.Namespace,
    *,
    metrics: TextIO | None = None,
) -> None:
    """Run training, periodic validation, and checkpointing."""
    current_step = 0
    if args.resume:
        current_step = load_checkpoint(args.resume, model, optimizer) + 1
    started_at = perf_counter()
    for step in range(current_step, args.max_steps):
        lr = get_lr_cosine_schedule(
            it=step,
            max_learning_rate=args.max_lr,
            min_learning_rate=args.min_lr,
            warmup_iters=args.warmup_steps,
            cosine_cycle_iters=args.max_steps - 1,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad()

        x_train, y_train = get_batch(train_data, args.batch_size, args.context_length, args.device)
        y_pred = model(x_train)
        loss_train = cross_entropy(y_pred, y_train)
        loss_train.backward()
        gradient_clipping(model.parameters(), args.max_grad_norm)
        optimizer.step()

        final_step = step + 1 == args.max_steps
        if final_step or (step != 0 and step % args.log_every == 0):
            train_loss = loss_train.item()
            log_loss(metrics, "train", step + 1, perf_counter() - started_at, train_loss, lr=lr)

        if final_step or (step != 0 and step % args.checkpoint_every == 0):
            args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            save_checkpoint(model, optimizer, step, args.checkpoint_dir / f"step_{step}.ckpt")

        if final_step or (step != 0 and step % args.eval_every == 0):
            with torch.no_grad():
                total_loss_val = 0.0
                for _ in range(args.eval_batches):
                    x_val, y_val = get_batch(val_data, args.batch_size, args.context_length, args.device)
                    y_val_pred = model(x_val)
                    loss_val = cross_entropy(y_val_pred, y_val)
                    total_loss_val += loss_val.item()
                val_loss = total_loss_val / args.eval_batches
                log_loss(metrics, "val", step + 1, perf_counter() - started_at, val_loss)

def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_data = np.load(args.train_data, mmap_mode="r", allow_pickle=False)
    val_data = np.load(args.val_data, mmap_mode="r", allow_pickle=False)

    model = TransformerLM(
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.rope_theta,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
    ).to(args.device)
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.betas[0], args.betas[1]),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )
    if args.run_dir is None:
        runs_root = Path("artifacts/runs")
        runs_root.mkdir(parents=True, exist_ok=True)
        args.run_dir = Path(mkdtemp(prefix="run-", dir=runs_root))
    else:
        try:
            args.run_dir.mkdir(parents=True)
        except FileExistsError:
            raise SystemExit(f"Run directory already exists: {args.run_dir}. Choose a new --run-dir.") from None

    config = json.dumps(vars(args), indent=2, default=str)
    (args.run_dir / "config.json").write_text(config + "\n", encoding="utf-8")
    print(f"Run directory: {args.run_dir}", flush=True)
    with (args.run_dir / "metrics.jsonl").open("x", encoding="utf-8", buffering=1) as metrics:
        train(model, optimizer, train_data, val_data, args, metrics=metrics)
    return args.run_dir


if __name__ == "__main__":
    main()
