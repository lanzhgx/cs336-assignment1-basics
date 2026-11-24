"""Command-line entry point for Transformer LM training."""

import argparse
from pathlib import Path
import random

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



def parse_args() -> argparse.Namespace:
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
    output.add_argument("--log-every", type=int, default=10)
    output.add_argument("--eval-every", type=int, default=100)
    output.add_argument("--eval-batches", type=int, default=10)
    output.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    output.add_argument("--checkpoint-every", type=int, default=100)
    output.add_argument("--resume", type=Path, help="Path to a checkpoint to resume from.")

    return parser.parse_args()


def train(
    model: TransformerLM,
    optimizer: AdamW,
    train_data: npt.NDArray[np.integer],
    val_data: npt.NDArray[np.integer],
    args: argparse.Namespace,
) -> None:
    """Run training, periodic validation, and checkpointing."""
    current_step = 0
    if args.resume:
        current_step = load_checkpoint(args.resume, model, optimizer) + 1
    training_loss = []
    validation_loss = []
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

        if step % args.log_every == 0 and step != 0:
            print("training loss: ", loss_train.item())
            training_loss.append(loss_train.item())

        if step % args.checkpoint_every == 0 and step != 0:
            args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            save_checkpoint(model, optimizer, step, args.checkpoint_dir / f"step_{step}.ckpt")

        if step % args.eval_every == 0 and step != 0:
            with torch.no_grad():
                total_loss_val = 0.0
                for _ in range(args.eval_batches):
                    x_val, y_val = get_batch(val_data, args.batch_size, args.context_length, args.device)
                    y_val_pred = model(x_val)
                    loss_val = cross_entropy(y_val_pred, y_val)
                    total_loss_val += loss_val.item()
                print("validation loss: ", total_loss_val / args.eval_batches)
                validation_loss.append(total_loss_val / args.eval_batches)





def main() -> None:
    args = parse_args()
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
    train(model, optimizer, train_data, val_data, args)


if __name__ == "__main__":
    main()
