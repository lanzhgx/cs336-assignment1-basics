"""Train the TinyStories baseline and save loss curves in the run directory."""

import sys
from pathlib import Path

from cs336_basics.plot_losses import plot_losses
from cs336_basics.train import main as train_main


def main() -> None:
    root = Path(__file__).resolve().parents[1]

    # Assignment 1 §7.2.1 model and the Apple Silicon budget of 40,960,000 tokens.
    # Adam defaults from Kingma and Ba: peak LR 1e-3, betas (0.9, 0.999), epsilon 1e-8.
    # Baseline choices: 100 warmup steps, cosine decay to 1e-4, AdamW weight decay 0.01.
    defaults = [
        "--train-data", str(root / "artifacts/bpe/tinystories_train.npy"),
        "--val-data", str(root / "artifacts/bpe/tinystories_valid.npy"),
        "--device", "mps",
        "--vocab-size", "10000",
        "--context-length", "256",
        "--d-model", "512",
        "--d-ff", "1344",
        "--num-layers", "4",
        "--num-heads", "16",
        "--rope-theta", "10000",
        "--batch-size", "32",
        "--max-steps", "5000",
        "--max-lr", "1e-3",
        "--min-lr", "1e-4",
        "--warmup-steps", "100",
        "--betas", "0.9", "0.999",
        "--eps", "1e-8",
        "--weight-decay", "0.01",
        "--max-grad-norm", "1.0",
        "--seed", "42",
        "--log-every", "10",
        "--eval-every", "100",
        "--eval-batches", "10",
        "--checkpoint-every", "1000",
        "--checkpoint-dir", str(root / "checkpoints/tinystories-baseline"),
        "--run-dir", str(root / "artifacts/runs/tinystories-baseline"),
    ]
    run_dir = train_main(defaults + sys.argv[1:])
    print(f"Saved {plot_losses(run_dir)}")


if __name__ == "__main__":
    main()
