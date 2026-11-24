"""Plot training and validation loss against updates and elapsed time."""

import argparse
import json
from pathlib import Path

from matplotlib.figure import Figure


def plot_losses(run_dir: Path, output: Path | None = None) -> Path:
    """Plot a run's recorded losses and return the figure path."""
    records = {"train": [], "val": []}
    metrics_path = run_dir / "metrics.jsonl"
    with metrics_path.open(encoding="utf-8") as metrics:
        for line in metrics:
            record = json.loads(line)
            records[record["split"]].append(record)
    if not any(records.values()):
        raise ValueError(f"No loss measurements in {metrics_path}")

    figure = Figure(figsize=(10, 4), layout="constrained")
    by_step, by_time = figure.subplots(1, 2, sharey=True)
    for split, label in (("train", "Train"), ("val", "Validation")):
        rows = records[split]
        if not rows:
            continue
        losses = [row["loss"] if row["loss"] is not None else float("nan") for row in rows]
        by_step.plot([row["step"] for row in rows], losses, marker=".", label=label)
        by_time.plot([row["elapsed_sec"] for row in rows], losses, marker=".", label=label)

    by_step.set(xlabel="Completed gradient steps", ylabel="Cross-entropy loss")
    by_time.set(xlabel="Elapsed time (seconds)")
    for axis in (by_step, by_time):
        axis.grid(alpha=0.25)
        axis.legend()

    output = output or run_dir / "loss.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Directory containing metrics.jsonl.")
    parser.add_argument("--output", type=Path, help="Figure path. Defaults to loss.png in the run directory.")
    args = parser.parse_args()

    try:
        output = plot_losses(args.run_dir, args.output)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
