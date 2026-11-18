"""Encode the four datasets with train-split tokenizers into uint16 .npy files."""

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import os
from pathlib import Path
import shutil
import time

import numpy as np

from .tokenizer import Tokenizer

_tokenizer = None
_BOUNDARY = "<|endoftext|>"


def _init_worker(vocab_path: str, merges_path: str) -> None:
    global _tokenizer
    _tokenizer = Tokenizer.from_files(vocab_path, merges_path, [_BOUNDARY])
    if min(_tokenizer.vocab) < 0 or max(_tokenizer.vocab) > np.iinfo(np.uint16).max:
        raise ValueError("Vocabulary token IDs do not fit in uint16.")


def _encode_batch(text: str) -> tuple[np.ndarray, int, int]:
    ids = _tokenizer.encode(text)
    original = text.encode("utf-8")
    # Validate every batch, including whitespace and document separators.
    decoded = b"".join(_tokenizer.vocab[token_id] for token_id in ids)
    if decoded != original:
        raise ValueError("Encoded batch does not decode to its original bytes.")
    return np.asarray(ids, dtype=np.uint16), len(original), text.count(_BOUNDARY)


def _document_batches(path: Path, read_size: int):
    buffer = ""
    with path.open("r", encoding="utf-8", newline="") as source:
        while text := source.read(read_size):
            buffer += text
            end = buffer.rfind(_BOUNDARY)
            if end >= 0:
                end += len(_BOUNDARY)
                yield buffer[:end]
                buffer = buffer[end:]
        if buffer:
            yield buffer


def encode_file(
    input_path: Path, output_path: Path, tokenizer_dir: Path,
    workers: int = 8, read_size: int = 1024 * 1024,
) -> dict:
    if workers < 1 or read_size < 1:
        raise ValueError("workers and read_size must be positive")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output_path.with_suffix(".raw.partial")
    npy_path = output_path.with_suffix(".npy.partial")
    total_bytes = input_path.stat().st_size
    processed_bytes = num_tokens = num_markers = batches = 0
    started = last_report = time.perf_counter()
    pending = deque()
    print(f"\n[{output_path.name}] input={total_bytes / 2**30:.3f} GiB "
          f"tokenizer={tokenizer_dir} workers={workers}", flush=True)

    def consume(raw) -> None:
        nonlocal processed_bytes, num_tokens, num_markers, batches, last_report
        array, size, markers = pending.popleft().result()
        array.tofile(raw)
        processed_bytes += size
        num_tokens += array.size
        num_markers += markers
        batches += 1
        now = time.perf_counter()
        if now - last_report >= 5:
            rate = processed_bytes / (now - started)
            remaining = (total_bytes - processed_bytes) / rate if rate else float("inf")
            print(f"[{output_path.name}] {processed_bytes / 2**20:.1f}/{total_bytes / 2**20:.1f} MiB "
                  f"({100 * processed_bytes / total_bytes if total_bytes else 100:.1f}%) "
                  f"tokens={num_tokens:,} rate={rate / 2**20:.2f} MiB/s "
                  f"elapsed={now - started:.1f}s ETA={remaining / 60:.1f}min", flush=True)
            last_report = now

    with raw_path.open("wb") as raw, ProcessPoolExecutor(
        max_workers=workers, mp_context=get_context("spawn"), initializer=_init_worker,
        initargs=(str(tokenizer_dir / "vocab.pkl"), str(tokenizer_dir / "merges.pkl")),
    ) as pool:
        for batch in _document_batches(input_path, read_size):
            pending.append(pool.submit(_encode_batch, batch))
            if len(pending) >= 2 * workers:
                consume(raw)
        while pending:
            consume(raw)
    if processed_bytes != total_bytes:
        raise ValueError(f"Byte count mismatch: {processed_bytes} != {total_bytes}")

    # The shape is known only after encoding. Stream a standard NPY header and
    # payload to disk, keeping memory bounded rather than loading all IDs.
    with npy_path.open("wb") as target, raw_path.open("rb") as raw:
        np.lib.format.write_array_header_2_0(target, {
            "descr": np.dtype(np.uint16).str, "fortran_order": False,
            "shape": (num_tokens,),
        })
        shutil.copyfileobj(raw, target, length=8 * 1024 * 1024)
    array = np.load(npy_path, mmap_mode="r", allow_pickle=False)
    if array.dtype != np.uint16 or array.shape != (num_tokens,):
        raise ValueError("Saved NPY dtype/shape mismatch.")
    vocab = Tokenizer.from_files(str(tokenizer_dir / "vocab.pkl"),
                                 str(tokenizer_dir / "merges.pkl"), [_BOUNDARY]).vocab
    max_id = int(array.max()) if num_tokens else None
    if max_id is not None and max_id > max(vocab):
        raise ValueError("Saved token ID exceeds the vocabulary.")
    del array
    os.replace(npy_path, output_path)
    raw_path.unlink()
    result = {
        "input": str(input_path), "output": str(output_path),
        "tokenizer": str(tokenizer_dir), "vocab_size": len(vocab),
        "dtype": "uint16", "tokens": num_tokens, "max_id": max_id,
        "input_bytes": processed_bytes, "file_bytes": output_path.stat().st_size,
        "document_separators": num_markers, "validated_batches": batches,
        "seconds": time.perf_counter() - started,
    }
    print(f"[{output_path.name}] DONE tokens={num_tokens:,} "
          f"size={result['file_bytes'] / 2**30:.3f} GiB "
          f"time={result['seconds']:.1f}s; all batches round-trip verified", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--batch-chars", type=int, default=1024 * 1024)
    args = parser.parse_args()
    if args.workers < 1 or args.batch_chars < 1:
        parser.error("workers and batch-chars must be positive")
    root = Path(__file__).resolve().parents[1]
    out = root / "artifacts/bpe"
    jobs = [
        ("TinyStoriesV2-GPT4-valid.txt", "tinystories_valid.npy", "tinystories"),
        ("TinyStoriesV2-GPT4-train.txt", "tinystories_train.npy", "tinystories"),
        ("owt_valid.txt", "openwebtext_valid.npy", "openwebtext"),
        ("owt_train.txt", "openwebtext_train.npy", "openwebtext"),
    ]
    results = []
    for source, target, model in jobs:
        results.append(encode_file(root / "data" / source, out / target, out / model,
                                   args.workers, args.batch_chars))
        lines = [
            "# Encoded datasets", "",
            "Both development splits use the tokenizer trained on the corresponding training split.",
            "Every batch was decoded and compared with the exact input bytes; document separators are retained.",
            "", "| File | Tokenizer | Dtype | Token count | File bytes | Seconds |",
            "|---|---|---|---:|---:|---:|",
        ]
        for result in results:
            lines.append(f"| {Path(result['output']).name} | {Path(result['tokenizer']).name} "
                         f"({result['vocab_size']:,}) | uint16 | {result['tokens']:,} "
                         f"| {result['file_bytes']:,} | {result['seconds']:.1f} |")
        lines += ["", "Load without reading the entire file into memory:", "", "```python",
                  'ids = np.load("path/to/dataset.npy", mmap_mode="r")', "```", "",
                  "uint16 stores IDs 0–65,535 using two bytes per token; both vocabularies fit.", ""]
        (out / "encoded-datasets.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
