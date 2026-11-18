"""Train BPE and save vocab/merges as pickle files preserving Python types.

Run from the repository root: uv run python -m cs336_basics.train_bpe
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from multiprocessing import get_context
import os
import heapq
import pickle
from pathlib import Path
import time

from collections import Counter, deque

import regex as re

ENCODE_FORMAT = "utf-8"
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_HEAP_REBUILD_FACTOR = 3
_HEAP_REBUILD_MIN = 1024


class _PairCandidate:
    """A heap snapshot ordered by highest frequency, then largest byte pair."""

    __slots__ = ("pair", "count")

    def __init__(self, pair: tuple[bytes, bytes], count: int) -> None:
        self.pair = pair
        self.count = count

    def __lt__(self, other: "_PairCandidate") -> bool:
        if self.count != other.count:
            return self.count > other.count
        return self.pair > other.pair


def _count_text_batch(text: str, special_tokens: list[str]) -> Counter[str]:
    """Worker task: count strings in a batch ending at a document boundary."""
    pattern = "|".join(re.escape(t) for t in sorted(special_tokens, key=len, reverse=True))
    chunks = re.split(pattern, text) if special_tokens else [text]
    counts: Counter[str] = Counter()
    for chunk in chunks:
        counts.update(match.group() for match in re.finditer(PAT, chunk))
    return counts


def _count_pretokens(
    input_path: str | Path, special_tokens: list[str], read_size: int = 1024 * 1024,
    *, progress: bool = False, workers: int = 1,
) -> Counter[tuple[bytes, ...]]:
    """Count strings in batches, then encode each unique pre-token once.

    Memory still depends on unique pre-tokens and the longest span without a
    special token. With no special tokens, the whole text must be retained.
    """
    if read_size <= 0:
        raise ValueError("read_size must be positive")
    if workers < 1:
        raise ValueError("workers must be positive")
    if any(not token for token in special_tokens):
        raise ValueError("special tokens must be nonempty")
    separator = (
        re.compile("|".join(re.escape(t) for t in sorted(special_tokens, key=len, reverse=True)))
        if special_tokens else None
    )
    longest_special = max(map(len, special_tokens), default=0)
    counts: Counter[str] = Counter()
    pool = None
    pending = deque()
    started = time.perf_counter()
    last_report = started
    read_seconds = count_seconds = 0.0
    read_bytes = processed_bytes = batches = 0
    total_bytes = Path(input_path).stat().st_size if progress else 0
    if progress:
        import psutil
        process = psutil.Process()
        print(f"[pretoken] starting: file={total_bytes / 2**30:.2f} GiB; "
              f"batch={read_size:,} characters workers={workers} "
              f"max_pending={2 * workers if workers > 1 else 0}", flush=True)

    def report(force: bool = False) -> None:
        nonlocal last_report
        if not progress:
            return
        now = time.perf_counter()
        if not force and now - last_report < 5:
            return
        elapsed = now - started
        rate = processed_bytes / elapsed if elapsed else 0
        eta = max(0, total_bytes - processed_bytes) / rate if rate else float("inf")
        tree_rss = process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                tree_rss += child.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        print(
            f"[pretoken] {processed_bytes / 2**20:.1f}/{total_bytes / 2**20:.1f} MiB "
            f"({100 * processed_bytes / total_bytes if total_bytes else 100:.1f}%) "
            f"read={read_bytes / 2**20:.1f} MiB batches={batches:,} "
            f"unique={len(counts):,} tail={len(buffer):,} chars "
            f"RSS={process.memory_info().rss / 2**30:.2f} GiB "
            f"tree_RSS={tree_rss / 2**30:.2f} GiB pending={len(pending)} "
            f"swap={psutil.swap_memory().used / 2**30:.2f} GiB "
            f"elapsed={elapsed:.1f}s rate={rate / 2**20:.2f} MiB/s "
            f"ETA={eta / 60:.1f}min read_time={read_seconds:.2f}s "
            f"count_time={count_seconds:.2f}s",
            flush=True,
        )
        last_report = now

    def consume_one() -> None:
        nonlocal processed_bytes
        future, size = pending.popleft()
        counts.update(future.result())
        processed_bytes += size

    def count_text(text: str) -> None:
        nonlocal processed_bytes
        if not text:
            return
        size = len(text.encode(ENCODE_FORMAT)) if progress else 0
        if pool is not None:
            pending.append((pool.submit(_count_text_batch, text, special_tokens), size))
            # Backpressure keeps queued input and result Counters bounded.
            if len(pending) >= 2 * workers:
                consume_one()
            return
        chunks = separator.split(text) if separator is not None else [text]
        for chunk in chunks:
            counts.update(match.group() for match in re.finditer(PAT, chunk))
        processed_bytes += size

    buffer = ""
    with ExitStack() as stack:
        if workers > 1:
            pool = stack.enter_context(ProcessPoolExecutor(
                max_workers=workers, mp_context=get_context("spawn")
            ))
        f = stack.enter_context(open(input_path, "r", encoding=ENCODE_FORMAT))
        while True:
            read_started = time.perf_counter() if progress else 0
            text = f.read(read_size)
            if progress:
                read_seconds += time.perf_counter() - read_started
                read_bytes += len(text.encode(ENCODE_FORMAT))
            if not text:
                break
            batches += 1
            buffer += text
            if separator is None:
                report()
                continue
            end = 0
            for match in separator.finditer(buffer):
                # A shorter token at the read edge might be a prefix of a
                # longer special token. Wait for enough lookahead to decide.
                if len(buffer) - match.start() < longest_special:
                    break
                end = match.end()
            if end:
                count_started = time.perf_counter() if progress else 0
                count_text(buffer[:end])
                if progress:
                    count_seconds += time.perf_counter() - count_started
                buffer = buffer[end:]
            report()
        count_started = time.perf_counter() if progress else 0
        count_text(buffer)
        while pending:
            consume_one()
            report()
        if progress:
            count_seconds += time.perf_counter() - count_started
        buffer = ""
        report(force=True)

    conversion_started = time.perf_counter()
    if progress:
        print(f"[pretoken] converting {len(counts):,} unique strings to byte tuples", flush=True)
    byte_counts: Counter[tuple[bytes, ...]] = Counter()
    while counts:
        # Release string keys as we go instead of retaining both full mappings.
        pretoken, count = counts.popitem()
        encoded = pretoken.encode(ENCODE_FORMAT)
        tokens = tuple(encoded[i:i + 1] for i in range(len(encoded)))
        byte_counts[tokens] = count
    counts.clear()
    if progress:
        print(f"[pretoken] conversion={time.perf_counter() - conversion_started:.2f}s "
              f"total={time.perf_counter() - started:.2f}s "
              f"RSS={process.memory_info().rss / 2**30:.2f} GiB", flush=True)
    return byte_counts


def train_bpe(
    input_path: str | Path, vocab_size: int, special_tokens: list[str],
    *, progress: bool = False, workers: int = 1, metrics: dict | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train byte-level BPE and return its vocabulary and ordered merges."""
    pretoken_started = time.perf_counter()
    pretoken_counts = _count_pretokens(input_path, special_tokens, progress=progress, workers=workers)
    if metrics is not None:
        metrics.update(pretoken_seconds=time.perf_counter() - pretoken_started,
                       unique_pretokens=len(pretoken_counts),
                       initial_token_slots=sum(map(len, pretoken_counts)))
        serialize_started = time.perf_counter()
        serialized = pickle.dumps(pretoken_counts, protocol=pickle.HIGHEST_PROTOCOL)
        metrics.update(counter_pickle_seconds=time.perf_counter() - serialize_started,
                       counter_pickle_bytes=len(serialized))
        del serialized
        metrics["rounds"] = []
    unique_count = len(pretoken_counts)
    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode(ENCODE_FORMAT)
    merges = []

    pair_counts = Counter()
    pair_rows: dict[tuple[bytes, bytes], set[int]] = {}
    sequences: list[tuple[bytes, ...]] = []
    frequencies: list[int] = []
    pair_started = time.perf_counter()
    if progress:
        print(f"[pairs] starting count and index: {unique_count:,} unique pre-tokens", flush=True)
    while pretoken_counts:
        old, count = pretoken_counts.popitem()
        row = len(sequences)
        sequences.append(old)
        frequencies.append(count)
        for i in range(len(old)-1):
            pair = (old[i], old[i+1])
            pair_counts[pair] += count
            if pair not in pair_rows:
                pair_rows[pair] = set()
            pair_rows[pair].add(row)
    pretoken_counts.clear()
    candidates = [_PairCandidate(pair, count) for pair, count in pair_counts.items()]
    heapq.heapify(candidates)
    heap_rebuilds = 0

    pair_seconds = time.perf_counter() - pair_started
    if metrics is not None:
        metrics["initial_pair_seconds"] = pair_seconds
        import psutil
        metrics["rss_after_index_bytes"] = psutil.Process().memory_info().rss
        metrics["index_memberships"] = sum(map(len, pair_rows.values()))
    if progress:
        print(f"[pairs] complete: {len(pair_counts):,} pairs in "
              f"{pair_seconds:.2f}s", flush=True)

    merge_total = vocab_size - len(special_tokens) - 256
    merge_started = last_merge_report = time.perf_counter()
    selection_seconds = apply_seconds = 0.0
    for _ in range(merge_total):

        if not pair_counts:
            break
        selection_started = time.perf_counter()
        # Ignore snapshots whose frequencies have changed since insertion.
        while candidates:
            candidate = heapq.heappop(candidates)
            if pair_counts.get(candidate.pair, 0) == candidate.count:
                best_pair = candidate.pair
                break
        else:
            break
        selection_elapsed = time.perf_counter() - selection_started
        selection_seconds += selection_elapsed
        apply_started = time.perf_counter()
        # Snapshot because updating rows also changes this pair's membership.
        affected_rows = tuple(pair_rows[best_pair])
        affected = len(affected_rows)
        dirty_pairs = set()
        for row in affected_rows:
            old = sequences[row]
            count = frequencies[row]
            new = []
            i = 0
            while i < len(old):
                if i < len(old)-1 and best_pair == (old[i], old[i+1]):
                    new.append(old[i] + old[i+1])
                    i += 2
                else:
                    new.append(old[i])
                    i += 1
            old_pairs = Counter(zip(old, old[1:]))
            new_pairs = Counter(zip(new, new[1:]))
            for pair, occurrences in old_pairs.items():
                remaining = new_pairs.get(pair, 0)
                delta = remaining - occurrences
                if delta:
                    pair_counts[pair] += delta * count
                    dirty_pairs.add(pair)
                    if not pair_counts[pair]:
                        del pair_counts[pair]
                if not remaining:
                    rows = pair_rows[pair]
                    rows.remove(row)
                    if not rows:
                        del pair_rows[pair]
            for pair, occurrences in new_pairs.items():
                if pair not in old_pairs:
                    pair_counts[pair] += occurrences * count
                    dirty_pairs.add(pair)
                    if pair not in pair_rows:
                        pair_rows[pair] = set()
                    pair_rows[pair].add(row)
            sequences[row] = tuple(new)
        # Insert once per changed pair, after all rows have updated its count.
        for pair in dirty_pairs:
            count = pair_counts.get(pair, 0)
            if count:
                heapq.heappush(candidates, _PairCandidate(pair, count))
        # Bound stale heap entries rather than retaining all historical counts.
        if len(candidates) > max(_HEAP_REBUILD_MIN, _HEAP_REBUILD_FACTOR * len(pair_counts)):
            candidates = [_PairCandidate(pair, count) for pair, count in pair_counts.items()]
            heapq.heapify(candidates)
            heap_rebuilds += 1
        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]
        apply_elapsed = time.perf_counter() - apply_started
        apply_seconds += apply_elapsed
        if metrics is not None:
            metrics["rounds"].append({
                "round": _ + 1, "selection_seconds": selection_elapsed,
                "apply_seconds": apply_elapsed, "affected_pretokens": affected,
                "pair_keys": len(pair_counts),
            })
        if progress:
            now = time.perf_counter()
            if _ == 0 or _ + 1 == merge_total or now - last_merge_report >= 5:
                elapsed = now - merge_started
                eta = elapsed / (_ + 1) * (merge_total - _ - 1)
                print(f"[merge] {_ + 1:,}/{merge_total:,} elapsed={elapsed:.1f}s "
                      f"avg={elapsed / (_ + 1):.3f}s/merge ETA={eta / 3600:.2f}h "
                      f"unique={unique_count:,} affected={affected:,} pair_keys={len(pair_counts):,}", flush=True)
                last_merge_report = now

    if metrics is not None:
        metrics.update(merge_seconds=time.perf_counter() - merge_started,
                       selection_seconds=selection_seconds, apply_seconds=apply_seconds,
                       final_pair_keys=len(pair_counts),
                       positive_pair_keys=sum(count > 0 for count in pair_counts.values()),
                       heap_entries=len(candidates), heap_rebuilds=heap_rebuilds,
                       rss_after_merge_bytes=psutil.Process().memory_info().rss)
    if progress:
        print(f"[merge-summary] rounds={len(merges):,} select={selection_seconds:.2f}s "
              f"apply={apply_seconds:.2f}s", flush=True)
    return vocab, merges



def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=root / "data/TinyStoriesV2-GPT4-train.txt"
    )
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--progress", action="store_true", help="Print stage timings, memory, throughput, and ETA.")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1),
                        help="Pre-tokenization processes (default: up to 4); use 1 for serial.")
    parser.add_argument(
        "--output-dir", type=Path, default=root / "artifacts/bpe/tinystories"
    )
    parser.add_argument(
        "--special-token", action="append",
        help="Repeat for multiple special tokens; defaults to <|endoftext|>.",
    )
    args = parser.parse_args()
    special_tokens = args.special_token or ["<|endoftext|>"]
    if args.workers < 1:
        parser.error("--workers must be positive.")
    if not args.input.is_file():
        parser.error(f"Input file does not exist: {args.input}")
    if len(set(special_tokens)) != len(special_tokens) or "" in special_tokens:
        parser.error("Special tokens must be unique and nonempty.")
    if args.vocab_size < 256 + len(special_tokens):
        parser.error("Vocab size must accommodate 256 bytes and all special tokens.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input: {args.input.resolve()}", flush=True)
    print(f"Target vocabulary: {args.vocab_size}; special tokens: {special_tokens}", flush=True)
    vocab, merges = train_bpe(
        args.input, args.vocab_size, special_tokens, progress=args.progress, workers=args.workers
    )

    for name, content in {"vocab.pkl": vocab, "merges.pkl": merges}.items():
        path = args.output_dir / name
        with path.open("wb") as f:
            pickle.dump(content, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved: {path.resolve()}")

    metadata = {
        "serialization": "pickle",
        "input_path": str(args.input.resolve()),
        "target_vocab_size": args.vocab_size,
        "actual_vocab_size": len(vocab),
        "num_merges": len(merges),
        "special_tokens": special_tokens,
        "workers": args.workers,
    }
    metadata_path = args.output_dir / "metadata.pkl"
    with metadata_path.open("wb") as f:
        pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved: {metadata_path.resolve()}")
    print(f"Training complete: {len(vocab)} tokens, {len(merges)} merges")


if __name__ == "__main__":
    main()
