from collections import Counter
import importlib
import random

import pytest

training = importlib.import_module("cs336_basics.train_bpe")


def reference_train(counts, merge_limit):
    """Small oracle that recomputes every pair count after each merge."""
    vocab = {i: bytes([i]) for i in range(256)}
    vocab[256] = b"<|endoftext|>"
    merges = []
    for _ in range(merge_limit):
        pairs = Counter()
        for tokens, weight in counts.items():
            for pair in zip(tokens, tokens[1:]):
                pairs[pair] += weight
        if not pairs:
            break
        best = max(pairs, key=lambda p: (pairs[p], p))
        updated = Counter()
        for tokens, weight in counts.items():
            result = []
            i = 0
            while i < len(tokens):
                if tokens[i:i + 2] == best:
                    result.append(b"".join(best))
                    i += 2
                else:
                    result.append(tokens[i])
                    i += 1
            updated[tuple(result)] += weight
        counts = updated
        merges.append(best)
        vocab[len(vocab)] = b"".join(best)
    return vocab, merges


@pytest.mark.parametrize("words", [
    [], [(b"a", 3)], [(b"aaaaa", 3)], [(b"ababab", 5)],
    [(b"ab", 1), (b"ac", 1)],
    [(b"abab", 3), (b"xababy", 7), (b"aaaaaaa", 4)],
])
def test_indexed_merges_match_recounting(monkeypatch, words):
    counts = Counter()
    for word, frequency in words:
        counts[tuple(bytes([byte]) for byte in word)] += frequency
    monkeypatch.setattr(training, "_count_pretokens", lambda *a, **kw: counts.copy())
    assert training.train_bpe("unused", 287, ["<|endoftext|>"]) == reference_train(counts, 30)


@pytest.mark.parametrize("seed", range(20))
def test_indexed_random_merges_match_recounting(monkeypatch, seed):
    rng = random.Random(seed)
    alphabet = [b"a", b"b", b"c", b" ", b"\xc3", b"\xa9"]
    counts = Counter()
    for _ in range(40):
        tokens = tuple(rng.choice(alphabet) for _ in range(rng.randint(1, 12)))
        counts[tokens] += rng.randint(1, 9)
    monkeypatch.setattr(training, "_count_pretokens", lambda *a, **kw: counts.copy())
    assert training.train_bpe("unused", 307, ["<|endoftext|>"]) == reference_train(counts, 50)


def test_heap_compaction_preserves_merge_order(monkeypatch):
    monkeypatch.setattr(training, "_HEAP_REBUILD_FACTOR", 1)
    monkeypatch.setattr(training, "_HEAP_REBUILD_MIN", 0)
    rng = random.Random(336)
    counts = Counter(
        tuple(rng.choice([b"a", b"b", b"c"]) for _ in range(40))
        for _ in range(200)
    )
    monkeypatch.setattr(training, "_count_pretokens", lambda *a, **kw: counts.copy())
    metrics = {}
    actual = training.train_bpe("unused", 457, ["<|endoftext|>"], metrics=metrics)
    assert actual == reference_train(counts, 200)
    assert metrics["heap_rebuilds"] > 0
