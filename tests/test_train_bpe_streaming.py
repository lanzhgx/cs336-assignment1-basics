from collections import Counter

import pytest
import regex as re

from cs336_basics.train_bpe import PAT, _count_pretokens, train_bpe


@pytest.mark.parametrize("read_size", [1, 2, 7, 13, 32, 1024])
@pytest.mark.parametrize(
    "text,special_tokens",
    [
        ("hello<|endoftext|>你好 🌍\n\nworld<|endoftext|>unfinished", ["<|endoftext|>"]),
        ("<x>first<x><y>second<x><x><y>", ["<x>", "<x><y>"]),
        ("a<user>b<|endoftext|>c<user>", ["<|endoftext|>", "<user>"]),
        ("no delimiter here: café 你好\n  tail", ["<|endoftext|>"]),
        ("hello world\n\n你好", []),
        ("", ["<|endoftext|>"]),
    ],
)
def test_streaming_counts_match_whole_text(tmp_path, read_size, text, special_tokens):
    path = tmp_path / "corpus.txt"
    path.write_text(text, encoding="utf-8")
    pattern = "|".join(re.escape(t) for t in sorted(special_tokens, key=len, reverse=True))
    chunks = re.split(pattern, text) if special_tokens else [text]
    expected = Counter()
    for chunk in chunks:
        for match in re.finditer(PAT, chunk):
            encoded = match.group().encode("utf-8")
            expected[tuple(bytes([byte]) for byte in encoded)] += 1

    assert _count_pretokens(path, special_tokens, read_size) == expected


@pytest.mark.parametrize(
    "text,special_tokens",
    [
        ("hello<|endoftext|>你好 🌍<|endoftext|>tail" * 8, ["<|endoftext|>"]),
        ("a<x>b<x><y>c<x>tail" * 8, ["<x>", "<x><y>"]),
        ("no special tokens: café 你好\n", []),
    ],
)
def test_parallel_counts_match_serial(tmp_path, text, special_tokens):
    path = tmp_path / "corpus.txt"
    path.write_text(text, encoding="utf-8")
    expected = _count_pretokens(path, special_tokens, read_size=7)
    assert _count_pretokens(path, special_tokens, read_size=7, workers=2) == expected


def test_parallel_training_matches_serial():
    from pathlib import Path

    path = Path(__file__).parent / "fixtures/corpus.en"
    expected = train_bpe(path, 270, ["<|endoftext|>"], workers=1)
    assert train_bpe(path, 270, ["<|endoftext|>"], workers=2) == expected
