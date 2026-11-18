import pickle

import numpy as np
import pytest

from cs336_basics.encode_datasets import encode_file
from cs336_basics.tokenizer import Tokenizer


@pytest.mark.parametrize("read_size", [1, 7, 1024])
def test_streamed_npy_matches_whole_text(tmp_path, read_size):
    model = tmp_path / "model"
    model.mkdir()
    vocab = {i: bytes([i]) for i in range(256)}
    merges = [(b"a", b"b"), (b"ab", b"ab"), (b" ", b"a")]
    for left, right in merges:
        vocab[len(vocab)] = left + right
    for name, data in [("vocab.pkl", vocab), ("merges.pkl", merges)]:
        (model / name).write_bytes(pickle.dumps(data))
    text = "abab 你好 🌍\r\n<|endoftext|>\n a<|endoftext|><|endoftext|>tail"
    source = tmp_path / "source.txt"
    source.write_bytes(text.encode("utf-8"))
    target = tmp_path / "tokens.npy"
    result = encode_file(source, target, model, workers=2, read_size=read_size)
    tokenizer = Tokenizer.from_files(str(model / "vocab.pkl"), str(model / "merges.pkl"),
                                     ["<|endoftext|>"])
    saved = np.load(target, mmap_mode="r", allow_pickle=False)
    assert saved.dtype == np.uint16
    assert saved.tolist() == tokenizer.encode(text)
    assert result["input_bytes"] == len(text.encode("utf-8"))
    assert result["document_separators"] == 3
    assert not target.with_suffix(".raw.partial").exists()
    assert not target.with_suffix(".npy.partial").exists()
