from collections.abc import Iterable, Iterator
import argparse
import pickle
import heapq
from functools import lru_cache
from pathlib import Path
from .train_bpe import PAT, ENCODE_FORMAT
import regex as re

_BYTE_TOKENS = tuple(bytes([i]) for i in range(256))

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        """Construct a tokenizer from a vocabulary, merges, and special tokens."""
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []
        self.reversed_vocab: dict[bytes, int] = {}
        for key, value in vocab.items():
            self.reversed_vocab[value] = key

        for special_token in self.special_tokens:
            special_token_byte = special_token.encode(ENCODE_FORMAT)
            if special_token_byte not in self.reversed_vocab:
                id = len(self.vocab)
                self.vocab[id] = special_token_byte
                self.reversed_vocab[special_token_byte] = id

        self._merge_ranks = {}
        for rank, pair in enumerate(merges):
            self._merge_ranks.setdefault(pair, rank)
        self._special_ids = {
            token: self.reversed_vocab[token.encode(ENCODE_FORMAT)]
            for token in self.special_tokens
        }
        self._special_pattern = (
            re.compile("(" + "|".join(
                re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True)
            ) + ")") if self.special_tokens else None
        )
        self._pretoken_pattern = re.compile(PAT)
        self._encode_piece_cached = lru_cache(maxsize=65_536)(self._encode_piece)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        """Construct a tokenizer from serialized vocabulary and merge files."""
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)

        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)
        
        return cls(vocab, merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
        """Encode text into token IDs."""
        chunks = self._special_pattern.split(text) if self._special_pattern else [text]
        result = []
        for chunk in chunks:
            if chunk in self._special_ids:
                result.append(self._special_ids[chunk])
                continue
            for match in self._pretoken_pattern.finditer(chunk):
                result.extend(self._encode_piece_cached(match.group()))
        return result

    def _encode_piece(self, text: str) -> tuple[int, ...]:
        tokens = [_BYTE_TOKENS[byte] for byte in text.encode(ENCODE_FORMAT)]
        if len(tokens) > 64:
            return self._encode_long_piece(tokens)
        ranks = self._merge_ranks
        while len(tokens) > 1:
            best_pair = None
            best_rank = len(self.merges)
            for pair in zip(tokens, tokens[1:]):
                rank = ranks.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank, best_pair = rank, pair
            if best_pair is None:
                break
            merged = best_pair[0] + best_pair[1]
            updated = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best_pair:
                    updated.append(merged)
                    i += 2
                else:
                    updated.append(tokens[i])
                    i += 1
            tokens = updated
        return tuple(self.reversed_vocab[token] for token in tokens)

    def _encode_long_piece(self, tokens: list[bytes]) -> tuple[int, ...]:
        """Update adjacent candidates locally for unusually long pre-tokens."""
        n = len(tokens)
        previous = list(range(-1, n - 1))
        following = list(range(1, n)) + [-1]
        candidates = []

        def add_candidate(left: int) -> None:
            if left < 0:
                return
            right = following[left]
            if right < 0:
                return
            rank = self._merge_ranks.get((tokens[left], tokens[right]))
            if rank is not None:
                heapq.heappush(candidates, (rank, left, right))

        for left in range(n - 1):
            add_candidate(left)
        while candidates:
            rank, left, right = heapq.heappop(candidates)
            if not tokens[left] or not tokens[right] or following[left] != right:
                continue
            if self._merge_ranks.get((tokens[left], tokens[right])) != rank:
                continue
            tokens[left] += tokens[right]
            tokens[right] = b""
            following[left] = following[right]
            if following[right] >= 0:
                previous[following[right]] = left
            add_candidate(previous[left])
            add_candidate(left)
        return tuple(self.reversed_vocab[token] for token in tokens if token)

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily encode an iterable of strings into token IDs."""
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs into text."""
        return b"".join(self.vocab[id] for id in ids).decode(ENCODE_FORMAT, errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode, decode, or test a tokenizer round trip.")
    parser.add_argument("--mode", required=True, choices=["encode", "decode", "roundtrip"])
    parser.add_argument("--vocab", required=True, help="Path to vocab.pkl.")
    parser.add_argument("--merges", required=True, help="Path to merges.pkl.")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--text", help="Input text for encode or roundtrip mode.")
    inputs.add_argument("--text-path", type=Path, help="UTF-8 text file for encode or roundtrip mode.")
    inputs.add_argument("--ids", type=int, nargs="*", help="Token IDs for decode mode.")
    parser.add_argument(
        "--special-token", action="append",
        help="Repeat for multiple special tokens; defaults to <|endoftext|>.",
    )
    args = parser.parse_args()
    if args.mode == "decode" and args.ids is None:
        parser.error("decode mode requires --ids.")
    if args.mode != "decode" and args.text is None and args.text_path is None:
        parser.error(f"{args.mode} mode requires --text or --text-path.")

    text = args.text
    if args.text_path is not None:
        try:
            text = args.text_path.read_text(encoding=ENCODE_FORMAT)
        except (OSError, UnicodeError) as error:
            parser.error(f"Cannot read {args.text_path}: {error}")

    tokenizer = Tokenizer.from_files(
        args.vocab,
        args.merges,
        special_tokens=args.special_token or ["<|endoftext|>"],
    )
    if args.mode == "decode":
        print(tokenizer.decode(args.ids), flush=True)
        return

    print(f"Input: {text!r}", flush=True)
    ids = tokenizer.encode(text)
    print(f"Token IDs: {ids}", flush=True)
    if args.mode == "encode":
        return
    decoded = tokenizer.decode(ids)
    print(f"Decoded: {decoded!r}", flush=True)
    if decoded != text:
        raise SystemExit("FAIL: decoded text differs from input.")
    print("PASS: encode/decode round trip.")


if __name__ == "__main__":
    main()
