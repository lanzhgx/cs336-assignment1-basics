"""Generate text from a trained Transformer LM checkpoint."""

import argparse
from math import isfinite
from pathlib import Path

import torch

from .tokenizer import Tokenizer
from .transformer import TransformerLM


END_OF_TEXT = "<|endoftext|>"


def _sample_next_token(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    probabilities = torch.softmax(logits.float() / temperature, dim=-1)
    if top_p == 1.0:
        return torch.multinomial(probabilities, num_samples=1).item()

    sorted_probs, sorted_ids = probabilities.sort(descending=True)
    cumulative_probs = sorted_probs.cumsum(dim=-1)
    # Keep the token that first brings cumulative probability to top_p.
    sorted_probs[1:] = sorted_probs[1:].masked_fill(cumulative_probs[:-1] >= top_p, 0.0)
    sampled_index = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_ids[sampled_index].item()


@torch.inference_mode()
def generate(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    *,
    context_length: int,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> str:
    """Sample and decode a completion, stopping before the end-of-text marker."""
    if context_length < 1:
        raise ValueError("context_length must be positive")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative")
    if not isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")

    eos_ids = tokenizer.encode(END_OF_TEXT)
    if END_OF_TEXT not in tokenizer.special_tokens or len(eos_ids) != 1:
        raise ValueError("The tokenizer must register <|endoftext|> as a special token")
    eos_id = eos_ids[0]
    token_ids = tokenizer.encode(prompt) or [eos_id]
    prompt_length = len(token_ids)
    device = next(model.parameters()).device

    for _ in range(max_new_tokens):
        input_ids = torch.tensor([token_ids[-context_length:]], dtype=torch.long, device=device)
        next_token_logits = model(input_ids)[0, -1]
        next_token = _sample_next_token(next_token_logits, temperature, top_p)
        if next_token == eos_id:
            break
        token_ids.append(next_token)

    return tokenizer.decode(token_ids[prompt_length:])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocab", required=True, help="Path to vocab.pkl.")
    parser.add_argument("--merges", required=True, help="Path to merges.pkl.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int)

    model_args = parser.add_argument_group("model settings used during training")
    model_args.add_argument("--context-length", type=int, default=128)
    model_args.add_argument("--d-model", type=int, default=128)
    model_args.add_argument("--num-heads", type=int, default=4)
    model_args.add_argument("--num-layers", type=int, default=4)
    model_args.add_argument("--d-ff", type=int, default=320)
    model_args.add_argument("--rope-theta", type=float, default=10_000.0)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_files(args.vocab, args.merges, special_tokens=[END_OF_TEXT])
    model = TransformerLM(
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.rope_theta,
        vocab_size=len(tokenizer.vocab),
        context_length=args.context_length,
        num_layers=args.num_layers,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    model.to(args.device).eval()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    completion = generate(
        model,
        tokenizer,
        args.prompt,
        context_length=args.context_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(args.prompt + completion)


if __name__ == "__main__":
    main()
