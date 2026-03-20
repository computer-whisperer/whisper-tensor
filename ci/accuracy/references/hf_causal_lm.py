#!/usr/bin/env python3
"""Generate golden reference outputs for HuggingFace causal language models.

Uses transformers AutoModelForCausalLM as the first-party reference.
Works for any model that follows the standard HF causal LM interface:
Qwen2, Llama, Gemma, Mistral, Phi, etc.

The whisper-tensor ONNX graph operates with seq_len=1 and KV cache state,
but for a single-token forward pass with no prior context, this is
equivalent to a standard HF forward pass with a single token.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import save_snapshot


def main():
    parser = argparse.ArgumentParser(
        description="Generate golden outputs for an HF causal LM"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to HuggingFace model directory (with config.json + safetensors)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for golden snapshot",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "bfloat16", "float16"],
        help="Dtype to run inference in (default: float32 for reference accuracy)",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    torch_dtype = dtype_map[args.dtype]

    print(f"Loading model: {model_path} (dtype={args.dtype})")

    from transformers import AutoModelForCausalLM, AutoConfig

    config = AutoConfig.from_pretrained(str(model_path))
    print(f"Model type: {config.model_type}")
    print(f"Layers: {config.num_hidden_layers}, "
          f"Heads: {config.num_attention_heads}, "
          f"Vocab: {config.vocab_size}")

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=torch_dtype,
        trust_remote_code=False,
    ).cpu()
    model.eval()

    with torch.no_grad():
        # Single token forward pass — matches whisper-tensor's RNN mode
        # with empty KV cache (first token in sequence).
        # Use token_id=1 (avoids special tokens like BOS/EOS/PAD).
        token_ids = [1]
        input_tensor = torch.tensor([token_ids], dtype=torch.long)

        print(f"Input tokens: {token_ids}")
        outputs = model(input_tensor)
        logits = outputs.logits  # [batch=1, seq_len=1, vocab_size]

        # Squeeze batch dim to match whisper-tensor output shape [seq_len, vocab_size]
        logits_squeezed = logits.squeeze(0)
        print(f"Logits shape: {logits_squeezed.shape}")

        # Always save logits as float32 for comparison accuracy
        logits_f32 = logits_squeezed.float()

    # The whisper-tensor ONNX graph uses I32 for token input
    inputs = {
        "input_ids": input_tensor.numpy().astype(np.int32),
    }
    outputs = {
        "logits": logits_f32.numpy(),
    }

    save_snapshot(args.output, inputs, outputs)


if __name__ == "__main__":
    main()
