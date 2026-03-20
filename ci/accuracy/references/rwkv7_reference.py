#!/usr/bin/env python3
"""Generate golden reference outputs for RWKV-7 using the official first-party code.

Uses the RWKV-7 model classes from https://github.com/BlinkDL/RWKV-LM (RWKV-v7/rwkv_v7_demo.py),
cloned into the container at /opt/rwkv-lm. The demo file runs everything at module level and
requires CUDA by default, so we exec a patched version that disables CUDA and skips the
inference/eval code.

The whisper-tensor ONNX graph operates in RNN mode (seq_len=1 with explicit state
inputs/outputs). We run the reference in GPT mode with seq_len=1 which is equivalent
when states are initialized to zeros.
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import save_snapshot


def load_rwkv7_model_class(rwkv_lm_path: str):
    """Load the RWKV-7 model class from the official demo source.

    Reads rwkv_v7_demo.py, patches it to disable CUDA and JIT, exec's the
    class definitions, and returns the RWKV model class.
    """
    demo_path = os.path.join(rwkv_lm_path, "RWKV-v7", "rwkv_v7_demo.py")
    if not os.path.exists(demo_path):
        raise FileNotFoundError(
            f"RWKV-LM demo not found at {demo_path}. "
            f"Set RWKV_LM_PATH env var or clone https://github.com/BlinkDL/RWKV-LM.git"
        )

    with open(demo_path) as f:
        source = f.read()

    # --- Patch the source to make it importable and CPU-only ---

    # 1. Force USE_CUDA_KERNEL = False (the demo defaults to True)
    source = source.replace("USE_CUDA_KERNEL = True", "USE_CUDA_KERNEL = False")

    # 2. Disable JIT (we want plain nn.Module, not ScriptModule)
    source = source.replace(
        "MyModule = torch.jit.ScriptModule",
        "MyModule = torch.nn.Module",
    )
    source = source.replace(
        "MyFunction = torch.jit.script_method",
        "MyFunction = lambda ob: ob",
    )
    source = source.replace(
        "MyStatic = torch.jit.script",
        "MyStatic = lambda ob: ob",
    )

    # 3. Remove side-effect code while keeping all class/function definitions.
    #    Structure of rwkv_v7_demo.py:
    #      - imports, config, globals
    #      - class RWKV_TOKENIZER (don't need, but harmless to define)
    #      - tokenizer = RWKV_TOKENIZER(...) — SKIP (side effect, needs vocab file)
    #      - CUDA kernel code + RWKV7_OP function
    #      - Model classes (RWKV_Tmix_x070, RWKV_CMix_x070, Block, RWKV)
    #      - model_params = torch.load(...) — TRUNCATE here (inference code)

    # Remove the tokenizer instantiation line
    source = re.sub(
        r'^tokenizer = RWKV_TOKENIZER\(.*\).*$',
        '# tokenizer instantiation removed',
        source,
        flags=re.MULTILINE,
    )

    # Truncate at the inference code (model_params = torch.load(...))
    marker = "\nmodel_params = "
    idx = source.find(marker)
    if idx != -1:
        source = source[:idx]

    # 4. Execute in a controlled namespace
    namespace = {"__name__": "__rwkv7_ref__", "__builtins__": __builtins__}
    exec(compile(source, demo_path, "exec"), namespace)

    rwkv_class = namespace.get("RWKV")
    if rwkv_class is None:
        raise RuntimeError(
            "Failed to extract RWKV class from demo source. "
            "The file format may have changed."
        )

    # Also grab the args namespace for config
    args_ns = namespace.get("args")
    return rwkv_class, args_ns


def main():
    parser = argparse.ArgumentParser(description="Generate RWKV-7 golden outputs")
    parser.add_argument(
        "--model",
        type=str,
        default="test_models/RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth",
        help="Path to RWKV-7 .pth model file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for golden snapshot",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    rwkv_lm_path = os.environ.get("RWKV_LM_PATH", "/opt/rwkv-lm")
    print(f"Loading RWKV-7 reference from {rwkv_lm_path}")

    RWKV, ref_args = load_rwkv7_model_class(rwkv_lm_path)

    # Auto-detect model config from weights
    print(f"Loading weights: {model_path}")
    weights = torch.load(str(model_path), map_location="cpu", weights_only=True)

    n_embd = weights["emb.weight"].shape[1]
    vocab_size = weights["emb.weight"].shape[0]
    n_layer = max(int(k.split(".")[1]) for k in weights if k.startswith("blocks.")) + 1
    head_size = weights["blocks.0.att.r_k"].shape[1]

    # Configure the reference model
    ref_args.n_layer = n_layer
    ref_args.n_embd = n_embd
    ref_args.vocab_size = vocab_size
    ref_args.head_size_a = head_size

    print(f"Config: n_layer={n_layer} n_embd={n_embd} vocab_size={vocab_size} head_size={head_size}")

    with torch.no_grad():
        model = RWKV(ref_args).float()
        model.load_state_dict(weights, strict=False)  # layer 0 has no v0/v1/v2
        model.eval()

        # Run with a single token (seq_len=1), matching whisper-tensor's RNN mode.
        # With seq_len=1 and zero states, GPT mode time_shift is equivalent to
        # RNN mode with zero previous state.
        token_ids = [1]
        input_tensor = torch.tensor([token_ids], dtype=torch.long)

        print(f"Input tokens: {token_ids}")
        output = model.forward(input_tensor)
        print(f"Output shape: {output.shape}")  # [1, 1, vocab_size]

    # The whisper-tensor ONNX graph uses I32 for token input
    inputs = {
        "token_input": input_tensor.numpy().astype(np.int32),
    }
    outputs = {
        "output": output.numpy(),
    }

    save_snapshot(args.output, inputs, outputs)


if __name__ == "__main__":
    main()
