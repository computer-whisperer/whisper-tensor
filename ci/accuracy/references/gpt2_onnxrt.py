#!/usr/bin/env python3
"""Generate golden reference outputs for GPT-2 ONNX model using ONNX Runtime.

The in-repo model (test_models/gpt2-lm-head-10.onnx) is an ONNX export,
so the correct first-party reference is ONNX Runtime itself.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

# Allow importing common.py from the same directory
sys.path.insert(0, str(Path(__file__).parent))
from common import save_snapshot


def main():
    parser = argparse.ArgumentParser(description="Generate GPT-2 golden outputs")
    parser.add_argument(
        "--model",
        type=str,
        default="test_models/gpt2-lm-head-10.onnx",
        help="Path to GPT-2 ONNX model file",
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

    print(f"Loading model: {model_path}")
    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )

    # Discover input/output metadata from the model
    input_meta = session.get_inputs()
    output_meta = session.get_outputs()

    print("Model inputs:")
    for inp in input_meta:
        print(f"  {inp.name}: {inp.type} {inp.shape}")
    print("Model outputs:")
    for out in output_meta:
        print(f"  {out.name}: {out.type} {out.shape}")

    # Build concrete inputs: replace dynamic dims with small test values
    DYNAMIC_DIM_SIZE = 4
    inputs = {}
    for inp in input_meta:
        shape = []
        for dim in inp.shape:
            if isinstance(dim, int):
                shape.append(dim)
            else:
                shape.append(DYNAMIC_DIM_SIZE)

        onnx_type = inp.type
        if "int64" in onnx_type:
            # Token IDs: use small valid values (GPT-2 vocab is 0..50256)
            arr = np.random.RandomState(42).randint(0, 1000, size=shape).astype(np.int64)
        elif "int32" in onnx_type:
            arr = np.random.RandomState(42).randint(0, 1000, size=shape).astype(np.int32)
        elif "float16" in onnx_type:
            arr = np.random.RandomState(42).randn(*shape).astype(np.float16)
        else:
            arr = np.random.RandomState(42).randn(*shape).astype(np.float32)

        inputs[inp.name] = arr
        print(f"Input '{inp.name}': shape={arr.shape} dtype={arr.dtype}")

    # Run inference
    print("Running ONNX Runtime inference...")
    output_names = [out.name for out in output_meta]
    results = session.run(output_names, inputs)

    outputs = {}
    for meta, result in zip(output_meta, results):
        outputs[meta.name] = result
        print(f"Output '{meta.name}': shape={result.shape} dtype={result.dtype}")

    save_snapshot(args.output, inputs, outputs)


if __name__ == "__main__":
    main()
