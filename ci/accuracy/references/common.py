"""Shared utilities for reference output generation.

Each reference script produces a golden snapshot directory containing:
  - manifest.json: metadata for all inputs and outputs
  - *.npy: individual tensor files in NumPy format
"""

import json
import os
from pathlib import Path

import numpy as np


def save_snapshot(output_dir: str, inputs: dict, outputs: dict):
    """Save golden inputs and outputs as .npy files with a manifest.

    Args:
        output_dir: Directory to write snapshot files into.
        inputs: Dict mapping tensor name -> numpy array.
        outputs: Dict mapping tensor name -> numpy array.
    """
    os.makedirs(output_dir, exist_ok=True)

    manifest = {"inputs": [], "outputs": []}

    for i, (name, arr) in enumerate(inputs.items()):
        filename = f"input_{i}.npy"
        np.save(os.path.join(output_dir, filename), arr)
        manifest["inputs"].append({
            "name": name,
            "file": filename,
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
        })

    for i, (name, arr) in enumerate(outputs.items()):
        filename = f"output_{i}.npy"
        np.save(os.path.join(output_dir, filename), arr)
        manifest["outputs"].append({
            "name": name,
            "file": filename,
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
        })

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved {len(inputs)} inputs, {len(outputs)} outputs to {output_dir}")
