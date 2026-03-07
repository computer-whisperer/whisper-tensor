#!/usr/bin/env python3
"""Generate reference outputs for SD 1.5 components using onnxruntime.

Saves inputs and outputs as f32 .npy files (even for f16 models) so Rust
can load them with ndarray-npy without needing f16 support.
Original dtypes are preserved in filenames as _f16/_i32 suffixes.
"""

import numpy as np
import onnxruntime as ort
import os
import time

SD_BASE = "/mnt/secondary/neural_networks/stable-diffusion-1.5-onnx-fp16"
OUT_DIR = "/tmp/sd_reference"
os.makedirs(OUT_DIR, exist_ok=True)

def save(name, arr):
    """Save array as f32 npy, with dtype suffix in filename."""
    dtype_name = str(arr.dtype)
    path = os.path.join(OUT_DIR, f"{name}_{dtype_name}.npy")
    np.save(path, arr.astype(np.float32))
    print(f"  Saved {name}: dtype={arr.dtype}, shape={arr.shape}")

# ---- Text Encoder ----
print("=== Text Encoder ===")
sess = ort.InferenceSession(os.path.join(SD_BASE, "text_encoder", "model.onnx"))

input_ids = np.zeros((1, 77), dtype=np.int32)
input_ids[0, 0] = 49406
input_ids[0, 1] = 320
input_ids[0, 2] = 1125
input_ids[0, 3] = 539
input_ids[0, 4] = 320
input_ids[0, 5] = 2368
input_ids[0, 6] = 49407
save("text_encoder_input_ids", input_ids)

t = time.time()
outputs = sess.run(None, {"input_ids": input_ids})
print(f"  Inference: {time.time()-t:.2f}s")
for i, out in enumerate(sess.get_outputs()):
    save(f"text_encoder_{out.name}", outputs[i])

cond_hidden = outputs[0]

# Unconditional
uncond_ids = np.zeros((1, 77), dtype=np.int32)
uncond_ids[0, 0] = 49406
uncond_ids[0, 1] = 49407
save("text_encoder_uncond_input_ids", uncond_ids)
outputs_uncond = sess.run(None, {"input_ids": uncond_ids})
for i, out in enumerate(sess.get_outputs()):
    save(f"text_encoder_uncond_{out.name}", outputs_uncond[i])
print()

# ---- UNet (single step) ----
print("=== UNet (single step) ===")
sess = ort.InferenceSession(os.path.join(SD_BASE, "unet", "model.onnx"))

H, W = 8, 8
np.random.seed(42)
sample = np.random.randn(1, 4, H, W).astype(np.float16) * np.float16(0.1)
timestep = np.array([500], dtype=np.float16)
encoder_hidden_states = cond_hidden

save("unet_sample", sample)
save("unet_timestep", timestep)
save("unet_encoder_hidden_states", encoder_hidden_states)

t = time.time()
outputs = sess.run(None, {
    "sample": sample,
    "timestep": timestep,
    "encoder_hidden_states": encoder_hidden_states,
})
print(f"  Inference: {time.time()-t:.2f}s")
for i, out in enumerate(sess.get_outputs()):
    save(f"unet_{out.name}", outputs[i])
print()

# ---- VAE Decoder ----
print("=== VAE Decoder ===")
sess = ort.InferenceSession(os.path.join(SD_BASE, "vae_decoder", "model.onnx"))

np.random.seed(123)
latent_sample = np.random.randn(1, 4, H, W).astype(np.float16) * np.float16(0.5)
save("vae_decoder_latent_sample", latent_sample)

t = time.time()
outputs = sess.run(None, {"latent_sample": latent_sample})
print(f"  Inference: {time.time()-t:.2f}s")
for i, out in enumerate(sess.get_outputs()):
    save(f"vae_decoder_{out.name}", outputs[i])
print()

# ---- VAE Encoder ----
print("=== VAE Encoder ===")
sess = ort.InferenceSession(os.path.join(SD_BASE, "vae_encoder", "model.onnx"))

np.random.seed(456)
image = np.random.randn(1, 3, 64, 64).astype(np.float16) * np.float16(0.5)
save("vae_encoder_sample", image)

t = time.time()
outputs = sess.run(None, {"sample": image})
print(f"  Inference: {time.time()-t:.2f}s")
for i, out in enumerate(sess.get_outputs()):
    save(f"vae_encoder_{out.name}", outputs[i])
print()

print(f"All reference data saved to {OUT_DIR}/")
