#!/usr/bin/env python3
"""Generate golden reference outputs for RWKV-7 using the official PyTorch implementation.

Uses the reference forward pass from https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/rwkv_v7_demo.py
to validate whisper-tensor's .pth → ONNX import and eval pipeline.

The whisper-tensor ONNX graph operates in RNN mode (seq_len=1 with explicit state
inputs/outputs). We run the reference in GPT mode with seq_len=1 which is equivalent
when states are initialized to zeros.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).parent))
from common import save_snapshot

# ---------------------------------------------------------------------------
# Official RWKV-7 "Goose" x070 forward pass (from rwkv_v7_demo.py)
# CPU-only, no CUDA kernel, no JIT — pure PyTorch for reference accuracy.
# ---------------------------------------------------------------------------

HEAD_SIZE = 64  # fixed for all RWKV-7 models


def RWKV7_OP(r, w, k, v, a, b):
    """Non-CUDA WKV-7 operator (reference implementation)."""
    B, T, C = r.size()
    H = C // HEAD_SIZE
    N = HEAD_SIZE
    r = r.view(B, T, H, N).float()
    k = k.view(B, T, H, N).float()
    v = v.view(B, T, H, N).float()
    a = a.view(B, T, H, N).float()
    b = b.view(B, T, H, N).float()
    w = torch.exp(-torch.exp(w.view(B, T, H, N).float()))
    out = torch.zeros((B, T, H, N), device=r.device, dtype=torch.float)
    state = torch.zeros((B, H, N, N), device=r.device, dtype=torch.float)

    for t in range(T):
        kk = k[:, t, :].view(B, H, 1, N)
        rr = r[:, t, :].view(B, H, N, 1)
        vv = v[:, t, :].view(B, H, N, 1)
        aa = a[:, t, :].view(B, H, N, 1)
        bb = b[:, t, :].view(B, H, 1, N)
        state = state * w[:, t, :, None, :] + state @ aa @ bb + vv @ kk
        out[:, t, :] = (state @ rr).view(B, H, N)

    return out.view(B, T, C).float()


class RWKV_Tmix_x070(nn.Module):
    def __init__(self, n_embd, n_head, layer_id, d_decay_lora, d_aaa_lora, d_mv_lora, d_gate_lora):
        super().__init__()
        self.layer_id = layer_id
        self.head_size = HEAD_SIZE
        self.n_head = n_head
        C = n_embd
        H = n_head

        self.x_r = nn.Parameter(torch.empty(1, 1, C))
        self.x_w = nn.Parameter(torch.empty(1, 1, C))
        self.x_k = nn.Parameter(torch.empty(1, 1, C))
        self.x_v = nn.Parameter(torch.empty(1, 1, C))
        self.x_a = nn.Parameter(torch.empty(1, 1, C))
        self.x_g = nn.Parameter(torch.empty(1, 1, C))

        self.w0 = nn.Parameter(torch.empty(1, 1, C))
        self.w1 = nn.Parameter(torch.empty(C, d_decay_lora))
        self.w2 = nn.Parameter(torch.empty(d_decay_lora, C))

        self.a0 = nn.Parameter(torch.empty(1, 1, C))
        self.a1 = nn.Parameter(torch.empty(C, d_aaa_lora))
        self.a2 = nn.Parameter(torch.empty(d_aaa_lora, C))

        self.v0 = nn.Parameter(torch.empty(1, 1, C))
        self.v1 = nn.Parameter(torch.empty(C, d_mv_lora))
        self.v2 = nn.Parameter(torch.empty(d_mv_lora, C))

        self.g1 = nn.Parameter(torch.empty(C, d_gate_lora))
        self.g2 = nn.Parameter(torch.empty(d_gate_lora, C))

        self.k_k = nn.Parameter(torch.empty(1, 1, C))
        self.k_a = nn.Parameter(torch.empty(1, 1, C))
        self.r_k = nn.Parameter(torch.empty(H, HEAD_SIZE))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)
        self.ln_x = nn.GroupNorm(H, C, eps=64e-5)

    def forward(self, x, v_first):
        B, T, C = x.size()
        H = self.n_head
        xx = self.time_shift(x) - x

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(
                self.v0 + (xv @ self.v1) @ self.v2
            )
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)

        x = RWKV7_OP(r, w, k, v, -kk, kk * a)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + (
            (r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.r_k)
            .sum(dim=-1, keepdim=True)
            * v.view(B, T, H, -1)
        ).view(B, T, C)
        x = self.output(x * g)
        return x, v_first


class RWKV_CMix_x070(nn.Module):
    def __init__(self, n_embd, dim_ffn):
        super().__init__()
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.x_k = nn.Parameter(torch.empty(1, 1, n_embd))
        self.key = nn.Linear(n_embd, dim_ffn, bias=False)
        self.value = nn.Linear(dim_ffn, n_embd, bias=False)

    def forward(self, x):
        xx = self.time_shift(x) - x
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2
        return self.value(k)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, layer_id, d_decay_lora, d_aaa_lora, d_mv_lora, d_gate_lora, dim_ffn):
        super().__init__()
        self.layer_id = layer_id
        self.ln0 = nn.LayerNorm(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.att = RWKV_Tmix_x070(n_embd, n_head, layer_id, d_decay_lora, d_aaa_lora, d_mv_lora, d_gate_lora)
        self.ffn = RWKV_CMix_x070(n_embd, dim_ffn)

    def forward(self, x, v_first):
        if self.layer_id == 0:
            x = self.ln0(x)
        xx, v_first = self.att(self.ln1(x), v_first)
        x = x + xx
        x = x + self.ffn(self.ln2(x))
        return x, v_first


class RWKV7(nn.Module):
    def __init__(self, n_layer, n_embd, vocab_size, n_head,
                 d_decay_lora, d_aaa_lora, d_mv_lora, d_gate_lora):
        super().__init__()
        dim_ffn = n_embd * 4
        self.emb = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, i, d_decay_lora, d_aaa_lora, d_mv_lora, d_gate_lora, dim_ffn)
            for i in range(n_layer)
        ])
        self.ln_out = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx):
        x = self.emb(idx)
        v_first = torch.empty_like(x)
        for block in self.blocks:
            x, v_first = block(x, v_first)
        x = self.ln_out(x)
        x = self.head(x)
        return x


def detect_model_config(weights):
    """Auto-detect model dimensions from weight shapes."""
    n_embd = weights["emb.weight"].shape[1]
    vocab_size = weights["emb.weight"].shape[0]

    n_layer = 0
    for key in weights:
        if key.startswith("blocks."):
            layer_id = int(key.split(".")[1])
            n_layer = max(n_layer, layer_id + 1)

    r_k = weights["blocks.0.att.r_k"]
    n_head = r_k.shape[0]

    # Detect LoRA dimensions from weight shapes
    d_decay_lora = weights["blocks.0.att.w1"].shape[1]
    d_aaa_lora = weights["blocks.0.att.a1"].shape[1]
    d_gate_lora = weights["blocks.0.att.g1"].shape[1]

    # d_mv_lora only exists from layer 1 onward (layer 0 has no v0/v1/v2)
    if "blocks.1.att.v1" in weights:
        d_mv_lora = weights["blocks.1.att.v1"].shape[1]
    else:
        d_mv_lora = 32  # default

    return {
        "n_layer": n_layer,
        "n_embd": n_embd,
        "vocab_size": vocab_size,
        "n_head": n_head,
        "d_decay_lora": d_decay_lora,
        "d_aaa_lora": d_aaa_lora,
        "d_mv_lora": d_mv_lora,
        "d_gate_lora": d_gate_lora,
    }


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

    print(f"Loading weights: {model_path}")
    weights = torch.load(str(model_path), map_location="cpu", weights_only=True)

    config = detect_model_config(weights)
    print(f"Model config: {config}")

    # Build model and load weights
    with torch.no_grad():
        model = RWKV7(**config).float()
        model.load_state_dict(weights, strict=False)  # layer 0 has no v0/v1/v2
        model.eval()

        # Run with a single token (seq_len=1), matching whisper-tensor's RNN mode.
        # With seq_len=1 and zero states, GPT mode time_shift is equivalent to
        # RNN mode with zero previous state.
        #
        # Use token_id=1 (not 0, which could be special in some tokenizers)
        token_ids = [1]
        input_tensor = torch.tensor([token_ids], dtype=torch.long)  # [1, 1]

        print(f"Input tokens: {token_ids}")
        print(f"Input shape: {input_tensor.shape}")

        output = model.forward(input_tensor)
        print(f"Output shape: {output.shape}")  # [1, 1, vocab_size]

    # The whisper-tensor ONNX graph uses I32 for token input
    inputs = {
        "token_input": input_tensor.numpy().astype(np.int32),
    }

    outputs = {
        "output": output.numpy(),
    }

    # Note: we only compare the logit output, not the intermediate states.
    # The state tensors (time_mixer_x_out, channel_mixer_x_out, vk_state_out)
    # are internal to the graph and would need a modified reference to extract.
    # Matching logits validates the full forward pass.

    save_snapshot(args.output, inputs, outputs)


if __name__ == "__main__":
    main()
