# Model Roster

> **Honesty notice.** Most model implementations in this project were generated
> by LLM agents (Claude) with limited manual review and weak end-to-end testing.
> The graph builders produce ONNX-compatible graphs that *compile and pass shape
> inference*, but many have never been run against real weights to confirm
> numerically correct output.
>
> The **Verified** column tracks which models have been confirmed to produce
> correct results against reference implementations. Until a model is marked
> verified, treat its output as *plausible but unconfirmed*. We plan to build a
> per-model verification pipeline; until then, this roster keeps honest track of
> what we know.

## Text generation (LLMs)

| Model | Variants | Format | Verified | Notes |
|-------|----------|--------|:--------:|-------|
| GPT-2 | — | ONNX | **Yes** | Loaded via `OnnxLoader`. End-to-end correct on NDArray and Vulkan. Used as primary integration test. |
| RWKV-7 | Any size | PyTorch `.pth` | **Yes** | Recurrent (non-transformer) architecture. Per-layer state carried across timesteps. Bit-perfect against reference at nano-graph level. |
| Llama 3 | Any size | HF Transformers / GGUF | No | GQA, RoPE. Fused QKV in GGUF path. |
| Gemma | Any size | HF Transformers | No | Embedding scale (multiply by sqrt(hidden_size)). GeGLU FFN. |
| Gemma 2 | Any size | HF Transformers | No | Adds attention logit softcapping and final logit softcapping over Gemma 1. |
| Gemma 3 | Any size | HF Transformers | No | Mixed sliding/full attention per layer. Per-head Q/K norm. Dual RoPE configs (separate theta for sliding vs full). |
| Qwen 2 / 3 | Any size | HF Transformers / GGUF | No | High rope_theta (1M) for long context. Qwen 3 adds per-head QK norm. |
| Phi 3 | Any size | HF Transformers / GGUF | No | Fused QKV and gate-up projections (split via Slice). |
| DeepSeek V2 | Any size | HF Transformers | No | MLA (Multi-head Latent Attention) with LoRA-compressed Q/KV. Hybrid dense + MoE FFN layers. |

### LLM architecture notes

All transformer LLMs follow the same pattern: token embedding → N transformer blocks (norm → attention → norm → FFN) → final norm → logit projection. Differences are in attention variant (MHA/GQA/MLA), FFN variant (SwiGLU/GeGLU), positional encoding (RoPE params), and normalization (RMSNorm, with or without weight+1 offset). KV cache state pairs are detected automatically from weight names.

The GGUF loader (`GgufLoader`) handles Llama 3, Qwen 2/3, and Phi 3 via the `general.architecture` metadata field. Dequantization happens at load time.

The Transformers loader (`TransformersLoader`) dispatches to the appropriate graph builder based on `config.json` → `model_type`.

## Image generation

| Model | Variants | Format | Verified | Notes |
|-------|----------|--------|:--------:|-------|
| Stable Diffusion 1.5 | — | Single `.safetensors` | No | CLIP text encoder + UNet (320 base, 4 levels) + VAE decoder. |
| Stable Diffusion 2.x | — | Single `.safetensors` | No | OpenCLIP ViT-H/14 encoder. Per-channel attention head dim. |
| Stable Diffusion XL | — | Single `.safetensors` | No | Dual text encoders (CLIP ViT-L + OpenCLIP ViT-bigG). 3-level UNet with variable transformer depth [0, 2, 10]. ADM conditioning. |
| Stable Diffusion 3.5 | Small / Large | ONNX or HF Diffusers | No | Triple text encoding (CLIP + OpenCLIP + T5-XXL). DiT with joint attention. |
| Flux | Schnell / Dev | Single or multi-file `.safetensors` | No | 19 double-stream + 38 single-stream DiT blocks. 3D RoPE. 16 latent channels. T5 + CLIP pooled text. Schnell has no guidance; Dev has guidance embedding. |

### Image generation architecture notes

SD 1.5 / 2 / XL use the classic UNet denoiser architecture with cross-attention to text embeddings. SD 3.5 and Flux use Diffusion Transformers (DiT). All image models use Euler discrete scheduling except Flux which uses rectified flow.

Loaders decompose single-checkpoint formats (SD 1.5, SD 2, SDXL) into separate text-encoder, denoiser, and VAE graphs by matching weight name prefixes. Multi-component formats (SD 3.5 Diffusers, Flux multi-file) load each component from its own directory.

## Video generation

| Model | Variants | Format | Verified | Notes |
|-------|----------|--------|:--------:|-------|
| CogVideoX | 2B / 5B / 1.5-5B | HF Diffusers | No | T5-XXL text encoder. DiT with joint spatial-temporal attention. 3D VAE. DDIM v-prediction scheduler. 5B+ variants use RoPE. |
| HunyuanVideo | — | HF Diffusers | No | LLaMA 3 + CLIP text encoders. 20 dual-stream + 40 single-stream DiT blocks with spatial refiner. 3D RoPE with per-axis dims [16, 56, 56]. |
| LTX-Video | 2B / 13B | HF Diffusers | No | T5-XXL text encoder. Cross-attention DiT. Rectified flow scheduler. |
| Mochi 1 | Preview | HF Diffusers | No | T5-XXL + pooled text embeddings. Separate video/text FFN dimensions. Temporal + spatial patch expansion. |
| Wan 2.1 | 1.3B / 14B | HF Diffusers | No | UMT5-XXL text encoder. Frequency-based embeddings. RoPE. Rectified flow scheduler. |
| Allegro | 2.8B | HF Diffusers | No | T5-XXL text encoder. 3D RoPE (per T/H/W axes). Temporal + spatial VAE upsampling. Rectified flow scheduler. |

### Video generation architecture notes

All video models follow the DiT (Diffusion Transformer) pattern: text encoder(s) → DiT denoiser with timestep conditioning → 3D VAE decoder. They differ in attention structure (self-attention vs cross-attention vs dual-stream), positional encoding (RoPE variants, frequency embeddings), and scheduler type (DDIM v-prediction for CogVideoX, rectified flow for the rest).

Each loader expects a HuggingFace Diffusers directory layout with `transformer/`, `vae/`, `text_encoder/`, and `scheduler/` subdirectories containing safetensors and config JSON files. Variant detection is automatic from config parameters.

## Speech

| Model | Task | Format | Verified | Notes |
|-------|------|--------|:--------:|-------|
| Whisper | Speech-to-text | HF Transformers | No | Encoder: Conv1d front-end (stride-2 downsampling) + transformer. Decoder: autoregressive with KV cache + cross-attention. 80 mel bins, 16 kHz. Fixed-step decoding with forced prefix tokens. |
| Kokoro | Text-to-speech | ONNX + voice `.bin` | No | Phoneme-based synthesis. Pre-computed voice embeddings with style conditioning. Speed control. Multiple ONNX precision variants (fp32, fp16, quantized). |
| Piper VITS | Text-to-speech | ONNX + config JSON | No | Espeak-based phoneme conversion. Multi-speaker support via speaker ID. VITS vocoder architecture. |
| F5-TTS | Text-to-speech | 3-model ONNX pipeline | No | Non-autoregressive flow-matching TTS. 3 stages: preprocess → diffusion transformer (32 Euler ODE steps) → vocoder. Reference audio for acoustic style transfer. 24 kHz output. |

## Weight format support

| Format | Loader | Notes |
|--------|--------|-------|
| ONNX `.onnx` | `OnnxLoader` | Direct graph import. Optional tokenizer for LLM interface. |
| HF Transformers | `TransformersLoader` | `config.json` + `.safetensors`. Auto-detects architecture from `model_type`. |
| GGUF | `GgufLoader` | llama.cpp quantized format. Dequantized at load time. Auto-detects architecture from metadata. |
| PyTorch `.pth` | `Rwkv7Loader` | Currently only used for RWKV-7. |
| HF Diffusers | Various | Directory layout with per-component subdirectories. Used by all diffusion models. |
| Single `.safetensors` | `SD15Loader`, `SD2Loader`, `SDXLLoader` | Checkpoint decomposed into components by weight name prefix matching. |

An `AutoLoader` attempts format detection and delegates to the appropriate loader.
