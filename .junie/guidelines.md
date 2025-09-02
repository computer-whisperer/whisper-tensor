Whisper Tensor – Development Guidelines (project‑specific)

Audience: experienced Rust developers working on this repository. Focus is on repo‑specific build knobs, backends, tests, and debugging tips.

1) Build and configuration
- Toolchain: Rust 1.80+ (Edition 2024). The crate compiles on stable; resolver = 2.
- Workspace layout:
  - Root crate: whisper-tensor (core runtime, backends, IRs).
  - Workspace members: crates/whisper-tensor-import (model loaders to ONNX), crates/whisper-tensor-server (host), crates/whisper-tensor-webui (WASM UI).
  - Vendored ONNX protobufs live under libs/onnx/onnx.
- Protobuf/ONNX: build.rs invokes prost-build on libs/onnx/onnx/onnx.proto3. No external onnx/protobuf install is required; the repo contains the .proto and prost-build compiles it during cargo build.
- Features of the root crate (see Cargo.toml):
  - default = [tokenizers, rwkv-tokenizer, rand, http].
  - Optional backends: vulkan (via vulkano 0.35), candle, tch. Backends can be combined; execution has per‑op fallback to NDArray when a kernel is missing on a chosen backend.
  - Other: tokenizers (HuggingFace), rwkv-tokenizer, http.
- Platform: Linux is the primary/only target at the moment.
- Vulkan specifics (only if you need GPU paths or Vulkan examples/tests):
  - Ensure a Vulkan loader/driver is available (e.g., Mesa’s lavapipe for headless CI/dev or a native driver). On Debian/Ubuntu, mesa-vulkan-drivers and vulkan-tools packages are typically sufficient for CPU‑based validation.
  - Enable with cargo feature flag: `--features vulkan`.

Common build commands (verified)
- Build workspace: `cargo build`
- Build with Vulkan enabled: `cargo build --features vulkan`
- Build only the root crate: `cargo build -p whisper-tensor`

2) Testing
There are two major test harnesses in this repository:
- Lightweight numeric tensor tests: tests/numeric_tensor_testing.rs (+ tests/numeric_tensor_tests/*). These validate arithmetic, matmul, reshape, etc., across backends.
- ONNX conformance node tests: tests/onnx_testing.rs. These enumerate a large subset of official ONNX backend tests under libs/onnx/onnx/backend/test/data/node/**. NDArray always runs; Vulkan variants are included when the feature is enabled.

How to run tests (verified)
- Full test suite with default features (NDArray backend): `cargo test`
- Run a minimal subset (example we verified): `cargo test ndarray_test_add -- --nocapture`
  - Expected result: a few add tests execute from both harnesses; sample run produced 3/3 passing in onnx_testing plus the basic numeric tests for add.
- Enable Vulkan tests: `cargo test --features vulkan`
  - To run a single Vulkan ONNX node test: `cargo test --features vulkan vulkan_test_add -- --nocapture`
- Filter by module or name substring as usual with cargo test; the generated test names follow the pattern `<backend>_<onnx_case_name>` for ONNX tests, and `<backend>_<unit_case_name>` for numeric tensor tests. Backends: ndarray, vulkan, tch (if enabled).

Adding new tests
- Numeric tensor tests (recommended for unit‑level coverage):
  - Place new cases under tests/numeric_tensor_tests/ (see existing basic_arith.rs, basic_matmul.rs, reshape.rs). Export them and include in tests/numeric_tensor_testing.rs via the existing macro runner. Tests are backend‑agnostic: the harness runs them against specified backends.
- ONNX node tests (for conformance):
  - tests/onnx_testing.rs uses macros to enumerate a curated list of official node test directories under libs/onnx/onnx/backend/test/data/node/.
  - To extend coverage, add another `do_test!(..., test_<name>)` line inside the macro list (ensure the corresponding directory with .pb inputs/outputs exists under libs/onnx/onnx/backend/test/data/node/<name>/). Avoid large batches unless needed; the list is already extensive.
- Per‑crate/unit tests: You can also add `#[cfg(test)]` module tests within src/** files; these build with the crate and can use internal APIs.

Running examples
- Examples declared in Cargo.toml require features as annotated:
  - `vulkan_test`: `cargo run --example vulkan_test --features vulkan`
  - `super_graph_test_3`: `cargo run --example super_graph_test_3 --features vulkan`
  - `matmul_perf_test`: `cargo run --example matmul_perf_test --features vulkan`
- There are additional example files (e.g., super_graph_test.rs) that may not be listed with required-features. Consult each file for backend usage and input assets. For GPT‑2 demo (NDArray path), ensure gpt2-lm-head-10.onnx is present (it is checked in at repo root) and run `cargo run --example super_graph_test`.

3) Development notes and debugging tips
- NumericTensor accuracy workflow (tch control backend):
  - Purpose: the optional `tch` backend is used as a control reference to validate that our unit tests in `tests/numeric_tensor_tests/**` are themselves correct. We do not optimize for `tch`; we use it to cross-check semantics, especially for bf16.
  - Enabling: build with `--features tch`. Then the numeric tensor harness will auto-generate a `tch_*` variant for each unit test (see tests/numeric_tensor_testing.rs). Example: `cargo test --features tch tch_test_add_bf16 -- --nocapture`.
  - Recommended workflow to resolve accuracy issues:
    * Write/extend a backend-agnostic unit test under `tests/numeric_tensor_tests/` that captures the failing op pattern (keep sizes small; cover edge cases like NaNs, denormals, broadcast, negative strides if applicable). Use bf16/f16/f32 variants. 
    * Run on NDArray (reference) and `tch` to ensure both agree within a strict tolerance before touching Vulkan or other implementations: `cargo test ndarray_<case>` and `cargo test --features tch tch_<case>`.
    * Only after NDArray ⇔ tch agreement, run Vulkan: `cargo test --features vulkan vulkan_<case>`. If Vulkan disagrees, fix the Vulkan kernel or the cast/pack/unpack path; if both disagree with each other but agree with NDArray, update Vulkan. If NDArray disagrees with `tch`, re-check the test logic and ONNX semantics first (the root cause might be the unit test or conversion/casting rules).
    * Logging: `RUST_LOG=whisper_tensor=debug` for rich traces of dtype conversions, shape inference, and dispatch.
  - Tolerances: prefer exact equality for integer and bit-preserving paths; for fp:
    * f32: atol=1e-6..1e-5, rtol=1e-6..1e-5 depending on op.
    * f16/bf16: pick op-specific tolerances; typical bf16 atol≈2e-3..4e-3 (1 ulp at exponent range), f16 can be a bit tighter on some ops. Document any expanded tolerance in the test name or a comment.
  - Known gaps in ONNX node tests: bf16 coverage is sparse upstream; prioritize explicit bf16 unit tests here to lock semantics before enabling corresponding ONNX cases.
- Graph pipeline:
  - Ingest: ONNX only. whisper-tensor-import converts weights (e.g., .safetensors, .pth) to canonical ONNX graphs.
  - IRs: Symbolic Graph (preserves ONNX semantics, supports symbolic dims) → Milli‑Op Graph (lowered simple op set) → backend execution.
  - Backends: NDArray (reference), Vulkan (partial); unimplemented Vulkan ops automatically fall back to NDArray during a single execution.
- Logging and tracing:
  - The examples initialize tracing_subscriber. Use env filter via RUST_LOG, e.g., `RUST_LOG=info` or `RUST_LOG=whisper_tensor=debug` when running tests/examples to get structured logs. Example: `RUST_LOG=info cargo test ndarray_test_add -- --nocapture`.
- DTypes: bf16/f16/f32 are first‑class. Expect special‑case handling and tests around these. Beware platform differences for f16/bf16; use the NDArray reference to validate semantics when optimizing other backends.
- Protobuf regeneration: build.rs automatically compiles ONNX .proto via prost-build on each fresh build; no manual step is required.
- Vulkan executor:
  - Uses vulkano 0.35 with minimal defaults; some operations are implemented as SPIR‑V compute shaders and more are in progress. Unsupported ops are dispatched to NDArray transparently.
  - Headless validation can use lavapipe; any Vulkan loader/runtime that satisfies vulkano will work.
- Server/WebUI:
  - The server/webui crates are included as workspace members but not required for core runtime/testing. They are useful for graph exploration and debugging.

4) Quick recipes (copy/paste verified locally)
- All tests on CPU (NDArray): `cargo test`
- Single ONNX add test on CPU: `cargo test ndarray_test_add -- --nocapture`
- Vulkan tests (requires Vulkan runtime): `cargo test --features vulkan`
- Vulkan single test: `cargo test --features vulkan vulkan_test_add -- --nocapture`
- Run GPT‑2 Super Graph demo (NDArray): `cargo run --example super_graph_test`

5) Contribution hygiene
- Formatting/lints: standard rustfmt and clippy apply. Prefer small, targeted commits. Keep perf changes guarded by tests (numeric tensor and ONNX node tests).
- When touching Vulkan kernels, always add/enable a corresponding numeric tensor unit test to lock semantics, then enable the ONNX case if applicable.
- When changing dtype or shape semantics, run: `cargo test` (NDArray) and at least a small Vulkan subset if relevant.

Appendix: What we actually ran to validate these notes
- `cargo test ndarray_test_add -- --nocapture` succeeded with 3/3 passing ONNX add tests plus numeric tensor add tests on NDArray. This validates the filter guidance and the ONNX harness wiring on CPU.
