# Bug: VulkanTensor::to_ndarray reads stale data during mid-computation downloads

## Status
Open — blocking Vulkan trig-to-NDArray dispatch (and by extension, fixing ONNX Mish precision).

## Symptoms
When `NumericTensor::trig()` is forced to NDArray (bypassing Vulkan), LSTM
Vulkan tests produce near-zero values (`-1.3e-23`) instead of correct results
(`0.095`). The computation chain is:

```
gates = matmul(x, w) + matmul(h, r) + bias   [Vulkan]
gc = gates.slice([0..batch, 3*hs..4*hs])       [Vulkan view]
c_cand = gc.trig(Tanh, backend)                 [downloads gc, computes on NDArray]
```

The `gc.trig(Tanh, backend)` call with NDArray dispatch does:
1. `self.to_ndarray()` — downloads the sliced Vulkan tensor to CPU
2. `.trig(Tanh)` — computes tanh on NDArray

Step 1 produces near-zero values, suggesting the GPU compute (matmul/add)
hasn't completed when the download occurs.

## Evidence

- **NDArray LSTM tests pass** — computation is correct on CPU.
- **Vulkan LSTM tests pass when trig stays on Vulkan** — no mid-computation
  download occurs, so no sync issue.
- **Vulkan LSTM tests fail when trig is forced to NDArray** — the download
  reads stale (zero-initialized) data from the Vulkan buffer.
- **`lstm_with_initial_bias` passes** even with trig-to-NDArray — this variant
  may have different timing that avoids the race.
- **The from_bytes stride fix is correct** — verified by LSTM passing when
  trig stays on Vulkan (non-contiguous slice data is read correctly).

## Hypothesis

`VulkanTensor::to_ndarray()` issues a `copy_buffer` command and waits on its
fence. But this fence only synchronizes the copy, not the preceding compute
commands that produced the data. The `VulkanImmediateExecutor` should ensure
all prior commands complete before new ones execute, but there may be a gap:

1. The matmul/add that produces `gates` is submitted via the immediate executor
2. The `to_ndarray` copy is submitted as a **new** command buffer via
   `AutoCommandBufferBuilder::primary` directly (not through the executor)
3. The new command buffer may execute before the matmul/add completes

The `to_ndarray` copy at `tensor.rs:330-347` creates its own command buffer
and submits it via `vulkano::sync::now(device).then_execute(queue, cb)`. This
doesn't wait for prior work on the queue to finish — it starts a new
submission chain from "now" rather than from the executor's last future.

## Affected tests (when trig forced to NDArray)
- `vulkan_test_lstm_defaults`
- `vulkan_test_lstm_batchwise`
- `vulkan_test_lstm_with_peepholes`

## Blocked fix
SPIR-V `Tanh` precision is implementation-defined. On lavapipe, the accumulated
error through the Mish chain (`softplus → tanh → mul`) exceeds ONNX test
tolerance by ~0.05e-7. Forcing trig to NDArray (libm) fixes Mish but triggers
this sync bug. Once the sync issue is resolved, trig can safely dispatch to
NDArray.

## Currently affected ONNX tests (accepted failures)
- `vulkan_test_mish` (SPIR-V tanh precision, 2 tests)

## Files
- `src/backends/vulkan_backend/tensor.rs:330-383` — `to_ndarray` copy+fence
- `src/numeric_tensor.rs` — `trig()` dispatch
- `src/symbolic_graph/ops/lstm.rs:262` — `gc.trig(Tanh, backend)`
