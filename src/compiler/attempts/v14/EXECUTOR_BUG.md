# B Executor Correctness Bug — Investigation Notes

## The Problem

Partitioner B produces structurally valid span NanoGraphs that pass every
validation check. When evaluated individually with correct inputs, each
span produces bit-perfect results. But when the executor runs all 173
phases sequentially — feeding each phase's outputs into the next phase's
inputs via a shared value store — the final model outputs are wrong.

Nearly all output elements mismatch the milli-op reference. The error is
not small numerical drift — it's large (max_abs 5-18) across 85-98% of
elements.

## What's Been Proven

1. **Nano lowering is bit-perfect.** The working test (nano_graph_model_test)
   compares 3707 intermediate tensors between milli-eval and nano-eval,
   all with max_abs=0.0.

2. **Direct eval is bit-perfect.** Running `eval::eval()` on the full
   NanoGraph with all 13 GPT-2 output ranges produces max_abs=0.0 for
   every output.

3. **Span subgraphs are faithful.** `validate_spans()` checks every span
   group against the main graph (op kind, dtype, InputRefs, count) and
   verifies all InputRef source atoms are covered. 0 errors on GPT-2.

4. **Single-span eval is bit-perfect.** Phase 0's span, evaluated directly
   with the same inputs the executor provides, produces 0 mismatches.

5. **All span inputs resolve.** Every `span.inputs` AtomRange base maps
   to an entry in `span.graph.input_tensors()` via `find_input_idx`.

6. **No count mismatches (after fixes).** Store entry sizes match span
   input range counts for phases 0-5.

7. **No panics (after fixes).** `populate_from_tensor` clamped to buffer
   length. No "atom not found" panics.

## What's Been Fixed Along the Way

- **IndirectLoad table_base:** BFS missed input_tensor tables (only used
  find_group_idx, not find_input_idx). Embedding table wasn't declared
  as external input. Fixed.

- **External range recording:** BFS recorded full producer group count,
  but producing span might output a different count. Changed to
  InputRef source range analysis (compute exact [lo, hi] from InputRef
  parameters, then walk main-graph groups/inputs in that range).

- **needs_output for Literals:** Literal groups adjacent to compute groups
  get merged by merge_external_ranges. If the Literal wasn't output by
  its span, the merged range was partially empty. Fixed by allowing
  Literal groups with later-phase successors to be output.

- **populate_from_tensor overflow:** Clamped writes to buffer length.
  The executor's multi-entry store scan can pass tensors larger than the
  span's declared input count.

- **Multi-entry store scan:** Executor now iterates all store entries
  overlapping each span input range (not just exact base match). Handles
  cases where a merged input range spans multiple store entries.

- **Segmented output extraction:** Concat (present KV cache) tensors have
  atoms scattered across multiple segments. Comparison uses
  atom_id_for_element for correct reassembly.

## Remaining Hypotheses

### H1: merge_external_ranges creates ranges the store can't fully satisfy

Even after the multi-entry scan, the merged range [A, A+N) might span
atoms from groups that were never output. The merge combines adjacent
groups into one input_tensor, but some of those groups might be Literals
that were inlined into the PRODUCING span (not output). The consuming
span declares [A, A+N) as an input, but only [A, A+K) exists in the
store. The remaining [A+K, A+N) stays at the default 0.0.

Evidence: early runs showed count mismatches (expects 3072 but store has
768). After the Literal-output fix, count mismatches disappeared for
phases 0-5, but we didn't check all 173 phases.

Test: extend the count mismatch check to all phases, or disable merging
and use per-group input_tensor entries (requires handling RangeMap
overlap constraints).

### H2: Eval refcount frees a buffer that a Reduce stride still needs

The eval's refcount system (`remaining[]`) tracks how many downstream
groups need each group's buffer. When the count hits 0, the buffer is
freed (`group_buffers[pi] = None`). But Reduce ops access atoms via
stride across a RANGE of the input group — they don't declare each
strided atom as a separate dependency. If the producer's refcount hits
0 after the direct consumer is processed but before the Reduce stride
finishes reading, the buffer is freed too early.

Evidence: none directly. The direct eval works (same refcount logic),
so this would have to be a span-specific issue where the subset of
groups changes the refcount arithmetic.

Test: disable refcount freeing in the eval (set all remaining[] to
u32::MAX) and rerun. If values become correct, the refcount is the bug.

### H3: Store entry overwrite between lanes in the same phase

Phase outputs are collected into `phase_outputs: Vec<(AtomId, tensor)>`
across all lanes, then committed to the store after the phase completes.
If two lanes produce outputs at the same AtomId base (e.g., both output
a duplicated Literal group), the second write overwrites the first. The
diagnostic checked for duplicate bases in phase_outputs and found none —
but only for the first 5 phases.

Test: extend duplicate check to all phases.

### H4: The multi-entry store scan passes atoms at wrong offsets

When the scan finds a store entry at base=X+768 that overlaps an input
range starting at X, it passes `(AtomId(X+768), tensor)` to the eval.
The eval calls `find_input_idx(AtomId(X+768))` and gets offset=768
within the input_tensor. `populate_from_tensor(buf, 768, tensor)` fills
buf[768..768+n]. This should be correct — but if the input_tensor in the
span graph was registered with `insert_input_tensor_at(AtomId(X), ...)`,
the RangeMap has a range [X, X+count). `find_input_idx(X+768)` returns
(ti, 768). This IS correct.

Unless `insert_input_tensor_at` created an entry at a different base
than X (because of the `would_overlap` check or gap-filling logic in
span construction).

Test: add a diagnostic that verifies, for each multi-entry match, that
find_input_idx returns the expected offset.

### H5: Shape-only lowering produces a different group structure than eval expects

The scaffold passes full-data TensorInfo for user inputs but shape-only
for weights. This causes `infer_all` to propagate differently than the
full-data path. Some downstream ops might get different shapes or counts,
leading to NanoGraph groups with different counts than the milli-eval
produces.

Evidence: the direct eval (same NanoGraph) is bit-perfect, so the
NanoGraph itself is correct. This hypothesis is unlikely.

## Recommended Next Steps

1. **Phase-by-phase value comparison.** Run the full-graph eval ONCE
   to get all atom values (already takes 15 min). Store in a HashMap.
   Then run the executor phase by phase, after each phase compare a
   sample of store values against the reference HashMap. This pinpoints
   the first divergent phase.

2. **Disable eval refcount freeing.** Set `remaining[pi] = u32::MAX`
   to prevent any buffer freeing. If values become correct, the refcount
   logic has a span-specific bug.

3. **Log multi-entry scan offsets.** For the first few phases, print
   each (base, offset, count) tuple the scan produces and verify the
   offset matches expectations.

## Test Commands

```bash
# Shape-only lowering, skip direct eval, 1hr timeout:
LOWER_SHAPE_ONLY=1 SKIP_DIRECT=1 timeout 3600 \
  ./target/release/examples/v14_scaffold test_models/gpt2-lm-head-10.onnx

# Full-data lowering (NOT recommended — inlines 54K weight Literals):
SKIP_DIRECT=1 timeout 3600 \
  ./target/release/examples/v14_scaffold test_models/gpt2-lm-head-10.onnx

# Run working test (bit-perfect reference):
cargo run --release --example nano_graph_model_test -- test_models/gpt2-lm-head-10.onnx

# Run just partitioner structural tests:
RUN_STRUCTURAL=1 PARTITIONER=B timeout 60 \
  ./target/release/examples/v14_scaffold test_models/gpt2-lm-head-10.onnx
```
