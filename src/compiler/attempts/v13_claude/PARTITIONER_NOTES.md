# Lane+Barrier Partitioner: Design Notes

## Current Architecture (v2c + nano_codegen_v2)

The v2c planner + codegen pipeline works end-to-end on GPT-2 (45,921 groups).
Balance is perfect (1.0x) but there are cross-lane dependency violations (2,201)
when using multiple lanes. The core issue: groups that depend on each other get
split across lanes without barriers between them.

## Next Architecture: Emit Separate NanoGraphs

Instead of tracking sub-ranges of the original graph, the partitioner should
emit N distinct NanoGraphs — one per lane per phase (or one per span). Each
sub-graph is self-contained:

- Has its own groups, atom IDs, and InputRefs
- Declares explicit inputs (atoms it reads from external sources / other spans)
- Declares explicit outputs (atoms it produces that other spans will read)
- Can be independently validated for correctness
- Can duplicate small computations (like Gather index math) rather than sharing

### Why this is better

1. **No cross-lane dependency ambiguity.** If a span needs a value, it either
   computes it internally or declares it as an input. The "is this a cross-lane
   read?" question doesn't arise — everything is explicit.

2. **Allows recomputation.** If computing a Gather offset takes 3 Binary ops,
   each lane can independently compute those 3 ops. There's no need to share
   them or worry about which lane "owns" the computation. This is cheaper than
   a barrier sync for 3 ops.

3. **Validates naturally.** Each sub-graph can be validated independently:
   all inputs are declared, all outputs are produced, topological order is
   correct within the sub-graph. No need for cross-span validation.

4. **Clean codegen interface.** Each sub-graph compiles to one Cranelift
   function with a clear signature: `fn(inputs: &[f32], outputs: &mut [f32])`.

### Output format

```rust
struct ExecutionPlan {
    num_lanes: usize,
    phases: Vec<Phase>,
}
struct Phase {
    spans: Vec<Span>,  // one per lane (may be empty for idle lanes)
}
struct Span {
    /// Self-contained NanoGraph for this span's computation.
    graph: NanoGraph,
    /// Which atoms from the shared values buffer this span reads.
    inputs: Vec<(AtomId, u64)>,  // (source atom in main graph, local atom in span graph)
    /// Which atoms this span writes back to the shared values buffer.
    outputs: Vec<(u64, AtomId)>,  // (local atom in span graph, target atom in main graph)
}
```

### Partitioner algorithm

1. **Find barrier positions.** Identify points where the computation
   reconverges (matmul reductions that read the full prior output).

2. **For each phase, identify the work to split across lanes.**
   Matmul rows are the primary split axis. Elementwise ops split
   proportionally.

3. **For each lane's slice, emit a new NanoGraph:**
   - Copy the relevant groups (or sub-ranges of groups)
   - For atoms that come from other lanes or earlier phases: create
     Literal input groups in the sub-graph
   - For atoms needed by later phases: mark as outputs
   - For small dependency chains (like Gather index computation):
     DUPLICATE them into each lane's sub-graph rather than sharing

4. **Validate each sub-graph** independently.

## Lessons from Previous Attempts

### What worked
- StridedBroadcast for matmul compression (800x group reduction)
- Zero-cost view ops (Transpose, Split, Slice, Concat)
- ReduceSum with known reduce_count/reduce_stride (no SymDim)
- Modular InputRef for cyclic broadcasts
- compress_explicit() pattern detection
- Live-set analysis for finding phase boundaries
- Group splitting via LaneWork sub-ranges (perfect balance)

### What failed
- Acyclic kernel DAG model (too restrictive, kills parallelism)
- Contiguous topo range partitions (sequential only)
- Treating every ReduceSum as a barrier (too many phases)
- Monolithic elementwise groups (can't split across lanes)
- Sampling-based producer lookup (misses spanning dependencies)
- Cross-lane same-phase dependencies (the current bug)

### Key invariant
Within a phase, lanes are INDEPENDENT. No lane may read an atom
produced by another lane in the same phase. All cross-lane data
flows through barriers (the shared values buffer, read between phases).

## Files

- `problem_shape.md` — execution model (lanes, phases, barriers)
- `nano_plan_v2c.rs` — current best planner (v2c with group splitting)
- `nano_codegen_v2.rs` — codegen for execution plans
- `nano_part_creative.rs` — reference for chain extraction + clustering
- `nano_execute.rs` — dtype-correct reference interpreter
- `nano_graph/pattern.rs` — NanoGraph, AtomGroup, InputRef types
- `nano_graph/ops.rs` — ScalarOp variants
- `nano_graph/eval.rs` — trusted NanoGraph evaluator
