# Lane+Barrier Partitioner: Lessons Learned

## The Execution Model

See `problem_shape.md` for full details. Summary:
- **Lanes** = persistent threads pinned to cores with cache affinity
- **Phases** = work between barrier sync points
- **Barriers** = sync points where all lanes wait before proceeding
- Lane assignment IS the tiling decision
- Cyclic dependencies between lanes are fine (barriers manage sync)
- Within a phase, lanes must be independent

## Output Format

```rust
struct ExecutionPlan {
    num_lanes: usize,
    phases: Vec<Phase>,
}
struct Phase {
    lane_work: Vec<Vec<usize>>,  // lane_idx -> group indices
}
```

## GPT-2 NanoGraph Structure (45,921 groups, 8.1B atoms)

Op breakdown:
- 21,801 Mul (Binary) — matmul products (StridedBroadcast input, count=K*N)
- 21,630 ReduceSum — matmul contractions + LayerNorm mean/variance
- 2,082 Literal — weights (globally shared, not assigned to lanes)
- ~400 other (Add, Div, Sub, Pow, Sqrt, Exp, Tanh, Select, ReduceMax)
- 12 Identity — dtype casts

### Matmul structure
For C[M,N] = A[M,K] @ B[K,N] (e.g. M=64, K=768, N=768):
- M Mul groups, each count=K*N=589,824 atoms
- M ReduceSum groups, each count=N=768 atoms
- Mul groups have StridedBroadcast (repeat=N) + Affine inputs
- ReduceSum has reduce_count=K, reduce_stride=N
- The M row pairs (Mul, ReduceSum) are INDEPENDENT — perfect for lanes

### Elementwise structure
Between matmuls: LayerNorm, residual adds, activations.
- **ONE group per op** covering the entire output (e.g., count=49,152)
- These are NOT split into rows by the lowering
- A LayerNorm has ~4 groups: Sub, Pow, ReduceSum(mean), ReduceSum(var), etc.

### The sync structure
- Matmul N+1's inputs depend on ALL of matmul N's outputs (the reduce)
- LayerNorm ReduceSums (mean/variance) also read the full input — they're
  mini sync points within a layer, not just at matmul boundaries
- Between two sync points, all the row-level work is independent

## What Previous Attempts Got Wrong

### Round 1: Acyclic kernel partitioners (nano_part_*)
- Tried to produce acyclic kernel DAGs
- Either killed parallelism (contiguous topo ranges = sequential only)
  or produced cycles (interleaved ranges without checking)
- The acyclic constraint was wrong — barriers manage sync, cycles are ok

### Round 2: Lane+barrier planners (nano_plan_*)
Three attempts, all finding barriers but failing at within-phase distribution:

**creative (147 phases, 1536x imbalance):**
- Over-eager barriers: put one at EVERY ReduceSum on critical path,
  including LayerNorm internal reduces. Created 49 tiny phases with
  only 4 groups each.
- The 4-group phases have one 49,152-atom group and one 64-atom group —
  impossible to balance across 8 lanes.

**critical (73 phases, 9826x imbalance):**
- Better barrier count (73 ≈ 1 per matmul) but still terrible balance
- Same root cause: phases with monolithic elementwise groups

**iterative (49 phases, 6686x imbalance):**
- Fewest phases, but still can't split the monolithic groups

### Root cause analysis

TWO problems compound:

1. **AtomGroups are treated as indivisible.** Elementwise ops produce ONE
   group of 49,152 atoms. The planners can't split this across lanes.
   But AtomGroups are compression artifacts — the problem_shape.md says
   the partitioner SHOULD be able to split groups.

2. **Barrier placement doesn't distinguish matmul sync from internal sync.**
   LayerNorm's ReduceSum (mean/variance) reads the full tensor but is
   NOT a cross-lane sync point in the same way a matmul output is. If
   lanes each have their row slice of the input, each lane can compute
   its own slice's contribution to the mean, then sync to combine.

   Actually: ReduceSum with reduce_count=768 reading from a 49,152-atom
   elementwise group IS a full-width reduce — it needs all 768 elements
   per position to compute the mean. But those 768 elements are within
   one "position" (one batch*seq element), not across lanes. If the
   49,152-atom group were split into 64 groups of 768, each lane could
   handle its positions independently, and the ReduceSum would be local.

## What the Next Attempt Should Do

### Group splitting

The partitioner MUST be able to split AtomGroups. For a group with
count=N and Affine(stride=1) inputs, splitting into K sub-groups of
count=N/K each is straightforward:
- Sub-group j: base_id = original.base_id + j*(N/K), count = N/K
- InputRef::Affine { base, stride } → Affine { base: base + j*(N/K)*stride, stride }
- InputRef::Broadcast(id) → still Broadcast(id)
- InputRef::StridedBroadcast { base, stride, repeat } → need to adjust
  based on which sub-range of atoms we're taking

For ReduceSum with reduce_count/reduce_stride: splitting along the
output dimension (count) is safe — each output atom's reduce is
independent. Sub-group j gets output atoms [j*chunk..(j+1)*chunk].

### Barrier placement

Barriers should be placed where the computation STRUCTURALLY requires
all lanes' results — not at every ReduceSum. The structural requirement:
a downstream op reads atoms produced by MULTIPLE lanes.

Specifically: if a ReduceSum reads from a Mul group that's on ONE lane,
no barrier needed — the reduce is lane-local. If a Binary::Add reads
from TWO groups that are on DIFFERENT lanes, a barrier is needed before
the Add.

The barrier decision should be AFTER lane assignment, not before. Or
iterative: tentatively assign to lanes, find where cross-lane reads
happen, place barriers there, re-assign.

### Cache affinity

Lane 0 should consistently get the same "row slice" across all phases.
This keeps data hot in L1/L2 across the entire model execution.

## Files to reference

- `src/compiler/problem_shape.md` — execution model
- `src/nano_graph/pattern.rs` — AtomGroup, InputRef, NanoGraph
- `src/nano_graph/ops.rs` — ScalarOp (especially ReduceSum fields)
- `src/compiler/attempts/v13_claude/nano_plan_creative.rs` — best attempt (147 phases, 1536x imbalance)
- `src/compiler/attempts/v13_claude/nano_plan_critical.rs` — critical path approach
- `src/compiler/attempts/v13_claude/nano_plan_iterative.rs` — iterative refinement approach
- `examples/nano_lower_test.rs` — GPT-2 diagnostic test
