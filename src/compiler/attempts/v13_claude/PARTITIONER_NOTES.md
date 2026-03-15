# Span-Based Partitioner: Design Notes

## The Goal

Split a NanoGraph into an `ExecutionPlan`: a sequence of phases separated
by barrier sync points. Each phase has N spans (one per lane). Each span
is a **self-contained NanoGraph** with declared inputs and outputs. Spans
in the same phase execute in parallel — they MUST NOT read each other's
outputs.

## Output Format

```rust
struct AtomMapping {
    main_base: AtomId,   // start in the main graph's atom space
    span_base: AtomId,   // start in the span graph's atom space
    count: u64,          // number of contiguous atoms
}
struct Span {
    graph: NanoGraph,             // self-contained sub-graph
    inputs: Vec<AtomMapping>,     // ranges read from shared buffer
    outputs: Vec<AtomMapping>,    // ranges written to shared buffer
}
struct Phase { spans: Vec<Span> }
struct ExecutionPlan { num_lanes: usize, phases: Vec<Phase> }
```

**Use ranges (AtomMapping), not per-atom pairs.** Previous attempts used
`Vec<(AtomId, AtomId)>` which caused OOM on GPT-2 (589K entries per weight
matrix × 8 lanes).

## GPT-2 NanoGraph Profile

- 45,921 groups, 8.1B atoms, 0 symbolic
- 21,801 Mul + 21,630 ReduceSum (matmuls), 2,082 Literal (weights)
- ~400 other compute (Add, Div, Sub, Exp, Tanh, Select, etc.)
- 12 Identity (dtype casts only), 1 IndirectLoad (embedding)
- Matmul structure: M Mul groups (StridedBroadcast, count=K*N) + M ReduceSum
  groups (reduce_count=K, reduce_stride=N)
- 10 transformer layers, each with ~7 matmuls in sequence
- Elementwise ops: ONE group per op, count=49,152 (batch×seq×hidden)
- Select groups: attention masks, count=3,072

## The Hard Problem

The partitioner must handle TWO simultaneous concerns:

### 1. Group splitting
Elementwise groups are monolithic (49,152 atoms). To balance across N lanes,
they must be split into sub-ranges. Each lane gets count/N atoms. This is
straightforward for Affine(stride=1) inputs but requires care for other
InputRef types.

ReduceSum groups with reduce_stride CANNOT be naively split — the stride
assumes a specific atom layout. The ReduceSum's count dimension (output
atoms) CAN be split, but each output atom's reduction accesses a RANGE of
input atoms determined by reduce_count × reduce_stride.

### 2. Cross-lane independence within a phase
After splitting, NO span in a phase may read atoms produced by another
span in the same phase. This means:

- If group A produces atoms that group B reads, and both are in the same
  phase, they MUST be on the same lane (in the same span).
- If group A is SPLIT across lanes, its atoms are produced by multiple spans.
  Any group B that reads the FULL range of A's output needs ALL spans'
  results — it can only execute AFTER a barrier.

**This is where every previous attempt has failed.** v2c's phase assignment
allows same-phase dependencies between groups, then splits them across lanes,
creating cross-lane reads. Patching individual violation patterns doesn't
work — the fundamental approach is wrong.

## What Previous Attempts Got Wrong

### v2c + spans_c approach (current best, 96 violations on GPT-2)
- v2c assigns phases, spans_c builds span NanoGraphs
- v2c's phase assignment was designed for within-lane sequential execution,
  not for independent parallel spans
- When v2c puts two dependent groups in the same phase (expecting same-lane
  ordering), and the span builder splits them across lanes, the dependency
  becomes a cross-lane violation
- Narrow pattern-matching fixes (detect specific InputRef patterns) always
  miss edge cases

### The root cause pattern
1. Group A (e.g., Select, count=3072) is classified as AllRows → split across lanes
2. Group B in the SAME phase reads the FULL output of A
3. B's span declares A's atoms as external inputs
4. But A's atoms in other lanes aren't available until the phase completes
5. Violation: B reads atoms not yet produced

### Correct approaches must do ONE of:
a. **Never split A** — keep it whole on one lane (hurts balance)
b. **Move B to a later phase** — barrier after A, then B reads full A output
c. **Duplicate A into each span** — each lane independently computes A (works
   for small groups like the 3072-atom Select, wasteful for large groups)
d. **Split B differently** — so each lane's slice of B only reads the
   corresponding lane's slice of A (only works if the dependency is
   lane-aligned)

The ideal partitioner considers all four options and picks the cheapest.

## Key Invariants

1. **Within a phase, spans are INDEPENDENT.** No cross-span reads.
2. **All inputs/outputs are declared as AtomMapping RANGES.** O(groups) not O(atoms).
3. **Each span's NanoGraph is independently valid.** All InputRefs resolve within the span or to declared inputs.
4. **Literal groups < 1024 atoms** may be duplicated into spans. Large Literals (weights) are external inputs.
5. **All original compute atoms appear in exactly one span.**

## Validation Checklist (must pass on GPT-2)

1. Every compute (non-Literal) atom assigned to exactly one span
2. Each span's inputs are covered by earlier phases' outputs OR Literal groups
3. Within each span, groups are in valid topological order
4. No cross-span same-phase reads (the hard one)
5. Plan runs in < 10 seconds on GPT-2
6. Reasonable balance (max_imbalance < 2x for matmul phases)

## Files

- `problem_shape.md` — execution model (lanes, phases, barriers)
- `nano_plan_spans_c.rs` — current best attempt (96 violations remain)
- `nano_plan_spans_a.rs` — creative approach (39s, 1580x imbalance)
- `nano_plan_spans_b.rs` — creative approach (>120s, broken)
- `nano_plan_spans_d.rs` — DAG slicing (95ms, 6281x imbalance, no splitting)
- `nano_plan_v2c.rs` — underlying phase planner (used by spans_c)
- `nano_codegen_v2.rs` — JIT codegen for execution plans
- `nano_execute.rs` — dtype-correct reference interpreter
- `nano_graph/pattern.rs` — NanoGraph, AtomGroup, InputRef
- `nano_graph/ops.rs` — ScalarOp variants
- `examples/nano_lower_test.rs` — GPT-2 diagnostic (SPAN_PLANNER env var, topology validation)
