---
name: compiler_state_march_2026
description: Current state of the v13 compiler pipeline — what works, what's broken, and the specific bug blocking progress
type: project
---

## v13 Compiler Pipeline Status (March 15, 2026)

### What Works End-to-End (Proven Correct)

**NanoGraph IR + Lowering:**
- GPT-2 lowers completely: 0 unsupported ops, 45,921 groups, 8.1B atoms, 0.27s
- 6 InputRef types: Broadcast, Affine, StridedBroadcast, Modular, SymAffine, Explicit
- 8 ScalarOp types: Literal, Identity, Binary (16 ops), Unary (9 ops), Select, ReduceSum, ReduceMax, IndirectLoad
- Zero-cost view ops: Transpose (permuted strides), Split (non-contiguous strides), Slice (step-scaled strides), Concat (segmented views)
- ReduceSum/ReduceMax carry reduce_count + reduce_stride directly (no SymDim)
- Explicit InputRefs minimized: only 13 groups (86K entries) in GPT-2
- BF16/F16 rounding at store boundaries in codegen
- u64 AtomIds for model-scale graphs

**Small-Scale Execution:**
- single_relu.onnx: zero error through both interpreter and JIT
- 2-layer MLP: zero error through v4b span plan + NanoEval
- Cranelift JIT codegen works for individual spans (nano_codegen_v2)

### The Best Partitioner: v4b (nano_plan_v4b.rs)

**Partition quality on GPT-2 (proven):**
- 50 phases (fewest barriers of any attempt — roughly 1 per matmul boundary)
- 400 spans (8 lanes × 50 phases), 0 empty
- 1.0x perfect balance across lanes
- ZERO topology violations (validated: no cross-lane same-phase reads)
- 532ms construction time
- Self-contained span NanoGraphs with AtomMapping ranges

**Algorithm:**
1. Group-level dependency DAG (all InputRef types + ReduceSum stride + IndirectLoad)
2. Dependency classification: LaneAligned vs AllRows (determines if splitting is safe)
3. Phase assignment via depth + convergence detection
4. Lane assignment with cross-lane violation repair (duplication for small groups, phase bumping for large)
5. Span NanoGraph construction with transitive dependency closure

### The Blocking Bug

**Symptom:** When building span NanoGraphs for GPT-2, `remap_atom` panics:
```
InputRef remap failed: main-graph atom 290010615 not in span atom map
(Explicit[4] atom=290010615)
```

**Location:** Phase 0, all 8 lanes. The Select group (attention mask, count=3072) is duplicated into each span. Its Explicit InputRef references atom 290010615.

**The specific issue:** Atom 290010615 falls numerically within the Select group's OWN output range (base=290009607, count=3072). The span builder's `find_missing_external_atoms` safety net checks `atom_available()`, which returns true because `main_to_local` already maps this atom (from the duplicated Select group). But the Explicit InputRef is referencing an INPUT atom, not the Select's own output — the atom belongs to a different group (the Identity/attention mask) that happens to have been allocated adjacent atom IDs by the lowering.

**Root cause hypothesis:** The lowering allocated the Identity mask group and the Select group with overlapping or adjacent base_ids. When the Select was duplicated into the span, its output range was added to `main_to_local`. The Select's Explicit InputRef entries reference atoms from the Identity group, but because the Identity group has atom IDs within the same numerical range as the Select, `main_to_local.get()` returns the Select's own span-local atom instead of recognizing the need for a separate external input.

**Why this is tricky:** The `RangeAtomMap` maps ranges, not individual atoms. When two different main-graph groups have overlapping atom ranges (which shouldn't happen if IDs are globally unique), the range map can't distinguish them. This suggests either:
1. The atom IDs truly don't overlap and my hypothesis is wrong — need to verify
2. The remapping logic has a subtler bug in how it handles Explicit InputRefs

**What would help:** Print the actual group structure around atom 290010615 — which group contains it, what's the Select group's actual Explicit vec, and verify whether there's truly an atom ID collision.

### File Inventory

**Core (keep):**
- `src/nano_graph/pattern.rs` — NanoGraph, AtomGroup, InputRef (6 types)
- `src/nano_graph/ops.rs` — ScalarOp (8 types, reduce_count/reduce_stride)
- `src/nano_graph/lower.rs` — MilliOpGraph → NanoGraph lowering
- `src/nano_graph/eval.rs` — Trusted NanoGraph evaluator
- `src/compiler/problem_shape.md` — Execution model (lanes + phases + barriers)
- `src/compiler/attempts/v13_claude/PARTITIONER_NOTES.md` — Lessons learned

**Best partitioner:**
- `src/compiler/attempts/v13_claude/nano_plan_v4b.rs` — THE WINNER (50 phases, 1.0x, 0 violations, but span build bug)

**Other partitioners (reference/comparison):**
- `nano_plan_v4a.rs` — Edge classification (22 tests, OOM on GPT-2 span build)
- `nano_plan_v4c.rs` — Closure-first (15 tests, >300s on GPT-2)
- `nano_plan_v3c.rs` — v2c salvage (0 topo violations, 96 span input errors)
- `nano_plan_v3d.rs` — Build-verify (91ms, 55 violations)
- `nano_plan_spans_c.rs` — Earlier span attempt (2.1s, 0 violations, 96 span input errors)
- `nano_part_creative.rs` — Acyclic kernel partitioner (reference for chain extraction)

**Codegen:**
- `nano_codegen.rs` — Original single-function Cranelift JIT
- `nano_codegen_v2.rs` — Phase-based Cranelift JIT (compiles LaneWork sub-ranges)

**Execution/testing:**
- `nano_execute.rs` — dtype-correct reference interpreter (NumericScalar)
- `examples/nano_lower_test.rs` — GPT-2 diagnostic (SPAN_PLANNER + EVAL_SPANS env vars)

### Performance Constraints (Learned the Hard Way)

1. NEVER iterate individual atoms — O(groups) only. GPT-2 has 8.1B atoms.
2. NEVER store per-atom data structures (HashMap<AtomId>, Vec indexed by atom).
3. Use AtomMapping ranges for span I/O, not per-atom pairs.
4. Explicit InputRefs (13 groups, 86K entries) ARE iterable — that's their purpose.
5. Span construction must be O(span_groups × inputs_per_group).
6. Target: < 10 seconds total for GPT-2 partitioning + span construction.

### Next Steps

1. **Debug the Select/Explicit remap bug** — verify whether atom IDs actually overlap, or if the remapping logic has a different bug
2. **Once span construction works:** run span-by-span NanoEval on GPT-2 to validate correctness
3. **Build span-aware codegen** — compile each span's NanoGraph to Cranelift, execute phases with barriers
4. **Multi-threaded execution** — the payoff for all this partitioning work
5. **Reduce phase count** — 50 phases may still be improvable (merge lane-local chains)
