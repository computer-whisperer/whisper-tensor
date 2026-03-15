---
name: compiler_state_march_2026
description: Current state of the v13 compiler pipeline and what needs doing next
type: project
---

## v13 Compiler Pipeline Status (March 15, 2026)

### What Works End-to-End
- MilliOpGraph → NanoGraph lowering (0.3s for GPT-2, zero unsupported ops)
- NanoGraph IR: 6 InputRef types, 8 ScalarOp types, u64 AtomIds
- Zero-cost view ops (Transpose, Split, Slice, Concat)
- GPT-2 lowers to 45.9K groups, 8.1B atoms
- Span-based planner (spans_c): 209 phases, 8 lanes, 1.0x balance, 4.7s
- Cranelift JIT codegen (nano_codegen_v2) for sub-range execution
- Validated correct on 2-layer MLP (small scale)

### What Needs Work Next
1. **Validate spans_c on GPT-2**: each span's NanoGraph needs independent verification
2. **Build span-aware codegen**: compile each span's NanoGraph, execute phases with barriers
3. **Fix GPT-2 numerical accuracy**: 1-lane JIT had errors (~1772 max_abs). Need to determine if this is plan ordering, codegen, or dtype handling
4. **spans_b still broken** (>120s on GPT-2)
5. **209 phases may be too many barriers** — room to merge lane-local chains

### Key Files
- `src/compiler/problem_shape.md` — execution model (lanes + phases + barriers)
- `src/compiler/attempts/v13_claude/PARTITIONER_NOTES.md` — lessons learned
- `src/compiler/attempts/v13_claude/nano_plan_spans_c.rs` — best planner (4.7s, 1.0x balance)
- `src/compiler/attempts/v13_claude/nano_codegen_v2.rs` — JIT for execution plans
- `src/nano_graph/lower.rs` — NanoGraph lowering (StridedBroadcast, Modular, zero-cost views)
- `examples/nano_lower_test.rs` — GPT-2 diagnostic (SPAN_PLANNER env var)

### Architecture Decisions
- Spans emit separate NanoGraphs (self-contained, independently validatable)
- AtomMapping ranges for I/O (not per-atom pairs — learned the hard way)
- Literals < 1024 atoms duplicated into spans; large ones are external inputs
- Lane pinning for cache affinity across phases
- Barriers align with matmul reduction boundaries
