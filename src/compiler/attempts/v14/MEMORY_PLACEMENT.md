# Memory Placement (working design)

> Status: design in progress. This document is a living working draft
> of the buffer-placement rework opened on 2026-04-10. Items listed
> under "Open questions" are unresolved and will be updated as we
> work through them.

## Motivation

The v14 executor is memory-bandwidth bound at runtime: cores spend
most of their time waiting on loads that miss to RAM. The root cause
is not that we do too much computation, nor even that we memcpy too
much — it is that **the current architecture makes cache residency
impossible**.

Today a cross-span value is produced at address A in some span's
private working buffer, memcpy'd out to address B in a `NumericTensor`
held by the `PhaseStore`, memcpy'd in to address C in the next span's
private working buffer, and finally loaded from C by the consumer's
JIT body. Four different physical addresses in four different
allocations, all carrying bit-identical data. Each hop forces the
line through main memory: the writing core's copy at A sits in L1/L2
in `Modified` state, gets read exactly once (by the memcpy that
copies it to B), and is then dead — the hardware eventually evicts
it. The next hop's destination address has never been in any cache
so the store allocates a fresh line; the one after that misses on
the read. Repeat for every cross-span atom, every layer, every
token. The working set cycles through RAM even when it would fit
in L3.

The fix is not to count memcpys — it is to **put the value in one
place and leave it there**. Producer writes to address X in a shared
buffer. The writing core's L1/L2 holds it in `Modified` state. At
the phase barrier, all cores synchronize. The consumer span on a
(possibly different) core reads address X; the coherence protocol
migrates the line into the reader's L1 via the on-chip interconnect,
never touching RAM. If a third span on another core reads X while
it is still hot, the line transitions to `Shared` and serves all
readers from their local caches. If X's lifetime ends and the same
offset is reused for atom Y, the first write of Y invalidates X's
stale copies in any cache that still holds them — exactly the
behavior the hardware is designed for.

This is what x86's cache system does for any ordinary shared buffer.
We do not need to invent a data-sharing mechanism — we need to
**stop breaking the one the hardware already provides**. The
lane/span partitioner already guarantees no intra-phase cross-lane
data hazards, so the only invariant the hardware still needs from
us is "don't false-share cache lines," which is a placement
constraint rather than an execution one.

Secondary wins (eliminating per-execute allocations, `literal_template`
cloning, `PhaseStore` bookkeeping, and the marshal-in/marshal-out
memcpys) fall out for free once the layout is fixed, but they are
secondary. The primary goal is **cache residency of cross-span
values from producer to last consumer**.

## High-level design

```
NanoGraph
   │
   ▼
partitioner_m  ──→  Vec<Phase{spans:[Span]}>      (unchanged shape;
   │                                               adds split-alignment
   │                                               hints)
   ▼
global placer  ──→  AtomPlacementMap              (NEW)
   │                {atom → (buffer_id, byte_offset)}
   │                + buffer sizes
   │                + per-lane scratch high-water marks
   ▼
per-span codegen  ──→  CompiledSpan               (uses placement map
   │                                                for cross-span atoms;
   │                                                allocates own scratch
   │                                                slots locally)
   ▼
executor        ──→  reuses pre-allocated buffers, no PhaseStore
```

The placer is the only new pass. Codegen and the executor change to
match its outputs.

## Buffers

Four kinds of buffers, all addressed by an integer `buffer_id` known at
compile time:

| kind          | count            | lifetime   | source                            |
|---------------|------------------|------------|-----------------------------------|
| Input         | one per input    | per-call   | caller-owned (often borrowed)     |
| Output        | one per output   | per-call   | pool-allocated, returned          |
| Intermediate  | one (shared)     | per-plan   | pool-allocated at build time      |
| Scratch       | one per lane     | per-plan   | pool-allocated at build time      |

- **Input buffers** are pass-through: the executor takes whatever the
  caller hands in (a `NumericTensorCOW`, possibly borrowed) and exposes
  its byte pointer to the JIT. Input tensors are **assumed to arrive
  in the layout the placer planned for** — primarily weight tensors,
  which dominate by volume. Nano-op lowering is responsible for
  emitting a graph that matches whatever layout the weights actually
  live in; if execute-time ever has to re-layout a weight, that's a
  lowering bug upstream, not a placer concern. Dynamic user inputs
  (activations, tokens) are handled by the existing `relayout_to_flat`
  path, which preserves borrows whenever the layout already matches.
- **Output buffers** are allocated by the executor before each execute
  call (using the pool) and returned to the caller via the same handle
  the JIT wrote into — no copy at extract time.
- **Intermediate buffer** is the shared cross-span data store. Holds
  every atom that needs to flow from one span to another (different
  `(phase, lane)` tuples). Sized to the **peak live cross-span data**
  via liveness-interval allocation.
- **Scratch buffers**: one per lane. Holds atoms that are produced and
  consumed within a single span. Sized to the maximum scratch demand
  of any span that lane will run.

## Atom classification

Each atom in the lowered NanoGraph falls into exactly one of three
categories:

1. **Cross-span**: produced in one span, consumed in another (where
   "another" means a different `(phase, lane)`). Lives in the
   intermediate buffer (or in an input/output buffer if it is a model
   input/output). Has a `(buffer_id, byte_offset)` in the global
   placement map.
2. **Span-local, scratch**: produced and consumed within one span.
   Multi-use, can't be inlined. Lives in the lane's scratch arena at
   an offset chosen by the per-span codegen.
3. **Span-local, inlinable**: produced and consumed within one span,
   single consumer, expression can be folded into the consumer's
   loop body. No slot anywhere — value lives in registers. Existing
   `inlinable` mechanism in `compute_layout` handles this unchanged.

The classification is done by the global placer (which knows which
atoms cross spans) plus the per-span codegen (which decides scratch
vs inline for the rest).

### Why split scratch out

The current per-span working buffer holds *every* atom — including
the giant K×N matmul Mul intermediate that gets reduced to N within
the same span. Putting that in shared memory would be wasteful: it
never escapes the span, lanes never need to share it, and
post-rewrite it would dominate the intermediate buffer's size for
no benefit.

Span-local scratch keeps these atoms private to one lane. Each lane's
scratch arena is sized once at plan-build time to fit the worst-case
span on that lane and is reused on every execute call. There is no
per-execute alloc on the hot path.

## The global placer

### Inputs

- The full `Vec<Phase>` from the partitioner.
- The set of model inputs (already mapped via TAMI).
- The set of model output atom ranges (already pinned).
- Per-InputRef bounds (see "InputRef bounds" below).

### Outputs

- `AtomPlacementMap`: `HashMap<AtomId, (BufferId, byte_offset)>` for
  every cross-span atom.
- `intermediate_buffer_size`: bytes.
- One `(BufferId, size)` per input and output buffer, with input
  buffer_ids assigned at placer time (nailed down deterministically
  from the tensor_map) so execute-time just dumps the right view
  pointers into the matching array slots.
- **Peak live footprint diagnostic**: the high-water mark the placer
  hit during interval packing. Reported alongside scratch sizes so
  regressions / runaway working sets are visible immediately.
- A list of all coalesced slabs that crossed `>= 1` buffers, if any
  remain after lowering hardening (these would be a configuration
  error — see Open Questions).

### Algorithm sketch

1. **Identify cross-span atoms.** Walk every span. An atom is
   cross-span if it appears in any span's `outputs` list (the
   partitioner already declares these). Model inputs and outputs are
   trivially cross-span and assigned to their dedicated input/output
   buffers up front.

2. **Compute liveness intervals.** For each cross-span atom (or slab,
   see step 3), compute `[first_produce_phase, last_consume_phase]`
   where:
   - `first_produce_phase` is the lowest phase index in which any
     span declares the atom as an output.
   - `last_consume_phase` is the highest phase index in which any
     span declares the atom as an input. Model outputs are pinned to
     `phases.len() - 1` so they survive to extraction.

3. **Identify coalescing constraints.**

   *What slab coalescing is.* The JIT addresses strided reads with a
   single linear formula `base_bit + i * bit_stride`. That formula
   can only walk a flat, contiguous byte range. So if an InputRef's
   access range covers atoms that come from multiple producer groups
   (or from a producer group plus an adjacent input tensor), those
   atoms **must** live at contiguous, stride-compatible byte offsets
   in memory — otherwise the consumer's `[base + i*stride]` load
   lands at the wrong address at some `i`. The group of atoms forced
   to be co-located by such a constraint is a **slab**.

   Real examples: a Pad op produces `literal_zeros | identity_copy |
   literal_zeros` and the consumer reads all three via a single
   `stride=1` InputRef — all three groups must be placed contiguous
   and in that order. The matmul Mul→Reduce pair (see
   `relayout_matmul_groups`) is the other big source today, but
   those are span-local and handled in scratch, not here.

   *The pass.* For each span, walk every cross-span InputRef. If its
   `(min_atom, max_atom)` bound spans multiple cross-span atoms (or
   one cross-span atom plus an input-buffer atom), union them.
   Connected components are slabs. Union-find as in today's
   `compute_layout::Step 1`, but (a) at the global level and (b)
   restricted to cross-span atoms — most coalescing today is
   span-local (matmul Mul→Reduce) and stays in per-span scratch
   unchanged.

4. **Per-slab liveness.** A slab's interval is the *union* of its
   members' intervals. A slab can only be reused after every member
   is dead. Heavy coalescing inflates the intermediate buffer because
   a slab containing one long-lived atom pins space for every other
   member of the slab until that atom dies.

5. **Pack intervals into the intermediate buffer.** Linear-scan or
   first-fit allocator over slab-intervals, sorted by start phase.
   Free a slab's offset once `current_phase > slab.end`. Honor:
   - Cache-line alignment for slabs that will be split-written by
     multiple lanes (see "Cache-line discipline" below).
   - Slab elem-bytes alignment (so stride-based InputRefs work).

6. **Detect cross-buffer slab violations.** If a slab's coalescing
   constraint pulls in atoms from different buffer kinds (e.g., a
   weight tensor and an intermediate atom), that's currently a hard
   error — the lowering must be hardened to never produce such
   InputRefs. See Open Questions.

7. **Emit the placement map.**

### Split groups

A split group's atoms keep their parent's atom IDs (partitioner_m
preserves IDs across fragment boundaries). From the placer's point
of view, the parent atom range has a single global offset in the
intermediate buffer; lane k's fragment writes the sub-range
`[base + lane_lo .. base + lane_hi)` at the corresponding sub-offset.
The placer does not need per-lane offsets — lane k's compiled span
knows its fragment's `(atom_offset, count)` from the partitioner and
computes the byte address via `parent_base_offset + atom_offset *
elem_bytes`.

Cross-span liveness for a split parent spans from the first phase
in which any lane writes a fragment of it to the last phase in
which any lane reads a fragment — the parent is tracked as a whole,
not per-fragment.

### Duplicate groups with cross-span consumers

A duplicate group's atoms are produced by every lane within a
phase. When those atoms need to flow to a later phase, only **lane
0** actually stores to the intermediate buffer slot. Other lanes
still compute the value into their own scratch (they need it
inside their own span for local consumers) but emit no store to
the shared slot. This eliminates the write-write race on the
shared bytes.

The per-span codegen needs to know "am I the canonical writer for
this cross-span output?" The placer answers this by tagging each
cross-span output declaration with a `canonical_writer_lane: u8`
and the codegen skips store emission on non-canonical lanes.

### Properties

- **Reuse**: the buffer's high-water mark is bounded by the *peak
  live cross-span footprint*, not the total. Activations flowing
  through a deep network with short-lived chains pack tightly.
- **Determinism**: same NanoGraph + same partition → same placement.
- **Compile cost**: O(spans × atoms_per_span) for liveness, O(slabs²)
  worst case for first-fit — comparable to today's per-span
  `compute_layout` summed over spans.

## Per-span codegen changes

Each span's codegen still calls `compute_layout` (or a renamed
successor), but with two changes:

1. It is given the global `AtomPlacementMap`. For any atom it
   encounters that is in the map, it skips its own slot allocation
   and uses the map's `(buffer_id, byte_offset)` directly.
2. For atoms not in the map (span-local), the existing FreeList
   allocator runs as today, producing scratch offsets. The total
   scratch high-water mark for the span is recorded as the span's
   `scratch_bytes` requirement.

The existing `inlinable` machinery is untouched — it operates on
span-local atoms only, since cross-span atoms by definition have
multiple consumers (the spans that read them) and aren't inlinable.

The per-span layout is no longer responsible for slab coalescing
across cross-span atoms — the global placer has already done that.
It still does coalescing on span-local atoms (which is the common
case for matmul Mul intermediates).

**Compile pipeline order**: the global placer runs once,
sequentially, over the full `Vec<Phase>`; then per-span codegen
runs in parallel (`rayon::par_iter` over spans) with each worker
holding an immutable borrow of the finished `AtomPlacementMap`.
The placer is the only sequential step added to compile; the
existing parallel per-span compile path is preserved.

### Unified slot lookup

`address.rs` currently calls `layout.find(atom_id) -> SlotInfo` to
resolve every atom access. Under the new design this lookup queries
the global placement map first (returning a `(buffer_id, bit_offset,
bit_stride, dtype)` tuple) and falls back to the per-span scratch
layout for atoms not in the map. `SlotInfo` gains a `buffer_id`
field; the address-emit functions take a "base-register-for-this-
buffer" argument they pull from the prologue's loaded bases.

This keeps address-computation code paths unified — the orch layer
does not branch on "cross-span vs scratch," it just asks the lookup
where the atom lives and then emits the same load/store sequence
with whichever base register the lookup named.

## JIT ABI

```c
extern "C" void compiled_span(const void * const * buffer_ptrs);
```

- Single argument in `rdi`: a pointer to an array of `void *`.
- The JIT bakes `buffer_ptrs[i]` accesses into its prologue: load
  each buffer base it cares about into a stack slot or callee-saved
  register.
- Every atom access at codegen time knows
  `(buffer_id, byte_offset, dtype, bit_stride)`. Codegen emits:
  - Load buffer base from its assigned register/stack slot.
  - Add the immediate `byte_offset` (or compute via bit/stride math
    for runtime-iterated accesses).
  - Bit-aware codec read or write.
- Number of buffers a span actually touches is small (intermediate,
  scratch, plus a handful of inputs/outputs). Common case fits in 4-6
  base registers.

The buffer-id-to-slot-in-rdi-array mapping is part of the
`AtomPlacementMap` — known at codegen time, no runtime lookup.

## Executor changes

Plan-build time:
- Receive the placement map and per-span scratch sizes from the
  compiler.
- Allocate the intermediate buffer (one big pool allocation).
- Allocate per-lane scratch arenas (sized to
  `max(span.scratch_bytes for span in this lane's spans)`).
- Write literal values into their (non-reusable) slots in the
  intermediate buffer and the lane scratch arenas.

Per-execute hot path:
- Allocate output buffers via the pool (one per output, sized from
  the placement map).
- Build a small `[*mut u8; N]` buffer-pointer-array **on the
  executor's stack frame** for this execute call. Fixed-address
  slots (intermediate + scratch) come from the persistent
  allocations; input slots get filled with the caller's view
  pointers; output slots get filled with the freshly-allocated
  output buffer pointers. The array is a handful of pointers —
  stack-only, no heap involvement on the hot path.
- For each phase: rayon-broadcast the lanes, each calling its
  compiled-span fn with a pointer to the stack array. (Per-lane
  variance is limited to the scratch slot, since each lane has its
  own scratch arena — either a per-lane copy of the array or a
  shared array with per-lane scratch patched in before dispatch.)
- After all phases: hand the output buffers back to the caller.
  No extract-copy step — output tensors *are* the output buffers,
  wrapped in `NumericTensor`s with the right layout metadata.

`PhaseStore`, `insert_batch`, `gather`, `evict`, `output_liveness`,
`pinned_output_atoms`: all deleted.

### Buffers are not wiped between executes

The intermediate buffer and scratch arenas are left in whatever state
the previous execute call left them in. Correctness is guaranteed by
a simple invariant: **every atom is written by its producer before
any consumer reads it**, which holds by construction in a dataflow
graph. No zero-ing, no scrubbing, no reset step. Literals are the
only values expected to survive across executes, and they live in
non-reusable slots the JIT never writes to after plan-build.

The JIT is responsible for respecting the producer-before-reader
invariant — i.e., for never emitting a load of an atom whose
producer hasn't run yet in the current execute. This is already
implicit in how codegen walks the NanoGraph in topological order.

## Cache-line discipline

The intermediate buffer is read by all cores and written by their
respective lanes. False sharing happens when two lanes' write ranges
share a cache line, even if the atoms themselves are disjoint at the
atom-id level.

Two tiers of enforcement:

1. **Partitioner** ensures every Split group's per-lane fragment is a
   multiple of `cache_line_bytes / atom.bytes_per_element`. For
   typical 64B lines and f32 atoms, that's 16-element minimum
   per-lane chunks. Splits below this minimum get demoted to Whole.
2. **Placer** rounds every slab whose members are split-written to a
   64B-aligned start, so the per-lane chunk boundaries fall on cache
   line boundaries.

Inter-phase writes (producer in phase k, reader in phase k+1) carry no
false-sharing risk because the barrier between phases ensures all
caches are settled before the next phase starts. The only concern is
intra-phase, and intra-phase cross-lane data flow is already forbidden
by the partitioner — so the only remaining risk is the sub-cache-line
boundary case at split-fragment edges, handled by the alignment rules
above.

Duplicate groups whose atoms cross spans get a single canonical
writer (lane 0); other lanes still compute the value into their
local scratch (because they need it inside their own span) but do
not store to the shared slot. No write-write race.

## Literals

Today literals are written into the per-span `literal_template` at
compile time and the template is cloned per execute. Under the new
design there is no per-span template, so literal values need a home
somewhere that the JIT can read from.

Two categories:

1. **Span-local literals**: produced and consumed inside one span.
   The per-span codegen assigns them a scratch offset and writes
   their values into the scratch arena **once at plan-build time**.
   The scratch arena is persistent across executes, so the literal
   stays valid as long as the plan lives — no per-execute rewrite.
   If the per-span FreeList ever reuses that scratch offset (because
   the literal's lifetime ended), the producer of the reusing atom
   overwrites it, and subsequent executes would see garbage there
   next time that phase runs. To avoid this, **literal scratch
   offsets are marked non-reusable** — their bytes stay live for the
   plan's lifetime. Literals are almost always small, so the waste
   is negligible.

2. **Cross-span literals** (rare but possible — e.g., a padding
   literal consumed in a later phase): go in the intermediate buffer
   at an offset chosen by the global placer, also marked
   non-reusable for the same reason. Written once at plan-build time.

In both cases the placer/codegen distinguishes "literal-backed"
atoms from regular produced atoms and excludes them from liveness
reuse. The total non-reusable footprint is `sum(literal_bytes)`,
which is tiny relative to activation data.

`LiteralSpan` (large literal tensors embedded in the graph — weight
constants that were folded into the NanoGraph rather than kept as
model inputs) follows the same rules but could in principle be huge.
If this becomes a footprint issue, a follow-up would move them to
their own dedicated read-only buffer. Not worth solving up front.

## Access bounds

The placer needs a tight upper/lower atom ID per access in order to
(a) detect when a single access crosses buffer boundaries and (b)
build coalescing slabs. For most accesses this is cheap to compute
from the existing `InputRef` + `(count, atom_offset)` context — the
stride patterns are all closed-form and the placer can get exact
bounds from a single method call:

```rust
impl InputRef {
    /// Exact (min_atom, max_atom) for a consumer group of `count` atoms
    /// resolved at `atom_offset`. No stored state; computed on demand.
    pub fn access_bounds(&self, count: u64, atom_offset: u64) -> (AtomId, AtomId);
}
```

This replaces today's conservative `input_ref_range` in
`layout.rs:417` (which deliberately over-approximates for safety
margin). The placer calls `access_bounds` directly rather than
re-deriving bounds from the stride math.

### IndirectLoad needs an explicit range

`ScalarOp::IndirectLoad { table_base }` is the exception: its access
range is determined by a **runtime** index, not a compile-time
stride. Today the partitioner and placer have to guess by looking
at the full extent of the group or input tensor that `table_base`
points into (see `partitioner_m.rs:2131`), which is a conservative
default but requires a graph-wide lookup to find the table's size.

The fix is to add an explicit `index_range` field:

```rust
IndirectLoad {
    table_base: AtomId,
    /// The index input is bounded to `[0, index_range)`. The op
    /// reads atoms in `[table_base, table_base + index_range)`.
    index_range: u64,
}
```

Lowering populates `index_range` at construction time — both
`gather.rs` and `gather_elements.rs` already know the full table
size (`data_map.count`) at the point they build the IndirectLoad.
The placer then has a local `(table_base, index_range)` pair and
doesn't need to traverse graph state to find the bounds.

This is a small, surgical change — three construction sites plus
the enum definition — and is prerequisite for the placer's
cross-buffer detection on IndirectLoad ops.

## Testing and validation

The placer is isolatable — it takes `Vec<Phase>` + InputRef bounds
and emits an `AtomPlacementMap`. Unit tests can feed it synthetic
phase structures and check:

- Every cross-span atom has exactly one `(buffer_id, offset)`.
- No two atoms with overlapping live ranges occupy overlapping
  byte ranges.
- All slab members are contiguous and in atom-id order.
- No slab crosses a buffer boundary.
- Split-written slab starts are cache-line aligned.
- Peak live footprint matches hand-computed expectations on small
  graphs.

For integration validation, a diff-mode test runs the same model
through today's per-span layout and the new placer, and compares
executed outputs. Any divergence is a placement bug (or a codegen
bug in the multi-buffer ABI). The existing cross-lane validation
in `compiled_eval.rs` transplants directly.

## Concurrency

`ExecutablePlan::execute` becomes `&mut self` because the
intermediate buffer and scratch arenas are mutated. An `Arc<Plan>`
shared across threads for concurrent execution is no longer
supported without additional machinery. Callers that need
concurrent execution of the same plan can hold multiple plans or
wrap execution in a mutex.

## Opaque ops (PoolEvalSpan) and pool_eval

Opaque ops are the slow path: ops the JIT can't compile, evaluated
via `pool_eval` against a real `NumericTensor`. Their cross-span
inputs and outputs go through the intermediate buffer like any
other span; `PoolEvalSpan::execute` does the memcpy at its boundary
(it already does today, just against `PhaseStore` gather slices
instead of a shared buffer). The boundary copy stays but the
source/destination changes from `StoreSlice` to
`(buffer_ptr, byte_offset, count)`.

**pool_eval itself needs updating** to match the new inter-span
communication format. Today it allocates tensors for each op's
inputs and outputs from the pool, independently of any buffer
layout. In the new world, pool_eval's entry/exit boundaries read
from / write to the shared intermediate buffer (via small
marshalling copies), so the interface becomes: "here are pointers
into the shared buffer for the inputs I need, here's space for
outputs, run the opaque op." Internally pool_eval can keep doing
whatever allocation it wants — only its edges need to change.

Opaque spans are typically a tiny fraction of execution time, so
the marshalling copy at the boundary is acceptable cost for the
first cut.

## Implementation order (proposed)

1. **IndirectLoad index_range + access_bounds method**: add
   `index_range: u64` to `ScalarOp::IndirectLoad` and populate it at
   the three lowering sites (`gather.rs` ×2, `gather_elements.rs`).
   Add an `InputRef::access_bounds(count, atom_offset) -> (AtomId,
   AtomId)` method that replaces today's conservative
   `input_ref_range` helper with an exact computation. No behavior
   change yet — still using today's per-span layout. Verify the new
   bounds match (or tighten) what `input_ref_range` computed.
2. **Audit cross-span slab coalescing**: with bounds available,
   instrument today's layout pass to count how often coalescing
   actually fires on cross-span atoms in real models (vs. span-local,
   which is the common matmul case and stays in scratch). The
   interesting subset is (a) cross-span coalescing at all, and (b)
   the further subset where a slab pulls atoms from different buffer
   kinds (e.g., weight ↔ intermediate). If (a) is rare the global
   placer's coalescing logic becomes near-dead code and simplifies;
   if (b) fires, we need to decide between hardening lowering,
   copying weight regions into the intermediate buffer at startup,
   or letting codegen handle cross-buffer stride reads.
3. **Standalone placer**: implement the global placement pass over a
   `Vec<Phase>` and emit an `AtomPlacementMap`. Test in isolation
   against real model partitions; verify peak live footprint is
   reasonable.
4. **Multi-buffer JIT ABI**: extend `SlotInfo` with `buffer_id`,
   thread it through the address.rs emit functions, change the
   prologue to load multiple bases. Run the existing executor with
   per-span layouts but the new ABI to confirm no behavioral
   regression.
5. **Wire the placer in**: per-span layout consults the placement map
   for cross-span atoms; total scratch bytes recorded per span.
6. **Replace the executor**: delete `PhaseStore`, allocate
   intermediate + scratch buffers at plan-build, simplify the per-
   execute hot path. Output tensors hand back directly.
7. **Update pool_eval boundaries**: rework `PoolEvalSpan::execute`
   and pool_eval's input/output boundary to read/write directly
   against the shared intermediate buffer instead of through
   `StoreSlice` / `PhaseStore`. Internal pool_eval allocation is
   unchanged.
8. **Cache-line alignment**: enforce in partitioner (split granularity)
   and placer (slab start alignment). Verify with perf measurement.

Each step is independently reviewable and reverts cleanly if a deeper
issue surfaces.

## Open questions

- **Cross-buffer slab coalescing in real models** (step 2 above). If
  it fires regularly, options are: (a) harden lowering to avoid it,
  (b) emit a one-time memcpy at executor build to copy weight regions
  into the intermediate buffer at slab-adjacent positions, (c) demote
  the affected accesses to non-coalesced and let codegen handle
  cross-buffer reads. Need data before deciding.
- **`aligned_splits` cooperation between partitioner and placer**:
  partitioner_m already aligns Mul→Reduce split boundaries via
  `aligned_splits` logic tied to reduce stride. Under the new
  design, does this aligning stay in the partitioner, move to the
  placer, or become cooperative? Needs independent research —
  probably wants a separate investigation pass before committing.
- **Spans with no cross-span I/O**: edge case in the ABI array
  layout. Need to figure out the convention (empty array vs
  single-entry-for-scratch vs always-include-intermediate-ptr).
  Solve when the first such span shows up in testing.
- **Slab membership across phases**: a slab built for one phase's
  consumer might benefit from absorbing an atom produced 5 phases
  later if a stride pattern reaches that far. Rare but possible. We'll
  see how the audit goes.
- **Debug overrun detection**: with a shared buffer, a buggy span
  could corrupt unrelated atoms invisibly. Worth a debug-only
  per-slot-canary mode or a span-write-bound checker.
