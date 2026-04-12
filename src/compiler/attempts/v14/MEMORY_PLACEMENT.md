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
   │                + intermediate / literal buffer sizes
   │                + per-lane scratch high-water marks
   ▼
per-span codegen  ──→  CompiledSpan               (uses placement map
   │                                                for cross-span + literal
   │                                                atoms; allocates own
   │                                                scratch slots locally)
   ▼
plan-build       ──→  ExecutablePlan              (immutable; owns compiled
   │                                                spans + literal buffer)
   ▼
execute(&self)   ──→  pool-allocated intermediate + scratch + outputs,
                      no PhaseStore, output tensors returned directly
```

The placer is the only new pass. Codegen and the executor change to
match its outputs.

## Buffers

Five kinds of buffers, all addressed by an integer `buffer_id` known at
compile time:

| kind          | count            | lifetime   | source                              |
|---------------|------------------|------------|-------------------------------------|
| Input         | one per input    | per-call   | caller-owned (often borrowed)       |
| Output        | one per output   | per-call   | pool-allocated per execute, returned |
| Intermediate  | one (shared)     | per-call   | pool-allocated per execute          |
| Scratch       | one per lane     | per-call   | pool-allocated per execute          |
| Literal       | one (shared)     | per-plan   | plan-owned, filled at plan-build    |

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
- **Output buffers** are allocated from the pool at the top of each
  execute call and returned to the caller via the same handle the JIT
  wrote into — no copy at extract time.
- **Intermediate buffer** is the shared cross-span data store. Holds
  every atom that needs to flow from one span to another (different
  `(phase, lane)` tuples). Sized at compile time to the **peak live
  cross-span data** via liveness-interval allocation; allocated fresh
  from the pool at the top of each execute call. The pool makes this
  cheap — the same backing chunk typically recycles across calls — and
  it keeps `ExecutablePlan` free of mutable state.
- **Scratch buffers**: one per lane. Holds atoms that are produced and
  consumed within a single span. Sized at compile time to the maximum
  scratch demand of any span that lane will run; allocated from the
  pool at the top of each execute call, same as the intermediate buffer.
- **Literal buffer** is a single plan-wide allocation holding every
  `Literal` / `LiteralSpan` group's bytes. Sized at compile time from
  the placer's assignments, populated once at plan-build by walking the
  main graph, and **never written to again** — the JIT emits loads
  against it but never stores. Because it's immutable, the plan owns
  it directly (`Box<[u8]>`) and drops its pointer into the buffer_ptrs
  template on every call. `execute` takes `&self`; plans are free to
  share across threads. Literal groups do not participate in liveness
  analysis or slot reuse — every literal byte stays valid for the
  plan's lifetime.

## Atom classification

Each atom in the lowered NanoGraph falls into exactly one of four
categories:

1. **Literal**: produced by a `Literal` or `LiteralSpan` op. Lives in
   the plan-wide literal buffer at an offset chosen by the global
   placer. No JIT stores emitted for these groups — the bytes are
   written once at plan-build and the consumer reads them via a
   normal `buffer_ptrs[literal_buf_id]` load.
2. **Cross-span**: produced in one span, consumed in another (where
   "another" means a different `(phase, lane)`). Lives in the
   intermediate buffer (or in an input/output buffer if it is a model
   input/output). Has a `(buffer_id, byte_offset)` in the global
   placement map.
3. **Span-local, scratch**: produced and consumed within one span.
   Multi-use, can't be inlined. Lives in the lane's scratch arena at
   an offset chosen by the per-span codegen.
4. **Span-local, inlinable**: produced and consumed within one span,
   single consumer, expression can be folded into the consumer's
   loop body. No slot anywhere — value lives in registers. Existing
   `inlinable` mechanism in `compute_layout` handles this unchanged.

The classification is done by the global placer (which knows which
atoms are literals and which cross spans) plus the per-span codegen
(which decides scratch vs inline for the rest).

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
  every cross-span atom and every literal group.
- `intermediate_buffer_size`: bytes.
- `literal_buffer_size`: bytes. The literal buffer is a plan-owned,
  immutable `Box<[u8]>` filled at plan-build.
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

1. **Assign literal groups to the literal buffer.** Walk every group
   in the main graph; for each `Literal` / `LiteralSpan` op, append
   its atoms to the literal buffer at the current high-water mark
   (aligned to `elem_bytes`). Record the assigned offset in the
   placement map. Literal groups do not participate in liveness
   analysis — their slots are permanent.

2. **Identify cross-span atoms.** Walk every span. An atom is
   cross-span if it appears in any span's `outputs` list (the
   partitioner already declares these). Model inputs and outputs are
   trivially cross-span and assigned to their dedicated input/output
   buffers up front. Literal groups are already assigned by step 1
   and skip this pass.

3. **Compute liveness intervals.** For each cross-span atom (or slab,
   see step 4), compute `[first_produce_phase, last_consume_phase]`
   where:
   - `first_produce_phase` is the lowest phase index in which any
     span declares the atom as an output.
   - `last_consume_phase` is the highest phase index in which any
     span declares the atom as an input. Model outputs are pinned to
     `phases.len() - 1` so they survive to extraction.

4. **Identify coalescing constraints.**

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

5. **Per-slab liveness.** A slab's interval is the *union* of its
   members' intervals. A slab can only be reused after every member
   is dead. Heavy coalescing inflates the intermediate buffer because
   a slab containing one long-lived atom pins space for every other
   member of the slab until that atom dies.

6. **Pack intervals into the intermediate buffer.** Linear-scan or
   first-fit allocator over slab-intervals, sorted by start phase.
   Free a slab's offset once `current_phase > slab.end`. Honor:
   - Cache-line alignment for slabs that will be split-written by
     multiple lanes (see "Cache-line discipline" below).
   - Slab elem-bytes alignment (so stride-based InputRefs work).

7. **Detect cross-buffer slab violations.** If a slab's coalescing
   constraint pulls in atoms from different buffer kinds (e.g., a
   weight tensor and an intermediate atom), that's currently a hard
   error — the lowering must be hardened to never produce such
   InputRefs. See Open Questions.

8. **Emit the placement map.**

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

`ExecutablePlan` is a frozen compilation artifact. It owns:

- The compiled spans (`Vec<Box<dyn CompiledSpanFn>>`, grouped by phase
  and lane).
- The placement map and the buffer-size metadata it needs at
  execute time.
- The plan-wide literal buffer (`Box<[u8]>`, immutable after
  plan-build).
- A fixed-worker rayon `ThreadPool` for lane dispatch.

It does **not** own the intermediate or scratch buffers. Every field
is immutable; `execute(&self, ...)` takes a shared reference, and
plans can be shared across threads freely.

Plan-build time:
- Receive the placement map + per-span scratch sizes from the
  compiler.
- Allocate the literal buffer (`Box<[u8]>` sized to
  `placement.literal_buffer_size`) and walk the main graph once,
  writing each `Literal` / `LiteralSpan` group's bytes into its
  placer-assigned offset.

Per-execute hot path:
- Allocate the intermediate buffer from the pool (sized from the
  placement map). Pool reuse keeps this cheap; the same backing
  chunk typically recycles across consecutive executes.
- Allocate per-lane scratch arenas from the pool (sized to the
  per-lane high-water mark the compiler recorded at build time).
- Allocate output tensors from the pool, one per output range,
  using the declared shape + dtype. These `NumericTensor`s are
  what `execute` will return — there is no separate extract step.
- Build a small `[*mut u8; N]` buffer-pointer-array on the stack
  for this call. Literal slot gets the plan's persistent literal
  pointer; intermediate and output slots get the freshly-allocated
  pointers; input slots get the caller's view pointers; the scratch
  slot gets patched per-lane inside the dispatch closure.
- For each phase: rayon-broadcast the lanes, each calling its
  compiled-span fn with the stack array. Per-lane variance is
  limited to the scratch slot, so each worker copies the template
  once and overwrites `[scratch_buf_id]` with its own lane's
  pointer.
- After all phases return: hand the output tensors back to the
  caller directly. Output tensors **are** the output buffers,
  wrapped in `NumericTensor`s with the right layout metadata. No
  memcpy, no hashmap round-trip.

`PhaseStore`, `insert_batch`, `gather`, `evict`, `output_liveness`,
`pinned_output_atoms`: all deleted.

The producer-before-reader invariant is guaranteed by the
topologically-ordered JIT emission within each span combined with
the phase barrier between spans. Intermediate and scratch buffers
start out whatever the pool hands us (typically zeroed on first
alloc, recycled on subsequent calls); no scrubbing step is needed
because every byte a consumer reads was written by a producer
earlier in the same execute call.

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

Every `Literal` / `LiteralSpan` group in the main graph has its
source bytes stored in a **single plan-wide literal buffer**. The
buffer is allocated and fully populated at plan-build time and
never mutated again — the executor links it into `buffer_ptrs` on
every `execute` call as a **read-only source**. There is no
execution-time literal-handling logic of any kind: no replay, no
initialization pass, no conditional writes.

### Placer outputs

The placer emits two pieces of data for literal groups:

1. **`literal_sources: HashMap<AtomId, u64>`** — for **every**
   `Literal` / `LiteralSpan` group, the byte offset in the literal
   buffer where that group's source bytes live. This includes
   groups whose primary placement is in a non-literal buffer; the
   source bytes always live in the literal buffer regardless.

2. **`PlacementEntry`s for primary-in-literal groups** — i.e., the
   common case where a group's atoms don't overlap a model output
   range. These entries point at `LITERAL_BUFFER` with the same
   offset recorded in `literal_sources[base]`.

`literal_buffer_size` is the watermark after walking all literal
groups; the executor allocates a `Box<[u8]>` of this size at
plan-build.

### Plan-build

`ExecutablePlanBuilder::build` walks the main graph once, iterates
every `Literal` / `LiteralSpan` group, and writes its bytes into
the literal buffer at `literal_sources[base]`. After this walk the
literal buffer is frozen.

### Execute

`execute(&self)` contains **zero** literal-aware code. The literal
buffer pointer is dropped into `buffer_ptrs[LITERAL_BUFFER]`
alongside the intermediate / input / output / scratch pointers,
and that's it.

### JIT codegen

`emit_group` handles `Literal` / `LiteralSpan` groups in two
branches depending on the group's primary slot:

1. **Primary slot is `LITERAL_BUFFER`** (the common case): no code
   is emitted. Source == destination; the bytes are already in
   place. Consumers read them via ordinary
   `buffer_ptrs[LITERAL_BUFFER]` loads.

2. **Primary slot is a non-literal buffer** (Pad-style lowering
   where the literal's atoms overlap a model output range): emit
   an ordinary copy loop that reads from the literal buffer at
   `literal_sources[base] + iter * bit_stride` and stores into the
   destination slot at `slot.bit_offset + iter * bit_stride`. Same
   shape as an `Identity` group — two bit-offset computations, a
   load, and a store — but the source side is synthesized from the
   placer's `literal_sources` map rather than resolved through an
   `InputRef`. Each span that needs this path forces
   `LITERAL_BUFFER` into its `BufferBases` table so the prologue
   loads `buffer_ptrs[LITERAL_BUFFER]` into a callee-saved GPR.

No special case in the executor, no execution-time literal writes,
no `output_literal_writes` replay. The literal buffer is allocated
once, written once, read many times.

### Why both branches

Splitting literal groups across two primary buffers (literal vs
output) is deliberate. The alternative — always placing literal
bytes in the literal buffer and having the output-extraction path
read from there — would either require aliasing a read-only
allocation into a per-execute output `NumericTensor` (breaks
ownership) or adding a copy at extract time (reintroduces the
extract-time memcpy the design is meant to eliminate). Having
the JIT emit a cheap in-phase copy for the overlap case lets the
output tensor own its allocation cleanly and keeps all literal
handling on the JIT side of the boundary.

Both `Literal` (scalar) and `LiteralSpan` (embedded constant
tensor) go through this path. If `LiteralSpan` footprint ever
becomes a real memory-pressure problem — e.g., a model folds large
weight constants into the NanoGraph rather than keeping them as
inputs — the followup is to split the literal buffer into "small
literals" and "literal spans," with the span buffer coming from
mmap-backed read-only storage. Not worth solving up front; the
current RWKV / GPT-2 profiles show literals in the low-MB range.

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
2. **Audit cross-span slab coalescing** — **done**. See `audit.rs`
   for the instrumentation; enable with `WT_AUDIT_SLABS=1` on any
   compiled-eval run to re-measure.

   Measured on two models:

   | metric                               | GPT-2 (lm-head-10) | RWKV 0.1B |
   |--------------------------------------|-------------------:|----------:|
   | InputRefs scanned                    |             24,160 |   453,934 |
   | coalescing triggers (>1 item)        |           260 (1.1%) | 148 (0.03%) |
   | all-scratch (span-local)             |                  0 |  48 (32%) |
   | all-input (weight-only)              |                  0 |         0 |
   | all-intermediate (pure cross-span)   |         260 (100%) | 100 (68%) |
   | all-output                           |                  0 |         0 |
   | **mixed input+intermediate** (cross-buffer) |                  **0** |     **0** |
   | other mixed                          |                  0 |         0 |

   **Headline: zero cross-buffer hits on either model.** The placer's
   global coalescing logic stays entirely within one buffer at a
   time — no weight ↔ intermediate mixing, no startup copies needed.
   The "mixed input+intermediate" handling in the placer can be a
   hard assertion rather than a supported code path.

   Other findings:
   - Total coalescing triggers are low: 260 on GPT-2, 148 on RWKV.
     Placer coalescing is non-trivial but bounded — not a perf
     concern at any plausible model size.
   - GPT-2 has zero all-scratch cases; RWKV has 48 (~32%). The RWKV
     span-local constraints come from its different lowering
     patterns (time mixing / state evolution) and stay inside
     per-span `compute_layout`. The global placer ignores them.
   - Spot-checked GPT-2 examples look like ~768 single-atom
     intermediate groups strided by 1 — typical LayerNorm/ReduceMean
     patterns where lowering produced many 1-atom groups that a
     consumer reads as a flat range. Benign; the placer just bundles
     them into one intermediate-buffer slab.
   - The matmul Mul→Reduce pattern I was expecting to dominate
     coalescing does NOT trigger at the InputRef-bound level — its
     stride reads stay within a single producer group. Coalescing
     still happens inside `compute_layout::Step 1` per-span for
     other lowering artifacts, but those stay in scratch.
3. **Standalone placer** — **done** (see `placer.rs`; enable with
   `WT_PRINT_PLACEMENT=1` on any compiled-eval run to see the
   summary). Implements classification → liveness → coalescing
   (union-find over Intermediate groups only, with a hard assertion
   against cross-buffer slabs per the step 2 audit) → first-fit
   interval packing → per-buffer emission.

   Measured peak footprints (default 4-lane partition, post
   partitioner barrier fix — see section below):

   | model    | input buffers | output buffers | intermediate peak | slabs | scratch groups |
   |----------|--------------:|---------------:|------------------:|------:|---------------:|
   | GPT-2    |      634 MiB |         1.3 MB |         **0.2 MB** | 2034 |          2819 |
   | RWKV 0.1B|      367 MiB |         2.4 MB |         **0.9 MB** | 19167 |         38918 |

   Both models fit the cache-residency sweet spot the design is
   aiming for: sub-MB cross-span intermediate buffer, with everything
   else either in per-lane scratch (reused via liveness), inlined
   into a downstream op (see the matmul case below), or in a
   dedicated input/output buffer.

   The placer's output includes:
   - `AtomPlacementMap` with per-atom `(buffer_id, byte_offset)`.
   - Per-buffer metadata (size, kind, debug name).
   - `intermediate_peak_bytes` diagnostic.
   - `intermediate_slab_count`, `scratch_group_count`,
     `top_slabs` summary for postmortem.

   #### GPT-2 LM head investigation and partitioner fix

   The *initial* GPT-2 placer run reported a 736 MB intermediate
   peak — 5 slabs of 147 MB each, live concurrently in phases 307–
   308, corresponding to the LM head matmul Mul intermediates for
   the 5 sequence positions (38.6M atoms = 50257 vocab × 768
   hidden, one per position). The executor's `PhaseStore` held the
   same 737 MB at the same phase, so the placer was matching
   today's actual footprint — but the partitioner was leaving a
   huge amount of memory on the table.

   Root cause: two `lane_local_access_check` limitations in
   `partitioner_m.rs` combined to force the Mul→Reduce pair across
   a phase barrier whenever the consumer count didn't divide evenly
   by `num_lanes`:

   1. The check computed `prod_chunk = producer.count / num_lanes`
      via floor division but the consumer footprint via ceil
      (`max_chunk`), so for `50257 % 4 != 0` the biggest lane's
      reduce window appeared to overflow the smallest producer
      chunk by one stride unit. In reality, **aligned splitting**
      makes the producer chunk grow in lock-step with the consumer
      chunk — the partition stays exact by construction — but the
      check didn't model that.
   2. `aligned_splits` in `build_phase` required divisibility even
      though `split_range_aligned` → `split_count` already handles
      uneven splits correctly.

   The fix adds a perfect-tile fast path in
   `lane_local_access_check` for Reduce consumers where
   `reduce_extent ≤ abs_stride` (i.e., each output's reduce window
   fits within one stride unit of the producer), and lifts the
   divisibility gate in `aligned_splits`. Both are guarded by a
   **safety gate** that protects against a subtle interaction:

   > A group P can be safely aligned-split (for a downstream
   > perfect-tile Reduce) only when
   > (a) P has a unique consumer in the whole graph — no other
   >     consumer sees P's per-lane layout, and
   > (b) P has no same-phase Split producers of its own —
   >     otherwise aligning P creates cross-lane reads upstream
   >     against those producers' default-split chunks.

   Condition (b) is tracked per-group in `assign_phases` via a
   `has_same_phase_producer` bool vector (Duplicate producers are
   exempt, since they broadcast the full tensor to every lane).
   The first version of the fix only had (a), and introduced 20
   cross-lane violations on GPT-2 when LayerNorm's `(x-mean)^2 →
   variance` pattern triggered the fast path: `(x-mean)^2` has a
   unique consumer (`variance`), but its input `(x-mean)` is
   multi-consumer default-split, so aligning `(x-mean)^2` tore
   holes in `(x-mean)`'s lane layout.

   Condition (b) precisely identifies the matmul Mul→Reduce pattern
   (the Mul is structurally always a single-use intermediate, and
   its own inputs — weights via input tensor, activations via a
   cross-phase modular-barrier edge — are always either flat or
   cross-phase). LayerNorm's squared-diff fails the gate because
   its producer is same-phase Split.

   #### Interaction with `compute_layout`'s reduce-fold inlining

   The barrier fix alone achieves the 736 MB → 0.2 MB drop because
   `compute_layout::Step 3b` in `layout.rs` already has a
   "reduce-fold inlining" pass: when a pure-scalar producer has a
   single intra-span consumer that is a Reduce with `reduce_stride
   == 1` and a matching stride-K affine InputRef, the producer's
   expression is folded directly into the Reduce's loop body. The
   materialized Mul intermediate vanishes entirely — each Reduce
   output is computed by reading A and W, multiplying, and
   accumulating, with no buffer roundtrip for the product.

   With aligned splitting, the Mul and Reduce share matching
   per-lane `atom_offset` values (`mul.atom_offset == K *
   reduce.atom_offset` for every lane), which is exactly the
   inlining precondition. The 5 × 147 MB Muls are not just
   span-local — they're **never stored anywhere**.

   Execution-side confirmation on GPT-2:

   | metric                    | pre-fix | post-fix |
   |---------------------------|--------:|---------:|
   | intermediate buffer peak  |  736 MB | **0.2 MB** |
   | `PhaseStore` data peak    |  737 MB | **1 MB** |
   | RSS at peak phase         | 2732 MB | **2069 MB** |
   | one-step execute time     | 2140 ms | **1771 ms** |
   | cross-lane violations     |       0 | **0** |

   The design's cache-residency story works for the LM head: each
   lane streams `A·W` into a local accumulator, no cross-core
   traffic on the hot bytes, the working set fits in L2/L3.
4. **Multi-buffer JIT ABI** — **done** (3 commits, landed as
   `971e110`, `d1b81c4`, `34d2fd8`).
   - *4a*: added `buffer_id: u8` to `SlotInfo` and `AddressInfo`,
     defaulting to 0 everywhere. No behavior change.
   - *4b*: replaced every `BUFFER_REG` in the `orch/group.rs` and
     `orch/reduce.rs` `emit_load_bits` / `emit_store_bits` call
     sites with `buffer_base_reg(info.buffer_id)`. `emit_indirect_load_iter`
     grew a `table_buffer_id` parameter. Under phase 4 the helper
     debug-asserts `buffer_id == 0` and returns `BUFFER_REG`; step 5
     will turn this into a real dispatch table.
   - *4c*: flipped the external ABI from
     `fn(buffer: *mut u8)` to `fn(buffer_ptrs: *const *mut u8)`.
     The prologue's `mov r12, rdi` became `mov r12, QWORD [rdi]` —
     it now loads `buffer_ptrs[0]` into r12. `X86JitSpan::execute`
     assembles a one-entry stack array (`[buffer.as_mut_ptr(); 1]`)
     and passes its pointer. Spans still only use buffer_id 0; the
     indirection is in place for step 5 to populate more slots.

   All 160 v14 tests and the JIT-compiled partitioned `test_set`
   categories (matmul, reduce, elementwise, …) remain green across
   all three commits. End-to-end GPT-2 runs cleanly through the new
   ABI with the same compile/execute footprint as before step 4.
5. **Wire the placer in + replace the executor + update pool_eval
   boundaries** — **done**. First attempt was scrapped during design
   review (it accreted `Mutex<PlanBuffers>`, three separate
   literal-population codepaths, a pointer-aliasing hack, a
   zero-filled input stub, and a `HashMap<AtomId, NumericTensor>` +
   second-copy extract path). All of that fell out of one early wrong
   turn: treating the intermediate and scratch buffers as plan-owned
   state instead of execute-scope allocations. The corrected shape:

   - `ExecutablePlan` is fully immutable. It owns the compiled spans,
     the placement map, the fixed-worker lane `ThreadPool`, and a
     single `Box<[u8]>` holding the plan-wide literal buffer (the
     only persistent storage the plan needs). No `Mutex`, no
     interior mutability.
   - `execute(&self, input_ptrs, pool)` is the one entry point. It
     allocates the intermediate buffer, per-lane scratch, and output
     tensors from the pool inside the call; builds the stack
     `buffer_ptrs` template; dispatches phases; and returns the
     output `NumericTensor`s directly in declaration order. No
     hashmap, no extract-time copy.
   - Literal handling is a single plan-wide literal buffer,
     allocated and pre-filled at plan-build, linked in as a
     read-only source at every `execute` call. The placer builds a
     `literal_sources: HashMap<AtomId, u64>` covering **every**
     `Literal` / `LiteralSpan` group regardless of its primary
     placement; plan-build walks the main graph once and writes
     each group's bytes into that offset; after plan-build the
     literal buffer is frozen and never touched again. `execute`
     has zero literal-aware code. Literal groups whose primary
     slot is the literal buffer are no-ops at JIT emission time
     (source == destination, consumers read directly from the
     literal buffer). Literal groups whose primary slot is a
     non-literal buffer — Pad-style lowering where the literal
     atoms overlap a model output range — get an ordinary copy
     loop emitted by the JIT: read from the literal buffer at
     `literal_sources[base]`, store into the destination slot.
     Structurally identical to an `Identity` group with the source
     synthesized from the placer's map. No special case in the
     executor, no replay loop. (The first corrected attempt used
     an `output_literal_writes` replay path in the executor; that
     was replaced with the JIT copy path when the design review
     caught that "no execute-time literal logic" wasn't being
     honored.)
   - `compute_layout` takes `&AtomPlacementMap` and assigns each
     atom a `(buffer_id, byte_offset)` from the placer when the span
     is the canonical writer for that atom, otherwise runs the
     FreeList with `buffer_id = scratch_buffer_id`. The
     non-canonical-writer rule for duplicate groups stays.
   - The multi-buffer JIT prologue / `BufferBases` register pool
     from step 4 is extended: each span touches N distinct buffer
     ids (intermediate, literal, its inputs, its output, scratch),
     and the prologue loads each into a callee-saved GPR.
   - `CompiledSpanFn` is `fn scratch_bytes(&self) -> usize` plus
     `fn execute(&self, buffer_ptrs: &[*mut u8])`. Nothing else.
   - `PoolEvalSpan` resolves each input/output range to
     `(buffer_id, byte_offset)` at construction and
     gathers/scatters against `buffer_ptrs` at execute time. Unused
     input pointers are tolerated: pool_eval receives a zeroed
     tensor for the input, matching the old PhaseStore behavior
     where missing atoms read as zero (test-only quirk: a
     constant-folded input leaves the `input_tensors()` entry
     declared but never bound by the caller).
   - Lane dispatch uses Rayon's `broadcast`, with the `buffer_ptrs`
     template and per-lane scratch pointers transported via
     `Vec<AtomicPtr<u8>>` (which is genuinely `Sync`, vs the
     pointer-aliasing hack from the first attempt).
   - `PhaseStore`, `StoreSlice`, `SpanOutput`, `insert_batch`,
     `gather`, `evict`, `output_liveness`, `pinned_output_atoms`,
     `write_store_slice_to_buffer`, `read_buffer_to_output`:
     deleted.

   Cache-residency story is unchanged: the intermediate buffer still
   holds cross-span atoms through a phase barrier; lanes still read
   the producer's writes out of shared L2/L3. Allocating it from the
   pool per-execute doesn't defeat that — pool recycling keeps the
   backing chunk stable across calls, and the hardware cache state
   only needs to persist within one execute, not across them.

   **Test results**: 156/156 v14 unit tests pass. 117/119 test_set
   cases pass through the trivial (1-phase) compiled-eval path;
   115/119 through the 8-lane partitioned path. The remaining 4
   failures are all pre-existing conv-partitioner issues flagged
   in memory prior to this work: `conv_3x3_same_pad` and
   `conv_with_bias` fail in both trivial and partitioned;
   `conv_3x3_no_pad` and `conv_1x1` fail only in partitioned (race
   on a shared intermediate buffer the partitioner didn't prevent).
   No new regressions from the rewrite.

6. **Cache-line alignment**: enforce in partitioner (split granularity)
   and placer (slab start alignment). Verify with perf measurement.

Each step is independently reviewable and reverts cleanly if a deeper
issue surfaces.

## Open questions

- **Cross-buffer slab coalescing in real models** — **resolved**:
  zero hits on both GPT-2 (260 coalescing constraints, 0 mixed) and
  RWKV 0.1B (148 coalescing constraints, 0 mixed). The placer's
  "mixed input+intermediate" code path becomes a debug assertion
  rather than a supported case.
- **`aligned_splits` cooperation between partitioner and placer** —
  **partially resolved**: the partitioner now aligns Mul→Reduce
  split boundaries for the perfect-tile case even when counts
  don't divide by `num_lanes`, gated on the producer having a
  unique consumer *and* no same-phase Split producers (see the
  GPT-2 LM head investigation above). A looser gate that admits
  multi-consumer producers — provided all their consumers agree
  on the aligned layout — would let LayerNorm-style patterns
  collapse too, but that's chain-wide alignment and is not
  implemented. The placer does not need to re-derive alignment:
  it consumes the partitioner's output as-is.
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
