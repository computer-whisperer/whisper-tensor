# Heterogeneous Compiled Execution

A working design document. The eventual goal is a runtime that can
distribute compiled inference work across multiple backends (CPU,
Vulkan, future targets) simultaneously, picking the distribution that
best fits the problem at hand. This document captures the design as
it is being worked out — decisions that are settled, points still
open, and the reasoning behind both.

## Motivation and Sequencing

The immediate need is a Vulkan backend alongside the existing v14 CPU
backend. The longer-term need is heterogeneous execution across
multiple backends at once. Building "switch between CPU and GPU"
first and then retrofitting heterogeneous scheduling on top would
require tearing up both backends once the distribution logic arrives.
Instead we build the heterogeneous layer first, in front of the
existing CPU backend, with CPU as the only initial implementation.
When Vulkan lands it slots into a layer that has already settled.

The risk in this ordering is leaking CPU assumptions (synchronous
dispatch, unified memory, no command queues, no real "transfer") into
the abstraction. To avoid that, the layer is designed *targeting* GPU
semantics — async dispatch, explicit transfers, separate memory pools
— even when the only backend implementing them is CPU. The CPU
backend implements those primitives as cheap synchronous calls, but
the shape of the API is GPU-shaped from day one.

## Grounding in v14

The CPU backend (`compiler/attempts/v14/`) provides the abstractions
the heterogeneous layer extends. The relevant concepts:

- **AtomGroup** — compression unit; a packet of structurally
  identical scalar atoms with one op and structured input refs.
- **Span** — one lane × one phase; the smallest JIT-dispatched unit.
- **Phase** — barriered work. All lanes sync at the end of a phase.
- **Plan** (`ExecutablePlan`) — the whole compiled artifact;
  immutable, with `execute(&self, input_ptrs, pool)` as the entry.
- **Buffers** — five kinds (`Input`, `Output`, `Intermediate`,
  `Scratch`, `Literal`), each with a compile-time-known
  `(BufferId, byte_offset)` for every atom that lives in it.
- **JIT ABI** — `extern "C" fn(buffer_ptrs: *const *mut u8)`. The
  only runtime tunable is the array of buffer pointers.
- **Placer** — global compile pass producing the `AtomPlacementMap`.

Crucially, v14 already adopts the discipline that maps cleanly onto
heterogeneous execution: *put the value in one place and leave it
there; let the interconnect handle delivery to consumers*. On a
single device the "interconnect" is on-chip cache coherence; the
producer writes once and consumers on other cores read from where
it sits. The heterogeneous extension is the same idea: producer
writes to its device's memory; if the consumer is on the same
device, hardware coherence delivers; if on a different device, an
explicit transfer at the piece boundary delivers.

There is no tensor abstraction inside a compiled piece. The
addressable unit is the working buffer; the meaningful unit at the
compiler's level is the AtomGroup. The user-facing API does have a
tensor-shaped boundary (model inputs and outputs), but that lives
above the heterogeneous layer.

## The Two-Layer Partitioning Model

Partitioning happens at two layers, and the two levels mirror each
other in shape — the heterogeneous layer is structurally a *macro
version* of v14's x86 JIT partitioner.

1. **Global partitioner.** Breaks the problem into **global phases**
   (cross-device synchronization points) and **global spans** (the
   parallel work within a phase, distributable across devices). At
   the global level, "lane" means "device-instance" — lane 1 might
   be the CPU, lanes 2–8 might be Vulkan queues. Each global span
   is compiled for every capable backend; the runtime scheduler
   chooses which backend's compiled form to dispatch and to which
   device-lane, with the option to rebalance under uneven load.

2. **Per-backend partitioner.** Each backend is given a global span
   and produces its own compiled artifact. The CPU backend takes a
   global span, runs today's v14 partitioner over it (producing
   internal phases, internal spans, lane assignment, placement),
   and emits the resulting ExecutablePlan-shaped artifact. Vulkan
   does whatever shape makes sense for its target. From the global
   layer's view, a global span is a **black box** that takes input
   buffers, produces output buffers, runs to completion.

The global partitioner's job — split parallel work across N
processors, balance per phase, respect data dependencies — is the
same work v14's partitioner does internally to divide work between
CPU cores. The differences between the two levels are
quantitative, not structural:

- Global lanes are heterogeneous (CPU vs GPU vs accelerator) with
  different cost models per backend.
- Global lane assignment is **runtime-dynamic** (rebalance under
  uneven load), where v14's CPU lane assignment is **static** for
  cache affinity.
- Inter-lane data transfer at the global level goes through PCIe /
  NVLink / explicit copies, not L3 cache coherence.

The two levels of cache discipline don't conflict: within a global
span, the CPU backend's internal lane affinity holds for the
duration of one execution. Across global spans, cross-device
transfers dominate so cache affinity is not the optimization target.

The contract between the layers is: **a backend never executes a
partial global span.** Spans are atomic at the dispatch level.
Cross-device transfers happen only at global phase barriers, never
inside a span. This is the same rule v14 applies internally — no
cross-lane data flow within a phase — promoted to the global level.

This atomic-dispatch rule is the load-bearing simplification of the
whole design. It eliminates a large class of cross-device
synchronization complexity and lets the scheduler reason about
opaque global spans dispatched to opaque device-lanes.

## Decisions Settled

### Compile every global span for every capable backend

The runtime distribution decision is late-bound. Each global span
carries a set of compiled implementations — one per backend that
could compile it — and the scheduler picks among them at dispatch
time. This is more compile work than committing statically, but
inference is repeated many times against the same compiled
artifacts, so the trade-off is sound. Compile-for-all cost is not a
primary concern; caching will matter, but compile-time
hyper-optimization is not the optimization target.

### Backends are not required to support every global span

Not every backend can compile every global span. Vulkan in
particular will not run everything — opaque ops, certain
symbolic-dim shapes, and ragged or scan-like operations may simply
not have a tractable GPU form. Global spans therefore carry a set
of "backends that compiled me," and the scheduler chooses among the
available ones.

CPU is the universal fallback. Any global span that no other
backend can compile must still be runnable on CPU. This naturally
accommodates the existing opaque-op boundary: opaque ops currently
dispatch through CPU paths, and that remains true under the
heterogeneous layer.

The global partitioner does not need to guarantee universal
compilability, but it does need to guarantee CPU-compilability for
every global span. Backends advertise which spans they could
compile; the scheduler's choice space is the intersection of
"compiled successfully" and "available at runtime."

### Global span boundaries are layout-neutral

Internal layout is each backend's freedom — Vulkan may prefer one
arrangement, CPU another — but at global span boundaries they have
to agree on a canonical form. Otherwise the cost of a cross-device
transfer carries an implicit layout transform that the backend
about to run could have done better itself. The canonical boundary
form is part of the global-span interface contract; backends
translate to and from it internally.

### Symbolic-dim parity is each backend's responsibility

The v14 CPU backend has a careful symbolic-dim story (see
`symbolic_dims_nano.md`). Any new backend needs an analogous one. If
a backend cannot represent a sym-dim pattern that appears in a
global span, it advertises that span as uncompilable and the
scheduler routes it to a backend that can — ultimately CPU. This is
the escape valve for sym-dim coverage gaps, not a problem to design
around at the global layer.

### Initial global partitioner: coarse configuration

Global-span boundary selection is an optimization problem with
genuinely conflicting pressures (small spans favor load balance,
large spans favor transfer amortization). It will need real work
eventually, but a rough first version is sufficient to put the
overall system well above the standard. Refinement comes later,
once we have measurements to refine against.

## Memory Model: Pyramid With Limitless Tail

The compute device sits at the top of a storage pyramid. Below it:
system DRAM. Below that: disk. Future levels (NVMe pools, NVLink
peers, distributed memory) extend the pyramid downward without
disturbing what is above. The bottom of the pyramid is treated as
functionally limitless — the system as a whole never runs out of
storage. It only runs out of *fast* storage, at the top.

Every tensor has a canonical home somewhere in the pyramid. Device
memory is a cache on top of that home. The question "do we have
room?" collapses into "what is the cache discipline?" — a
better-shaped problem.

A useful side effect: even in the CPU-only bootstrap phase, this
framing has real work to do. CPU's "device" is its working DRAM, and
the level below is disk. Disk-backed weights for a model that
doesn't fit in DRAM exercise the same paging and streaming machinery
that will later move data between system DRAM and a GPU. We get a
meaningful test target before any GPU code exists.

## Buffer Kinds and Global-Span Boundaries

Within a global span, the v14 buffer taxonomy applies unchanged:
`Input`, `Output`, `Intermediate`, `Scratch`, `Literal`. The
backend compiling the span does not distinguish whether each
`Input` is a model-level input (user-supplied), the output of an
upstream span on the same device, or the output of an upstream span
on another device. All arrive as buffer pointers in the span's
`buffer_ptrs` array at execute time. Same for `Output`: the span
doesn't know whether its outputs go to the user or to a downstream
span on this or another device.

Buffer kinds at the global-span-boundary level:

- **Input** (span-level) — either a model input or an upstream
  global span's output. Caller-supplied pointer at execute time.
- **Output** (span-level) — either a model output or a downstream
  global span's input. Pool-allocated by the runtime per execute.
- **Intermediate** — within-span cross-(internal-span) data.
  Backend-internal. v14's cache-coherence story stays exactly as is
  for cores within a CPU span.
- **Scratch** — within-(internal-span) data. Backend-internal.
- **Literal** — constants the span's compilation embedded.
  Per-span, per-backend; if multiple global spans on multiple
  backends reference the same literal bytes, each backend gets its
  own copy resident on its target device. One-time cost at
  plan-build per backend.

The heterogeneous layer's runtime concern is the device residency
of span-level `Input` and `Output` buffers — which device each
currently lives on, and whether a transfer is needed before the
next consuming span can run. `Intermediate`, `Scratch`, and
`Literal` are entirely the backend's concern and do not appear in
the heterogeneous layer's bookkeeping.

## The Two Regimes: Cached and Streamed

From a global span's point of view, a span-level input or output
buffer is in one of two regimes:

1. **Cached.** The span sees a fully materialized buffer at
   execution time — a contiguous byte range with the agreed shape
   and dtype, addressable through a single `buffer_ptrs` slot. The
   placer (or its per-backend equivalent) decided everything about
   layout at compile time. Cross-invocation reuse of cached buffers
   (weights staying on device across calls) emerges from runtime
   bookkeeping: if the input the user supplies on call N+1 is the
   same logical buffer as on call N, and the device-side copy
   wasn't displaced, the transfer is skipped. No "stickiness" tag,
   no model-specific path; just buffer identity.

2. **Streamed.** The buffer is too large to materialize at any
   single pyramid level. Data flows through the device in chunks
   while the span executes. v14's placer already does
   interval-packing of working buffers along the time axis;
   streaming is the same discipline applied when the working set
   doesn't fit and the planner has to schedule atoms in time-shared
   regions of a smaller working buffer. Streaming may require a
   span to be written with that chunking in mind — not every span
   has a streamed compiled form.

The atomic-dispatch rule still holds in both regimes. For a
streamed span, the backend still owns its execution from first
chunk to last; transfer wall-clock overlaps with execution rather
than preceding it, but transfer *responsibility* still changes
hands only at global phase boundaries.

## No Model-Specific Hints

A foundational design choice: the engine accepts no model-specific
"please keep this on device" / "this is a weight" / "stream this
input" tags. The compiler is expected to do the right thing from the
graph structure and the inputs it is given. A weight is just an
input number; if it is best kept on-device, that should be the
natural outcome of compilation and cache discipline working
together, not the result of someone tagging it.

This is a deliberate contrast with engines like TensorRT or vLLM
that build extensive machinery for per-model behavior. Whisper
Tensor instead invests that effort into a more capable general
compiler. When a class of model performs poorly, the response is to
improve the compiler's handling of the pattern — not to add a hint
that special-cases it.

## Cross-Invocation Buffer Identity

Today, model `Input` buffers in v14 are caller-owned: `execute`
takes whatever pointer the caller hands in (often a borrowed
`NumericTensorCOW`) and exposes it directly to the JIT. For weights
to remain resident on a device across calls, the heterogeneous
layer needs **stable identity** for input buffers — some way to
recognize "this is the same logical buffer as last call's slot N,"
so the device-resident copy can be reused.

The engine itself does not invent this identity. The supergraph
layer already manages model-level state (weights, persistent KV,
session bindings); whatever stable handle it uses to refer to
those values is what the heterogeneous layer looks up in its
device-residency table. A binding registered once and reused
across many `execute` calls — the typical inference-loop pattern —
naturally lets the device-side copy stay live across calls.

For inputs without stable identity (fresh activations, new tokens
each call), there is no cross-invocation reuse and no harm done —
the buffer is fresh on every call, transfer happens once per call,
the residency table records a transient entry that gets recycled.

**Output buffers** are returned by `execute` directly — today as
`NumericTensor`s wrapping the pool-allocated output bytes. Under
the heterogeneous design, an output's bytes may live on a device
rather than host DRAM, which means materializing them to a
host-readable form may require an explicit transfer. The output
handle the user receives is opaque enough to hide where the bytes
actually live; reading them through the host API triggers a
transfer if needed.

This is the same handle-shape that PyTorch/JAX device tensors use:
opaque to the user, transparent to the engine, host-materialization
on demand.

## Open Problems

### Scheduler decisions

Within a global phase, the scheduler assigns each runnable global
span to a device-lane. The decisions are:

- Which device-lane to dispatch to (the heterogeneous decision).
- Which compiled form (cached or streamed) to use, when the chosen
  backend produced multiple forms for that span.
- Whether to rebalance mid-phase if one device-lane is starving
  while another still has unfinished spans.

Past decisions affect future ones — a span dispatched to device D
leaves its outputs in D's memory, biasing future dispatch toward D
for consumers of those outputs. This is a sequential planning
problem across phases, not a span-by-span one within a phase.
Initial scheduling will be static and conservative; a real
cost-model-driven scheduler waits for multiple backends and real
timing data.

### Async/transfer primitives

The abstraction every backend implements must be GPU-shaped:
explicit transfer submission, completion signaling, command-queue
ordering. CPU implements these as cheap synchronous calls but the
shape of the API is async from day one. Concrete primitive set is
TBD; needs to support both cached-staging (whole-buffer transfers
at phase boundaries) and streamed (chunked transfers with overlap
during span execution).

### Global span boundary selection (deferred)

The conflicting-pressures problem from the partitioner section.
The coarse first version buys time to learn what good boundaries
look like in practice. Worth revisiting once we have real
cross-backend timings.

### Backend capability advertisement — mechanism TBD

Backends need to declare which global spans they could compile.
Whether this is a try-and-fail model (attempt compile, record
which succeeded) or a structural pre-check (analyze the span,
decide without compiling) is not yet decided. Try-and-fail is
simpler; pre-check is faster but requires backends to maintain a
separate capability model that may drift from their actual
compiler.

## What This Layer Does Not Do

- **It does not replace the v14 compiler.** v14 remains the CPU
  backend; the heterogeneous layer sits in front of it and other
  future backends. v14's internal partitioner, placer, and JIT
  continue to operate on a global-span-shaped subgraph rather than
  the whole nano-graph; the JIT ABI (`fn(buffer_ptrs)`) is
  unchanged.
- **It does not redo per-backend optimization.** Fusion, layout
  choice, tiling, internal lane/phase scheduling within a global
  span are each backend's own concern.
- **It is not a graph rewriter.** The global partitioner draws
  span and phase boundaries; it does not transform the underlying
  graph.
- **It is not a research-grade scheduler on day one.** Early
  scheduling will be simple (likely static assignment with optional
  profiling feedback). A real cost-model-driven scheduler waits for
  multiple backends and real timing data.
