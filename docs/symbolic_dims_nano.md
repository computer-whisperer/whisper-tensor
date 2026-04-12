# Symbolic Dimensions in the NanoGraph

## Core Concept

A NanoGraph is a DAG of **atoms**, where each atom computes a scalar
operation over its inputs.  Known dimensions of a tensor are expanded
into separate atoms — a tensor `[512]` becomes 512 atoms.  Dimensions
whose size is not known at graph-build time remain as **symbolic
dimensions** attached to the atom's group.

Each atom's value is an n-dimensional array whose shape is described
entirely by its symbolic dimensions:

- 0 sym\_dims: the atom is a plain scalar.
- 1 sym\_dim (e.g. `batch`): the atom is a 1-D vector of length
  `batch`.
- 2 sym\_dims (e.g. `seq_q, seq_k`): the atom is a 2-D matrix of
  shape `seq_q * seq_k`.

Known dimensions never appear as sym\_dims — they are already
expanded into the atom count.  Sym\_dims are the *remaining*
dimensions that could not be expanded because their size is unknown.

## GraphConstant

A **GraphConstant** is a graph-level unknown scalar: an integer value
that is not known when the graph is constructed but is filled in at
execution time.  The NanoGraph declares a set of GraphConstants and
references them wherever it needs a value that depends on runtime
context.

Each symbolic dimension axis of an atom group has a GraphConstant
that describes its runtime size.  Two different axes that reference
the same GraphConstant have the same runtime extent.  For example, in
self-attention the score matrix has two sym\_dim axes — both reference
the same GraphConstant `seq_len`, producing a `[seq_len, seq_len]`
matrix.

The axes themselves are fungible — their identity is determined by
their position in the group's sym\_dims list and the GraphConstant
that sizes them.  Any metadata about their original tensor-level
meaning (e.g. "this was dim 0 of the attention output") is a concern
of the input/output boundary (TensorAtomMapInfo), not of the
NanoGraph evaluation.

At runtime, the caller provides a concrete value for each
GraphConstant.  The evaluator and compiler use these to determine
iteration extents, buffer sizes, and addressing math.

## Relationship to InputRef

InputRef (Broadcast, Strided, Explicit) operates entirely in
**atom-ID space**.  It maps a consumer atom index `i` to a producer
atom ID.  Symbolic dimensions do not change InputRef — it continues
to address atoms by their known-dimension index.

What changes is that each atom referenced by an InputRef is no longer
a single scalar but an n-dimensional array.  The **op** must declare
how it operates across the sym\_dim axes of its inputs and output.

## Per-Input Sym\_dim Mapping

Each input of a group is a wrapper pairing an InputRef (atom-space
addressing) with a sym\_dim mapping (how the consumer's sym\_dim axes
correspond to the producer's sym\_dim axes):

- **Identity(producer\_axis)**: consumer axis `j` reads from producer
  axis `k` at the same index.  The producer's axis `k` must have the
  same GraphConstant (same runtime size) as the consumer's axis `j`.
- **Broadcast**: the producer does not vary along this consumer axis.
  The same value is used for every index.

The mapping is a `Vec<SymDimMap>` parallel to the consumer group's
sym\_dims.  It tells evaluation (and codegen) how to select the right
element from each input's sym\_dim array for a given point in the
output's sym\_dim space.

## Ops and Symbolic Dimensions

Elementwise ops (Binary, Unary, Select, Cast, Identity) operate
pointwise across the sym\_dim space: for each point in the output's
sym\_dim array, the op reads the corresponding point from each input
(via the per-input mapping) and computes the scalar result.

**SymReduce** is a `ScalarOp` variant that collapses one sym\_dim
axis by accumulating over it:

```rust
ScalarOp::SymReduce {
    kind: ReduceKind,       // Sum, Max, Mean, etc.
    axis: usize,            // index into the group's sym_dims list
    compute_dtype: NumericDType,
}
```

`axis` identifies which sym\_dim to reduce by position in the
group's `sym_dims` vector — not by GraphConstant, since the same
GraphConstant can appear on multiple axes (e.g. attention's
`[seq_len, seq_len]`).  The reduction extent is
`runtime(sym_dims[axis].graph_constant)`.  The output group has one
fewer sym\_dim than the input (the axis at that index is removed).

Existing known-dim **Reduce** is unchanged — it accumulates over a
fixed `reduce_count` of atoms and is orthogonal to sym\_dims.

## Memory Layout and Buffer Sizing

Symbolic dimensions are **innermost** (fastest-moving strides) in
memory.  Known dimensions (atom index) are outermost.  This ensures
that when the compiler partitions work across known dims (assigning
atoms to lanes), each lane gets a contiguous sym\_dim block with good
cache locality.

**Buffer sizing is compile-time, strides are runtime.**

Buffers are allocated at compile time using GraphConstant maximum
values:

```
slot_max_bytes = count * product(gc.max for gc in sym_dims) * elem_size
```

Slot byte offsets within the buffer are assigned by the placer at
compile time, spaced apart by worst-case sizes.  Unused capacity
between slots (when runtime values are smaller than maximums) is
acceptable — it is wasted buffer space, not wasted data stride.

Data strides within a slot are computed at **runtime** from actual
GraphConstant values:

```
atom_stride = runtime(gc0) * runtime(gc1) * elem_bytes
addr = slot_offset
     + atom_i * atom_stride
     + s0 * runtime(gc1) * elem_bytes
     + s1 * elem_bytes
```

This is critical: baking max-bound strides at compile time would
inflate each atom's footprint even when runtime values are small,
destroying cache coherency for the common case where symbolic dims
are much smaller than their maximum.  With runtime strides, the data
within each tensor is always packed tightly regardless of buffer
over-allocation.

Two spans sharing a buffer naturally agree on layout because they
reference the same GraphConstants and compute the same runtime
strides.

## Evaluation Model

Symbolic dimension iteration happens in the **inner loop**, not as
an outer dispatch.

The evaluator stores each group's values as a
`NumericTensor<DynRank>` where dimension 0 spans atom IDs and
remaining dimensions correspond to the group's sym\_dim axes in
order.  A group with no sym\_dims stores a rank-1 tensor of shape
`[count]`; a group with two sym\_dims stores a rank-3 tensor of
shape `[count, runtime(gc0), runtime(gc1)]`.

Evaluation of a group iterates sym\_dims inside the atom loop:

```
for atom_i in 0..count:
    for s0_val in 0..runtime(sym_dims[0].graph_constant):
        for s1_val in 0..runtime(sym_dims[1].graph_constant):
            // load each input at (atom_i's InputRef, s0_val, s1_val)
            // with per-input sym_dim mapping applied
            // compute op
            // store result at (atom_i, s0_val, s1_val)
```

The innermost sym\_dim should be the one that benefits most from
memory locality.  For weight-sharing patterns (batched inference),
the batch dim should be innermost so that weights are loaded once and
applied across all batch elements.

## Input/Output Boundary (TensorAtomMapInfo)

TensorAtomMapInfo handles the boundary between the tensor world
(flat element arrays with mixed known and symbolic dimensions) and
the atom world (atom-id + sym\_dim point).

A tensor with original shape `[batch, 512, seq_len, 64]` where
batch and seq\_len are symbolic has:

- known\_dims: [512, 64] → 32768 atoms
- sym\_dims: [batch, seq\_len] → per-atom 2-D array

To map flat tensor elements to `(atom_idx, sym_dim_point)` pairs,
TAMI records the original shape interleaving — which positions in
the full shape are known vs symbolic:

```
dim_layout: [Sym(0), Known(0), Sym(1), Known(1)]
```

Element `[b, i, s, j]` maps to atom `i * 64 + j` at sym point
`(b, s)`.  TAMI is the authoritative source for this mapping — it
carries enough information for the executor to load input tensors
into the evaluator's n-d atom stores and to extract output tensors
back.  `InputTensor` itself remains a plain atom range; the
sym\_dim structure is entirely described by TAMI's dim\_layout and
the group's sym\_dims.  The NanoGraph evaluation itself only works
with atom indices and sym\_dim points.

## Symbolic Dims in test\_set

Symbolic dimensions are a consequence of incomplete information at
lowering time.  Test infrastructure controls this by withholding
shape information when calling the lowering: a test that provides
full concrete shapes produces a graph with zero sym\_dims (all
known), while the same test withholding batch size produces a graph
where batch is symbolic.

The test API specifies which dimensions to withhold.  The test
runner sweeps across different sets of withheld information to
exercise the lowering system under varying levels of knowledge.
Concrete tensor data is always available for comparison — only the
*lowering's view* of the shape is partial.

## Limitations and Non-goals

- **Ragged/variable-length sequences**: this model assumes every
  instance of a GraphConstant has the same value everywhere in the
  graph.  Ragged tensors (where different batch elements have
  different seq\_lens) require padding + masking, which is standard
  practice in the target models.

- **Data-dependent output shapes** (NonZero, unique, NMS): the output
  atom count depends on tensor values, not on GraphConstants.  These
  are preprocessing ops, not part of standard transformer inference.

Note that "reshaping" between symbolic and known dims is not a
concern.  Known dims are spans of atom IDs addressed by InputRef;
symbolic dims are per-atom unknown-dimensional arrays handled by op
semantics and sym\_dim mappings.  These are orthogonal —
reinterpreting how known dims are addressed is just a different
InputRef, and reinterpreting how symbolic dims are accessed is just a
different sym\_dim mapping.  No data movement or conversion between
the two categories is needed.
