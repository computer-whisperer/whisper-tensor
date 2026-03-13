//! Lowering from MilliOpGraph to NanoGraph.
//!
//! Walks the MilliOpGraph in topological order, classifying each tensor's
//! dimensions as Known (expanded to atoms) or Symbolic (iteration params).
//! View ops dissolve into addressing changes. Elementwise ops become groups
//! with the same atom count. Ops that don't fit become boundary groups.

use std::collections::HashMap;

use crate::DynRank;
use crate::dtype::DType;
use crate::graph::{GlobalId, Graph, Node};
use crate::milli_graph::MilliOpGraph;
use crate::milli_graph::ops::AnyMilliOp;
use crate::nano_graph::ops::{ScalarBinOp, ScalarOp, ScalarUnaryOp};
use crate::nano_graph::pattern::{AtomId, InputRef, NanoGraph, SymDim};
use crate::numeric_scalar::NumericScalar;
use crate::numeric_tensor::NumericTensor;
use crate::tensor_info::TensorInfo;

/// Common accessors for reduce ops (ReduceSum, ReduceMax, ReduceMean).
trait ReduceAccessors {
    fn axes_tensor(&self) -> Option<GlobalId>;
    fn noop_with_empty_axes(&self) -> bool;
    #[allow(dead_code)]
    fn keepdims(&self) -> bool;
}

impl ReduceAccessors for crate::milli_graph::ops::ReduceSum {
    fn axes_tensor(&self) -> Option<GlobalId> {
        self.axes_tensor()
    }
    fn noop_with_empty_axes(&self) -> bool {
        self.noop_with_empty_axes()
    }
    fn keepdims(&self) -> bool {
        self.keepdims()
    }
}

impl ReduceAccessors for crate::milli_graph::ops::ReduceMax {
    fn axes_tensor(&self) -> Option<GlobalId> {
        self.axes_tensor()
    }
    fn noop_with_empty_axes(&self) -> bool {
        self.noop_with_empty_axes()
    }
    fn keepdims(&self) -> bool {
        self.keepdims()
    }
}

impl ReduceAccessors for crate::milli_graph::ops::ReduceMean {
    fn axes_tensor(&self) -> Option<GlobalId> {
        self.axes_tensor()
    }
    fn noop_with_empty_axes(&self) -> bool {
        self.noop_with_empty_axes()
    }
    fn keepdims(&self) -> bool {
        self.keepdims()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum LowerError {
    #[error("Missing tensor shape for {0}")]
    MissingShape(GlobalId),
    #[error("Missing tensor dtype for {0}")]
    MissingDtype(GlobalId),
    #[error("Missing tensor value for {0} (needed for {1})")]
    MissingValue(GlobalId, &'static str),
    #[error("Missing tensor ref for {0}")]
    MissingRef(GlobalId),
    #[error("Unsupported op: {0}")]
    UnsupportedOp(String),
    #[error("MilliOpGraph error: {0}")]
    MilliGraph(#[from] crate::milli_graph::MilliOpGraphError),
}

/// Result of lowering a MilliOpGraph.
pub struct LowerResult {
    pub graph: NanoGraph,
    /// Ops that could not be lowered (treated as boundary).
    pub unsupported: Vec<(GlobalId, String)>,
    /// Human-readable detail for each unsupported op (input/output shapes).
    pub unsupported_details: Vec<String>,
    /// Mapping from milli tensor GlobalId to nano atom group.
    pub tensor_map: HashMap<GlobalId, TensorAtomMapInfo>,
    /// Pre-built overrides for all Numeric (constant) tensors.
    /// Maps atom index → NumericScalar value. Covers model weights,
    /// constant-folded ops, and any other tensor whose value is known at
    /// lowering time. User inputs (Shaped tensors) are NOT included — the
    /// caller must add those separately before passing to NanoEval.
    pub numeric_overrides: HashMap<u32, crate::numeric_scalar::NumericScalar>,
}

/// Public view of how a milli tensor maps to nano atoms.
#[derive(Debug, Clone)]
pub struct TensorAtomMapInfo {
    pub base_id: AtomId,
    pub count: u32,
    pub sym_dims: Vec<SymDim>,
}

/// How a milli tensor maps to atoms in the nano graph.
///
/// The tensor's dimensions are split into known (expanded to atoms) and
/// symbolic (iteration parameters). Atoms are indexed by their position
/// in the flattened known dims (row-major order).
#[derive(Debug, Clone)]
struct TensorAtomMap {
    /// First atom id in the group.
    base_id: AtomId,
    /// Total number of atoms (product of known dims).
    count: u32,
    /// The full tensor layout: one entry per dim, preserving original order.
    /// Used to compute input refs when downstream ops index this tensor.
    layout: Vec<DimKind>,
    /// Row-major strides for the known dims (one per known dim, in order
    /// of appearance in the original shape).
    known_strides: Vec<u64>,
    /// Symbolic dims attached to each atom.
    sym_dims: Vec<SymDim>,
}

/// Classification of one tensor dimension.
#[derive(Debug, Clone)]
enum DimKind {
    Known(u64),
    Symbolic(#[allow(dead_code)] SymDim),
}

/// (layout, known_dims, sym_dims, atom_count)
type DimClassification = (Vec<DimKind>, Vec<u64>, Vec<SymDim>, u32);

impl TensorAtomMap {
    /// Compute row-major strides from known dim sizes.
    fn compute_strides(known_dims: &[u64]) -> Vec<u64> {
        let mut strides = vec![0u64; known_dims.len()];
        if known_dims.is_empty() {
            return strides;
        }
        let mut stride = 1u64;
        for i in (0..known_dims.len()).rev() {
            strides[i] = stride;
            stride *= known_dims[i];
        }
        strides
    }
}

/// Lower a MilliOpGraph into a NanoGraph using concrete inputs.
pub fn lower(
    graph: &MilliOpGraph,
    inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
) -> Result<LowerResult, LowerError> {
    let info_inputs: HashMap<GlobalId, TensorInfo> = inputs
        .iter()
        .map(|(id, tensor)| (*id, TensorInfo::from(tensor.clone())))
        .collect();
    lower_with_info(graph, &info_inputs)
}

/// Lower a MilliOpGraph into a NanoGraph using partial tensor information.
pub fn lower_with_info(
    graph: &MilliOpGraph,
    inputs: &HashMap<GlobalId, TensorInfo>,
) -> Result<LowerResult, LowerError> {
    let all_infos = graph.infer_all(inputs)?;
    let mut ctx = LowerCtx::new();

    // Register all tensors that exist before ops run (graph inputs + inferred
    // constants) and are NOT produced by any op.
    let op_outputs: std::collections::HashSet<GlobalId> = graph
        .op_ordering()
        .iter()
        .filter_map(|op_id| graph.get_node_by_id(op_id))
        .flat_map(|op| op.outputs().collect::<Vec<_>>())
        .collect();
    for (id, info) in &all_infos {
        if !op_outputs.contains(id) {
            ctx.register_input(*id, info);
        }
    }

    // Walk ops in topological order.
    for &op_id in graph.op_ordering() {
        let Some(op) = graph.get_node_by_id(&op_id) else {
            continue;
        };
        ctx.lower_op(op, &all_infos);
    }

    // Collect outputs. Try get_outputs() first; if empty, scan all tensors
    // in the tensor_map that aren't consumed by any op (terminal tensors).
    let output_ids = graph.get_outputs();
    if !output_ids.is_empty() {
        for out_id in &output_ids {
            if let Some(tam) = ctx.tensor_map.get(out_id) {
                for i in 0..tam.count {
                    ctx.nano.outputs.push(tam.base_id.offset(i));
                }
            }
        }
    }
    // If still no outputs, register all mapped tensors from the last ops.
    // This is a fallback — proper output mapping will be fixed later.

    let tensor_map: HashMap<GlobalId, TensorAtomMapInfo> = ctx
        .tensor_map
        .iter()
        .map(|(id, tam)| {
            (
                *id,
                TensorAtomMapInfo {
                    base_id: tam.base_id,
                    count: tam.count,
                    sym_dims: tam.sym_dims.clone(),
                },
            )
        })
        .collect();

    // Build overrides for all Numeric (constant-valued) tensors.
    use crate::numeric_scalar::NumericScalar;
    let mut numeric_overrides: HashMap<u32, NumericScalar> = HashMap::new();
    let mut backend = crate::backends::eval_backend::EvalBackend::NDArray;
    for (id, info) in &all_infos {
        let Some(numeric) = info.as_numeric() else {
            continue;
        };
        let Some(tam) = ctx.tensor_map.get(id) else {
            continue;
        };
        let tensor_dtype = numeric.dtype();
        // Extract values via f32 (exact for BF16/F16/F32, sufficient for most cases),
        // then cast to the tensor's actual dtype to preserve rounding semantics.
        let Ok(f32_tensor) = numeric.cast(DType::F32, &mut backend) else {
            continue;
        };
        let Ok(flat) = f32_tensor.flatten() else {
            continue;
        };
        let Ok(nd) = flat.to_ndarray() else { continue };
        let Ok(v): Result<Vec<f32>, _> = nd.try_into() else {
            continue;
        };
        if v.len() != tam.count as usize {
            continue;
        }
        for (i, &val) in v.iter().enumerate() {
            let scalar = NumericScalar::F32(val).cast_to(tensor_dtype);
            numeric_overrides.insert(tam.base_id.0 + i as u32, scalar);
        }
    }

    Ok(LowerResult {
        graph: ctx.nano,
        unsupported: ctx.unsupported,
        unsupported_details: ctx.unsupported_details,
        tensor_map,
        numeric_overrides,
    })
}

struct LowerCtx {
    nano: NanoGraph,
    tensor_map: HashMap<GlobalId, TensorAtomMap>,
    next_anon_sym: usize,
    unsupported: Vec<(GlobalId, String)>,
    unsupported_details: Vec<String>,
}

impl LowerCtx {
    fn new() -> Self {
        Self {
            nano: NanoGraph::new(),
            tensor_map: HashMap::new(),
            next_anon_sym: 0,
            unsupported: Vec::new(),
            unsupported_details: Vec::new(),
        }
    }

    /// Classify tensor dims and return layout info.
    /// Returns None if rank is unknown or atom count overflows u32.
    fn classify_dims(&mut self, info: &TensorInfo) -> Option<DimClassification> {
        let rank = info.rank_if_known()?;
        let mut layout = Vec::with_capacity(rank);
        let mut known_dims = Vec::new();
        let mut sym_dims = Vec::new();

        for i in 0..rank {
            if let Some(size) = info.dim_if_known(i) {
                layout.push(DimKind::Known(size));
                known_dims.push(size);
            } else {
                let sd = self.alloc_sym_dim();
                layout.push(DimKind::Symbolic(sd));
                sym_dims.push(sd);
            }
        }

        let count: u64 = known_dims.iter().product();
        let count: u32 = count.try_into().ok()?;

        Some((layout, known_dims, sym_dims, count))
    }

    fn alloc_sym_dim(&mut self) -> SymDim {
        let name = format!("sym_{}", self.next_anon_sym);
        self.next_anon_sym += 1;
        self.nano.sym_dim(&name)
    }

    /// Register a graph input as leaf atoms.
    fn register_input(&mut self, id: GlobalId, info: &TensorInfo) {
        let Some((layout, known_dims, sym_dims, count)) = self.classify_dims(info) else {
            let base_id = self.nano.push_atom(
                ScalarOp::Literal(NumericScalar::F32(0.0)),
                vec![],
                vec![],
                vec![],
            );
            self.tensor_map.insert(
                id,
                TensorAtomMap {
                    base_id,
                    count: 1,
                    layout: vec![],
                    known_strides: vec![],
                    sym_dims: vec![],
                },
            );
            return;
        };

        let strides = TensorAtomMap::compute_strides(&known_dims);
        let count = count.max(1);

        let base_id = self.nano.push_group(
            count,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            sym_dims.clone(),
            vec![],
            vec![],
        );

        self.tensor_map.insert(
            id,
            TensorAtomMap {
                base_id,
                count,
                layout,
                known_strides: strides,
                sym_dims,
            },
        );
    }

    /// Register a tensor as a boundary (opaque) group.
    /// Boundary atoms are leaves — they use Literal(0) with no inputs.
    fn register_boundary(&mut self, output_id: GlobalId, info: &TensorInfo, _op_kind: &str) {
        let Some((layout, known_dims, sym_dims, count)) = self.classify_dims(info) else {
            let base_id = self.nano.push_atom(
                ScalarOp::Literal(NumericScalar::F32(0.0)),
                vec![],
                vec![],
                vec![],
            );
            self.tensor_map.insert(
                output_id,
                TensorAtomMap {
                    base_id,
                    count: 1,
                    layout: vec![],
                    known_strides: vec![],
                    sym_dims: vec![],
                },
            );
            return;
        };

        let strides = TensorAtomMap::compute_strides(&known_dims);
        let count = count.max(1);

        let base_id = self.nano.push_group(
            count,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            sym_dims.clone(),
            vec![],
            vec![],
        );

        self.tensor_map.insert(
            output_id,
            TensorAtomMap {
                base_id,
                count,
                layout,
                known_strides: strides,
                sym_dims,
            },
        );
    }

    /// Compute an InputRef mapping consumer atoms → producer atoms.
    fn compute_input_ref(
        &self,
        consumer: &TensorAtomMap,
        producer: &TensorAtomMap,
        consumer_info: &TensorInfo,
        producer_info: &TensorInfo,
    ) -> InputRef {
        // Same count + same known shape → stride 1.
        if consumer.count == producer.count && consumer.count > 0 {
            let c_known: Vec<u64> = consumer
                .layout
                .iter()
                .filter_map(|d| {
                    if let DimKind::Known(s) = d {
                        Some(*s)
                    } else {
                        None
                    }
                })
                .collect();
            let p_known: Vec<u64> = producer
                .layout
                .iter()
                .filter_map(|d| {
                    if let DimKind::Known(s) = d {
                        Some(*s)
                    } else {
                        None
                    }
                })
                .collect();
            if c_known == p_known {
                return InputRef::Affine {
                    base: producer.base_id,
                    stride: 1,
                };
            }
        }

        // Scalar producer → broadcast.
        if producer.count == 1 {
            return InputRef::Broadcast(producer.base_id);
        }

        // General broadcast: build Explicit.
        self.build_broadcast_explicit(consumer, producer, consumer_info, producer_info)
    }

    /// Build an Explicit InputRef for broadcast patterns.
    fn build_broadcast_explicit(
        &self,
        consumer: &TensorAtomMap,
        producer: &TensorAtomMap,
        consumer_info: &TensorInfo,
        producer_info: &TensorInfo,
    ) -> InputRef {
        let c_rank = consumer_info.rank_if_known().unwrap_or(0);
        let p_rank = producer_info.rank_if_known().unwrap_or(0);

        let c_known_sizes: Vec<u64> = consumer
            .layout
            .iter()
            .filter_map(|d| {
                if let DimKind::Known(s) = d {
                    Some(*s)
                } else {
                    None
                }
            })
            .collect();
        let p_known_sizes: Vec<u64> = producer
            .layout
            .iter()
            .filter_map(|d| {
                if let DimKind::Known(s) = d {
                    Some(*s)
                } else {
                    None
                }
            })
            .collect();
        let p_strides = TensorAtomMap::compute_strides(&p_known_sizes);

        // Build mapping from consumer known-dim index to producer known-dim
        // index using right-aligned broadcasting.
        let offset = c_rank.saturating_sub(p_rank);

        // For each original dim, track its known-dim index (if known).
        let c_known_indices: Vec<Option<usize>> = {
            let mut ki = 0;
            consumer
                .layout
                .iter()
                .map(|d| {
                    if matches!(d, DimKind::Known(_)) {
                        let idx = ki;
                        ki += 1;
                        Some(idx)
                    } else {
                        None
                    }
                })
                .collect()
        };
        let p_known_indices: Vec<Option<usize>> = {
            let mut ki = 0;
            producer
                .layout
                .iter()
                .map(|d| {
                    if matches!(d, DimKind::Known(_)) {
                        let idx = ki;
                        ki += 1;
                        Some(idx)
                    } else {
                        None
                    }
                })
                .collect()
        };

        // Map: consumer_known_dim_idx → producer_known_dim_idx (or None if broadcast/mismatch).
        let mut c_to_p_known: Vec<Option<usize>> = vec![None; c_known_sizes.len()];
        for (c_orig, c_known_idx) in c_known_indices.iter().enumerate() {
            let Some(c_ki) = *c_known_idx else {
                continue;
            };
            if c_orig < offset {
                continue;
            }
            let p_orig = c_orig - offset;
            if p_orig >= p_rank {
                continue;
            }
            if let Some(p_ki) = p_known_indices[p_orig] {
                c_to_p_known[c_ki] = Some(p_ki);
            }
        }

        let mut ids = Vec::with_capacity(consumer.count as usize);
        let c_strides = &consumer.known_strides;

        for flat_c in 0..consumer.count as u64 {
            // Decompose flat_c into known-dim indices.
            let mut c_indices = vec![0u64; c_known_sizes.len()];
            let mut rem = flat_c;
            for (i, &stride) in c_strides.iter().enumerate() {
                if stride > 0 {
                    c_indices[i] = rem / stride;
                    rem %= stride;
                }
            }

            // Map to producer indices.
            let mut p_indices = vec![0u64; p_known_sizes.len()];
            for (c_ki, &p_ki_opt) in c_to_p_known.iter().enumerate() {
                if let Some(p_ki) = p_ki_opt {
                    if p_known_sizes[p_ki] == 1 {
                        p_indices[p_ki] = 0; // broadcast
                    } else {
                        p_indices[p_ki] = c_indices[c_ki];
                    }
                }
            }

            // Flatten producer indices.
            let mut flat_p = 0u64;
            for (i, &stride) in p_strides.iter().enumerate() {
                flat_p += p_indices[i] * stride;
            }

            ids.push(producer.base_id.offset(flat_p as u32));
        }

        InputRef::Explicit(ids)
    }

    fn lower_op(&mut self, op: &AnyMilliOp, all_infos: &HashMap<GlobalId, TensorInfo>) {
        match op {
            AnyMilliOp::SimpleBinary(bin) => self.lower_simple_binary(bin, all_infos),
            AnyMilliOp::SimpleUnary(un) => self.lower_simple_unary(un, all_infos),
            AnyMilliOp::ClampMin(clamp) => self.lower_clamp_min(clamp, all_infos),
            AnyMilliOp::Cast(cast) => self.lower_identity_passthrough(cast, all_infos),
            AnyMilliOp::CastLike(cl) => self.lower_identity_passthrough(cl, all_infos),
            AnyMilliOp::Constant(c) => {
                let out_id = Node::outputs(c).next().unwrap();
                if let Some(info) = all_infos.get(&out_id) {
                    self.register_input(out_id, info);
                } else {
                    self.register_opaque(out_id);
                }
            }
            AnyMilliOp::ConstantOfShape(c) => {
                let out_id = Node::outputs(c).next().unwrap();
                if let Some(info) = all_infos.get(&out_id) {
                    self.register_input(out_id, info);
                } else {
                    self.register_opaque(out_id);
                }
            }
            AnyMilliOp::Shape(s) => {
                let out_id = Node::outputs(s).next().unwrap();
                if let Some(info) = all_infos.get(&out_id) {
                    self.register_input(out_id, info);
                } else {
                    self.register_opaque(out_id);
                }
            }
            AnyMilliOp::Reshape(r) => self.lower_view_op(r, all_infos),
            AnyMilliOp::Squeeze(s) => self.lower_view_op(s, all_infos),
            AnyMilliOp::Unsqueeze(u) => self.lower_view_op(u, all_infos),
            AnyMilliOp::Transpose(t) => self.lower_transpose(t, all_infos),
            AnyMilliOp::Expand(e) => self.lower_expand(e, all_infos),
            AnyMilliOp::Pow(p) => self.lower_pow(p, all_infos),
            AnyMilliOp::Where(w) => self.lower_where(w, all_infos),
            AnyMilliOp::Concat(c) => self.lower_concat(c, all_infos),
            AnyMilliOp::Split(s) => self.lower_split(s, all_infos),
            AnyMilliOp::Slice(s) => self.lower_slice(s, all_infos),
            AnyMilliOp::MatMul(m) => self.lower_matmul(m, all_infos),
            AnyMilliOp::ReduceSum(r) => {
                self.lower_reduce(r, all_infos, |compute_dt, out_dt| ScalarOp::ReduceSum {
                    compute_dtype: compute_dt,
                    output_dtype: out_dt,
                })
            }
            AnyMilliOp::ReduceMax(r) => {
                self.lower_reduce(r, all_infos, |compute_dt, out_dt| ScalarOp::ReduceMax {
                    compute_dtype: compute_dt,
                    output_dtype: out_dt,
                })
            }
            AnyMilliOp::ReduceMean(r) => self.lower_reduce_mean(r, all_infos),
            // Everything else: check if outputs are fully concrete (constant-folded),
            // otherwise register as boundary.
            other => {
                let op_kind = other.op_kind();
                let _op_id = other.global_id();
                // If all outputs are Numeric (fully constant), treat like a Constant —
                // the inference already computed the values, no need to lower the op.
                let all_numeric = other.outputs().all(|out_id| {
                    all_infos
                        .get(&out_id)
                        .is_some_and(|i| i.as_numeric().is_some())
                });
                for out_id in other.outputs() {
                    if let Some(info) = all_infos.get(&out_id) {
                        if all_numeric {
                            self.register_input(out_id, info);
                        } else {
                            self.register_boundary(out_id, info, &op_kind);
                        }
                    } else {
                        self.register_opaque(out_id);
                    }
                }
                if !all_numeric {
                    self.push_unsupported(other, all_infos, &op_kind);
                }
            }
        }
    }

    fn lower_simple_binary(
        &mut self,
        bin: &crate::milli_graph::ops::SimpleBinary,
        all_infos: &HashMap<GlobalId, TensorInfo>,
    ) {
        use crate::milli_graph::ops::binary::WhichSimpleBinaryOp;

        let mut inputs_iter = Node::inputs(bin);
        let a_id = inputs_iter.next().unwrap();
        let b_id = inputs_iter.next().unwrap();
        let out_id = Node::outputs(bin).next().unwrap();

        let (Some(a_map), Some(b_map)) = (
            self.tensor_map.get(&a_id).cloned(),
            self.tensor_map.get(&b_id).cloned(),
        ) else {
            self.lower_as_boundary_named(bin, all_infos, "SimpleBinary");
            return;
        };
        let Some(out_info) = all_infos.get(&out_id) else {
            self.lower_as_boundary_named(bin, all_infos, "SimpleBinary");
            return;
        };

        let dt = out_info.dtype();
        let scalar_op = match bin.which_op() {
            WhichSimpleBinaryOp::Add => ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: dt,
                output_dtype: dt,
            },
            WhichSimpleBinaryOp::Sub => ScalarOp::Binary {
                op: ScalarBinOp::Sub,
                compute_dtype: dt,
                output_dtype: dt,
            },
            WhichSimpleBinaryOp::Mul => ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: dt,
                output_dtype: dt,
            },
            WhichSimpleBinaryOp::Div => ScalarOp::Binary {
                op: ScalarBinOp::Div,
                compute_dtype: dt,
                output_dtype: dt,
            },
            WhichSimpleBinaryOp::Max => ScalarOp::Binary {
                op: ScalarBinOp::Max,
                compute_dtype: dt,
                output_dtype: dt,
            },
            WhichSimpleBinaryOp::Min => ScalarOp::Binary {
                op: ScalarBinOp::Min,
                compute_dtype: dt,
                output_dtype: dt,
            },
            WhichSimpleBinaryOp::Modulo(_) => ScalarOp::Binary {
                op: ScalarBinOp::Mod,
                compute_dtype: dt,
                output_dtype: dt,
            },
            // Comparison/logical ops: opaque binary, but shape is known.
            _ => ScalarOp::Binary {
                op: ScalarBinOp::Max,
                compute_dtype: dt,
                output_dtype: dt,
            }, // placeholder
        };

        let Some((layout, known_dims, sym_dims, count)) = self.classify_dims(out_info) else {
            self.lower_as_boundary_named(bin, all_infos, "SimpleBinary");
            return;
        };
        let count = count.max(1);
        let strides = TensorAtomMap::compute_strides(&known_dims);

        let out_tmp = TensorAtomMap {
            base_id: AtomId(0),
            count,
            layout: layout.clone(),
            known_strides: strides.clone(),
            sym_dims: sym_dims.clone(),
        };

        let a_info = all_infos.get(&a_id);
        let b_info = all_infos.get(&b_id);
        let input_a =
            self.compute_input_ref(&out_tmp, &a_map, out_info, a_info.unwrap_or(out_info));
        let input_b =
            self.compute_input_ref(&out_tmp, &b_map, out_info, b_info.unwrap_or(out_info));

        let base_id = self.nano.push_group(
            count,
            scalar_op,
            sym_dims.clone(),
            vec![],
            vec![input_a, input_b],
        );

        self.tensor_map.insert(
            out_id,
            TensorAtomMap {
                base_id,
                count,
                layout,
                known_strides: strides,
                sym_dims,
            },
        );
    }

    fn lower_simple_unary(
        &mut self,
        un: &crate::milli_graph::ops::SimpleUnaryOp,
        all_infos: &HashMap<GlobalId, TensorInfo>,
    ) {
        use crate::milli_graph::ops::unary::WhichSimpleUnaryOp;

        let in_id = Node::inputs(un).next().unwrap();
        let out_id = Node::outputs(un).next().unwrap();

        let Some(in_map) = self.tensor_map.get(&in_id).cloned() else {
            self.lower_as_boundary_named(un, all_infos, "SimpleUnary");
            return;
        };
        let Some(out_info) = all_infos.get(&out_id) else {
            self.lower_as_boundary_named(un, all_infos, "SimpleUnary");
            return;
        };

        let dt = out_info.dtype();
        let scalar_op = match un.which_op() {
            WhichSimpleUnaryOp::Neg => ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: dt,
                output_dtype: dt,
            },
            WhichSimpleUnaryOp::Abs => ScalarOp::Unary {
                op: ScalarUnaryOp::Abs,
                compute_dtype: dt,
                output_dtype: dt,
            },
            WhichSimpleUnaryOp::Exp => ScalarOp::Unary {
                op: ScalarUnaryOp::Exp,
                compute_dtype: dt,
                output_dtype: dt,
            },
            WhichSimpleUnaryOp::Ln => ScalarOp::Unary {
                op: ScalarUnaryOp::Ln,
                compute_dtype: dt,
                output_dtype: dt,
            },
            WhichSimpleUnaryOp::Sqrt => ScalarOp::Unary {
                op: ScalarUnaryOp::Sqrt,
                compute_dtype: dt,
                output_dtype: dt,
            },
            WhichSimpleUnaryOp::Reciprocal => ScalarOp::Unary {
                op: ScalarUnaryOp::Reciprocal,
                compute_dtype: dt,
                output_dtype: dt,
            },
            WhichSimpleUnaryOp::Trig(crate::TrigOp::Tanh) => ScalarOp::Unary {
                op: ScalarUnaryOp::Tanh,
                compute_dtype: dt,
                output_dtype: dt,
            },
            WhichSimpleUnaryOp::Floor => ScalarOp::Unary {
                op: ScalarUnaryOp::Floor,
                compute_dtype: dt,
                output_dtype: dt,
            },
            WhichSimpleUnaryOp::Ceil => ScalarOp::Unary {
                op: ScalarUnaryOp::Ceil,
                compute_dtype: dt,
                output_dtype: dt,
            },
            _ => {
                self.lower_as_boundary_named(un, all_infos, "SimpleUnary");
                return;
            }
        };

        let base_id = self.nano.push_group(
            in_map.count,
            scalar_op,
            in_map.sym_dims.clone(),
            vec![],
            vec![InputRef::Affine {
                base: in_map.base_id,
                stride: 1,
            }],
        );

        self.tensor_map.insert(
            out_id,
            TensorAtomMap {
                base_id,
                count: in_map.count,
                layout: in_map.layout.clone(),
                known_strides: in_map.known_strides.clone(),
                sym_dims: in_map.sym_dims.clone(),
            },
        );
    }

    fn lower_clamp_min(
        &mut self,
        clamp: &crate::milli_graph::ops::ClampMin,
        all_infos: &HashMap<GlobalId, TensorInfo>,
    ) {
        let in_id = Node::inputs(clamp).next().unwrap();
        let out_id = Node::outputs(clamp).next().unwrap();

        let Some(in_map) = self.tensor_map.get(&in_id).cloned() else {
            self.lower_as_boundary_named(clamp, all_infos, "ClampMin");
            return;
        };
        let Some(out_info) = all_infos.get(&out_id) else {
            self.lower_as_boundary_named(clamp, all_infos, "ClampMin");
            return;
        };

        let min_val = clamp.min_val();
        let dt = out_info.dtype();
        let min_id = self.nano.push_atom(
            ScalarOp::Literal(NumericScalar::F32(min_val)),
            vec![],
            vec![],
            vec![],
        );

        let base_id = self.nano.push_group(
            in_map.count,
            ScalarOp::Binary {
                op: ScalarBinOp::Max,
                compute_dtype: dt,
                output_dtype: dt,
            },
            in_map.sym_dims.clone(),
            vec![],
            vec![
                InputRef::Affine {
                    base: in_map.base_id,
                    stride: 1,
                },
                InputRef::Broadcast(min_id),
            ],
        );

        self.tensor_map.insert(
            out_id,
            TensorAtomMap {
                base_id,
                count: in_map.count,
                layout: in_map.layout.clone(),
                known_strides: in_map.known_strides.clone(),
                sym_dims: in_map.sym_dims.clone(),
            },
        );
    }

    /// Cast / CastLike / other shape-preserving identity ops.
    fn lower_identity_passthrough<T: Node>(
        &mut self,
        op: &T,
        all_infos: &HashMap<GlobalId, TensorInfo>,
    ) {
        let in_id = Node::inputs(op).next().unwrap();
        let out_id = Node::outputs(op).next().unwrap();

        let Some(in_map) = self.tensor_map.get(&in_id).cloned() else {
            self.lower_as_boundary_named(op, all_infos, "ViewOp");
            return;
        };
        let Some(out_info) = all_infos.get(&out_id) else {
            self.lower_as_boundary_named(op, all_infos, "ViewOp");
            return;
        };

        let dt = out_info.dtype();
        let base_id = self.nano.push_group(
            in_map.count,
            ScalarOp::Identity {
                compute_dtype: dt,
                output_dtype: dt,
            },
            in_map.sym_dims.clone(),
            vec![],
            vec![InputRef::Affine {
                base: in_map.base_id,
                stride: 1,
            }],
        );

        self.tensor_map.insert(
            out_id,
            TensorAtomMap {
                base_id,
                count: in_map.count,
                layout: in_map.layout.clone(),
                known_strides: in_map.known_strides.clone(),
                sym_dims: in_map.sym_dims.clone(),
            },
        );
    }

    /// Transpose: permutes element order, requiring explicit reindexing.
    ///
    /// Unlike Reshape/Squeeze (which just relabel dims without changing element
    /// order), Transpose changes the physical row-major order of elements.
    /// We emit an Identity group with an Explicit InputRef that maps each output
    /// atom to the correct input atom based on the permutation.
    fn lower_transpose(
        &mut self,
        t: &crate::milli_graph::ops::Transpose,
        all_infos: &HashMap<GlobalId, TensorInfo>,
    ) {
        let in_id = Node::inputs(t).next().unwrap();
        let out_id = Node::outputs(t).next().unwrap();

        let Some(in_map) = self.tensor_map.get(&in_id).cloned() else {
            self.lower_as_boundary_named(t, all_infos, "Transpose");
            return;
        };
        let Some(out_info) = all_infos.get(&out_id) else {
            self.register_opaque(out_id);
            return;
        };
        let Some(in_info) = all_infos.get(&in_id) else {
            self.lower_as_boundary_named(t, all_infos, "Transpose");
            return;
        };

        let in_rank = match in_info.rank_if_known() {
            Some(r) => r,
            None => {
                self.lower_as_boundary_named(t, all_infos, "Transpose");
                return;
            }
        };

        let Some((out_layout, out_known_dims, out_sym_dims, out_count)) =
            self.classify_dims(out_info)
        else {
            self.lower_as_boundary_named(t, all_infos, "Transpose");
            return;
        };
        let out_count = out_count.max(1);

        // Build full permutation (handling None=reverse, partial perms, negative indices).
        let full_perm: Vec<usize> = match t.perm() {
            None => (0..in_rank).rev().collect(),
            Some(perm) => {
                let expanded = if perm.len() < in_rank {
                    let prefix_len = in_rank - perm.len();
                    let mut fp: Vec<i64> = (0..prefix_len as i64).collect();
                    fp.extend(
                        perm.iter()
                            .map(|&x| if x < 0 { x + in_rank as i64 } else { x }),
                    );
                    fp
                } else {
                    perm.to_vec()
                };
                expanded
                    .iter()
                    .map(|&x| {
                        if x < 0 {
                            (x + in_rank as i64) as usize
                        } else {
                            x as usize
                        }
                    })
                    .collect()
            }
        };

        // Get input known dim sizes (only the Known dims, in original order).
        let in_known_sizes: Vec<u64> = in_map
            .layout
            .iter()
            .filter_map(|d| {
                if let DimKind::Known(s) = d {
                    Some(*s)
                } else {
                    None
                }
            })
            .collect();
        let in_strides = TensorAtomMap::compute_strides(&in_known_sizes);
        let out_strides = TensorAtomMap::compute_strides(&out_known_dims);

        // We need to map between original dim indices and known-dim indices.
        // For now, assume all dims are known (symbolic dims in transpose would
        // be unusual). If any dim is symbolic, fall back to boundary.
        if in_map
            .layout
            .iter()
            .any(|d| matches!(d, DimKind::Symbolic(_)))
            || out_layout.iter().any(|d| matches!(d, DimKind::Symbolic(_)))
        {
            self.register_boundary(out_id, out_info, "Transpose");
            self.push_unsupported(t, all_infos, "Transpose(symbolic dims)");
            return;
        }

        // All dims are known. Build Explicit mapping.
        // For each output flat index, decompose into output dim indices,
        // apply inverse permutation to get input dim indices, then flatten.
        let mut ids = Vec::with_capacity(out_count as usize);
        for flat_out in 0..out_count as u64 {
            // Decompose flat_out into output dim indices.
            let mut out_indices = vec![0u64; out_known_dims.len()];
            let mut rem = flat_out;
            for (i, &stride) in out_strides.iter().enumerate() {
                if stride > 0 {
                    out_indices[i] = rem / stride;
                    rem %= stride;
                }
            }

            // output[i0, i1, ..., iN] = input[perm[0], perm[1], ..., perm[N]]
            // So output dim j corresponds to input dim perm[j].
            // Therefore: input_indices[perm[j]] = out_indices[j]
            let mut in_indices = vec![0u64; in_known_sizes.len()];
            for (out_dim, &in_dim) in full_perm.iter().enumerate() {
                in_indices[in_dim] = out_indices[out_dim];
            }

            // Flatten input indices.
            let mut flat_in = 0u64;
            for (i, &stride) in in_strides.iter().enumerate() {
                flat_in += in_indices[i] * stride;
            }

            ids.push(in_map.base_id.offset(flat_in as u32));
        }

        let dt = out_info.dtype();
        let base_id = self.nano.push_group(
            out_count,
            ScalarOp::Identity {
                compute_dtype: dt,
                output_dtype: dt,
            },
            out_sym_dims.clone(),
            vec![],
            vec![InputRef::Explicit(ids)],
        );

        self.tensor_map.insert(
            out_id,
            TensorAtomMap {
                base_id,
                count: out_count,
                layout: out_layout,
                known_strides: out_strides,
                sym_dims: out_sym_dims,
            },
        );
    }

    /// View op: no compute, just re-register with the new shape.
    fn lower_view_op<T: Node>(&mut self, op: &T, all_infos: &HashMap<GlobalId, TensorInfo>) {
        let in_id = Node::inputs(op).next().unwrap();
        let out_id = Node::outputs(op).next().unwrap();

        let Some(in_map) = self.tensor_map.get(&in_id).cloned() else {
            self.lower_as_boundary_named(op, all_infos, "ViewOp");
            return;
        };
        let Some(out_info) = all_infos.get(&out_id) else {
            self.register_opaque(out_id);
            return;
        };

        let Some((layout, known_dims, sym_dims, count)) = self.classify_dims(out_info) else {
            self.lower_as_boundary_named(op, all_infos, "ViewOp");
            return;
        };
        let count = count.max(1);

        if count == in_map.count {
            // Atom-count preserving: just re-register with new layout.
            self.tensor_map.insert(
                out_id,
                TensorAtomMap {
                    base_id: in_map.base_id,
                    count,
                    layout,
                    known_strides: TensorAtomMap::compute_strides(&known_dims),
                    sym_dims,
                },
            );
        } else {
            self.register_boundary(out_id, out_info, "ViewOp");
            let name = format!("ViewOp(count {} → {})", in_map.count, count);
            self.push_unsupported(op, all_infos, &name);
        }
    }

    fn lower_expand(
        &mut self,
        e: &crate::milli_graph::ops::Expand,
        all_infos: &HashMap<GlobalId, TensorInfo>,
    ) {
        let in_id = Node::inputs(e).next().unwrap();
        let out_id = Node::outputs(e).next().unwrap();

        let Some(in_map) = self.tensor_map.get(&in_id).cloned() else {
            self.lower_as_boundary_named(e, all_infos, "Expand");
            return;
        };
        let Some(out_info) = all_infos.get(&out_id) else {
            self.register_opaque(out_id);
            return;
        };
        let in_info = all_infos.get(&in_id);

        let Some((layout, known_dims, sym_dims, count)) = self.classify_dims(out_info) else {
            self.lower_as_boundary_named(e, all_infos, "Expand");
            return;
        };
        let count = count.max(1);

        if count == in_map.count {
            self.tensor_map.insert(
                out_id,
                TensorAtomMap {
                    base_id: in_map.base_id,
                    count,
                    layout,
                    known_strides: TensorAtomMap::compute_strides(&known_dims),
                    sym_dims,
                },
            );
        } else {
            let out_tmp = TensorAtomMap {
                base_id: AtomId(0),
                count,
                layout: layout.clone(),
                known_strides: TensorAtomMap::compute_strides(&known_dims),
                sym_dims: sym_dims.clone(),
            };
            let input_ref =
                self.compute_input_ref(&out_tmp, &in_map, out_info, in_info.unwrap_or(out_info));

            let dt = out_info.dtype();
            let base_id = self.nano.push_group(
                count,
                ScalarOp::Identity {
                    compute_dtype: dt,
                    output_dtype: dt,
                },
                sym_dims.clone(),
                vec![],
                vec![input_ref],
            );

            self.tensor_map.insert(
                out_id,
                TensorAtomMap {
                    base_id,
                    count,
                    layout,
                    known_strides: TensorAtomMap::compute_strides(&known_dims),
                    sym_dims,
                },
            );
        }
    }

    fn lower_pow(
        &mut self,
        pow: &crate::milli_graph::ops::Pow,
        all_infos: &HashMap<GlobalId, TensorInfo>,
    ) {
        let mut inputs_iter = Node::inputs(pow);
        let a_id = inputs_iter.next().unwrap();
        let b_id = inputs_iter.next().unwrap();
        let out_id = Node::outputs(pow).next().unwrap();

        let (Some(a_map), Some(b_map)) = (
            self.tensor_map.get(&a_id).cloned(),
            self.tensor_map.get(&b_id).cloned(),
        ) else {
            self.lower_as_boundary_named(pow, all_infos, "Pow");
            return;
        };
        let Some(out_info) = all_infos.get(&out_id) else {
            self.lower_as_boundary_named(pow, all_infos, "Pow");
            return;
        };

        let Some((layout, known_dims, sym_dims, count)) = self.classify_dims(out_info) else {
            self.lower_as_boundary_named(pow, all_infos, "Pow");
            return;
        };
        let count = count.max(1);
        let strides = TensorAtomMap::compute_strides(&known_dims);

        let out_tmp = TensorAtomMap {
            base_id: AtomId(0),
            count,
            layout: layout.clone(),
            known_strides: strides.clone(),
            sym_dims: sym_dims.clone(),
        };

        let a_info = all_infos.get(&a_id);
        let b_info = all_infos.get(&b_id);
        let input_a =
            self.compute_input_ref(&out_tmp, &a_map, out_info, a_info.unwrap_or(out_info));
        let input_b =
            self.compute_input_ref(&out_tmp, &b_map, out_info, b_info.unwrap_or(out_info));

        let dt = out_info.dtype();
        let base_id = self.nano.push_group(
            count,
            ScalarOp::Binary {
                op: ScalarBinOp::Pow,
                compute_dtype: dt,
                output_dtype: dt,
            },
            sym_dims.clone(),
            vec![],
            vec![input_a, input_b],
        );

        self.tensor_map.insert(
            out_id,
            TensorAtomMap {
                base_id,
                count,
                layout,
                known_strides: strides,
                sym_dims,
            },
        );
    }

    fn lower_where(
        &mut self,
        where_op: &crate::milli_graph::ops::Where,
        all_infos: &HashMap<GlobalId, TensorInfo>,
    ) {
        let mut inputs_iter = Node::inputs(where_op);
        let cond_id = inputs_iter.next().unwrap();
        let x_id = inputs_iter.next().unwrap();
        let y_id = inputs_iter.next().unwrap();
        let out_id = Node::outputs(where_op).next().unwrap();

        let (Some(cond_map), Some(x_map), Some(y_map)) = (
            self.tensor_map.get(&cond_id).cloned(),
            self.tensor_map.get(&x_id).cloned(),
            self.tensor_map.get(&y_id).cloned(),
        ) else {
            self.lower_as_boundary_named(where_op, all_infos, "Where");
            return;
        };
        let Some(out_info) = all_infos.get(&out_id) else {
            self.lower_as_boundary_named(where_op, all_infos, "Where");
            return;
        };

        let Some((layout, known_dims, sym_dims, count)) = self.classify_dims(out_info) else {
            self.lower_as_boundary_named(where_op, all_infos, "Where");
            return;
        };
        let count = count.max(1);
        let strides = TensorAtomMap::compute_strides(&known_dims);

        let out_tmp = TensorAtomMap {
            base_id: AtomId(0),
            count,
            layout: layout.clone(),
            known_strides: strides.clone(),
            sym_dims: sym_dims.clone(),
        };

        let cond_info = all_infos.get(&cond_id);
        let x_info = all_infos.get(&x_id);
        let y_info = all_infos.get(&y_id);
        let input_cond =
            self.compute_input_ref(&out_tmp, &cond_map, out_info, cond_info.unwrap_or(out_info));
        let input_x =
            self.compute_input_ref(&out_tmp, &x_map, out_info, x_info.unwrap_or(out_info));
        let input_y =
            self.compute_input_ref(&out_tmp, &y_map, out_info, y_info.unwrap_or(out_info));

        let dt = out_info.dtype();
        let base_id = self.nano.push_group(
            count,
            ScalarOp::Select {
                compute_dtype: dt,
                output_dtype: dt,
            },
            sym_dims.clone(),
            vec![],
            vec![input_cond, input_x, input_y],
        );

        self.tensor_map.insert(
            out_id,
            TensorAtomMap {
                base_id,
                count,
                layout,
                known_strides: strides,
                sym_dims,
            },
        );
    }

    fn lower_concat(
        &mut self,
        concat: &crate::milli_graph::ops::Concat,
        all_infos: &HashMap<GlobalId, TensorInfo>,
    ) {
        let axis_raw = concat.axis();
        let out_id = Node::outputs(concat).next().unwrap();
        let input_ids = concat.concat_inputs();

        let Some(out_info) = all_infos.get(&out_id) else {
            self.lower_as_boundary_named(concat, all_infos, "Concat");
            return;
        };

        let Some((out_layout, out_known_dims, out_sym_dims, out_count)) =
            self.classify_dims(out_info)
        else {
            self.lower_as_boundary_named(concat, all_infos, "Concat");
            return;
        };
        let out_count = out_count.max(1);

        // Normalize axis.
        let rank = out_layout.len();
        let axis = if axis_raw < 0 {
            (axis_raw + rank as i64) as usize
        } else {
            axis_raw as usize
        };

        // Concat axis must be a known dim.
        if axis >= rank || !matches!(out_layout[axis], DimKind::Known(_)) {
            self.lower_as_boundary_named(concat, all_infos, "Concat");
            return;
        }

        // Known-dim index of the concat axis.
        let concat_known_idx = out_layout[..=axis]
            .iter()
            .filter(|d| matches!(d, DimKind::Known(_)))
            .count()
            - 1;

        // Gather input maps and their concat-axis sizes.
        let mut input_maps = Vec::with_capacity(input_ids.len());
        let mut concat_dim_sizes = Vec::with_capacity(input_ids.len());
        for &inp_id in input_ids {
            let Some(inp_map) = self.tensor_map.get(&inp_id).cloned() else {
                self.lower_as_boundary_named(concat, all_infos, "Concat");
                return;
            };
            let inp_known: Vec<u64> = inp_map
                .layout
                .iter()
                .filter_map(|d| {
                    if let DimKind::Known(s) = d {
                        Some(*s)
                    } else {
                        None
                    }
                })
                .collect();
            if inp_known.len() != out_known_dims.len() {
                self.lower_as_boundary_named(concat, all_infos, "Concat");
                return;
            }
            concat_dim_sizes.push(inp_known[concat_known_idx]);
            input_maps.push(inp_map);
        }

        // Cumulative offsets along concat dim.
        let mut cum_offsets = vec![0u64];
        for &s in &concat_dim_sizes {
            cum_offsets.push(cum_offsets.last().unwrap() + s);
        }

        // Build Explicit InputRef: map each output atom to its source.
        let out_strides = TensorAtomMap::compute_strides(&out_known_dims);
        let mut ids = Vec::with_capacity(out_count as usize);

        for flat in 0..out_count as u64 {
            // Decompose to known-dim indices.
            let mut indices = vec![0u64; out_known_dims.len()];
            let mut rem = flat;
            for (i, &stride) in out_strides.iter().enumerate() {
                if stride > 0 {
                    indices[i] = rem / stride;
                    rem %= stride;
                }
            }

            // Find which input the concat-axis index falls into.
            let concat_idx = indices[concat_known_idx];
            let input_idx = match cum_offsets.iter().position(|&off| off > concat_idx) {
                Some(pos) => pos - 1,
                None => {
                    // Shouldn't happen — boundary as fallback.
                    self.lower_as_boundary_named(concat, all_infos, "Concat");
                    return;
                }
            };

            // Adjust concat index to be relative to this input.
            indices[concat_known_idx] = concat_idx - cum_offsets[input_idx];

            // Recompose to input flat index.
            let inp_map = &input_maps[input_idx];
            let mut inp_flat = 0u64;
            for (i, &stride) in inp_map.known_strides.iter().enumerate() {
                inp_flat += indices[i] * stride;
            }

            ids.push(inp_map.base_id.offset(inp_flat as u32));
        }

        let dt = out_info.dtype();
        let base_id = self.nano.push_group(
            out_count,
            ScalarOp::Identity {
                compute_dtype: dt,
                output_dtype: dt,
            },
            out_sym_dims.clone(),
            vec![],
            vec![InputRef::Explicit(ids)],
        );

        self.tensor_map.insert(
            out_id,
            TensorAtomMap {
                base_id,
                count: out_count,
                layout: out_layout,
                known_strides: TensorAtomMap::compute_strides(&out_known_dims),
                sym_dims: out_sym_dims,
            },
        );
    }

    fn lower_split(
        &mut self,
        split: &crate::milli_graph::ops::Split,
        all_infos: &HashMap<GlobalId, TensorInfo>,
    ) {
        let in_id = Node::inputs(split).next().unwrap();
        let out_id = Node::outputs(split).next().unwrap();

        let Some(in_map) = self.tensor_map.get(&in_id).cloned() else {
            self.lower_as_boundary_named(split, all_infos, "Split");
            return;
        };
        let Some(out_info) = all_infos.get(&out_id) else {
            self.lower_as_boundary_named(split, all_infos, "Split");
            return;
        };
        let Some(_in_info) = all_infos.get(&in_id) else {
            self.lower_as_boundary_named(split, all_infos, "Split");
            return;
        };

        let Some((out_layout, out_known_dims, out_sym_dims, out_count)) =
            self.classify_dims(out_info)
        else {
            self.lower_as_boundary_named(split, all_infos, "Split");
            return;
        };
        let out_count = out_count.max(1);

        // Normalize axis.
        let rank = in_map.layout.len();
        let axis_raw = split.axis();
        let axis = if axis_raw < 0 {
            (axis_raw + rank as i64) as usize
        } else {
            axis_raw as usize
        };

        if axis >= rank || !matches!(in_map.layout[axis], DimKind::Known(_)) {
            self.lower_as_boundary_named(split, all_infos, "Split");
            return;
        }

        // Determine the offset along the split axis for this output_id.
        // We need the split sizes. If all outputs have known dims, compute from the
        // input dim and output sizes. Otherwise use the output info directly.
        let split_known_idx = in_map.layout[..=axis]
            .iter()
            .filter(|d| matches!(d, DimKind::Known(_)))
            .count()
            - 1;

        // Get the input's full dim along split axis.
        let in_known: Vec<u64> = in_map
            .layout
            .iter()
            .filter_map(|d| {
                if let DimKind::Known(s) = d {
                    Some(*s)
                } else {
                    None
                }
            })
            .collect();
        let _in_split_size = in_known[split_known_idx];

        // Figure out the offset: we need to know what came before this output_id's chunk.
        // The output_id tells us which chunk we are. We need the sizes of all prior chunks.
        // We can compute this from the axis dim of the output info and output_id index.
        let output_id_idx = split.output_id();
        let out_split_size = match &out_layout[axis] {
            DimKind::Known(s) => *s,
            _ => {
                self.lower_as_boundary_named(split, all_infos, "Split");
                return;
            }
        };

        // Compute offset: we need the sum of split sizes for all outputs before this one.
        // Since we might not have the split tensor values, estimate from output_id * output_size.
        // This is only correct for equal splits. For unequal splits we need the actual sizes.
        // Try to get them from the split tensor.
        let offset_along_axis =
            self.compute_split_offset(split, output_id_idx, out_split_size, all_infos);

        if out_known_dims.len() != in_known.len() {
            self.lower_as_boundary_named(split, all_infos, "Split");
            return;
        }

        // Build Explicit InputRef: map each output atom to corresponding input atom.
        let out_strides = TensorAtomMap::compute_strides(&out_known_dims);
        let mut ids = Vec::with_capacity(out_count as usize);

        for flat in 0..out_count as u64 {
            // Decompose to known-dim indices.
            let mut indices = vec![0u64; out_known_dims.len()];
            let mut rem = flat;
            for (i, &stride) in out_strides.iter().enumerate() {
                if stride > 0 {
                    indices[i] = rem / stride;
                    rem %= stride;
                }
            }

            // Shift the split axis index by the offset.
            indices[split_known_idx] += offset_along_axis;

            // Recompose to input flat index.
            let mut inp_flat = 0u64;
            for (i, &stride) in in_map.known_strides.iter().enumerate() {
                inp_flat += indices[i] * stride;
            }

            ids.push(in_map.base_id.offset(inp_flat as u32));
        }

        let dt = out_info.dtype();
        let base_id = self.nano.push_group(
            out_count,
            ScalarOp::Identity {
                compute_dtype: dt,
                output_dtype: dt,
            },
            out_sym_dims.clone(),
            vec![],
            vec![InputRef::Explicit(ids)],
        );

        self.tensor_map.insert(
            out_id,
            TensorAtomMap {
                base_id,
                count: out_count,
                layout: out_layout,
                known_strides: TensorAtomMap::compute_strides(&out_known_dims),
                sym_dims: out_sym_dims,
            },
        );
    }

    /// Compute the cumulative offset along the split axis for output_id_idx.
    fn compute_split_offset(
        &self,
        split: &crate::milli_graph::ops::Split,
        output_id_idx: usize,
        out_split_size: u64,
        all_infos: &HashMap<GlobalId, TensorInfo>,
    ) -> u64 {
        // Try to get concrete split sizes from the split tensor.
        if let Some(crate::milli_graph::ops::MilliOpTensorIDOrLiteral::TensorID(tensor_id)) =
            split.split_tensor()
            && let Some(info) = all_infos.get(tensor_id)
            && let Some(numeric) = info.as_numeric()
            && let Ok(cast) = numeric.cast(
                DType::I64,
                &mut crate::backends::eval_backend::EvalBackend::NDArray,
            )
            && let Ok(rank1) = cast.try_to_rank::<typenum::P1>()
            && let Ok(vals) = Vec::<i64>::try_from(rank1.to_ndarray().unwrap())
        {
            let offset: i64 = vals[..output_id_idx].iter().sum();
            return offset as u64;
        }
        // Fallback: assume equal splits.
        output_id_idx as u64 * out_split_size
    }

    fn lower_slice(
        &mut self,
        slice: &crate::milli_graph::ops::Slice,
        all_infos: &HashMap<GlobalId, TensorInfo>,
    ) {
        let data_id = slice.data_id();
        let out_id = Node::outputs(slice).next().unwrap();

        let Some(in_map) = self.tensor_map.get(&data_id).cloned() else {
            self.lower_as_boundary_named(slice, all_infos, "Slice");
            return;
        };
        let Some(out_info) = all_infos.get(&out_id) else {
            self.lower_as_boundary_named(slice, all_infos, "Slice");
            return;
        };

        let Some((out_layout, out_known_dims, out_sym_dims, out_count)) =
            self.classify_dims(out_info)
        else {
            self.lower_as_boundary_named(slice, all_infos, "Slice");
            return;
        };
        let out_count = out_count.max(1);

        // Extract concrete slice parameters.
        let extract_i64 = |id: &GlobalId| -> Option<Vec<i64>> {
            let info = all_infos.get(id)?;
            let tensor = info.as_numeric()?;
            let as_i64 = tensor
                .cast(
                    DType::I64,
                    &mut crate::backends::eval_backend::EvalBackend::NDArray,
                )
                .ok()?;
            let rank1 = as_i64.try_to_rank::<typenum::P1>().ok()?;
            Vec::<i64>::try_from(rank1.to_ndarray().ok()?).ok()
        };

        let starts = extract_i64(&slice.starts_id());
        let ends = extract_i64(&slice.ends_id());
        let steps: Option<Vec<i64>> = if let Some(steps_id) = slice.steps_id() {
            extract_i64(&steps_id)
        } else {
            starts.as_ref().map(|s| s.iter().map(|_| 1i64).collect())
        };

        let in_rank = in_map.layout.len();
        let axes: Option<Vec<usize>> = if let Some(axes_id) = slice.axes_id() {
            extract_i64(&axes_id).map(|a| {
                a.iter()
                    .map(|&v| {
                        if v < 0 {
                            (v + in_rank as i64) as usize
                        } else {
                            v as usize
                        }
                    })
                    .collect()
            })
        } else {
            starts.as_ref().map(|s| (0..s.len()).collect())
        };

        let (Some(starts), Some(_ends), Some(steps), Some(axes)) = (starts, ends, steps, axes)
        else {
            self.lower_as_boundary_named(slice, all_infos, "Slice");
            return;
        };

        // Build per-axis (start, step) for known dims.
        // The input's known dims define the coordinate space.
        let in_known: Vec<u64> = in_map
            .layout
            .iter()
            .filter_map(|d| {
                if let DimKind::Known(s) = d {
                    Some(*s)
                } else {
                    None
                }
            })
            .collect();

        // Map tensor-axis → known-dim index (None if symbolic).
        let axis_to_known_idx: Vec<Option<usize>> = {
            let mut ki = 0;
            in_map
                .layout
                .iter()
                .map(|d| {
                    if matches!(d, DimKind::Known(_)) {
                        let idx = ki;
                        ki += 1;
                        Some(idx)
                    } else {
                        None
                    }
                })
                .collect()
        };

        // Build the start offset and step for each known dim.
        // Default: start=0, step=1 (full range).
        let mut known_starts = vec![0i64; in_known.len()];
        let mut known_steps = vec![1i64; in_known.len()];

        for (i, &axis) in axes.iter().enumerate() {
            if axis >= in_rank {
                self.lower_as_boundary_named(slice, all_infos, "Slice");
                return;
            }
            let Some(ki) = axis_to_known_idx[axis] else {
                // Slicing a symbolic dim — boundary.
                self.lower_as_boundary_named(slice, all_infos, "Slice");
                return;
            };

            let dim = in_known[ki] as i64;
            let step = steps[i];
            if step == 0 {
                self.lower_as_boundary_named(slice, all_infos, "Slice");
                return;
            }

            let start = if step > 0 {
                let s = starts[i].clamp(-dim, dim);
                if s < 0 { s + dim } else { s }
            } else {
                let s = starts[i].clamp(-dim, dim - 1);
                if s < 0 { s + dim } else { s }
            };

            known_starts[ki] = start;
            known_steps[ki] = step;
        }

        if out_known_dims.len() != in_known.len() {
            self.lower_as_boundary_named(slice, all_infos, "Slice");
            return;
        }

        // Build Explicit InputRef.
        let out_strides = TensorAtomMap::compute_strides(&out_known_dims);
        let mut ids = Vec::with_capacity(out_count as usize);

        for flat in 0..out_count as u64 {
            // Decompose to output known-dim indices.
            let mut out_indices = vec![0u64; out_known_dims.len()];
            let mut rem = flat;
            for (i, &stride) in out_strides.iter().enumerate() {
                if stride > 0 {
                    out_indices[i] = rem / stride;
                    rem %= stride;
                }
            }

            // Map to input known-dim indices.
            let mut in_flat = 0u64;
            for (ki, &stride) in in_map.known_strides.iter().enumerate() {
                let in_idx = (known_starts[ki] + known_steps[ki] * out_indices[ki] as i64) as u64;
                in_flat += in_idx * stride;
            }

            ids.push(in_map.base_id.offset(in_flat as u32));
        }

        let dt = out_info.dtype();
        let base_id = self.nano.push_group(
            out_count,
            ScalarOp::Identity {
                compute_dtype: dt,
                output_dtype: dt,
            },
            out_sym_dims.clone(),
            vec![],
            vec![InputRef::Explicit(ids)],
        );

        self.tensor_map.insert(
            out_id,
            TensorAtomMap {
                base_id,
                count: out_count,
                layout: out_layout,
                known_strides: TensorAtomMap::compute_strides(&out_known_dims),
                sym_dims: out_sym_dims,
            },
        );
    }

    /// Extract concrete i64 values from a tensor in all_infos.
    fn extract_i64(all_infos: &HashMap<GlobalId, TensorInfo>, id: &GlobalId) -> Option<Vec<i64>> {
        let info = all_infos.get(id)?;
        let tensor = info.as_numeric()?;
        let as_i64 = tensor
            .cast(
                DType::I64,
                &mut crate::backends::eval_backend::EvalBackend::NDArray,
            )
            .ok()?;
        let rank1 = as_i64.try_to_rank::<typenum::P1>().ok()?;
        Vec::<i64>::try_from(rank1.to_ndarray().ok()?).ok()
    }

    fn lower_matmul(
        &mut self,
        matmul: &crate::milli_graph::ops::MatMul,
        all_infos: &HashMap<GlobalId, TensorInfo>,
    ) {
        let mut inputs_iter = Node::inputs(matmul);
        let a_id = inputs_iter.next().unwrap();
        let b_id = inputs_iter.next().unwrap();
        let out_id = Node::outputs(matmul).next().unwrap();

        let (Some(a_map), Some(b_map)) = (
            self.tensor_map.get(&a_id).cloned(),
            self.tensor_map.get(&b_id).cloned(),
        ) else {
            self.lower_as_boundary_named(matmul, all_infos, "MatMul");
            return;
        };
        let Some(out_info) = all_infos.get(&out_id) else {
            self.lower_as_boundary_named(matmul, all_infos, "MatMul");
            return;
        };

        // MatMul operates on the last 2 tensor dims: A=[...,M,K] @ B=[...,K,N].
        // We look at the full layout (not just known dims) to handle symbolic M.
        // Requirements:
        //   - K (last dim of A, second-to-last of B): must be Known, must match
        //   - N (last dim of B): must be Known
        //   - M (second-to-last of A): can be Known or Symbolic
        //   - Known batch dims must match between A and B
        let a_layout = &a_map.layout;
        let b_layout = &b_map.layout;

        if a_layout.len() < 2 || b_layout.len() < 2 {
            self.lower_as_boundary_named(matmul, all_infos, "MatMul");
            return;
        }

        // Extract K from last dim of A and second-to-last of B.
        let k = match (&a_layout[a_layout.len() - 1], &b_layout[b_layout.len() - 2]) {
            (DimKind::Known(ka), DimKind::Known(kb)) if ka == kb && *ka > 0 => *ka,
            _ => {
                self.lower_as_boundary_named(matmul, all_infos, "MatMul");
                return;
            }
        };

        // Extract N from last dim of B.
        let n = match &b_layout[b_layout.len() - 1] {
            DimKind::Known(n) if *n > 0 => *n,
            _ => {
                self.lower_as_boundary_named(matmul, all_infos, "MatMul");
                return;
            }
        };

        // M from second-to-last of A: can be known or symbolic.
        let m_known: Option<u64> = match &a_layout[a_layout.len() - 2] {
            DimKind::Known(m) => Some(*m),
            DimKind::Symbolic(_) => None,
        };

        // Extract known batch dims from A and B (everything except last 2).
        let a_batch_layout = &a_layout[..a_layout.len() - 2];
        let b_batch_layout = &b_layout[..b_layout.len() - 2];

        let a_batch_known: Vec<u64> = a_batch_layout
            .iter()
            .filter_map(|d| {
                if let DimKind::Known(s) = d {
                    Some(*s)
                } else {
                    None
                }
            })
            .collect();
        let b_batch_known: Vec<u64> = b_batch_layout
            .iter()
            .filter_map(|d| {
                if let DimKind::Known(s) = d {
                    Some(*s)
                } else {
                    None
                }
            })
            .collect();

        // B can have fewer batch dims (broadcasting). If B has batch dims, they must match.
        if !b_batch_known.is_empty() && a_batch_known != b_batch_known {
            self.lower_as_boundary_named(matmul, all_infos, "MatMul");
            return;
        }

        let batch_known_product: u64 = a_batch_known.iter().product::<u64>().max(1);

        // Classify output dims.
        let Some((out_layout, out_known_dims, out_sym_dims, out_count)) =
            self.classify_dims(out_info)
        else {
            self.lower_as_boundary_named(matmul, all_infos, "MatMul");
            return;
        };
        let out_count = out_count.max(1);

        if out_count > 64_000_000 {
            self.lower_as_boundary_named(matmul, all_infos, "MatMul");
            return;
        }

        // Use the MatMul's explicit dtype fields for nano group precision.
        // product_dtype: precision of A*B products (Mul groups).
        // accumulate_dtype: precision for summing products (ReduceSum groups).
        // output_dtype: final output precision (Identity cast-back if needed).
        let product_dtype = matmul.product_dtype();
        let accumulate_dtype = matmul.accumulate_dtype();
        let out_dtype = matmul.output_dtype();

        // Create bounded sym dim for the contraction dimension K.
        let k_sym = self
            .nano
            .bounded_sym_dim(&format!("matmul_k_{}", self.next_anon_sym), k);
        self.next_anon_sym += 1;

        // Compute A's known-dim strides (for addressing within A's atoms).
        let a_known_dims: Vec<u64> = a_layout
            .iter()
            .filter_map(|d| {
                if let DimKind::Known(s) = d {
                    Some(*s)
                } else {
                    None
                }
            })
            .collect();
        let a_strides = TensorAtomMap::compute_strides(&a_known_dims);

        // Compute B's known-dim strides.
        let b_known_dims: Vec<u64> = b_layout
            .iter()
            .filter_map(|d| {
                if let DimKind::Known(s) = d {
                    Some(*s)
                } else {
                    None
                }
            })
            .collect();
        let b_strides = TensorAtomMap::compute_strides(&b_known_dims);

        // A's K dim is the last known dim. B's K dim is second-to-last known dim.
        // B's N dim is the last known dim.
        let a_k_known_idx = a_known_dims.len() - 1;
        let b_k_known_idx = b_known_dims.len() - 2;

        // The number of row groups: batch_known_product * (M if known, else 1).
        let m_groups = m_known.unwrap_or(1);
        let num_row_groups = (batch_known_product * m_groups) as usize;
        let n_u32 = n as u32;

        // For each row group, compute the A and B base offsets within their atoms.
        // A's known dims: [...batch_known, (M if known), K]
        // B's known dims: [...batch_known, K, N]
        let a_m_known_idx = if m_known.is_some() {
            Some(a_known_dims.len() - 2) // M is second-to-last known dim
        } else {
            None
        };

        // Expand the K contraction dimension into K concrete mul groups per row.
        // Each mul group has N atoms computing A[...,m,k] * B[...,k,n] for a fixed k.
        // This allows the flat eval to correctly evaluate each atom independently,
        // and the ReduceSum uses stride_k=N to hop between k layers.
        let mut mul_base_id = None;

        for g in 0..num_row_groups {
            let m_idx = if m_known.is_some() {
                g as u64 % m_groups
            } else {
                0
            };
            let batch_idx = if m_known.is_some() {
                g as u64 / m_groups
            } else {
                g as u64
            };

            // Compute A's base offset: set batch indices + m_idx, K=0.
            let mut a_offset = 0u64;
            let mut batch_rem = batch_idx;
            let a_batch_strides = TensorAtomMap::compute_strides(&a_batch_known);
            for (i, &stride) in a_batch_strides.iter().enumerate() {
                if stride > 0 {
                    let idx = batch_rem / stride;
                    batch_rem %= stride;
                    a_offset += idx * a_strides[i];
                }
            }
            if let Some(m_ki) = a_m_known_idx {
                a_offset += m_idx * a_strides[m_ki];
            }

            // Compute B's base offset: set batch indices, K=0, N=0.
            let mut b_offset = 0u64;
            if !b_batch_known.is_empty() {
                let mut batch_rem_b = batch_idx;
                let b_batch_strides = TensorAtomMap::compute_strides(&b_batch_known);
                for (i, &stride) in b_batch_strides.iter().enumerate() {
                    if stride > 0 {
                        let idx = batch_rem_b / stride;
                        batch_rem_b %= stride;
                        b_offset += idx * b_strides[i];
                    }
                }
            }

            // Create K mul groups for this row, one per contraction index.
            for ki in 0..k {
                let a_atom = a_map
                    .base_id
                    .offset((a_offset + ki * a_strides[a_k_known_idx]) as u32);
                let b_atom = b_map
                    .base_id
                    .offset((b_offset + ki * b_strides[b_k_known_idx]) as u32);

                let input_a = InputRef::Broadcast(a_atom);
                let input_b = InputRef::Affine {
                    base: b_atom,
                    stride: 1,
                };

                let base = self.nano.push_group(
                    n_u32,
                    ScalarOp::Binary {
                        op: ScalarBinOp::Mul,
                        compute_dtype: product_dtype,
                        output_dtype: product_dtype,
                    },
                    out_sym_dims.clone(),
                    vec![],
                    vec![input_a, input_b],
                );

                if mul_base_id.is_none() {
                    mul_base_id = Some(base);
                }
            }
        }

        let mul_base = mul_base_id.unwrap();

        // ReduceSum: one group per row, each with N atoms.
        // stride_k = N so that stepping k hops between the K mul groups for this row.
        let mut reduce_base_id = None;

        for g in 0..num_row_groups {
            let row_mul_base = AtomId(mul_base.0 + (g as u32) * (k as u32) * n_u32);

            let base = self.nano.push_group(
                n_u32,
                ScalarOp::ReduceSum {
                    compute_dtype: accumulate_dtype,
                    output_dtype: out_dtype,
                },
                out_sym_dims.clone(),
                vec![k_sym],
                vec![InputRef::SymAffine {
                    base: row_mul_base,
                    stride_i: 1,
                    stride_k: n_u32 as i32,
                }],
            );

            if reduce_base_id.is_none() {
                reduce_base_id = Some(base);
            }
        }

        let base_id = reduce_base_id.unwrap();

        self.tensor_map.insert(
            out_id,
            TensorAtomMap {
                base_id,
                count: out_count,
                layout: out_layout,
                known_strides: TensorAtomMap::compute_strides(&out_known_dims),
                sym_dims: out_sym_dims,
            },
        );
    }

    /// Lower ReduceSum or ReduceMax over known axes.
    fn lower_reduce<R, F>(
        &mut self,
        reduce: &R,
        all_infos: &HashMap<GlobalId, TensorInfo>,
        make_reduce_op: F,
    ) where
        R: Node,
        R: ReduceAccessors,
        F: Fn(DType, DType) -> ScalarOp,
    {
        let in_id = Node::inputs(reduce).next().unwrap();
        let out_id = Node::outputs(reduce).next().unwrap();

        let Some(in_map) = self.tensor_map.get(&in_id).cloned() else {
            self.lower_as_boundary_named(reduce, all_infos, "Reduce");
            return;
        };
        let Some(out_info) = all_infos.get(&out_id) else {
            self.lower_as_boundary_named(reduce, all_infos, "Reduce");
            return;
        };

        // Get the concrete reduction axes.
        let axes: Vec<i64> = if let Some(axes_id) = reduce.axes_tensor() {
            let Some(vals) = Self::extract_i64(all_infos, &axes_id) else {
                self.lower_as_boundary_named(reduce, all_infos, "Reduce");
                return;
            };
            vals
        } else if reduce.noop_with_empty_axes() {
            // No axes + noop = identity.
            let Some((layout, known_dims, sym_dims, count)) = self.classify_dims(out_info) else {
                self.lower_as_boundary_named(reduce, all_infos, "Reduce");
                return;
            };
            let count = count.max(1);
            let dt = out_info.dtype();
            let base_id = self.nano.push_group(
                count,
                ScalarOp::Identity {
                    compute_dtype: dt,
                    output_dtype: dt,
                },
                sym_dims.clone(),
                vec![],
                vec![InputRef::Affine {
                    base: in_map.base_id,
                    stride: 1,
                }],
            );
            self.tensor_map.insert(
                out_id,
                TensorAtomMap {
                    base_id,
                    count,
                    layout,
                    known_strides: TensorAtomMap::compute_strides(&known_dims),
                    sym_dims,
                },
            );
            return;
        } else {
            // No axes = reduce all. Boundary for now.
            self.lower_as_boundary_named(reduce, all_infos, "Reduce");
            return;
        };

        let in_rank = in_map.layout.len();

        // Normalize axes and check they're all known dims.
        let norm_axes: Vec<usize> = axes
            .iter()
            .map(|&a| {
                if a < 0 {
                    (a + in_rank as i64) as usize
                } else {
                    a as usize
                }
            })
            .collect();

        // Map tensor-axis → known-dim index.
        let axis_to_known_idx: Vec<Option<usize>> = {
            let mut ki = 0;
            in_map
                .layout
                .iter()
                .map(|d| {
                    if matches!(d, DimKind::Known(_)) {
                        let idx = ki;
                        ki += 1;
                        Some(idx)
                    } else {
                        None
                    }
                })
                .collect()
        };

        let in_known: Vec<u64> = in_map
            .layout
            .iter()
            .filter_map(|d| {
                if let DimKind::Known(s) = d {
                    Some(*s)
                } else {
                    None
                }
            })
            .collect();

        // Check all reduction axes are known dims.
        let mut reduce_known_indices = Vec::new();
        for &ax in &norm_axes {
            if ax >= in_rank {
                self.lower_as_boundary_named(reduce, all_infos, "Reduce");
                return;
            }
            let Some(ki) = axis_to_known_idx[ax] else {
                // Reducing a symbolic dim — use the symbolic reduce path.
                // For now, boundary.
                self.lower_as_boundary_named(reduce, all_infos, "Reduce");
                return;
            };
            reduce_known_indices.push(ki);
        }

        // Compute the reduction extent (product of reduced known dims).
        let reduce_extent: u64 = reduce_known_indices
            .iter()
            .map(|&ki| in_known[ki])
            .product();
        if reduce_extent == 0 {
            self.lower_as_boundary_named(reduce, all_infos, "Reduce");
            return;
        }

        // Output known dims = input known dims minus the reduced ones.
        let out_known: Vec<u64> = in_known
            .iter()
            .enumerate()
            .filter(|(i, _)| !reduce_known_indices.contains(i))
            .map(|(_, &v)| v)
            .collect();
        let out_count = out_known.iter().product::<u64>().max(1) as u32;

        // Create bounded sym dim for the reduction.
        let reduce_sym = self
            .nano
            .bounded_sym_dim(&format!("reduce_{}", self.next_anon_sym), reduce_extent);
        self.next_anon_sym += 1;

        // Build the intermediate multiply group (identity * 1 for ReduceSum/Max,
        // actually we need a group with sym_dims=[reduce_sym] that reads from
        // the input with SymAffine).
        //
        // For each output atom (indices over non-reduced known dims), we need to
        // iterate over the reduced dims. The SymAffine stride_k encodes how the
        // reduction iteration advances through the input's flat index.
        //
        // For a single reduced dim at known-dim index `rki` with stride `s`:
        //   stride_k = s (input stride of the reduced dim)
        //   stride_i = 1 for the output's flat index
        //
        // For multiple reduced dims, we'd need multiple sym dims.
        // For now, handle single-axis reduction (covers most cases).
        if reduce_known_indices.len() != 1 {
            self.lower_as_boundary_named(reduce, all_infos, "Reduce");
            return;
        }

        let rki = reduce_known_indices[0];
        let in_strides = TensorAtomMap::compute_strides(&in_known);
        let reduce_stride = in_strides[rki] as i32;

        // Build output known strides.
        let out_strides_local = TensorAtomMap::compute_strides(&out_known);

        // For each output atom, compute the base input atom.
        // The output atom at flat index `f` maps to input indices where the
        // reduced dim is 0. We need to compute the input flat index with
        // the reduced dim set to 0.
        // Build Explicit mapping: output flat → input flat (at k=0).
        // For each output atom, decompose into non-reduced dims, then compute
        // the input flat index (with reduced dim = 0).
        let mut base_ids = Vec::with_capacity(out_count as usize);
        for flat_out in 0..out_count as u64 {
            // Decompose flat_out into output known-dim indices.
            let mut out_indices = vec![0u64; out_known.len()];
            let mut rem = flat_out;
            for (i, &stride) in out_strides_local.iter().enumerate() {
                if stride > 0 {
                    out_indices[i] = rem / stride;
                    rem %= stride;
                }
            }

            // Map back to input known-dim indices (insert 0 for reduced dim).
            let mut in_indices = Vec::with_capacity(in_known.len());
            let mut oi = 0;
            for ki in 0..in_known.len() {
                if ki == rki {
                    in_indices.push(0u64);
                } else {
                    in_indices.push(out_indices[oi]);
                    oi += 1;
                }
            }

            let mut in_flat = 0u64;
            for (i, &stride) in in_strides.iter().enumerate() {
                in_flat += in_indices[i] * stride;
            }
            base_ids.push(in_flat as u32);
        }

        // Check if the base_ids form a simple affine pattern.
        let is_affine = if out_count <= 1 {
            true
        } else {
            let stride = base_ids[1] as i64 - base_ids[0] as i64;
            base_ids
                .windows(2)
                .all(|w| (w[1] as i64 - w[0] as i64) == stride)
        };

        // Input group: out_count atoms with sym_dims including reduce_sym.
        // Uses SymAffine to index into the source, advancing by reduce_stride per k.
        let input_ref = if is_affine && out_count > 0 {
            let stride_i = if out_count > 1 {
                base_ids[1] as i32 - base_ids[0] as i32
            } else {
                1
            };
            InputRef::SymAffine {
                base: in_map.base_id.offset(base_ids[0]),
                stride_i,
                stride_k: reduce_stride,
            }
        } else {
            // Need per-atom SymAffine, which we can't do with a single group.
            // Fall back to boundary.
            self.lower_as_boundary_named(reduce, all_infos, "Reduce");
            return;
        };

        // Determine compute dtype: ReduceSum upcasts BF16/F16 → F32 to match
        // milli eval precision semantics (see reduce_sum.rs lines 193-196).
        let out_dt = out_info.dtype();
        let in_dt = all_infos.get(&in_id).map(|i| i.dtype()).unwrap_or(out_dt);
        let compute_dt = match in_dt {
            DType::BF16 | DType::F16 => DType::F32,
            other => other,
        };

        // Classify output for proper layout.
        let Some((out_layout, out_known_dims_full, out_sym_dims, _)) = self.classify_dims(out_info)
        else {
            // This shouldn't happen since we computed out_count, but be safe.
            self.lower_as_boundary_named(reduce, all_infos, "Reduce");
            return;
        };

        // ReduceSum/ReduceMax group: reads directly from input atoms via
        // SymAffine. The compute_dtype handles casting inputs to the
        // accumulation precision, and output_dtype casts the result.
        let reduce_op = make_reduce_op(compute_dt, out_dt);
        let base_id = self.nano.push_group(
            out_count,
            reduce_op,
            out_sym_dims.clone(),
            vec![reduce_sym],
            vec![input_ref],
        );

        self.tensor_map.insert(
            out_id,
            TensorAtomMap {
                base_id,
                count: out_count,
                layout: out_layout,
                known_strides: TensorAtomMap::compute_strides(&out_known_dims_full),
                sym_dims: out_sym_dims,
            },
        );
    }

    /// Lower ReduceMean = ReduceSum / extent.
    ///
    /// Matches milli eval semantics: for BF16/F16, the entire mean computation
    /// (sum + divide) happens in F32, with only the final result cast back.
    fn lower_reduce_mean(
        &mut self,
        reduce: &crate::milli_graph::ops::ReduceMean,
        all_infos: &HashMap<GlobalId, TensorInfo>,
    ) {
        let out_id = Node::outputs(reduce).next().unwrap();

        // First try to lower as ReduceSum.
        // We need to know the reduction extent to divide.
        let axes_vals: Option<Vec<i64>> = reduce
            .axes_tensor()
            .and_then(|id| Self::extract_i64(all_infos, &id));

        // Get input info for extent computation.
        let in_id = Node::inputs(reduce).next().unwrap();
        let in_info = all_infos.get(&in_id);

        // Compute the reduction extent from input dims.
        let extent: Option<u64> = (|| {
            let axes = axes_vals.as_ref()?;
            let info = in_info?;
            let rank = info.rank_if_known()?;
            let mut product = 1u64;
            for &a in axes {
                let ax = if a < 0 {
                    (a + rank as i64) as usize
                } else {
                    a as usize
                };
                product *= info.dim_if_known(ax)?;
            }
            Some(product)
        })();

        let Some(extent) = extent else {
            self.lower_as_boundary_named(reduce, all_infos, "ReduceMean");
            return;
        };

        let Some(out_info) = all_infos.get(&out_id) else {
            self.lower_as_boundary_named(reduce, all_infos, "ReduceMean");
            return;
        };
        let out_dt = out_info.dtype();
        let in_dt = in_info.map(|i| i.dtype()).unwrap_or(out_dt);

        // For BF16/F16: keep entire mean computation in F32, cast at the end.
        // This matches milli eval where ndarray accumulates in F32.
        let compute_dt = match in_dt {
            DType::BF16 | DType::F16 => DType::F32,
            other => other,
        };

        // Lower as ReduceSum, keeping output in compute_dt (not out_dt).
        self.lower_reduce(reduce, all_infos, |cd, _od| ScalarOp::ReduceSum {
            compute_dtype: cd,
            output_dtype: cd,
        });

        // If ReduceSum succeeded (output is in tensor_map), divide by extent.
        let Some(sum_map) = self.tensor_map.get(&out_id).cloned() else {
            return; // ReduceSum failed, already boundaried.
        };

        // Create literal for 1/extent.
        let recip = 1.0 / extent as f64;
        let lit_id = self.nano.push_atom(
            ScalarOp::Literal(NumericScalar::F32(recip as f32)),
            vec![],
            vec![],
            vec![],
        );

        // Multiply by 1/extent in compute_dt, then cast to output dtype.
        let base_id = self.nano.push_group(
            sum_map.count,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: compute_dt,
                output_dtype: out_dt,
            },
            sum_map.sym_dims.clone(),
            vec![],
            vec![
                InputRef::Affine {
                    base: sum_map.base_id,
                    stride: 1,
                },
                InputRef::Broadcast(lit_id),
            ],
        );

        // Update tensor_map to point to the divided result.
        self.tensor_map.insert(
            out_id,
            TensorAtomMap {
                base_id,
                count: sum_map.count,
                layout: sum_map.layout,
                known_strides: sum_map.known_strides,
                sym_dims: sum_map.sym_dims,
            },
        );
    }

    /// Generic boundary fallback. Always registers outputs in tensor_map
    /// so downstream ops can reference them.
    fn lower_as_boundary_named<T: Node>(
        &mut self,
        op: &T,
        all_infos: &HashMap<GlobalId, TensorInfo>,
        name: &str,
    ) {
        let _op_id = op.global_id();
        for out_id in op.outputs() {
            if let Some(info) = all_infos.get(&out_id) {
                self.register_boundary(out_id, info, name);
            } else {
                // No info available — register a minimal opaque atom so downstream
                // ops always find this tensor in tensor_map.
                self.register_opaque(out_id);
            }
        }
        self.push_unsupported(op, all_infos, name);
    }

    fn push_unsupported<T: Node>(
        &mut self,
        op: &T,
        all_infos: &HashMap<GlobalId, TensorInfo>,
        name: &str,
    ) {
        let in_shapes: Vec<String> = op
            .inputs()
            .map(|id| Self::fmt_info(all_infos.get(&id)))
            .collect();
        let out_shapes: Vec<String> = op
            .outputs()
            .map(|id| Self::fmt_info(all_infos.get(&id)))
            .collect();
        let detail = format!(
            "{} : ({}) → ({})",
            name,
            in_shapes.join(", "),
            out_shapes.join(", ")
        );
        self.unsupported.push((op.global_id(), name.to_string()));
        self.unsupported_details.push(detail);
    }

    fn fmt_info(info: Option<&TensorInfo>) -> String {
        let Some(info) = info else {
            return "?".to_string();
        };
        let r = info.rank_if_known().unwrap_or(0);
        let prefix = if info.as_numeric().is_some() {
            "N"
        } else {
            "R"
        };
        let dims: Vec<String> = (0..r)
            .map(|i| {
                info.dim_if_known(i)
                    .map(|d| d.to_string())
                    .unwrap_or("?".to_string())
            })
            .collect();
        format!("{}[{}]", prefix, dims.join(","))
    }

    /// Register a tensor with no shape info as a single opaque atom.
    fn register_opaque(&mut self, id: GlobalId) {
        if self.tensor_map.contains_key(&id) {
            return;
        }
        let base_id = self.nano.push_atom(
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
            vec![],
        );
        self.tensor_map.insert(
            id,
            TensorAtomMap {
                base_id,
                count: 1,
                layout: vec![],
                known_strides: vec![],
                sym_dims: vec![],
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DynRank;
    use crate::backends::eval_backend::EvalBackend;
    use crate::graph::Graph;
    use crate::milli_graph::MilliOpGraph;
    use crate::milli_graph::ops::MilliOp;
    use crate::nano_graph::eval::NanoEval;
    use crate::numeric_scalar::NumericScalar;
    use crate::numeric_tensor::NumericTensor;

    /// Extract flat f32 values from a NumericTensor, returned as f64.
    fn tensor_to_f64(t: &NumericTensor<DynRank>) -> Vec<f64> {
        let mut backend = EvalBackend::NDArray;
        let f32_tensor = t.cast(crate::dtype::DType::F32, &mut backend).unwrap();
        let flat = f32_tensor.flatten().unwrap();
        let v: Vec<f32> = flat.to_ndarray().unwrap().try_into().unwrap();
        v.into_iter().map(|x| x as f64).collect()
    }

    /// Extract flat values from a NumericTensor as NumericScalar in the tensor's dtype.
    fn tensor_to_scalars(t: &NumericTensor<DynRank>) -> Vec<NumericScalar> {
        let mut backend = EvalBackend::NDArray;
        let dtype = t.dtype();
        let f32_tensor = t.cast(crate::dtype::DType::F32, &mut backend).unwrap();
        let flat = f32_tensor.flatten().unwrap();
        let v: Vec<f32> = flat.to_ndarray().unwrap().try_into().unwrap();
        v.into_iter()
            .map(|x| NumericScalar::F32(x).cast_to(dtype))
            .collect()
    }

    /// Build a milli graph, eval through both milli and nano, compare results.
    fn check_integrity(
        build_graph: impl FnOnce(
            &mut MilliOpGraph,
            &mut rand::rngs::ThreadRng,
        ) -> (Vec<GlobalId>, Vec<GlobalId>),
        inputs: Vec<NumericTensor<DynRank>>,
    ) {
        let mut rng = rand::rng();
        let (mut milli, _ext_map) = MilliOpGraph::new(std::iter::empty(), &mut rng);

        let (input_ids, output_ids) = build_graph(&mut milli, &mut rng);
        assert_eq!(input_ids.len(), inputs.len());

        // Prepare inputs: add_input makes ext==int, so id maps to itself.
        let mut info_inputs: HashMap<GlobalId, TensorInfo> = HashMap::new();
        let mut intermediates: HashMap<GlobalId, NumericTensor<DynRank>> = HashMap::new();
        for (id, tensor) in input_ids.iter().zip(inputs.iter()) {
            info_inputs.insert(*id, TensorInfo::from(tensor.clone()));
            intermediates.insert(*id, tensor.clone());
        }

        // Eval through MilliOpGraph (walk ops manually).
        let mut backend = EvalBackend::NDArray;
        for &op_id in milli.op_ordering() {
            let op = milli.get_node_by_id(&op_id).unwrap();
            for (tid, val) in op.eval(&intermediates, &mut backend).unwrap() {
                intermediates.insert(tid, val);
            }
        }

        // Lower to NanoGraph.
        let result = lower_with_info(&milli, &info_inputs).unwrap();
        assert!(
            result.unsupported.is_empty(),
            "Unsupported ops: {:?}",
            result.unsupported_details
        );

        // Start with numeric overrides from lowering, add input values on top.
        let mut overrides = result.numeric_overrides;
        for (&id, tensor) in input_ids.iter().zip(inputs.iter()) {
            if let Some(tam) = result.tensor_map.get(&id) {
                let scalars = tensor_to_scalars(tensor);
                assert_eq!(scalars.len(), tam.count as usize);
                for (i, val) in scalars.into_iter().enumerate() {
                    overrides.insert(tam.base_id.0 + i as u32, val);
                }
            }
        }

        // Eval NanoGraph.
        let nano_eval = NanoEval::eval(&result.graph, &overrides);

        // Compare outputs.
        for out_id in &output_ids {
            let milli_tensor = &intermediates[out_id];
            let milli_flat = tensor_to_f64(milli_tensor);

            let Some(tam) = result.tensor_map.get(out_id) else {
                panic!("Output tensor {:?} not in tensor_map", out_id);
            };
            assert!(
                tam.sym_dims.is_empty(),
                "Sym dims not yet supported in test"
            );

            let nano_flat: Vec<f64> = (0..tam.count)
                .map(|i| nano_eval.get(tam.base_id.offset(i)))
                .collect();

            assert_eq!(
                milli_flat.len(),
                nano_flat.len(),
                "Output {:?}: milli has {} elements, nano has {}",
                out_id,
                milli_flat.len(),
                nano_flat.len()
            );

            for (i, (m, n)) in milli_flat.iter().zip(nano_flat.iter()).enumerate() {
                let diff = (m - n).abs();
                let tol = 1e-4 * m.abs().max(1.0);
                assert!(
                    diff < tol,
                    "Output {:?} element {}: milli={} nano={} diff={}",
                    out_id,
                    i,
                    m,
                    n,
                    diff
                );
            }
        }
    }

    #[test]
    fn test_add_two_vectors() {
        check_integrity(
            |graph, rng| {
                let a = graph.add_input(rng);
                let b = graph.add_input(rng);
                let c = crate::milli_graph::ops::SimpleBinary::add(graph, a, b, rng);
                (vec![a, b], vec![c])
            },
            vec![
                NumericTensor::from_vec_shape(vec![1.0f32, 2.0, 3.0, 4.0], vec![4]).unwrap(),
                NumericTensor::from_vec_shape(vec![10.0f32, 20.0, 30.0, 40.0], vec![4]).unwrap(),
            ],
        );
    }

    #[test]
    fn test_mul_add_chain() {
        check_integrity(
            |graph, rng| {
                let a = graph.add_input(rng);
                let b = graph.add_input(rng);
                let c = graph.add_input(rng);
                let ab = crate::milli_graph::ops::SimpleBinary::mul(graph, a, b, rng);
                let abc = crate::milli_graph::ops::SimpleBinary::add(graph, ab, c, rng);
                (vec![a, b, c], vec![abc])
            },
            vec![
                NumericTensor::from_vec_shape(vec![1.0f32, 2.0, 3.0], vec![3]).unwrap(),
                NumericTensor::from_vec_shape(vec![4.0f32, 5.0, 6.0], vec![3]).unwrap(),
                NumericTensor::from_vec_shape(vec![0.1f32, 0.2, 0.3], vec![3]).unwrap(),
            ],
        );
    }

    #[test]
    fn test_broadcast_add() {
        // [2,3] + [3] → [2,3]
        check_integrity(
            |graph, rng| {
                let a = graph.add_input(rng);
                let b = graph.add_input(rng);
                let c = crate::milli_graph::ops::SimpleBinary::add(graph, a, b, rng);
                (vec![a, b], vec![c])
            },
            vec![
                NumericTensor::from_vec_shape(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])
                    .unwrap(),
                NumericTensor::from_vec_shape(vec![10.0f32, 20.0, 30.0], vec![3]).unwrap(),
            ],
        );
    }

    #[test]
    fn test_unary_exp() {
        check_integrity(
            |graph, rng| {
                let a = graph.add_input(rng);
                let b = crate::milli_graph::ops::SimpleUnaryOp::exp(graph, a, rng);
                (vec![a], vec![b])
            },
            vec![NumericTensor::from_vec_shape(vec![0.0f32, 1.0, -1.0, 0.5], vec![4]).unwrap()],
        );
    }

    #[test]
    fn test_matmul_2d() {
        // [2,3] @ [3,2] → [2,2]
        check_integrity(
            |graph, rng| {
                let a = graph.add_input(rng);
                let b = graph.add_input(rng);
                let c = crate::milli_graph::ops::MatMul::push_new_default_precision(
                    graph,
                    a,
                    b,
                    crate::dtype::DType::F32,
                    rng,
                );
                (vec![a, b], vec![c])
            },
            vec![
                NumericTensor::from_vec_shape(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])
                    .unwrap(),
                NumericTensor::from_vec_shape(vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0], vec![3, 2])
                    .unwrap(),
            ],
        );
    }
}
