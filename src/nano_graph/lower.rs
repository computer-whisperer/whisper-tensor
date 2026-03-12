//! Lowering from MilliOpGraph to NanoGraph.
//!
//! Walks the MilliOpGraph in topological order, classifying each tensor's
//! dimensions as Known (expanded to atoms) or Symbolic (iteration params).
//! View ops dissolve into addressing changes. Elementwise ops become groups
//! with the same atom count. Ops that don't fit become boundary groups.

use std::collections::HashMap;

use crate::graph::{GlobalId, Graph, Node};
use crate::milli_graph::MilliOpGraph;
use crate::milli_graph::ops::AnyMilliOp;
use crate::nano_graph::ops::{LiteralBits, ScalarBinOp, ScalarOp, ScalarUnaryOp};
use crate::nano_graph::pattern::{AtomId, InputRef, NanoGraph, SymDim};
use crate::numeric_tensor::NumericTensor;
use crate::tensor_info::TensorInfo;
use crate::DynRank;

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
    Symbolic(SymDim),
}

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

    // Register all graph inputs as leaf atom groups.
    for (id, info) in &all_infos {
        if inputs.contains_key(id) {
            ctx.register_input(*id, info);
        }
    }

    // Walk ops in topological order.
    for &op_id in graph.op_ordering() {
        let Some(op) = graph.get_node_by_id(&op_id) else { continue };
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

    Ok(LowerResult {
        graph: ctx.nano,
        unsupported: ctx.unsupported,
    })
}

struct LowerCtx {
    nano: NanoGraph,
    tensor_map: HashMap<GlobalId, TensorAtomMap>,
    next_anon_sym: usize,
    unsupported: Vec<(GlobalId, String)>,
}

impl LowerCtx {
    fn new() -> Self {
        Self {
            nano: NanoGraph::new(),
            tensor_map: HashMap::new(),
            next_anon_sym: 0,
            unsupported: Vec::new(),
        }
    }

    /// Classify tensor dims and return layout info.
    /// Returns None if rank is unknown or atom count exceeds limit.
    fn classify_dims(&mut self, info: &TensorInfo) -> Option<(Vec<DimKind>, Vec<u64>, Vec<SymDim>, u32)> {
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
        // Cap at 64M atoms per tensor to avoid OOM.
        if count > 64_000_000 {
            return None;
        }

        Some((layout, known_dims, sym_dims, count as u32))
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
                ScalarOp::Literal(LiteralBits::f32(0.0)),
                info.dtype(),
                vec![],
                vec![],
                vec![],
            );
            self.tensor_map.insert(id, TensorAtomMap {
                base_id, count: 1, layout: vec![], known_strides: vec![], sym_dims: vec![],
            });
            return;
        };

        let strides = TensorAtomMap::compute_strides(&known_dims);
        let count = count.max(1);

        let base_id = self.nano.push_group(
            count,
            ScalarOp::Literal(LiteralBits::f32(0.0)),
            info.dtype(),
            sym_dims.clone(),
            vec![],
            vec![],
        );

        self.tensor_map.insert(id, TensorAtomMap {
            base_id, count, layout, known_strides: strides, sym_dims,
        });
    }

    /// Register a tensor as a boundary (opaque) group.
    /// Boundary atoms are leaves — they use Literal(0) with no inputs.
    fn register_boundary(&mut self, output_id: GlobalId, info: &TensorInfo, _op_kind: &str) {
        let Some((layout, known_dims, sym_dims, count)) = self.classify_dims(info) else {
            let base_id = self.nano.push_atom(
                ScalarOp::Literal(LiteralBits::f32(0.0)),
                info.dtype(), vec![], vec![], vec![],
            );
            self.tensor_map.insert(output_id, TensorAtomMap {
                base_id, count: 1, layout: vec![], known_strides: vec![], sym_dims: vec![],
            });
            return;
        };

        let strides = TensorAtomMap::compute_strides(&known_dims);
        let count = count.max(1);

        let base_id = self.nano.push_group(
            count, ScalarOp::Literal(LiteralBits::f32(0.0)), info.dtype(),
            sym_dims.clone(), vec![], vec![],
        );

        self.tensor_map.insert(output_id, TensorAtomMap {
            base_id, count, layout, known_strides: strides, sym_dims,
        });
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
            let c_known: Vec<u64> = consumer.layout.iter()
                .filter_map(|d| if let DimKind::Known(s) = d { Some(*s) } else { None })
                .collect();
            let p_known: Vec<u64> = producer.layout.iter()
                .filter_map(|d| if let DimKind::Known(s) = d { Some(*s) } else { None })
                .collect();
            if c_known == p_known {
                return InputRef::Affine { base: producer.base_id, stride: 1 };
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

        let c_known_sizes: Vec<u64> = consumer.layout.iter()
            .filter_map(|d| if let DimKind::Known(s) = d { Some(*s) } else { None })
            .collect();
        let p_known_sizes: Vec<u64> = producer.layout.iter()
            .filter_map(|d| if let DimKind::Known(s) = d { Some(*s) } else { None })
            .collect();
        let p_strides = TensorAtomMap::compute_strides(&p_known_sizes);

        // Build mapping from consumer known-dim index to producer known-dim
        // index using right-aligned broadcasting.
        let offset = c_rank.saturating_sub(p_rank);

        // For each original dim, track its known-dim index (if known).
        let c_known_indices: Vec<Option<usize>> = {
            let mut ki = 0;
            consumer.layout.iter().map(|d| {
                if matches!(d, DimKind::Known(_)) {
                    let idx = ki;
                    ki += 1;
                    Some(idx)
                } else {
                    None
                }
            }).collect()
        };
        let p_known_indices: Vec<Option<usize>> = {
            let mut ki = 0;
            producer.layout.iter().map(|d| {
                if matches!(d, DimKind::Known(_)) {
                    let idx = ki;
                    ki += 1;
                    Some(idx)
                } else {
                    None
                }
            }).collect()
        };

        // Map: consumer_known_dim_idx → producer_known_dim_idx (or None if broadcast/mismatch).
        let mut c_to_p_known: Vec<Option<usize>> = vec![None; c_known_sizes.len()];
        for c_orig in 0..c_rank {
            let Some(c_ki) = c_known_indices[c_orig] else { continue };
            if c_orig < offset { continue }
            let p_orig = c_orig - offset;
            if p_orig >= p_rank { continue }
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
                }
            }
            AnyMilliOp::ConstantOfShape(c) => {
                let out_id = Node::outputs(c).next().unwrap();
                if let Some(info) = all_infos.get(&out_id) {
                    self.register_input(out_id, info);
                }
            }
            AnyMilliOp::Shape(s) => {
                // Shape produces a small constant tensor — treat as input leaf.
                let out_id = Node::outputs(s).next().unwrap();
                if let Some(info) = all_infos.get(&out_id) {
                    self.register_input(out_id, info);
                }
            }
            AnyMilliOp::Reshape(r) => self.lower_view_op(r, all_infos),
            AnyMilliOp::Squeeze(s) => self.lower_view_op(s, all_infos),
            AnyMilliOp::Unsqueeze(u) => self.lower_view_op(u, all_infos),
            AnyMilliOp::Transpose(t) => self.lower_view_op(t, all_infos),
            AnyMilliOp::Expand(e) => self.lower_expand(e, all_infos),
            // Everything else: boundary.
            other => {
                let op_kind = other.op_kind();
                let op_id = other.global_id();
                for out_id in other.outputs() {
                    if let Some(info) = all_infos.get(&out_id) {
                        self.register_boundary(out_id, info, &op_kind);
                    }
                }
                self.unsupported.push((op_id, op_kind));
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

        let scalar_op = match bin.which_op() {
            WhichSimpleBinaryOp::Add => ScalarOp::Binary(ScalarBinOp::Add),
            WhichSimpleBinaryOp::Sub => ScalarOp::Binary(ScalarBinOp::Sub),
            WhichSimpleBinaryOp::Mul => ScalarOp::Binary(ScalarBinOp::Mul),
            WhichSimpleBinaryOp::Div => ScalarOp::Binary(ScalarBinOp::Div),
            WhichSimpleBinaryOp::Max => ScalarOp::Binary(ScalarBinOp::Max),
            WhichSimpleBinaryOp::Min => ScalarOp::Binary(ScalarBinOp::Min),
            WhichSimpleBinaryOp::Modulo(_) => ScalarOp::Binary(ScalarBinOp::Mod),
            // Comparison/logical ops: opaque binary, but shape is known.
            _ => ScalarOp::Binary(ScalarBinOp::Max), // placeholder
        };

        let Some((layout, known_dims, sym_dims, count)) = self.classify_dims(out_info) else {
            self.lower_as_boundary_named(bin, all_infos, "SimpleBinary");
            return;
        };
        let count = count.max(1);
        let strides = TensorAtomMap::compute_strides(&known_dims);

        let out_tmp = TensorAtomMap {
            base_id: AtomId(0), count,
            layout: layout.clone(), known_strides: strides.clone(), sym_dims: sym_dims.clone(),
        };

        let a_info = all_infos.get(&a_id);
        let b_info = all_infos.get(&b_id);
        let input_a = self.compute_input_ref(&out_tmp, &a_map, out_info, a_info.unwrap_or(out_info));
        let input_b = self.compute_input_ref(&out_tmp, &b_map, out_info, b_info.unwrap_or(out_info));

        let base_id = self.nano.push_group(
            count, scalar_op, out_info.dtype(), sym_dims.clone(), vec![],
            vec![input_a, input_b],
        );

        self.tensor_map.insert(out_id, TensorAtomMap {
            base_id, count, layout, known_strides: strides, sym_dims,
        });
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

        let scalar_op = match un.which_op() {
            WhichSimpleUnaryOp::Neg => ScalarOp::Unary(ScalarUnaryOp::Neg),
            WhichSimpleUnaryOp::Abs => ScalarOp::Unary(ScalarUnaryOp::Abs),
            WhichSimpleUnaryOp::Exp => ScalarOp::Unary(ScalarUnaryOp::Exp),
            WhichSimpleUnaryOp::Ln => ScalarOp::Unary(ScalarUnaryOp::Ln),
            WhichSimpleUnaryOp::Sqrt => ScalarOp::Unary(ScalarUnaryOp::Sqrt),
            WhichSimpleUnaryOp::Reciprocal => ScalarOp::Unary(ScalarUnaryOp::Reciprocal),
            WhichSimpleUnaryOp::Trig(crate::TrigOp::Tanh) => ScalarOp::Unary(ScalarUnaryOp::Tanh),
            WhichSimpleUnaryOp::Floor => ScalarOp::Unary(ScalarUnaryOp::Floor),
            WhichSimpleUnaryOp::Ceil => ScalarOp::Unary(ScalarUnaryOp::Ceil),
            _ => {
                self.lower_as_boundary_named(un, all_infos, "SimpleUnary");
                return;
            }
        };

        let base_id = self.nano.push_group(
            in_map.count, scalar_op, out_info.dtype(), in_map.sym_dims.clone(), vec![],
            vec![InputRef::Affine { base: in_map.base_id, stride: 1 }],
        );

        self.tensor_map.insert(out_id, TensorAtomMap {
            base_id, count: in_map.count,
            layout: in_map.layout.clone(),
            known_strides: in_map.known_strides.clone(),
            sym_dims: in_map.sym_dims.clone(),
        });
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
        let min_id = self.nano.push_atom(
            ScalarOp::Literal(LiteralBits::f32(min_val)),
            out_info.dtype(), vec![], vec![], vec![],
        );

        let base_id = self.nano.push_group(
            in_map.count, ScalarOp::Binary(ScalarBinOp::Max), out_info.dtype(),
            in_map.sym_dims.clone(), vec![],
            vec![
                InputRef::Affine { base: in_map.base_id, stride: 1 },
                InputRef::Broadcast(min_id),
            ],
        );

        self.tensor_map.insert(out_id, TensorAtomMap {
            base_id, count: in_map.count,
            layout: in_map.layout.clone(),
            known_strides: in_map.known_strides.clone(),
            sym_dims: in_map.sym_dims.clone(),
        });
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

        let base_id = self.nano.push_group(
            in_map.count, ScalarOp::Identity, out_info.dtype(),
            in_map.sym_dims.clone(), vec![],
            vec![InputRef::Affine { base: in_map.base_id, stride: 1 }],
        );

        self.tensor_map.insert(out_id, TensorAtomMap {
            base_id, count: in_map.count,
            layout: in_map.layout.clone(),
            known_strides: in_map.known_strides.clone(),
            sym_dims: in_map.sym_dims.clone(),
        });
    }

    /// View op: no compute, just re-register with the new shape.
    fn lower_view_op<T: Node>(&mut self, op: &T, all_infos: &HashMap<GlobalId, TensorInfo>) {
        let in_id = Node::inputs(op).next().unwrap();
        let out_id = Node::outputs(op).next().unwrap();

        let Some(in_map) = self.tensor_map.get(&in_id).cloned() else {
            self.lower_as_boundary_named(op, all_infos, "ViewOp");
            return;
        };
        let Some(out_info) = all_infos.get(&out_id) else { return };

        let Some((layout, known_dims, sym_dims, count)) = self.classify_dims(out_info) else {
            self.lower_as_boundary_named(op, all_infos, "ViewOp");
            return;
        };
        let count = count.max(1);

        if count == in_map.count {
            // Atom-count preserving: just re-register with new layout.
            self.tensor_map.insert(out_id, TensorAtomMap {
                base_id: in_map.base_id, count,
                layout, known_strides: TensorAtomMap::compute_strides(&known_dims), sym_dims,
            });
        } else {
            self.register_boundary(out_id, out_info, "ViewOp");
            self.unsupported.push((op.global_id(), format!("ViewOp(count {} → {})", in_map.count, count)));
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
        let Some(out_info) = all_infos.get(&out_id) else { return };
        let in_info = all_infos.get(&in_id);

        let Some((layout, known_dims, sym_dims, count)) = self.classify_dims(out_info) else {
            self.lower_as_boundary_named(e, all_infos, "Expand");
            return;
        };
        let count = count.max(1);

        if count == in_map.count {
            self.tensor_map.insert(out_id, TensorAtomMap {
                base_id: in_map.base_id, count,
                layout, known_strides: TensorAtomMap::compute_strides(&known_dims), sym_dims,
            });
        } else {
            let out_tmp = TensorAtomMap {
                base_id: AtomId(0), count,
                layout: layout.clone(),
                known_strides: TensorAtomMap::compute_strides(&known_dims),
                sym_dims: sym_dims.clone(),
            };
            let input_ref = self.compute_input_ref(
                &out_tmp, &in_map, out_info, in_info.unwrap_or(out_info),
            );

            let base_id = self.nano.push_group(
                count, ScalarOp::Identity, out_info.dtype(),
                sym_dims.clone(), vec![], vec![input_ref],
            );

            self.tensor_map.insert(out_id, TensorAtomMap {
                base_id, count, layout,
                known_strides: TensorAtomMap::compute_strides(&known_dims), sym_dims,
            });
        }
    }

    /// Generic boundary fallback.
    fn lower_as_boundary_named<T: Node>(
        &mut self,
        op: &T,
        all_infos: &HashMap<GlobalId, TensorInfo>,
        name: &str,
    ) {
        let op_id = op.global_id();
        for out_id in op.outputs() {
            if let Some(info) = all_infos.get(&out_id) {
                self.register_boundary(out_id, info, name);
            }
        }
        self.unsupported.push((op_id, name.to_string()));
    }
}
