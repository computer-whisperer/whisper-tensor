//! Lowering from MilliOpGraph to NanoGraph.
//!
//! This module walks a MilliOpGraph in topological order and emits
//! compressed PatternBlocks into a NanoGraph. View ops (reshape, transpose,
//! squeeze, unsqueeze, expand) are dissolved into index remapping rather
//! than creating compute blocks.

use std::collections::HashMap;

use crate::dtype::DType;
use crate::graph::{GlobalId, Graph, Node};
use crate::milli_graph::MilliOpGraph;
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::nano_graph::ops::{LiteralBits, ScalarBinOp, ScalarOp, ScalarUnaryOp};
use crate::nano_graph::pattern::{BlockId, Dim, InputRef, NanoGraph, SourceDimMap};
use crate::numeric_tensor::NumericTensor;
use crate::tensor_info::TensorInfo;
use crate::DynRank;

/// Tracks how a MilliOp tensor maps to the NanoGraph.
///
/// Every tensor is backed by a block, with an optional index permutation
/// from view ops. Metadata tensors (shape, axes) get constant literal blocks
/// whose values are extracted from TensorInfo at lowering time.
#[derive(Debug, Clone)]
struct TensorRef {
    block_id: BlockId,
    /// The logical shape of the tensor as seen by consumers.
    /// Dims may be Known (concrete) or Symbolic (unknown at graph-build time).
    shape: Vec<Dim>,
    /// Permutation from logical dims to source block dims.
    /// If None, logical dims == block dims (identity).
    /// perm[logical_dim] = block_dim
    perm: Option<Vec<usize>>,
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
    /// Ops that could not be lowered.
    pub unsupported: Vec<(GlobalId, String)>,
    /// Map from MilliOp output tensor GlobalId to NanoGraph BlockId.
    pub tensor_to_block: HashMap<GlobalId, BlockId>,
}

/// Lower a MilliOpGraph into a NanoGraph using concrete inputs.
///
/// Convenience wrapper that converts NumericTensors to TensorInfo and
/// calls `lower_with_info`.
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
///
/// Accepts `TensorInfo` inputs which may be concrete (Numeric), shape-only
/// (Shaped/Ranked), or dtype-only (Minimal). Uses `infer_all()` to propagate
/// shapes and dtypes through the graph. Concrete values are extracted where
/// available for meta tensors (axes, shape tensors for Reshape/Slice/etc).
pub fn lower_with_info(
    graph: &MilliOpGraph,
    inputs: &HashMap<GlobalId, TensorInfo>,
) -> Result<LowerResult, LowerError> {
    // Use infer_all to propagate shape/dtype info through the graph.
    let all_infos = graph.infer_all(inputs)?;

    let mut dims_map: HashMap<GlobalId, Vec<Dim>> = HashMap::new();
    let mut dtypes: HashMap<GlobalId, DType> = HashMap::new();
    for (id, info) in &all_infos {
        dtypes.insert(*id, info.dtype());
        if let Some(dims) = info.dims_for_nano() {
            dims_map.insert(*id, dims);
        }
    }

    let mut ctx = LowerCtx {
        nano: NanoGraph::new(),
        refs: HashMap::new(),
        dims: dims_map,
        dtypes,
        infos: all_infos,
        unsupported: Vec::new(),
    };

    // Register graph inputs as literal blocks (externally filled).
    for ext_id in inputs.keys() {
        let int_id = graph.input_map[ext_id];
        ctx.register_input(int_id)?;
    }

    // Walk ops in topological order.
    for &op_id in graph.op_ordering() {
        let op = graph.get_node_by_id(&op_id).unwrap();
        if let Err(e) = ctx.lower_op(op) {
            let kind_str = match &e {
                LowerError::UnsupportedOp(kind) => kind.clone(),
                LowerError::MissingRef(id) => {
                    format!("{} (missing ref {:?})", op.op_kind(), id)
                }
                LowerError::MissingValue(id, ctx_str) => {
                    format!("{} (missing value {:?} for {})", op.op_kind(), id, ctx_str)
                }
                LowerError::MissingShape(id) => {
                    format!("{} (missing shape {:?})", op.op_kind(), id)
                }
                LowerError::MissingDtype(id) => {
                    format!("{} (missing dtype {:?})", op.op_kind(), id)
                }
                LowerError::MilliGraph(err) => {
                    format!("{} (milli: {})", op.op_kind(), err)
                }
            };
            ctx.unsupported.push((op_id, kind_str));
            // Register outputs as opaque so downstream ops can still reference them.
            for out_id in op.outputs() {
                if !ctx.refs.contains_key(&out_id) {
                    if ctx.register_input(out_id).is_err() {
                        // Dims unknown (e.g., data-dependent output shape).
                        // Register with whatever info we have from infer.
                        ctx.register_input_best_effort(out_id);
                    }
                }
            }
        }
    }

    // Mark graph outputs.
    if let Some(ref output_map) = graph.output_map {
        for int_id in output_map.keys() {
            if let Some(tref) = ctx.refs.get(int_id) {
                ctx.nano.output_blocks.push(tref.block_id);
            }
        }
    }

    let tensor_to_block: HashMap<GlobalId, BlockId> = ctx
        .refs
        .iter()
        .map(|(gid, tref)| (*gid, tref.block_id))
        .collect();

    Ok(LowerResult {
        graph: ctx.nano,
        unsupported: ctx.unsupported,
        tensor_to_block,
    })
}

struct LowerCtx {
    nano: NanoGraph,
    refs: HashMap<GlobalId, TensorRef>,
    dims: HashMap<GlobalId, Vec<Dim>>,
    dtypes: HashMap<GlobalId, DType>,
    /// Full TensorInfo for each tensor — used to extract concrete values
    /// from Numeric tensors (for meta tensors like axes, shape, etc).
    infos: HashMap<GlobalId, TensorInfo>,
    unsupported: Vec<(GlobalId, String)>,
}

impl LowerCtx {
    fn get_dims(&self, id: &GlobalId) -> Result<&[Dim], LowerError> {
        self.dims
            .get(id)
            .map(|s| s.as_slice())
            .ok_or(LowerError::MissingShape(*id))
    }

    fn get_dtype(&self, id: &GlobalId) -> Result<DType, LowerError> {
        self.dtypes
            .get(id)
            .copied()
            .ok_or(LowerError::MissingDtype(*id))
    }

    fn get_ref(&self, id: &GlobalId) -> Result<&TensorRef, LowerError> {
        self.refs.get(id).ok_or(LowerError::MissingRef(*id))
    }

    /// Extract a concrete NumericTensor's values as Vec<i64>, casting if needed.
    /// Works for I64, I32, U64, U32, and other integer-compatible dtypes.
    fn tensor_to_i64_vec(
        tensor: &NumericTensor<DynRank>,
        id: GlobalId,
        context: &'static str,
    ) -> Result<Vec<i64>, LowerError> {
        use crate::backends::eval_backend::EvalBackend;
        use typenum::P1;
        // Cast to I64 if not already, then extract.
        let as_i64 = if tensor.dtype() == DType::I64 {
            tensor.clone()
        } else {
            tensor
                .cast(DType::I64, &mut EvalBackend::NDArray)
                .map_err(|_| LowerError::MissingValue(id, context))?
        };
        let rank1 = as_i64
            .try_to_rank::<P1>()
            .map_err(|_| LowerError::MissingValue(id, context))?;
        let ndarray = rank1
            .to_ndarray()
            .map_err(|_| LowerError::MissingValue(id, context))?;
        Vec::<i64>::try_from(ndarray).map_err(|_| LowerError::MissingValue(id, context))
    }

    /// Read a tensor's concrete value as a Vec<i64> (for axes, shape tensors).
    /// Only works if the tensor is Numeric (concrete). Returns MissingValue
    /// if the tensor is symbolic.
    fn read_i64_values(&self, id: &GlobalId, context: &'static str) -> Result<Vec<i64>, LowerError> {
        let info = self
            .infos
            .get(id)
            .ok_or(LowerError::MissingValue(*id, context))?;
        let tensor = info
            .as_numeric()
            .ok_or(LowerError::MissingValue(*id, context))?;
        Self::tensor_to_i64_vec(tensor, *id, context)
    }

    /// Best-effort registration when dims/dtype are incomplete.
    /// Uses symbolic dims if concrete dims are unavailable, or a
    /// rank-1 symbolic placeholder if even the rank is unknown.
    /// This prevents cascade failures where downstream ops fail with
    /// MissingRef just because an upstream op couldn't be lowered.
    fn register_input_best_effort(&mut self, tensor_id: GlobalId) {
        let dtype = self
            .dtypes
            .get(&tensor_id)
            .copied()
            .or_else(|| self.infos.get(&tensor_id).map(|i| i.dtype()))
            .unwrap_or(DType::F32); // last resort fallback

        let dims = if let Some(d) = self.dims.get(&tensor_id) {
            d.clone()
        } else if let Some(info) = self.infos.get(&tensor_id) {
            if let Some(rank) = info.rank_if_known() {
                (0..rank)
                    .map(|i| {
                        info.dim_if_known(i)
                            .map(Dim::Known)
                            .unwrap_or(Dim::Symbolic(0xF000 | i as u16))
                    })
                    .collect()
            } else {
                // Unknown rank — use rank-1 symbolic placeholder.
                vec![Dim::Symbolic(0xFFFF)]
            }
        } else {
            vec![Dim::Symbolic(0xFFFF)]
        };

        let block_id = self.nano.push(
            ScalarOp::Literal(LiteralBits::f32(0.0)),
            dtype,
            dims.clone(),
            vec![],
        );

        self.refs.insert(
            tensor_id,
            TensorRef {
                block_id,
                shape: dims,
                perm: None,
            },
        );
    }

    /// Register an external input or constant tensor.
    fn register_input(&mut self, tensor_id: GlobalId) -> Result<BlockId, LowerError> {
        let dims = self.get_dims(&tensor_id)?.to_vec();
        let dtype = self.get_dtype(&tensor_id)?;

        let block_id = self.nano.push(
            ScalarOp::Literal(LiteralBits::f32(0.0)), // sentinel; runtime fills
            dtype,
            dims.clone(),
            vec![],
        );

        self.refs.insert(
            tensor_id,
            TensorRef {
                block_id,
                shape: dims,
                perm: None,
            },
        );
        Ok(block_id)
    }

    /// Register a tensor as a constant block using its TensorInfo.
    /// Used for metadata tensors (shape outputs, axes, etc.) whose values
    /// are known from infer() but don't come from a lowered op.
    fn register_input_from_info(&mut self, tensor_id: GlobalId) -> Result<BlockId, LowerError> {
        // If already registered, return the existing block.
        if let Some(tref) = self.refs.get(&tensor_id) {
            return Ok(tref.block_id);
        }

        let info = self
            .infos
            .get(&tensor_id)
            .ok_or(LowerError::MissingValue(tensor_id, "register from info"))?;

        if let Some(tensor) = info.as_numeric() {
            use crate::backends::eval_backend::EvalBackend;
            let dtype = tensor.dtype();
            let dims: Vec<Dim> = tensor.shape().into_iter().map(Dim::Known).collect();

            // Cast to F32 to extract scalar value, regardless of original dtype.
            let as_f32 = tensor.cast(DType::F32, &mut EvalBackend::NDArray)
                .map_err(|_| LowerError::MissingValue(tensor_id, "info cast"))?;
            let flat = as_f32.flatten()
                .map_err(|_| LowerError::MissingValue(tensor_id, "info flatten"))?;
            let arr = flat.to_ndarray()
                .map_err(|_| LowerError::MissingValue(tensor_id, "info ndarray"))?;
            let vals: Vec<f32> = Vec::try_from(arr)
                .map_err(|_| LowerError::MissingValue(tensor_id, "info extract"))?;

            let literal = if vals.len() == 1 {
                LiteralBits::f32(vals[0])
            } else {
                LiteralBits::f32(0.0) // multi-element — sentinel
            };

            let block_id = self.nano.push(
                ScalarOp::Literal(literal),
                dtype,
                dims.clone(),
                vec![],
            );

            self.refs.insert(
                tensor_id,
                TensorRef {
                    block_id,
                    shape: dims,
                    perm: None,
                },
            );

            Ok(block_id)
        } else {
            // Non-numeric info — fall back to register_input (uses dims/dtype maps).
            self.register_input(tensor_id)
        }
    }

    /// Build an InputRef for an elementwise op where consumer and source have
    /// the same shape (possibly broadcast).
    fn build_elementwise_input(
        &self,
        source_id: &GlobalId,
        output_shape: &[Dim],
    ) -> Result<InputRef, LowerError> {
        let tref = self.get_ref(source_id)?;
        let (block_id, source_shape, perm) =
            (tref.block_id, tref.shape.clone(), tref.perm.clone());

        let out_ndim = output_shape.len();
        let src_ndim = source_shape.len();
        let source_block = self.nano.get(block_id).unwrap();
        let block_ndim = source_block.ndim();

        // Scalar source (no dims) → empty dim_map, broadcasts everywhere.
        if block_ndim == 0 {
            return Ok(InputRef {
                source_block: block_id,
                dim_map: vec![],
            });
        }

        // When block_ndim != src_ndim (reshape changed the logical shape),
        // we can't directly map logical dims to block dims. Fall back to
        // treating it as flat: if total elements match, use identity on
        // the block's own dims.
        if block_ndim != src_ndim && perm.is_none() {
            // Different rank due to reshape. Build a dim_map that tries
            // to align block dims with output dims from the right.
            let mut dim_map = vec![SourceDimMap::constant(0); block_ndim];
            let block_dims = &source_block.dims;
            let boffset = out_ndim.saturating_sub(block_ndim);
            for bd in 0..block_ndim {
                let out_dim = bd + boffset;
                if out_dim < out_ndim {
                    if block_dims[bd].is_one() {
                        dim_map[bd] = SourceDimMap::constant(0);
                    } else {
                        // Same size or symbolic — use identity mapping.
                        dim_map[bd] = SourceDimMap::identity(out_dim, out_ndim);
                    }
                }
            }
            return Ok(InputRef {
                source_block: block_id,
                dim_map,
            });
        }

        // Build affine maps from output dims to source block dims.
        // Handle numpy-style broadcasting: align from the right.
        let mut dim_map = vec![SourceDimMap::constant(0); block_ndim];

        let offset = out_ndim.saturating_sub(src_ndim);
        for src_logical in 0..src_ndim {
            let out_dim = src_logical + offset;
            let block_dim = if let Some(ref p) = perm {
                p[src_logical]
            } else {
                src_logical
            };

            if block_dim < block_ndim {
                if source_shape[src_logical].is_one() {
                    // Broadcast: source dim is 1, output dim may be larger.
                    dim_map[block_dim] = SourceDimMap::constant(0);
                } else {
                    // Same size, symbolic, or non-broadcast — identity map.
                    dim_map[block_dim] = SourceDimMap::identity(out_dim, out_ndim);
                }
            }
        }

        Ok(InputRef {
            source_block: block_id,
            dim_map,
        })
    }

    fn lower_op(&mut self, op: &AnyMilliOp) -> Result<(), LowerError> {
        let kind = op.op_kind();
        match kind.as_str() {
            "Constant" | "ConstantOfShape" | "Range" => self.lower_constant(op),
            "Add" | "Sub" | "Mul" | "Div" | "Max" | "Min" | "Modulo" | "Equal" | "Greater"
            | "GreaterOrEqual" | "Less" | "LessOrEqual" => self.lower_binary(op),
            "Pow" => self.lower_binary(op),
            "Neg" | "Abs" | "Exp" | "Ln" | "Sqrt" | "Reciprocal" | "Tanh" | "Floor" | "Ceil"
            | "Erf" | "Sign" => self.lower_unary(op),
            // Trig ops
            "Sin" | "Cos" | "Tan" | "Sinh" | "Cosh" | "Asin" | "Acos" | "Atan" | "Asinh"
            | "Acosh" | "Atanh" => self.lower_unary(op),
            "MatMul" => self.lower_matmul(op),
            "ReduceSum" | "ReduceMax" | "ReduceMin" | "ReduceMean" | "ReduceProd" => {
                self.lower_reduce(op)
            }
            "Reshape" => self.lower_reshape(op),
            "Transpose" => self.lower_transpose(op),
            "Squeeze" | "Unsqueeze" => self.lower_squeeze_unsqueeze(op),
            "Expand" => self.lower_expand(op),
            "Cast" | "CastLike" => self.lower_cast(op),
            "Clamp Min" => self.lower_clamp_min(op),
            "Where" => self.lower_where(op),
            "Concat" => self.lower_concat(op),
            "Slice" => self.lower_slice(op),
            "Split" => self.lower_split(op),
            "Gather" => self.lower_gather(op),
            "Shape" => self.lower_shape(op),
            "SumTo" => self.lower_reduce(op), // SumTo is effectively a reduce
            _ => Err(LowerError::UnsupportedOp(kind)),
        }
    }

    fn lower_constant(&mut self, op: &AnyMilliOp) -> Result<(), LowerError> {
        let out_id = op.outputs().next().unwrap();
        // Constants are already in the values map from the interpreter run.
        // Register them as input blocks (externally filled).
        self.register_input(out_id)?;
        Ok(())
    }

    fn lower_binary(&mut self, op: &AnyMilliOp) -> Result<(), LowerError> {
        let kind = op.op_kind();
        let inputs: Vec<GlobalId> = op.inputs().collect();
        let out_id = op.outputs().next().unwrap();
        let out_dims = self.get_dims(&out_id)?.to_vec();
        let out_dtype = self.get_dtype(&out_id)?;

        let scalar_op = match kind.as_str() {
            "Add" => ScalarBinOp::Add,
            "Sub" => ScalarBinOp::Sub,
            "Mul" | "Pow" => ScalarBinOp::Mul, // Pow is more complex but Mul for now
            "Div" => ScalarBinOp::Div,
            "Max" => ScalarBinOp::Max,
            "Min" => ScalarBinOp::Min,
            "Modulo" => ScalarBinOp::Mod,
            // Comparison ops — model as binary for now
            "Equal" | "Greater" | "GreaterOrEqual" | "Less" | "LessOrEqual" => ScalarBinOp::Sub,
            _ => return Err(LowerError::UnsupportedOp(kind)),
        };

        let inp_a = self.build_elementwise_input(&inputs[0], &out_dims)?;
        let inp_b = self.build_elementwise_input(&inputs[1], &out_dims)?;

        let block_id = self.nano.push(
            ScalarOp::Binary(scalar_op),
            out_dtype,
            out_dims.clone(),
            vec![inp_a, inp_b],
        );

        self.refs.insert(
            out_id,
            TensorRef {
                block_id,
                shape: out_dims,
                perm: None,
            },
        );
        Ok(())
    }

    fn lower_unary(&mut self, op: &AnyMilliOp) -> Result<(), LowerError> {
        let kind = op.op_kind();
        let inputs: Vec<GlobalId> = op.inputs().collect();
        let out_id = op.outputs().next().unwrap();
        let out_dims = self.get_dims(&out_id)?.to_vec();
        let out_dtype = self.get_dtype(&out_id)?;

        let scalar_op = match kind.as_str() {
            "Neg" => ScalarUnaryOp::Neg,
            "Abs" => ScalarUnaryOp::Abs,
            "Exp" => ScalarUnaryOp::Exp,
            "Ln" => ScalarUnaryOp::Ln,
            "Sqrt" => ScalarUnaryOp::Sqrt,
            "Reciprocal" => ScalarUnaryOp::Reciprocal,
            "Tanh" => ScalarUnaryOp::Tanh,
            "Floor" => ScalarUnaryOp::Floor,
            "Ceil" => ScalarUnaryOp::Ceil,
            // Approximate trig/special as unsupported for now
            _ => return Err(LowerError::UnsupportedOp(kind)),
        };

        let inp = self.build_elementwise_input(&inputs[0], &out_dims)?;

        let block_id = self.nano.push(
            ScalarOp::Unary(scalar_op),
            out_dtype,
            out_dims.clone(),
            vec![inp],
        );

        self.refs.insert(
            out_id,
            TensorRef {
                block_id,
                shape: out_dims,
                perm: None,
            },
        );
        Ok(())
    }

    fn lower_matmul(&mut self, op: &AnyMilliOp) -> Result<(), LowerError> {
        let inputs: Vec<GlobalId> = op.inputs().collect();
        let out_id = op.outputs().next().unwrap();

        let a_dims = self.get_dims(&inputs[0])?.to_vec();
        let b_dims = self.get_dims(&inputs[1])?.to_vec();
        let out_dims = self.get_dims(&out_id)?.to_vec();
        let out_dtype = self.get_dtype(&out_id)?;

        // Handle batched matmul: (..., M, K) x (..., K, N) → (..., M, N)
        let a_rank = a_dims.len();
        let b_rank = b_dims.len();

        if a_rank < 2 || b_rank < 2 {
            return Err(LowerError::UnsupportedOp(format!(
                "MatMul with rank < 2: A{:?} B{:?}",
                a_dims, b_dims
            )));
        }

        let m = a_dims[a_rank - 2];
        let k = a_dims[a_rank - 1];
        let n = b_dims[b_rank - 1];

        // Batch dims from output.
        let batch_dims: Vec<Dim> = out_dims[..out_dims.len() - 2].to_vec();
        let batch_count = batch_dims.len();

        // product block dims: [...batch, M, N, K]
        let mut product_dims: Vec<Dim> = batch_dims.clone();
        product_dims.push(m);
        product_dims.push(n);
        product_dims.push(k);
        let product_ndim = product_dims.len();

        // Input A: source dims [..., M, K], consumer dims [...batch, M, N, K]
        let a_ref = self.build_matmul_input(
            &inputs[0],
            &a_dims,
            product_ndim,
            batch_count,
            false,
        )?;

        // Input B: source dims [..., K, N], consumer dims [...batch, M, N, K]
        let b_ref = self.build_matmul_input(
            &inputs[1],
            &b_dims,
            product_ndim,
            batch_count,
            true,
        )?;

        let product_id = self.nano.push(
            ScalarOp::Binary(ScalarBinOp::Mul),
            out_dtype,
            product_dims,
            vec![a_ref, b_ref],
        );

        // Reduce block dims: [...batch, M, N] (reduce K)
        let mut reduce_dims: Vec<Dim> = batch_dims;
        reduce_dims.push(m);
        reduce_dims.push(n);
        let reduce_ndim = reduce_dims.len();

        // Map product dims to reduce dims: batch → batch, M → M, N → N, K → Reduce
        let mut reduce_dim_map = Vec::new();
        for i in 0..batch_count {
            reduce_dim_map.push(SourceDimMap::identity(i, reduce_ndim)); // batch
        }
        reduce_dim_map.push(SourceDimMap::identity(batch_count, reduce_ndim)); // M
        reduce_dim_map.push(SourceDimMap::identity(batch_count + 1, reduce_ndim)); // N
        reduce_dim_map.push(SourceDimMap::Reduce); // K

        let reduce_id = self.nano.push(
            ScalarOp::ReduceSum,
            out_dtype,
            reduce_dims,
            vec![InputRef {
                source_block: product_id,
                dim_map: reduce_dim_map,
            }],
        );

        self.refs.insert(
            out_id,
            TensorRef {
                block_id: reduce_id,
                shape: out_dims,
                perm: None,
            },
        );
        Ok(())
    }

    /// Build a matmul InputRef by constructing the logical→product dim mapping
    /// and then translating through any perm/reshape to get block dims.
    fn build_matmul_input(
        &self,
        tensor_id: &GlobalId,
        tensor_dims: &[Dim],
        product_ndim: usize,
        batch_count: usize,
        is_b: bool, // false = A input, true = B input
    ) -> Result<InputRef, LowerError> {
        let tref = self.get_ref(tensor_id)?;
        let block_id = tref.block_id;
        let perm = tref.perm.clone();

        let source_block = self.nano.get(block_id).unwrap();
        let block_ndim = source_block.ndim();
        let rank = tensor_dims.len();

        // Build logical_dim → consumer_dim mapping first.
        let mut logical_to_consumer = Vec::with_capacity(rank);
        for src_logical in 0..rank {
            let consumer_dim = if src_logical < rank - 2 {
                // Batch dim
                let batch_offset = batch_count.saturating_sub(rank - 2);
                src_logical + batch_offset
            } else if src_logical == rank - 2 {
                if is_b { batch_count + 2 } else { batch_count } // K or M
            } else if is_b { batch_count + 1 } else { batch_count + 2 }; // N or K
            logical_to_consumer.push((src_logical, consumer_dim));
        }

        // Now map to block dims. If block_ndim != rank (reshape happened),
        // align from the right.
        let mut dim_map = vec![SourceDimMap::constant(0); block_ndim];
        if let Some(ref p) = perm {
            // Permutation maps logical → block dim.
            for &(src_logical, consumer_dim) in &logical_to_consumer {
                if src_logical < p.len() {
                    let block_dim = p[src_logical];
                    if block_dim < block_ndim {
                        if tensor_dims[src_logical].is_one() && src_logical < rank - 2 {
                            dim_map[block_dim] = SourceDimMap::constant(0);
                        } else {
                            dim_map[block_dim] = SourceDimMap::identity(consumer_dim, product_ndim);
                        }
                    }
                }
            }
        } else {
            // No perm. Align logical dims to block dims.
            let offset = block_ndim.saturating_sub(rank);
            for &(src_logical, consumer_dim) in &logical_to_consumer {
                let block_dim = src_logical + offset;
                if block_dim < block_ndim {
                    if tensor_dims[src_logical].is_one() && src_logical < rank - 2 {
                        dim_map[block_dim] = SourceDimMap::constant(0);
                    } else {
                        dim_map[block_dim] = SourceDimMap::identity(consumer_dim, product_ndim);
                    }
                }
            }
        }

        Ok(InputRef {
            source_block: block_id,
            dim_map,
        })
    }

    fn lower_reduce(&mut self, op: &AnyMilliOp) -> Result<(), LowerError> {
        let kind = op.op_kind();
        let inputs: Vec<GlobalId> = op.inputs().collect();
        let out_id = op.outputs().next().unwrap();

        let data_id = inputs[0];
        let data_dims = self.get_dims(&data_id)?.to_vec();
        let out_dims = self.get_dims(&out_id)?.to_vec();
        let out_dtype = self.get_dtype(&out_id)?;

        // Determine reduce axes from the tensor value (if present) or from shape diff.
        let reduce_axes: Vec<usize> = if inputs.len() > 1 {
            // axes tensor provided
            let axes_vals = self.read_i64_values(&inputs[1], "reduce axes")?;
            let rank = data_dims.len() as i64;
            axes_vals
                .iter()
                .map(|&a| if a < 0 { (a + rank) as usize } else { a as usize })
                .collect()
        } else {
            // Reduce all axes or infer from shape difference.
            // For SumTo: reduce axes where output dim differs from input dim.
            if kind == "SumTo" {
                let mut axes = Vec::new();
                for (i, (ds, os)) in data_dims.iter().zip(out_dims.iter()).enumerate() {
                    if !ds.matches(os) {
                        axes.push(i);
                    }
                }
                axes
            } else {
                (0..data_dims.len()).collect()
            }
        };

        let reduce_op = match kind.as_str() {
            "ReduceSum" | "SumTo" | "ReduceMean" => ScalarOp::ReduceSum,
            "ReduceMax" => ScalarOp::ReduceMax,
            "ReduceMin" => ScalarOp::ReduceMax, // TODO: ReduceMin
            "ReduceProd" => ScalarOp::ReduceSum, // TODO: ReduceProd
            _ => return Err(LowerError::UnsupportedOp(kind)),
        };

        // The output shape may have keepdims=true (size-1 dims preserved).
        // Our reduce block's dims are the non-reduced output dims.
        // We need to figure out the mapping from data dims → (output dim or Reduce).

        // Compute the "effective" output: squeeze out keepdim=1 dims that are reduce axes.
        let mut out_dim_idx = 0;
        let mut consumer_to_source = Vec::new();
        let mut out_effective_dims = Vec::new();

        for (data_dim, &ds) in data_dims.iter().enumerate() {
            if reduce_axes.contains(&data_dim) {
                consumer_to_source.push(None); // Reduce
            } else {
                consumer_to_source.push(Some(out_dim_idx));
                out_effective_dims.push(ds);
                out_dim_idx += 1;
            }
        }

        let reduce_dims: Vec<Dim> = out_effective_dims.clone();
        let reduce_ndim = reduce_dims.len();

        let tref = self.get_ref(&data_id)?;
        let block_id = tref.block_id;
        let perm = tref.perm.clone();

        let source_block = self.nano.get(block_id).unwrap();
        let block_ndim = source_block.ndim();

        let mut dim_map = Vec::with_capacity(block_ndim);
        for block_dim in 0..block_ndim {
            let logical_dim = if let Some(ref p) = perm {
                p.iter().position(|&d| d == block_dim).unwrap_or(block_dim)
            } else {
                block_dim
            };

            if logical_dim < consumer_to_source.len() {
                match consumer_to_source[logical_dim] {
                    Some(consumer_dim) => {
                        dim_map.push(SourceDimMap::identity(consumer_dim, reduce_ndim));
                    }
                    None => {
                        dim_map.push(SourceDimMap::Reduce);
                    }
                }
            } else {
                dim_map.push(SourceDimMap::constant(0));
            }
        }

        let block_id = self.nano.push(
            reduce_op,
            out_dtype,
            reduce_dims,
            vec![InputRef {
                source_block: block_id,
                dim_map,
            }],
        );

        // If ReduceMean, we need to divide by the reduction count.
        let final_block = if kind == "ReduceMean" {
            // Try to compute reduce count from concrete dims.
            // If any reduce axis has a symbolic dim, use a symbolic count.
            let reduce_count: Option<u64> = reduce_axes
                .iter()
                .map(|&a| data_dims[a].as_known())
                .try_fold(1u64, |acc, d| d.map(|v| acc * v));

            let count_lit = if let Some(count) = reduce_count {
                self.nano.push(
                    ScalarOp::Literal(LiteralBits::f32(count as f32)),
                    out_dtype,
                    vec![], // scalar
                    vec![],
                )
            } else {
                // Symbolic reduce count — emit a sentinel literal.
                // TODO: proper symbolic reduce count handling.
                self.nano.push(
                    ScalarOp::Literal(LiteralBits::f32(1.0)),
                    out_dtype,
                    vec![],
                    vec![],
                )
            };

            let div_dims: Vec<Dim> = out_effective_dims;
            let div_ndim = div_dims.len();
            let sum_ref = InputRef {
                source_block: block_id,
                dim_map: (0..div_ndim)
                    .map(|i| SourceDimMap::identity(i, div_ndim))
                    .collect(),
            };
            let count_ref = InputRef {
                source_block: count_lit,
                dim_map: vec![], // scalar, no dims to map
            };
            self.nano.push(
                ScalarOp::Binary(ScalarBinOp::Div),
                out_dtype,
                div_dims,
                vec![sum_ref, count_ref],
            )
        } else {
            block_id
        };

        self.refs.insert(
            out_id,
            TensorRef {
                block_id: final_block,
                shape: out_dims,
                perm: None,
            },
        );
        Ok(())
    }

    fn lower_reshape(&mut self, op: &AnyMilliOp) -> Result<(), LowerError> {
        let inputs: Vec<GlobalId> = op.inputs().collect();
        let out_id = op.outputs().next().unwrap();
        let out_dims = self.get_dims(&out_id)?.to_vec();

        // Register shape tensor as a constant block.
        if inputs.len() > 1 {
            self.register_input_from_info(inputs[1])?;
        }

        // Reshape is a view op — no computation, just reindex.
        let tref = self.get_ref(&inputs[0])?.clone();
        let block_id = tref.block_id;
        let old_perm = tref.perm.clone();

        if old_perm.is_some() {
            // Transpose before reshape — flat ordering changed. Create an identity block.
            let old_logical_dims = self.get_dims(&inputs[0])?.to_vec();

            let inp = self.build_elementwise_input(&inputs[0], &old_logical_dims)?;
            let dtype = self.get_dtype(&out_id)?;
            let identity_block = self.nano.push(
                ScalarOp::Unary(ScalarUnaryOp::Neg), // TODO: Identity op
                dtype,
                old_logical_dims,
                vec![inp],
            );
            self.refs.insert(
                out_id,
                TensorRef {
                    block_id: identity_block,
                    shape: out_dims,
                    perm: None,
                },
            );
        } else {
            // No perm — just update the shape.
            self.refs.insert(
                out_id,
                TensorRef {
                    block_id,
                    shape: out_dims,
                    perm: None,
                },
            );
        }
        Ok(())
    }

    fn lower_transpose(&mut self, op: &AnyMilliOp) -> Result<(), LowerError> {
        let inputs: Vec<GlobalId> = op.inputs().collect();
        let out_id = op.outputs().next().unwrap();
        let out_dims = self.get_dims(&out_id)?.to_vec();

        let tref = self.get_ref(&inputs[0])?.clone();
        let in_rank = tref.shape.len();
        let perm_i64: Vec<i64> = match op {
            AnyMilliOp::Transpose(t) => match t.perm() {
                Some(p) => p.to_vec(),
                None => (0..in_rank as i64).rev().collect(),
            },
            _ => unreachable!(),
        };
        let new_perm: Vec<usize> = perm_i64.iter().map(|&p| p as usize).collect();

        let composed = if let Some(ref ep) = tref.perm {
            new_perm.iter().map(|&np| ep[np]).collect()
        } else {
            new_perm
        };

        self.refs.insert(
            out_id,
            TensorRef {
                block_id: tref.block_id,
                shape: out_dims,
                perm: Some(composed),
            },
        );
        Ok(())
    }

    fn lower_squeeze_unsqueeze(&mut self, op: &AnyMilliOp) -> Result<(), LowerError> {
        let inputs: Vec<GlobalId> = op.inputs().collect();
        let out_id = op.outputs().next().unwrap();
        let out_dims = self.get_dims(&out_id)?.to_vec();

        // Register axes tensor as constant block.
        if inputs.len() > 1 {
            self.register_input_from_info(inputs[1])?;
        }

        let tref = self.get_ref(&inputs[0])?.clone();
        // TODO: properly remap permutation through squeeze/unsqueeze.
        self.refs.insert(
            out_id,
            TensorRef {
                block_id: tref.block_id,
                shape: out_dims,
                perm: None,
            },
        );
        Ok(())
    }

    fn lower_expand(&mut self, op: &AnyMilliOp) -> Result<(), LowerError> {
        let inputs: Vec<GlobalId> = op.inputs().collect();
        let out_id = op.outputs().next().unwrap();
        let out_dims = self.get_dims(&out_id)?.to_vec();

        // Register shape tensor as constant block.
        if inputs.len() > 1 {
            self.register_input_from_info(inputs[1])?;
        }

        // Expand is broadcast — same block, bigger logical shape.
        let tref = self.get_ref(&inputs[0])?.clone();
        self.refs.insert(
            out_id,
            TensorRef {
                block_id: tref.block_id,
                shape: out_dims,
                perm: tref.perm,
            },
        );
        Ok(())
    }

    fn lower_cast(&mut self, op: &AnyMilliOp) -> Result<(), LowerError> {
        let inputs: Vec<GlobalId> = op.inputs().collect();
        let out_id = op.outputs().next().unwrap();
        let out_dims = self.get_dims(&out_id)?.to_vec();

        // Cast is a dtype change — pass through the block with updated shape.
        // TODO: proper Cast op for mixed-precision.
        let tref = self.get_ref(&inputs[0])?.clone();
        self.refs.insert(
            out_id,
            TensorRef {
                block_id: tref.block_id,
                shape: out_dims,
                perm: tref.perm,
            },
        );
        Ok(())
    }

    fn lower_clamp_min(&mut self, op: &AnyMilliOp) -> Result<(), LowerError> {
        let inputs: Vec<GlobalId> = op.inputs().collect();
        let out_id = op.outputs().next().unwrap();
        let out_dims = self.get_dims(&out_id)?.to_vec();
        let out_dtype = self.get_dtype(&out_id)?;

        // ClampMin(x, val) = max(x, val).
        let clamp_value: f32 = match op {
            AnyMilliOp::ClampMin(c) => c.min_val(),
            _ => unreachable!(),
        };

        let lit_block = self.nano.push(
            ScalarOp::Literal(LiteralBits::f32(clamp_value)),
            out_dtype,
            vec![], // scalar
            vec![],
        );

        let inp = self.build_elementwise_input(&inputs[0], &out_dims)?;
        let lit_ref = InputRef {
            source_block: lit_block,
            dim_map: vec![], // scalar broadcast
        };

        let block_id = self.nano.push(
            ScalarOp::Binary(ScalarBinOp::Max),
            out_dtype,
            out_dims.clone(),
            vec![inp, lit_ref],
        );

        self.refs.insert(
            out_id,
            TensorRef {
                block_id,
                shape: out_dims,
                perm: None,
            },
        );
        Ok(())
    }

    fn lower_where(&mut self, op: &AnyMilliOp) -> Result<(), LowerError> {
        // Where(cond, x, y) — treat as binary for now (lossy but gets us past it).
        // TODO: proper ternary op.
        let inputs: Vec<GlobalId> = op.inputs().collect();
        let out_id = op.outputs().next().unwrap();
        let out_dims = self.get_dims(&out_id)?.to_vec();
        let out_dtype = self.get_dtype(&out_id)?;

        let inp_x = self.build_elementwise_input(&inputs[1], &out_dims)?;
        let inp_y = self.build_elementwise_input(&inputs[2], &out_dims)?;

        let block_id = self.nano.push(
            ScalarOp::Binary(ScalarBinOp::Add), // placeholder
            out_dtype,
            out_dims.clone(),
            vec![inp_x, inp_y],
        );

        self.refs.insert(
            out_id,
            TensorRef {
                block_id,
                shape: out_dims,
                perm: None,
            },
        );
        Ok(())
    }

    fn lower_concat(&mut self, op: &AnyMilliOp) -> Result<(), LowerError> {
        let inputs: Vec<GlobalId> = op.inputs().collect();
        let out_id = op.outputs().next().unwrap();

        // Check if all inputs are concrete — if so, constant-fold via eval.
        let all_numeric = inputs.iter().all(|id| {
            self.infos
                .get(id)
                .map(|info| info.as_numeric().is_some())
                .unwrap_or(false)
        });

        if all_numeric {
            use crate::backends::eval_backend::EvalBackend;
            let mut eval_inputs: HashMap<GlobalId, NumericTensor<DynRank>> = HashMap::new();
            for &id in &inputs {
                if let Some(tensor) = self.infos.get(&id).and_then(|i| i.as_numeric()) {
                    eval_inputs.insert(id, tensor.clone());
                }
            }
            let mut backend = EvalBackend::NDArray;
            if let Ok(results) = op.eval(&eval_inputs, &mut backend) {
                for (tid, tensor) in results {
                    let result_dims: Vec<Dim> =
                        tensor.shape().into_iter().map(Dim::Known).collect::<Vec<Dim>>();
                    let dtype = tensor.dtype();
                    self.infos.insert(tid, TensorInfo::from(tensor));
                    self.dims.insert(tid, result_dims.clone());
                    self.dtypes.insert(tid, dtype);
                    self.register_input(tid)?;
                }
                return Ok(());
            }
        }

        // Concat can't be expressed as pure affine blocks — treat as boundary.
        self.register_input(out_id)?;
        self.unsupported
            .push((op.global_id(), "Concat (boundary)".to_string()));
        Ok(())
    }

    fn lower_slice(&mut self, op: &AnyMilliOp) -> Result<(), LowerError> {
        let inputs: Vec<GlobalId> = op.inputs().collect();
        let out_id = op.outputs().next().unwrap();
        let data_id = inputs[0];

        // Register parameter tensors as constant blocks.
        for &inp in &inputs[1..] {
            self.register_input_from_info(inp)?;
        }

        // Read starts, ends, steps, axes from constant tensors.
        let starts = self.read_i64_values(&inputs[1], "slice starts")?;
        let _ends = self.read_i64_values(&inputs[2], "slice ends")?;

        let steps = if inputs.len() > 3 {
            self.read_i64_values(&inputs[3], "slice steps")?
        } else {
            vec![1i64; starts.len()]
        };

        let data_dims = self.get_dims(&data_id)?.to_vec();
        let rank = data_dims.len();

        let axes: Vec<usize> = if inputs.len() > 4 {
            let raw = self.read_i64_values(&inputs[4], "slice axes")?;
            raw.iter()
                .map(|&a| if a < 0 { (a + rank as i64) as usize } else { a as usize })
                .collect()
        } else {
            (0..starts.len()).collect()
        };

        // Build per-axis (start, step) for the sliced axes.
        // Non-sliced axes are identity: start=0, step=1.
        let mut axis_start = vec![0i64; rank];
        let mut axis_step = vec![1i64; rank];

        for (i, &axis) in axes.iter().enumerate() {
            let mut s = starts[i];
            let step = steps[i];

            // Clamp start per ONNX spec — requires concrete dim size.
            if let Some(dim_size) = data_dims[axis].as_known() {
                let dim = dim_size as i64;
                if s < 0 { s += dim; }
                if step > 0 {
                    s = s.clamp(0, dim);
                } else {
                    s = s.clamp(-1, dim - 1);
                }
            }

            axis_start[axis] = s;
            axis_step[axis] = step;
        }

        let out_dims = self.get_dims(&out_id)?.to_vec();
        let out_dtype = self.get_dtype(&out_id)?;
        let out_ndim = out_dims.len();

        // Build InputRef: for each source dim, output_idx → start + output_idx * step.
        let tref = self.get_ref(&data_id)?.clone();
        let block_id = tref.block_id;
        let perm = tref.perm.clone();

        let source_block = self.nano.get(block_id).unwrap();
        let block_ndim = source_block.ndim();

        let mut dim_map = Vec::with_capacity(block_ndim);
        for block_dim in 0..block_ndim {
            let logical_dim = if let Some(ref p) = perm {
                p.iter().position(|&d| d == block_dim).unwrap_or(block_dim)
            } else {
                block_dim
            };

            if logical_dim < rank {
                let consumer_dim = logical_dim;
                let base = axis_start[logical_dim];
                let step = axis_step[logical_dim];

                if step == 1 && base == 0 {
                    dim_map.push(SourceDimMap::identity(consumer_dim, out_ndim));
                } else {
                    dim_map.push(SourceDimMap::Affine(
                        crate::nano_graph::AffineDimMap::strided(consumer_dim, step, out_ndim)
                            .with_base(base),
                    ));
                }
            } else {
                dim_map.push(SourceDimMap::constant(0));
            }
        }

        let block_id = self.nano.push(
            ScalarOp::Identity,
            out_dtype,
            out_dims.clone(),
            vec![InputRef {
                source_block: block_id,
                dim_map,
            }],
        );

        self.refs.insert(
            out_id,
            TensorRef {
                block_id,
                shape: out_dims,
                perm: None,
            },
        );
        Ok(())
    }

    fn lower_split(&mut self, op: &AnyMilliOp) -> Result<(), LowerError> {
        let AnyMilliOp::Split(split_op) = op else {
            unreachable!()
        };

        let inputs: Vec<GlobalId> = op.inputs().collect();
        let out_id = op.outputs().next().unwrap();
        let data_id = inputs[0];

        // Register split-sizes tensor as constant block if present.
        if inputs.len() > 1 {
            self.register_input_from_info(inputs[1])?;
        }

        let data_dims = self.get_dims(&data_id)?.to_vec();
        let rank = data_dims.len();
        let axis_raw = split_op.axis();
        let axis = if axis_raw < 0 {
            (axis_raw + rank as i64) as usize
        } else {
            axis_raw as usize
        };

        // Determine split sizes (same logic as Split::eval).
        let split_sizes: Vec<usize> = if let Some(split_tensor) = split_op.split_tensor() {
            use crate::milli_graph::ops::MilliOpTensorIDOrLiteral;
            match split_tensor {
                MilliOpTensorIDOrLiteral::TensorID(id) => {
                    let vals = self.read_i64_values(id, "split sizes")?;
                    vals.iter().map(|&v| v as usize).collect()
                }
                MilliOpTensorIDOrLiteral::Literal(lit) => {
                    let tensor: NumericTensor<DynRank> = lit.clone().into();
                    use typenum::P1;
                    let r1 = tensor
                        .try_to_rank::<P1>()
                        .map_err(|_| LowerError::MissingValue(data_id, "split literal"))?;
                    let nd = r1
                        .to_ndarray()
                        .map_err(|_| LowerError::MissingValue(data_id, "split literal ndarray"))?;
                    let vals: Vec<i64> = nd
                        .try_into()
                        .map_err(|_| LowerError::MissingValue(data_id, "split literal conv"))?;
                    vals.iter().map(|&v| v as usize).collect()
                }
            }
        } else if let Some(num_outputs) = split_op.num_outputs() {
            // Need concrete dim size for even split.
            let dim = data_dims[axis]
                .as_usize()
                .ok_or(LowerError::MissingShape(data_id))?;
            let base = dim / num_outputs;
            let remainder = dim % num_outputs;
            (0..num_outputs)
                .map(|i| base + if i < remainder { 1 } else { 0 })
                .collect()
        } else {
            return Err(LowerError::UnsupportedOp("Split: no split sizes or num_outputs".to_string()));
        };

        // Compute the start offset for this output_id.
        let output_id = split_op.output_id();
        let start: usize = split_sizes[..output_id].iter().sum();

        let out_dims = self.get_dims(&out_id)?.to_vec();
        let out_dtype = self.get_dtype(&out_id)?;
        let out_ndim = out_dims.len();

        let tref = self.get_ref(&data_id)?.clone();
        let block_id = tref.block_id;
        let perm = tref.perm.clone();

        let source_block = self.nano.get(block_id).unwrap();
        let block_ndim = source_block.ndim();

        let mut dim_map = Vec::with_capacity(block_ndim);
        for block_dim in 0..block_ndim {
            let logical_dim = if let Some(ref p) = perm {
                p.iter().position(|&d| d == block_dim).unwrap_or(block_dim)
            } else {
                block_dim
            };

            if logical_dim < rank {
                let consumer_dim = logical_dim;
                if logical_dim == axis && start > 0 {
                    dim_map.push(SourceDimMap::Affine(
                        crate::nano_graph::AffineDimMap::identity(consumer_dim, out_ndim)
                            .with_base(start as i64),
                    ));
                } else {
                    dim_map.push(SourceDimMap::identity(consumer_dim, out_ndim));
                }
            } else {
                dim_map.push(SourceDimMap::constant(0));
            }
        }

        let new_block_id = self.nano.push(
            ScalarOp::Identity,
            out_dtype,
            out_dims.clone(),
            vec![InputRef {
                source_block: block_id,
                dim_map,
            }],
        );

        self.refs.insert(
            out_id,
            TensorRef {
                block_id: new_block_id,
                shape: out_dims,
                perm: None,
            },
        );
        Ok(())
    }

    fn lower_gather(&mut self, op: &AnyMilliOp) -> Result<(), LowerError> {
        let AnyMilliOp::Gather(gather_op) = op else {
            unreachable!()
        };

        let data_id = gather_op.data_id();
        let indices_id = gather_op.indices_id();
        let out_id = gather_op.output_id();

        let indices_info = self.infos.get(&indices_id);

        // Check if indices are concrete (Numeric).
        let indices_numeric = indices_info.and_then(|info| info.as_numeric());

        // Case 2: Indices are concrete — can express as structured indexing.
        if let Some(indices_tensor) = indices_numeric {
            let indices_i64: Vec<i64> = {
                // Flatten indices to 1D for processing.
                let flat = indices_tensor
                    .flatten()
                    .map_err(|_| LowerError::MissingValue(indices_id, "gather indices flatten"))?;
                Self::tensor_to_i64_vec(&flat.to_dyn_rank(), indices_id, "gather indices")?
            };

            let data_dims = self.get_dims(&data_id)?.to_vec();
            let data_rank = data_dims.len();
            let axis_raw = gather_op.axis();
            let axis = if axis_raw < 0 {
                (axis_raw + data_rank as i64) as usize
            } else {
                axis_raw as usize
            };

            let axis_dim_size = data_dims[axis].as_known();

            // Normalize negative indices.
            let indices_normalized: Vec<i64> = indices_i64
                .iter()
                .map(|&idx| {
                    if idx < 0 {
                        if let Some(dim) = axis_dim_size {
                            idx + dim as i64
                        } else {
                            idx // Can't normalize without known dim size
                        }
                    } else {
                        idx
                    }
                })
                .collect();

            let indices_shape: Vec<u64> = indices_tensor.shape();
            let out_dims = self.get_dims(&out_id)?.to_vec();
            let out_dtype = self.get_dtype(&out_id)?;

            // Scalar index (indices shape is [] or [1] with one element):
            // Gather with a scalar index just selects a slice along the axis.
            // Output shape = data_shape[..axis] ++ indices_shape ++ data_shape[axis+1..]
            // For scalar index, this is data_shape[..axis] ++ data_shape[axis+1..].
            if indices_normalized.len() == 1 {
                let idx = indices_normalized[0];
                let is_scalar_index = indices_shape.is_empty();

                // This is a slice: data[..., idx, ...] along `axis`.
                // We can express this as an Identity block with an offset on the gather axis.
                let tref = self.get_ref(&data_id)?.clone();
                let block_id = tref.block_id;
                let perm = tref.perm.clone();

                let source_block = self.nano.get(block_id).unwrap();
                let block_ndim = source_block.ndim();
                let out_ndim = out_dims.len();

                // Build dim_map: for each block dim, map to the corresponding output dim.
                // The gather axis gets a constant (the selected index), and other dims
                // shift to account for the removed/kept axis.
                let mut dim_map = Vec::with_capacity(block_ndim);

                // Map from data logical dims to block dims.
                for block_dim in 0..block_ndim {
                    let logical_dim = if let Some(ref p) = perm {
                        p.iter().position(|&d| d == block_dim).unwrap_or(block_dim)
                    } else {
                        block_dim
                    };

                    if logical_dim == axis {
                        // This is the gather axis — fix to the selected index.
                        dim_map.push(SourceDimMap::constant(idx));
                    } else if logical_dim < data_rank {
                        // Map to the corresponding output dim.
                        // Output dims: data_shape[..axis] ++ indices_shape ++ data_shape[axis+1..]
                        // For scalar index: dims before axis keep their position,
                        // dims after axis shift left by 1 (if scalar) or stay (if indices_shape=[1]).
                        let out_dim = if is_scalar_index {
                            if logical_dim < axis {
                                logical_dim
                            } else {
                                logical_dim - 1
                            }
                        } else {
                            // indices_shape = [1], so it occupies one dim at position `axis`
                            if logical_dim < axis {
                                logical_dim
                            } else {
                                logical_dim // axis+1 in data maps to axis+1 in output
                            }
                        };

                        if out_dim < out_ndim {
                            dim_map.push(SourceDimMap::identity(out_dim, out_ndim));
                        } else {
                            dim_map.push(SourceDimMap::constant(0));
                        }
                    } else {
                        dim_map.push(SourceDimMap::constant(0));
                    }
                }

                let new_block = self.nano.push(
                    ScalarOp::Identity,
                    out_dtype,
                    out_dims.clone(),
                    vec![InputRef {
                        source_block: block_id,
                        dim_map,
                    }],
                );

                self.refs.insert(
                    out_id,
                    TensorRef {
                        block_id: new_block,
                        shape: out_dims,
                        perm: None,
                    },
                );
                // Register indices as constant block.
                self.register_input_from_info(indices_id)?;
                return Ok(());
            }

            // Multi-element concrete indices: fall through to constant-fold path.
            // If both data and indices are fully Numeric, we can eval the op.
            if let Some(data_tensor) = self.infos.get(&data_id).and_then(|i| i.as_numeric()) {
                use crate::backends::eval_backend::EvalBackend;
                let mut eval_inputs: HashMap<GlobalId, NumericTensor<DynRank>> = HashMap::new();
                eval_inputs.insert(data_id, data_tensor.clone());
                eval_inputs.insert(indices_id, indices_tensor.clone());
                let mut backend = EvalBackend::NDArray;
                if let Ok(results) = op.eval(&eval_inputs, &mut backend) {
                    for (tid, tensor) in results {
                        let result_dims: Vec<Dim> =
                            tensor.shape().into_iter().map(Dim::Known).collect::<Vec<Dim>>();
                        let dtype = tensor.dtype();
                        // Store the result as a TensorInfo for downstream ops.
                        self.infos.insert(tid, TensorInfo::from(tensor));
                        self.dims.insert(tid, result_dims.clone());
                        self.dtypes.insert(tid, dtype);
                        self.register_input(tid)?;
                    }
                    self.register_input_from_info(indices_id)?;
                    return Ok(());
                }
            }
        }

        // Case 3: Indices are runtime/symbolic — treat as boundary.
        let out_id = op.outputs().next().unwrap();
        self.register_input(out_id)?;
        self.unsupported
            .push((op.global_id(), "Gather (boundary)".to_string()));
        Ok(())
    }

    fn lower_shape(&mut self, op: &AnyMilliOp) -> Result<(), LowerError> {
        // Shape op outputs a 1D tensor of the input's dimensions.
        // Register as a constant block from infer results.
        let out_id = op.outputs().next().unwrap();
        self.register_input_from_info(out_id)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::milli_graph::MilliOpGraph;
    use crate::milli_graph::ops::{MatMul, SimpleBinary, SimpleUnaryOp};
    use crate::backends::ndarray_backend::NDArrayNumericTensor;

    fn make_id(rng: &mut impl rand::Rng) -> GlobalId {
        GlobalId::new(rng)
    }

    fn make_f32_tensor(shape: &[u64], val: f32) -> NumericTensor<DynRank> {
        let numel: usize = shape.iter().map(|&d| d as usize).product();
        let shape_vec: Vec<u64> = shape.to_vec();
        NDArrayNumericTensor::<DynRank>::from_vec_shape(vec![val; numel], &shape_vec)
            .unwrap()
            .into()
    }

    #[test]
    fn test_lower_add() {
        let mut rng = wyrand::WyRand::new(42);
        let ext_a = make_id(&mut rng);
        let ext_b = make_id(&mut rng);
        let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
        let int_a = input_map[&ext_a];
        let int_b = input_map[&ext_b];
        let _int_out = SimpleBinary::add(&mut graph, int_a, int_b, &mut rng);

        let mut inputs = HashMap::new();
        inputs.insert(ext_a, make_f32_tensor(&[2, 3], 1.0));
        inputs.insert(ext_b, make_f32_tensor(&[2, 3], 2.0));

        let result = lower(&graph, &inputs).unwrap();
        assert!(result.unsupported.is_empty());
        // 2 input blocks + 1 add block = 3
        assert_eq!(result.graph.len(), 3);

        let errors = result.graph.validate();
        assert!(errors.is_empty(), "Validation errors: {:?}", errors);
    }

    #[test]
    fn test_lower_matmul() {
        let mut rng = wyrand::WyRand::new(42);
        let ext_a = make_id(&mut rng);
        let ext_b = make_id(&mut rng);
        let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
        let int_a = input_map[&ext_a];
        let int_b = input_map[&ext_b];
        let _int_out = MatMul::push_new(&mut graph, int_a, int_b, &mut rng);

        let mut inputs = HashMap::new();
        inputs.insert(ext_a, make_f32_tensor(&[4, 3], 1.0));
        inputs.insert(ext_b, make_f32_tensor(&[3, 5], 1.0));

        let result = lower(&graph, &inputs).unwrap();
        assert!(result.unsupported.is_empty());
        // 2 inputs + 1 mul product + 1 reduce sum = 4
        assert_eq!(result.graph.len(), 4);

        let errors = result.graph.validate();
        assert!(errors.is_empty(), "Validation errors: {:?}", errors);

        let stats = result.graph.stats();
        // inputs: 4*3 + 3*5 = 27, product: 4*5*3 = 60, reduce: 4*5 = 20
        assert_eq!(stats.total_concrete_ops, 12 + 15 + 60 + 20);
    }

    #[test]
    fn test_lower_chain() {
        let mut rng = wyrand::WyRand::new(42);
        let ext_a = make_id(&mut rng);
        let ext_b = make_id(&mut rng);
        let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
        let int_a = input_map[&ext_a];
        let int_b = input_map[&ext_b];
        let sum = SimpleBinary::add(&mut graph, int_a, int_b, &mut rng);
        let _neg = SimpleUnaryOp::neg(&mut graph, sum, &mut rng);

        let mut inputs = HashMap::new();
        inputs.insert(ext_a, make_f32_tensor(&[8], 1.0));
        inputs.insert(ext_b, make_f32_tensor(&[8], 2.0));

        let result = lower(&graph, &inputs).unwrap();
        assert!(result.unsupported.is_empty());
        // 2 inputs + add + neg = 4
        assert_eq!(result.graph.len(), 4);

        let errors = result.graph.validate();
        assert!(errors.is_empty(), "Validation errors: {:?}", errors);
    }

    #[test]
    fn test_lower_with_info_concrete() {
        // Same as test_lower_add but using lower_with_info directly.
        let mut rng = wyrand::WyRand::new(42);
        let ext_a = make_id(&mut rng);
        let ext_b = make_id(&mut rng);
        let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
        let int_a = input_map[&ext_a];
        let int_b = input_map[&ext_b];
        let _int_out = SimpleBinary::add(&mut graph, int_a, int_b, &mut rng);

        let mut inputs = HashMap::new();
        inputs.insert(ext_a, TensorInfo::from(make_f32_tensor(&[2, 3], 1.0)));
        inputs.insert(ext_b, TensorInfo::from(make_f32_tensor(&[2, 3], 2.0)));

        let result = lower_with_info(&graph, &inputs).unwrap();
        assert!(result.unsupported.is_empty());
        assert_eq!(result.graph.len(), 3);

        let errors = result.graph.validate();
        assert!(errors.is_empty(), "Validation errors: {:?}", errors);
    }

    #[test]
    fn test_lower_with_info_shaped() {
        // Use Shaped (non-concrete) inputs — shape known but no data.
        use crate::tensor_info::TensorInfo;

        let mut rng = wyrand::WyRand::new(42);
        let ext_a = make_id(&mut rng);
        let ext_b = make_id(&mut rng);
        let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
        let int_a = input_map[&ext_a];
        let int_b = input_map[&ext_b];
        let _int_out = SimpleBinary::add(&mut graph, int_a, int_b, &mut rng);

        // Provide shape-only info (no concrete values).
        let mut inputs = HashMap::new();
        inputs.insert(ext_a, TensorInfo::from_shape_u64(&[2, 3]));
        inputs.insert(ext_b, TensorInfo::from_shape_u64(&[2, 3]));

        let result = lower_with_info(&graph, &inputs).unwrap();
        assert!(result.unsupported.is_empty());
        assert_eq!(result.graph.len(), 3);

        let errors = result.graph.validate();
        assert!(errors.is_empty(), "Validation errors: {:?}", errors);
    }

    #[test]
    fn test_lower_with_info_symbolic_dims() {
        // Use inputs with a mix of known and symbolic dims.
        use crate::scalar_info::ScalarInfoTyped;
        use crate::symbolic_scalar::SymbolicScalarTyped;
        use crate::symbolic_scalar::SymbolicResolver;
        use crate::tensor_info::TensorInfo;

        let mut rng = wyrand::WyRand::new(42);
        let ext_a = make_id(&mut rng);
        let ext_b = make_id(&mut rng);
        let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
        let int_a = input_map[&ext_a];
        let int_b = input_map[&ext_b];
        let _int_out = SimpleBinary::add(&mut graph, int_a, int_b, &mut rng);

        // Create inputs with one symbolic dim (e.g., batch_size).
        let mut resolver = SymbolicResolver::new();
        let batch_sym: ScalarInfoTyped<u64> =
            ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(&mut resolver));
        let info_a = TensorInfo::from_shape_scalars(&[batch_sym.clone(), ScalarInfoTyped::Numeric(3)]);
        let info_b = TensorInfo::from_shape_scalars(&[batch_sym, ScalarInfoTyped::Numeric(3)]);

        let mut inputs = HashMap::new();
        inputs.insert(ext_a, info_a);
        inputs.insert(ext_b, info_b);

        let result = lower_with_info(&graph, &inputs).unwrap();
        assert!(result.unsupported.is_empty());
        assert_eq!(result.graph.len(), 3);

        // The add block should have 2 dims. The input blocks should carry
        // the symbolic+concrete dims from the TensorInfo inputs.
        let input_block_0 = result.graph.get(BlockId(0)).unwrap();
        assert_eq!(input_block_0.dims.len(), 2);
        assert!(matches!(input_block_0.dims[0], Dim::Symbolic(_)));
        assert_eq!(input_block_0.dims[1], Dim::Known(3));
    }
}
