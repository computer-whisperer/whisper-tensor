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
use crate::milli_graph::ops::AnyMilliOp;
use crate::nano_graph::ops::{LiteralBits, ScalarBinOp, ScalarOp, ScalarUnaryOp};
use crate::nano_graph::pattern::{BlockId, Dim, InputRef, NanoGraph, SourceDimMap};
use crate::numeric_tensor::NumericTensor;
use crate::DynRank;

/// Tracks how a MilliOp tensor maps to the NanoGraph.
///
/// A tensor is either backed by a block (with an optional index permutation
/// from view ops), or is a "meta" tensor (shape, axes) whose value we
/// resolved at lowering time and don't need in the NanoGraph.
#[derive(Debug, Clone)]
enum TensorRef {
    /// This tensor is produced by a block in the NanoGraph.
    Block {
        block_id: BlockId,
        /// The logical shape of the tensor as seen by consumers.
        shape: Vec<usize>,
        /// Permutation from logical dims to source block dims.
        /// If None, logical dims == block dims (identity).
        /// perm[logical_dim] = block_dim
        perm: Option<Vec<usize>>,
    },
    /// This tensor is metadata (shape tensor, axes tensor) — its value
    /// was consumed at lowering time and it has no NanoGraph block.
    Meta,
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

/// Lower a MilliOpGraph into a NanoGraph.
///
/// Requires concrete inputs to resolve all tensor shapes, dtypes, and
/// values (needed for constant folding and resolving shape/axes tensors).
pub fn lower(
    graph: &MilliOpGraph,
    inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
) -> Result<LowerResult, LowerError> {
    // Run the graph to collect all shapes, dtypes, and intermediate values.
    let all_values = graph.collect_all_intermediate_values(inputs)?;

    let mut shapes: HashMap<GlobalId, Vec<usize>> = HashMap::new();
    let mut dtypes: HashMap<GlobalId, DType> = HashMap::new();
    for (id, tensor) in &all_values {
        shapes.insert(*id, tensor.shape().iter().map(|&d| d as usize).collect());
        dtypes.insert(*id, tensor.dtype());
    }

    let mut ctx = LowerCtx {
        nano: NanoGraph::new(),
        refs: HashMap::new(),
        shapes,
        dtypes,
        values: all_values,
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
                LowerError::MissingRef(id) => format!("{} (missing ref {:?})", op.op_kind(), id),
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
                    ctx.register_input(out_id).ok();
                }
            }
        }
    }

    // Mark graph outputs.
    if let Some(ref output_map) = graph.output_map {
        for int_id in output_map.keys() {
            if let Some(TensorRef::Block { block_id, .. }) = ctx.refs.get(int_id) {
                ctx.nano.output_blocks.push(*block_id);
            }
        }
    }

    let tensor_to_block: HashMap<GlobalId, BlockId> = ctx
        .refs
        .iter()
        .filter_map(|(gid, tref)| match tref {
            TensorRef::Block { block_id, .. } => Some((*gid, *block_id)),
            TensorRef::Meta => None,
        })
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
    shapes: HashMap<GlobalId, Vec<usize>>,
    dtypes: HashMap<GlobalId, DType>,
    values: HashMap<GlobalId, NumericTensor<DynRank>>,
    unsupported: Vec<(GlobalId, String)>,
}

impl LowerCtx {
    fn get_shape(&self, id: &GlobalId) -> Result<&[usize], LowerError> {
        self.shapes
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

    /// Read a tensor's value as a Vec<i64> (for axes, shape tensors).
    fn read_i64_values(&self, id: &GlobalId, context: &'static str) -> Result<Vec<i64>, LowerError> {
        use typenum::P1;
        let tensor = self
            .values
            .get(id)
            .ok_or(LowerError::MissingValue(*id, context))?;
        // Convert to rank-1 NDArray then to Vec<i64>.
        let rank1 = tensor
            .try_to_rank::<P1>()
            .map_err(|_| LowerError::MissingValue(*id, context))?;
        let ndarray = rank1
            .to_ndarray()
            .map_err(|_| LowerError::MissingValue(*id, context))?;
        Vec::<i64>::try_from(ndarray).map_err(|_| LowerError::MissingValue(*id, context))
    }

    /// Register an external input or constant tensor.
    fn register_input(&mut self, tensor_id: GlobalId) -> Result<BlockId, LowerError> {
        let shape = self.get_shape(&tensor_id)?.to_vec();
        let dtype = self.get_dtype(&tensor_id)?;
        let dims: Vec<Dim> = shape.iter().map(|&s| Dim::Known(s as u64)).collect();

        let block_id = self.nano.push(
            ScalarOp::Literal(LiteralBits::f32(0.0)), // sentinel; runtime fills
            dtype,
            dims,
            vec![],
        );

        self.refs.insert(
            tensor_id,
            TensorRef::Block {
                block_id,
                shape: shape.clone(),
                perm: None,
            },
        );
        Ok(block_id)
    }

    /// Build an InputRef for an elementwise op where consumer and source have
    /// the same shape (possibly broadcast).
    fn build_elementwise_input(
        &self,
        source_id: &GlobalId,
        output_shape: &[usize],
    ) -> Result<InputRef, LowerError> {
        let tref = self.get_ref(source_id)?;
        let (block_id, source_shape, perm) = match tref {
            TensorRef::Block {
                block_id,
                shape,
                perm,
            } => (*block_id, shape.clone(), perm.clone()),
            TensorRef::Meta => return Err(LowerError::MissingRef(*source_id)),
        };

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
                    let block_size = block_dims[bd].as_known().unwrap_or(1) as usize;
                    if block_size == output_shape[out_dim] {
                        dim_map[bd] = SourceDimMap::identity(out_dim, out_ndim);
                    } else if block_size == 1 {
                        dim_map[bd] = SourceDimMap::constant(0);
                    } else {
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
                if source_shape[src_logical] == output_shape[out_dim] {
                    dim_map[block_dim] = SourceDimMap::identity(out_dim, out_ndim);
                } else if source_shape[src_logical] == 1 {
                    dim_map[block_dim] = SourceDimMap::constant(0);
                } else {
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
            "Gather" => self.lower_gather_boundary(op),
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
        let out_shape = self.get_shape(&out_id)?.to_vec();
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

        let dims: Vec<Dim> = out_shape.iter().map(|&s| Dim::Known(s as u64)).collect();
        let inp_a = self.build_elementwise_input(&inputs[0], &out_shape)?;
        let inp_b = self.build_elementwise_input(&inputs[1], &out_shape)?;

        let block_id = self.nano.push(
            ScalarOp::Binary(scalar_op),
            out_dtype,
            dims,
            vec![inp_a, inp_b],
        );

        self.refs.insert(
            out_id,
            TensorRef::Block {
                block_id,
                shape: out_shape,
                perm: None,
            },
        );
        Ok(())
    }

    fn lower_unary(&mut self, op: &AnyMilliOp) -> Result<(), LowerError> {
        let kind = op.op_kind();
        let inputs: Vec<GlobalId> = op.inputs().collect();
        let out_id = op.outputs().next().unwrap();
        let out_shape = self.get_shape(&out_id)?.to_vec();
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

        let dims: Vec<Dim> = out_shape.iter().map(|&s| Dim::Known(s as u64)).collect();
        let inp = self.build_elementwise_input(&inputs[0], &out_shape)?;

        let block_id = self.nano.push(
            ScalarOp::Unary(scalar_op),
            out_dtype,
            dims,
            vec![inp],
        );

        self.refs.insert(
            out_id,
            TensorRef::Block {
                block_id,
                shape: out_shape,
                perm: None,
            },
        );
        Ok(())
    }

    fn lower_matmul(&mut self, op: &AnyMilliOp) -> Result<(), LowerError> {
        let inputs: Vec<GlobalId> = op.inputs().collect();
        let out_id = op.outputs().next().unwrap();

        let a_shape = self.get_shape(&inputs[0])?.to_vec();
        let b_shape = self.get_shape(&inputs[1])?.to_vec();
        let out_shape = self.get_shape(&out_id)?.to_vec();
        let out_dtype = self.get_dtype(&out_id)?;

        // Handle batched matmul: (..., M, K) x (..., K, N) → (..., M, N)
        let a_rank = a_shape.len();
        let b_rank = b_shape.len();

        if a_rank < 2 || b_rank < 2 {
            return Err(LowerError::UnsupportedOp(format!(
                "MatMul with rank < 2: A{:?} B{:?}",
                a_shape, b_shape
            )));
        }

        let m = a_shape[a_rank - 2];
        let k = a_shape[a_rank - 1];
        let n = b_shape[b_rank - 1];

        // Batch dims from output.
        let batch_dims: Vec<usize> = out_shape[..out_shape.len() - 2].to_vec();

        // product block dims: [...batch, M, N, K]
        let mut product_dims: Vec<Dim> = batch_dims.iter().map(|&s| Dim::Known(s as u64)).collect();
        product_dims.push(Dim::Known(m as u64));
        product_dims.push(Dim::Known(n as u64));
        product_dims.push(Dim::Known(k as u64));
        let product_ndim = product_dims.len();
        let batch_count = batch_dims.len();

        // Input A: source dims [..., M, K], consumer dims [...batch, M, N, K]
        let a_ref = self.build_matmul_input(
            &inputs[0],
            &a_shape,
            product_ndim,
            batch_count,
            false,
        )?;

        // Input B: source dims [..., K, N], consumer dims [...batch, M, N, K]
        let b_ref = self.build_matmul_input(
            &inputs[1],
            &b_shape,
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
        let mut reduce_dims: Vec<Dim> = batch_dims.iter().map(|&s| Dim::Known(s as u64)).collect();
        reduce_dims.push(Dim::Known(m as u64));
        reduce_dims.push(Dim::Known(n as u64));
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
            TensorRef::Block {
                block_id: reduce_id,
                shape: out_shape,
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
        tensor_shape: &[usize],
        product_ndim: usize,
        batch_count: usize,
        is_b: bool, // false = A input, true = B input
    ) -> Result<InputRef, LowerError> {
        let tref = self.get_ref(tensor_id)?;
        let (block_id, _source_shape, perm) = match tref {
            TensorRef::Block {
                block_id,
                shape,
                perm,
            } => (*block_id, shape.clone(), perm.clone()),
            TensorRef::Meta => return Err(LowerError::MissingRef(*tensor_id)),
        };

        let source_block = self.nano.get(block_id).unwrap();
        let block_ndim = source_block.ndim();
        let rank = tensor_shape.len();

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
                        if tensor_shape[src_logical] == 1 && src_logical < rank - 2 {
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
                    if tensor_shape[src_logical] == 1 && src_logical < rank - 2 {
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
        let data_shape = self.get_shape(&data_id)?.to_vec();
        let out_shape = self.get_shape(&out_id)?.to_vec();
        let out_dtype = self.get_dtype(&out_id)?;

        // Determine reduce axes from the tensor value (if present) or from shape diff.
        let reduce_axes: Vec<usize> = if inputs.len() > 1 {
            // axes tensor provided
            let axes_vals = self.read_i64_values(&inputs[1], "reduce axes")?;
            let rank = data_shape.len() as i64;
            axes_vals
                .iter()
                .map(|&a| if a < 0 { (a + rank) as usize } else { a as usize })
                .collect()
        } else {
            // Reduce all axes or infer from shape difference.
            // For SumTo: reduce axes where output dim is 1 and input dim > 1.
            if kind == "SumTo" {
                let mut axes = Vec::new();
                for (i, (&ds, &os)) in data_shape.iter().zip(out_shape.iter()).enumerate() {
                    if ds != os {
                        axes.push(i);
                    }
                }
                axes
            } else {
                (0..data_shape.len()).collect()
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
        let mut out_effective_shape = Vec::new();

        for (data_dim, &ds) in data_shape.iter().enumerate() {
            if reduce_axes.contains(&data_dim) {
                consumer_to_source.push(None); // Reduce
            } else {
                consumer_to_source.push(Some(out_dim_idx));
                out_effective_shape.push(ds);
                out_dim_idx += 1;
            }
        }

        let reduce_dims: Vec<Dim> = out_effective_shape
            .iter()
            .map(|&s| Dim::Known(s as u64))
            .collect();
        let reduce_ndim = reduce_dims.len();

        let tref = self.get_ref(&data_id)?;
        let (block_id, perm) = match tref {
            TensorRef::Block {
                block_id, perm, ..
            } => (*block_id, perm.clone()),
            TensorRef::Meta => return Err(LowerError::MissingRef(data_id)),
        };

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
            let reduce_count: u64 = reduce_axes
                .iter()
                .map(|&a| data_shape[a] as u64)
                .product();
            let count_lit = self.nano.push(
                ScalarOp::Literal(LiteralBits::f32(reduce_count as f32)),
                out_dtype,
                vec![], // scalar
                vec![],
            );
            // div_block dims = reduce block dims
            let div_dims: Vec<Dim> = out_effective_shape
                .iter()
                .map(|&s| Dim::Known(s as u64))
                .collect();
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
            TensorRef::Block {
                block_id: final_block,
                shape: out_shape,
                perm: None,
            },
        );
        Ok(())
    }

    fn lower_reshape(&mut self, op: &AnyMilliOp) -> Result<(), LowerError> {
        let inputs: Vec<GlobalId> = op.inputs().collect();
        let out_id = op.outputs().next().unwrap();
        let out_shape = self.get_shape(&out_id)?.to_vec();

        // Reshape is a view op — no computation, just reindex.
        // Point the output tensor ref at the same block with updated shape.
        let tref = self.get_ref(&inputs[0])?.clone();
        match tref {
            TensorRef::Block {
                block_id,
                shape: _old_shape,
                perm: old_perm,
            } => {
                // If the source had a permutation AND the reshape changes rank,
                // we need to be careful. For now, reset perm on reshape (it's a
                // full reindex).
                // Mark the shape tensor as meta.
                if inputs.len() > 1 {
                    self.refs.insert(inputs[1], TensorRef::Meta);
                }

                // If there was a transpose before this reshape, we can't just
                // pass through — the flat ordering changed. Create an identity block.
                if old_perm.is_some() {
                    let source_block = self.nano.get(block_id).unwrap();
                    let _old_block_ndim = source_block.ndim();
                    let old_logical_shape = self.get_shape(&inputs[0])?.to_vec();

                    // Identity block with old logical shape.
                    let id_dims: Vec<Dim> =
                        old_logical_shape.iter().map(|&s| Dim::Known(s as u64)).collect();
                    let inp = self.build_elementwise_input(&inputs[0], &old_logical_shape)?;
                    let dtype = self.get_dtype(&out_id)?;
                    let identity_block = self.nano.push(
                        ScalarOp::Unary(ScalarUnaryOp::Neg), // TODO: Identity op
                        dtype,
                        id_dims,
                        vec![inp],
                    );
                    // Now the identity block has no perm, same flat order.
                    self.refs.insert(
                        out_id,
                        TensorRef::Block {
                            block_id: identity_block,
                            shape: out_shape,
                            perm: None,
                        },
                    );
                } else {
                    // No perm — just update the shape.
                    self.refs.insert(
                        out_id,
                        TensorRef::Block {
                            block_id,
                            shape: out_shape,
                            perm: None,
                        },
                    );
                }
            }
            TensorRef::Meta => {
                // Reshaping a meta tensor — just register the output.
                self.refs.insert(out_id, TensorRef::Meta);
            }
        }
        Ok(())
    }

    fn lower_transpose(&mut self, op: &AnyMilliOp) -> Result<(), LowerError> {
        let inputs: Vec<GlobalId> = op.inputs().collect();
        let out_id = op.outputs().next().unwrap();
        let out_shape = self.get_shape(&out_id)?.to_vec();

        let tref = self.get_ref(&inputs[0])?.clone();
        match tref {
            TensorRef::Block {
                block_id,
                shape: in_shape,
                perm: existing_perm,
            } => {
                // Get the transpose permutation.
                let in_rank = in_shape.len();
                let perm_i64: Vec<i64> = match op {
                    AnyMilliOp::Transpose(t) => match t.perm() {
                        Some(p) => p.to_vec(),
                        None => (0..in_rank as i64).rev().collect(), // reverse all
                    },
                    _ => unreachable!(),
                };
                let new_perm: Vec<usize> = perm_i64.iter().map(|&p| p as usize).collect();

                // Compose with existing permutation if any.
                let composed = if let Some(ref ep) = existing_perm {
                    // new_perm[out_dim] = logical_dim, ep[logical_dim] = block_dim
                    // composed[out_dim] = ep[new_perm[out_dim]]
                    new_perm.iter().map(|&np| ep[np]).collect()
                } else {
                    new_perm
                };

                self.refs.insert(
                    out_id,
                    TensorRef::Block {
                        block_id,
                        shape: out_shape,
                        perm: Some(composed),
                    },
                );
            }
            TensorRef::Meta => {
                self.refs.insert(out_id, TensorRef::Meta);
            }
        }
        Ok(())
    }

    fn lower_squeeze_unsqueeze(&mut self, op: &AnyMilliOp) -> Result<(), LowerError> {
        let inputs: Vec<GlobalId> = op.inputs().collect();
        let out_id = op.outputs().next().unwrap();
        let out_shape = self.get_shape(&out_id)?.to_vec();

        // Mark axes tensor as meta.
        if inputs.len() > 1 {
            self.refs.insert(inputs[1], TensorRef::Meta);
        }

        // Squeeze/unsqueeze just changes shape (removes/adds size-1 dims).
        // For NanoGraph, just update the TensorRef shape.
        let tref = self.get_ref(&inputs[0])?.clone();
        match tref {
            TensorRef::Block {
                block_id,
                ..
            } => {
                // TODO: properly remap permutation through squeeze/unsqueeze.
                let new_perm: Option<Vec<usize>> = None;

                self.refs.insert(
                    out_id,
                    TensorRef::Block {
                        block_id,
                        shape: out_shape,
                        perm: new_perm,
                    },
                );
            }
            TensorRef::Meta => {
                self.refs.insert(out_id, TensorRef::Meta);
            }
        }
        Ok(())
    }

    fn lower_expand(&mut self, op: &AnyMilliOp) -> Result<(), LowerError> {
        let inputs: Vec<GlobalId> = op.inputs().collect();
        let out_id = op.outputs().next().unwrap();
        let out_shape = self.get_shape(&out_id)?.to_vec();

        // Mark shape tensor as meta.
        if inputs.len() > 1 {
            self.refs.insert(inputs[1], TensorRef::Meta);
        }

        // Expand is broadcast — same block, bigger logical shape.
        let tref = self.get_ref(&inputs[0])?.clone();
        match tref {
            TensorRef::Block {
                block_id, perm, ..
            } => {
                self.refs.insert(
                    out_id,
                    TensorRef::Block {
                        block_id,
                        shape: out_shape,
                        perm,
                    },
                );
            }
            TensorRef::Meta => {
                self.refs.insert(out_id, TensorRef::Meta);
            }
        }
        Ok(())
    }

    fn lower_cast(&mut self, op: &AnyMilliOp) -> Result<(), LowerError> {
        let inputs: Vec<GlobalId> = op.inputs().collect();
        let out_id = op.outputs().next().unwrap();
        let out_shape = self.get_shape(&out_id)?.to_vec();

        // Cast is a dtype change. For the NanoGraph, just pass through the block
        // with updated shape. Actual dtype conversion is tracked on the block dtype.
        // TODO: proper Cast op for mixed-precision.
        let tref = self.get_ref(&inputs[0])?.clone();
        match tref {
            TensorRef::Block {
                block_id, perm, ..
            } => {
                self.refs.insert(
                    out_id,
                    TensorRef::Block {
                        block_id,
                        shape: out_shape,
                        perm,
                    },
                );
            }
            TensorRef::Meta => {
                self.refs.insert(out_id, TensorRef::Meta);
            }
        }
        Ok(())
    }

    fn lower_clamp_min(&mut self, op: &AnyMilliOp) -> Result<(), LowerError> {
        let inputs: Vec<GlobalId> = op.inputs().collect();
        let out_id = op.outputs().next().unwrap();
        let out_shape = self.get_shape(&out_id)?.to_vec();
        let out_dtype = self.get_dtype(&out_id)?;

        // ClampMin(x, val) = max(x, val).
        // Create a literal block for the clamp value, then a Max binary block.
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

        let dims: Vec<Dim> = out_shape.iter().map(|&s| Dim::Known(s as u64)).collect();
        let inp = self.build_elementwise_input(&inputs[0], &out_shape)?;
        let lit_ref = InputRef {
            source_block: lit_block,
            dim_map: vec![], // scalar broadcast
        };

        let block_id = self.nano.push(
            ScalarOp::Binary(ScalarBinOp::Max),
            out_dtype,
            dims,
            vec![inp, lit_ref],
        );

        self.refs.insert(
            out_id,
            TensorRef::Block {
                block_id,
                shape: out_shape,
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
        let out_shape = self.get_shape(&out_id)?.to_vec();
        let out_dtype = self.get_dtype(&out_id)?;

        let dims: Vec<Dim> = out_shape.iter().map(|&s| Dim::Known(s as u64)).collect();
        let inp_x = self.build_elementwise_input(&inputs[1], &out_shape)?;
        let inp_y = self.build_elementwise_input(&inputs[2], &out_shape)?;

        // Approximate: just use x (ignoring condition). Obviously wrong for compute
        // but structurally sound for testing the lowering.
        let block_id = self.nano.push(
            ScalarOp::Binary(ScalarBinOp::Add), // placeholder
            out_dtype,
            dims,
            vec![inp_x, inp_y],
        );

        self.refs.insert(
            out_id,
            TensorRef::Block {
                block_id,
                shape: out_shape,
                perm: None,
            },
        );
        Ok(())
    }

    fn lower_concat(&mut self, op: &AnyMilliOp) -> Result<(), LowerError> {
        // Concat is tough to express as pure affine blocks because it's
        // a piecewise function (different source depending on position along axis).
        // For now, treat as a boundary — register output as opaque input.
        let out_id = op.outputs().next().unwrap();
        self.register_input(out_id)?;
        self.unsupported
            .push((op.global_id(), "Concat (boundary)".to_string()));
        Ok(())
    }

    fn lower_slice(&mut self, op: &AnyMilliOp) -> Result<(), LowerError> {
        let inputs: Vec<GlobalId> = op.inputs().collect();
        let out_id = op.outputs().next().unwrap();
        let data_id = inputs[0];

        // Mark parameter tensors as meta.
        for &inp in &inputs[1..] {
            self.refs.insert(inp, TensorRef::Meta);
        }

        // Read starts, ends, steps, axes from constant tensors.
        let starts = self.read_i64_values(&inputs[1], "slice starts")?;
        let _ends = self.read_i64_values(&inputs[2], "slice ends")?;

        let steps = if inputs.len() > 3 {
            self.read_i64_values(&inputs[3], "slice steps")?
        } else {
            vec![1i64; starts.len()]
        };

        let data_shape = self.get_shape(&data_id)?.to_vec();
        let rank = data_shape.len();

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
            let dim = data_shape[axis] as i64;
            let mut s = starts[i];
            let step = steps[i];

            // Clamp start per ONNX spec.
            if s < 0 { s += dim; }
            if step > 0 {
                s = s.clamp(0, dim);
            } else {
                s = s.clamp(-1, dim - 1);
            }

            axis_start[axis] = s;
            axis_step[axis] = step;
        }

        let out_shape = self.get_shape(&out_id)?.to_vec();
        let out_dtype = self.get_dtype(&out_id)?;
        let out_ndim = out_shape.len();

        // Build InputRef: for each source dim, output_idx → start + output_idx * step.
        let tref = self.get_ref(&data_id)?.clone();
        let (block_id, perm) = match &tref {
            TensorRef::Block { block_id, perm, .. } => (*block_id, perm.clone()),
            TensorRef::Meta => return Err(LowerError::MissingRef(data_id)),
        };

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
                let consumer_dim = logical_dim; // identity mapping: output dim i → source dim i
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

        let dims: Vec<Dim> = out_shape.iter().map(|&s| Dim::Known(s as u64)).collect();
        let block_id = self.nano.push(
            ScalarOp::Identity,
            out_dtype,
            dims,
            vec![InputRef {
                source_block: block_id,
                dim_map,
            }],
        );

        self.refs.insert(
            out_id,
            TensorRef::Block {
                block_id,
                shape: out_shape,
                perm: None,
            },
        );
        Ok(())
    }

    fn lower_split(&mut self, op: &AnyMilliOp) -> Result<(), LowerError> {
        // Split is a Slice along one axis. We compute the cumulative offset
        // for this output_id and emit an Identity block with base offset.
        let AnyMilliOp::Split(split_op) = op else {
            unreachable!()
        };

        let inputs: Vec<GlobalId> = op.inputs().collect();
        let out_id = op.outputs().next().unwrap();
        let data_id = inputs[0];

        // Mark the split-sizes tensor as meta if present.
        if inputs.len() > 1 {
            self.refs.insert(inputs[1], TensorRef::Meta);
        }

        let data_shape = self.get_shape(&data_id)?.to_vec();
        let rank = data_shape.len();
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
            let dim = data_shape[axis];
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

        let out_shape = self.get_shape(&out_id)?.to_vec();
        let out_dtype = self.get_dtype(&out_id)?;
        let out_ndim = out_shape.len();

        let tref = self.get_ref(&data_id)?.clone();
        let (block_id, perm) = match &tref {
            TensorRef::Block { block_id, perm, .. } => (*block_id, perm.clone()),
            TensorRef::Meta => return Err(LowerError::MissingRef(data_id)),
        };

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

        let dims: Vec<Dim> = out_shape.iter().map(|&s| Dim::Known(s as u64)).collect();
        let new_block_id = self.nano.push(
            ScalarOp::Identity,
            out_dtype,
            dims,
            vec![InputRef {
                source_block: block_id,
                dim_map,
            }],
        );

        self.refs.insert(
            out_id,
            TensorRef::Block {
                block_id: new_block_id,
                shape: out_shape,
                perm: None,
            },
        );
        Ok(())
    }

    fn lower_gather_boundary(&mut self, op: &AnyMilliOp) -> Result<(), LowerError> {
        // Gather has data-dependent indexing — treat as boundary.
        let out_id = op.outputs().next().unwrap();
        self.register_input(out_id)?;
        self.unsupported
            .push((op.global_id(), "Gather (boundary)".to_string()));
        Ok(())
    }

    fn lower_shape(&mut self, op: &AnyMilliOp) -> Result<(), LowerError> {
        // Shape op outputs a 1D tensor of the input's dimensions — metadata only.
        let out_id = op.outputs().next().unwrap();
        self.refs.insert(out_id, TensorRef::Meta);
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
}
