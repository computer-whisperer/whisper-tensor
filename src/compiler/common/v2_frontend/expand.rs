//! Expander: walks a MilliOpGraph and produces OutputBindings.
//!
//! Matches on AnyMilliOp enum variants directly (no string matching).
//! View ops (Reshape, Transpose, Squeeze, Unsqueeze) dissolve into index
//! remapping — they don't produce ALU ops, they just change which flat
//! index of the source tensor an output element references.

use super::{OutputBinding, ScalarBinOp, ScalarExpr, ScalarUnaryOp};
use crate::graph::{GlobalId, Graph, Node};
use crate::milli_graph::MilliOpGraph;
use crate::milli_graph::ops::{AnyMilliOp, WhichSimpleBinaryOp, WhichSimpleUnaryOp};
use std::collections::HashMap;

#[derive(Debug, thiserror::Error)]
pub enum ExpandError {
    #[error("Missing shape for tensor {0}")]
    MissingShape(GlobalId),
    #[error("Unsupported milli op for expansion: {0}")]
    UnsupportedOp(String),
}

/// Expands a MilliOpGraph into OutputBindings — pure scalar expression trees.
pub struct ExprExpander {
    tensor_shapes: HashMap<GlobalId, Vec<usize>>,
    /// Max output elements to sample per op (0 = all).
    max_output_samples: usize,
    /// When true, unsupported ops are silently skipped (outputs become leaves).
    skip_unsupported: bool,
}

impl ExprExpander {
    pub fn new(tensor_shapes: HashMap<GlobalId, Vec<usize>>) -> Self {
        Self {
            tensor_shapes,
            max_output_samples: 0,
            skip_unsupported: false,
        }
    }

    pub fn new_sampled(
        tensor_shapes: HashMap<GlobalId, Vec<usize>>,
        max_output_samples: usize,
    ) -> Self {
        Self {
            tensor_shapes,
            max_output_samples,
            skip_unsupported: false,
        }
    }

    /// Enable skip-unsupported mode: unsupported ops are silently ignored,
    /// their outputs become leaf tensors that must be pre-filled by the caller.
    pub fn with_skip_unsupported(mut self, skip: bool) -> Self {
        self.skip_unsupported = skip;
        self
    }

    fn get_shape(&self, tensor: &GlobalId) -> Result<&[usize], ExpandError> {
        self.tensor_shapes
            .get(tensor)
            .map(|s| s.as_slice())
            .ok_or(ExpandError::MissingShape(*tensor))
    }

    /// Expand the entire graph into a Vec of OutputBindings.
    pub fn expand(&self, graph: &MilliOpGraph) -> Result<Vec<OutputBinding>, ExpandError> {
        let mut bindings = Vec::new();
        for op_id in graph.op_ordering() {
            let op = graph.get_node_by_id(op_id).unwrap();
            self.expand_op(op, &mut bindings)?;
        }
        Ok(bindings)
    }

    /// Expand a single milli op, appending bindings to `out`.
    pub(crate) fn expand_op(
        &self,
        op: &AnyMilliOp,
        out: &mut Vec<OutputBinding>,
    ) -> Result<(), ExpandError> {
        match op {
            // Constants: no ALU expression needed. The runtime pre-fills buffers.
            // We still emit no bindings — constants are leaves referenced by
            // downstream ops via Element nodes.
            AnyMilliOp::Constant(_) | AnyMilliOp::ConstantOfShape(_) => Ok(()),

            // Elementwise binary ops
            AnyMilliOp::SimpleBinary(bin) => {
                let scalar_op = match bin.which_op() {
                    WhichSimpleBinaryOp::Add => ScalarBinOp::Add,
                    WhichSimpleBinaryOp::Sub => ScalarBinOp::Sub,
                    WhichSimpleBinaryOp::Mul => ScalarBinOp::Mul,
                    WhichSimpleBinaryOp::Div => ScalarBinOp::Div,
                    WhichSimpleBinaryOp::Max => ScalarBinOp::Max,
                    WhichSimpleBinaryOp::Min => ScalarBinOp::Min,
                    other => {
                        if self.skip_unsupported {
                            return Ok(());
                        }
                        return Err(ExpandError::UnsupportedOp(format!(
                            "SimpleBinary variant {:?}",
                            other
                        )));
                    }
                };
                let inputs: Vec<GlobalId> = op.inputs().collect();
                let outputs: Vec<GlobalId> = op.outputs().collect();
                self.emit_binary(inputs[0], inputs[1], outputs[0], scalar_op, out)
            }

            // Elementwise unary ops
            AnyMilliOp::SimpleUnary(unary) => {
                let scalar_op = match unary.which_op() {
                    WhichSimpleUnaryOp::Neg => ScalarUnaryOp::Neg,
                    WhichSimpleUnaryOp::Abs => ScalarUnaryOp::Abs,
                    WhichSimpleUnaryOp::Exp => ScalarUnaryOp::Exp,
                    WhichSimpleUnaryOp::Ln => ScalarUnaryOp::Ln,
                    WhichSimpleUnaryOp::Sqrt => ScalarUnaryOp::Sqrt,
                    WhichSimpleUnaryOp::Reciprocal => ScalarUnaryOp::Reciprocal,
                    WhichSimpleUnaryOp::Trig(crate::TrigOp::Tanh) => ScalarUnaryOp::Tanh,
                    WhichSimpleUnaryOp::Floor => ScalarUnaryOp::Floor,
                    WhichSimpleUnaryOp::Ceil => ScalarUnaryOp::Ceil,
                    WhichSimpleUnaryOp::Erf => ScalarUnaryOp::Erf,
                    other => {
                        if self.skip_unsupported {
                            return Ok(());
                        }
                        return Err(ExpandError::UnsupportedOp(format!(
                            "SimpleUnary variant {:?}",
                            other
                        )));
                    }
                };
                let inputs: Vec<GlobalId> = op.inputs().collect();
                let outputs: Vec<GlobalId> = op.outputs().collect();
                self.emit_unary(inputs[0], outputs[0], scalar_op, out)
            }

            // MatMul (rank-2 only for now)
            AnyMilliOp::MatMul(_) => {
                let inputs: Vec<GlobalId> = op.inputs().collect();
                let outputs: Vec<GlobalId> = op.outputs().collect();
                self.emit_matmul(inputs[0], inputs[1], outputs[0], out)
            }

            // View ops: Reshape, Transpose, Squeeze, Unsqueeze
            // These dissolve into index remapping — output[i] = Element(input, remapped_i)
            AnyMilliOp::Reshape(_) => {
                let inputs: Vec<GlobalId> = op.inputs().collect();
                let outputs: Vec<GlobalId> = op.outputs().collect();
                // Reshape is a view — output[flat_i] = input[flat_i] (same flat layout)
                self.emit_identity_view(inputs[0], outputs[0], out)
            }

            AnyMilliOp::Squeeze(_) => {
                let inputs: Vec<GlobalId> = op.inputs().collect();
                let outputs: Vec<GlobalId> = op.outputs().collect();
                // Squeeze removes size-1 dims — flat layout unchanged
                self.emit_identity_view(inputs[0], outputs[0], out)
            }

            AnyMilliOp::Unsqueeze(_) => {
                let inputs: Vec<GlobalId> = op.inputs().collect();
                let outputs: Vec<GlobalId> = op.outputs().collect();
                // Unsqueeze adds size-1 dims — flat layout unchanged
                self.emit_identity_view(inputs[0], outputs[0], out)
            }

            AnyMilliOp::Transpose(t) => {
                let inputs: Vec<GlobalId> = op.inputs().collect();
                let outputs: Vec<GlobalId> = op.outputs().collect();
                self.emit_transpose(inputs[0], outputs[0], t.perm(), out)
            }

            // Cast / CastLike: dtype conversion — identity in the expression tree.
            // The actual load/store dtype difference is handled by codegen.
            AnyMilliOp::Cast(_) => {
                let inputs: Vec<GlobalId> = op.inputs().collect();
                let outputs: Vec<GlobalId> = op.outputs().collect();
                self.emit_identity_view(inputs[0], outputs[0], out)
            }
            AnyMilliOp::CastLike(_) => {
                let inputs: Vec<GlobalId> = op.inputs().collect();
                let outputs: Vec<GlobalId> = op.outputs().collect();
                // CastLike has two inputs: data and target_type. Only data flows through.
                self.emit_identity_view(inputs[0], outputs[0], out)
            }

            // ClampMin: max(input, threshold)
            AnyMilliOp::ClampMin(c) => {
                let inputs: Vec<GlobalId> = op.inputs().collect();
                let outputs: Vec<GlobalId> = op.outputs().collect();
                self.emit_clamp_min(inputs[0], outputs[0], c.min_val(), out)
            }

            // Reductions
            AnyMilliOp::ReduceSum(r) => {
                let inputs: Vec<GlobalId> = op.inputs().collect();
                let outputs: Vec<GlobalId> = op.outputs().collect();
                self.emit_reduce(
                    inputs[0],
                    outputs[0],
                    r.axes_tensor(),
                    r.keepdims(),
                    ScalarBinOp::Add,
                    0.0,
                    out,
                )
            }

            AnyMilliOp::ReduceMax(r) => {
                let inputs: Vec<GlobalId> = op.inputs().collect();
                let outputs: Vec<GlobalId> = op.outputs().collect();
                self.emit_reduce(
                    inputs[0],
                    outputs[0],
                    r.axes_tensor(),
                    r.keepdims(),
                    ScalarBinOp::Max,
                    f64::NEG_INFINITY,
                    out,
                )
            }

            AnyMilliOp::ReduceMean(r) => {
                let inputs: Vec<GlobalId> = op.inputs().collect();
                let outputs: Vec<GlobalId> = op.outputs().collect();
                self.emit_reduce_mean(inputs[0], outputs[0], r.axes_tensor(), r.keepdims(), out)
            }

            _ => {
                if self.skip_unsupported {
                    Ok(()) // Output tensors become leaves
                } else {
                    Err(ExpandError::UnsupportedOp(op.op_kind()))
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Emitters
    // ------------------------------------------------------------------

    fn emit_binary(
        &self,
        a_id: GlobalId,
        b_id: GlobalId,
        out_id: GlobalId,
        op: ScalarBinOp,
        out: &mut Vec<OutputBinding>,
    ) -> Result<(), ExpandError> {
        let out_shape = self.get_shape(&out_id)?;
        let a_shape = self.get_shape(&a_id)?;
        let b_shape = self.get_shape(&b_id)?;
        let a_strides = broadcast_strides(a_shape, out_shape);
        let b_strides = broadcast_strides(b_shape, out_shape);
        let _total: usize = out_shape.iter().product();
        let out_shape_owned = out_shape.to_vec();

        for flat_out in self.sample_indices_nd(&out_shape_owned) {
            let multi = flat_to_multi(flat_out, &out_shape_owned);
            let flat_a = multi_to_flat(&multi, &a_strides);
            let flat_b = multi_to_flat(&multi, &b_strides);

            let expr = ScalarExpr::Binary {
                op,
                a: Box::new(ScalarExpr::Element {
                    tensor: a_id,
                    flat_index: flat_a,
                }),
                b: Box::new(ScalarExpr::Element {
                    tensor: b_id,
                    flat_index: flat_b,
                }),
            };
            out.push(OutputBinding {
                output_tensor: out_id,
                flat_index: flat_out,
                expr,
            });
        }
        Ok(())
    }

    fn emit_unary(
        &self,
        input_id: GlobalId,
        out_id: GlobalId,
        op: ScalarUnaryOp,
        out: &mut Vec<OutputBinding>,
    ) -> Result<(), ExpandError> {
        let out_shape = self.get_shape(&out_id)?.to_vec();

        for flat_idx in self.sample_indices_nd(&out_shape) {
            let expr = ScalarExpr::Unary {
                op,
                input: Box::new(ScalarExpr::Element {
                    tensor: input_id,
                    flat_index: flat_idx,
                }),
            };
            out.push(OutputBinding {
                output_tensor: out_id,
                flat_index: flat_idx,
                expr,
            });
        }
        Ok(())
    }

    fn emit_matmul(
        &self,
        a_id: GlobalId,
        b_id: GlobalId,
        out_id: GlobalId,
        out: &mut Vec<OutputBinding>,
    ) -> Result<(), ExpandError> {
        let a_shape = self.get_shape(&a_id)?;
        let b_shape = self.get_shape(&b_id)?;
        let out_shape = self.get_shape(&out_id)?;

        // ONNX MatMul broadcasting: batch dims broadcast, last 2 dims do the matmul.
        // A[..., M, K] × B[..., K, N] = Out[..., M, N]
        if a_shape.len() < 2 || b_shape.len() < 2 || out_shape.len() < 2 {
            if self.skip_unsupported {
                return Ok(());
            }
            return Err(ExpandError::UnsupportedOp(format!(
                "MatMul needs rank >= 2; got A{:?}, B{:?}, Out{:?}",
                a_shape, b_shape, out_shape
            )));
        }

        let a_rank = a_shape.len();
        let b_rank = b_shape.len();
        let out_rank = out_shape.len();

        let m = a_shape[a_rank - 2];
        let k = a_shape[a_rank - 1];
        let n = b_shape[b_rank - 1];

        // Batch dimensions: everything except the last 2 dims of the output
        let batch_shape = &out_shape[..out_rank - 2];
        let batch_total: usize = batch_shape.iter().product::<usize>().max(1);

        // Strides for the batch dims of A, B, and Out in flat layout
        let _a_mat_size = m * k;
        let _b_mat_size = k * n;
        let out_mat_size = m * n;

        // Broadcast strides for batch dims: if A or B has fewer batch dims,
        // the missing leading dims are broadcast (stride 0).
        let a_batch = &a_shape[..a_rank - 2];
        let b_batch = &b_shape[..b_rank - 2];
        let a_batch_strides = broadcast_strides(a_batch, batch_shape);
        let b_batch_strides = broadcast_strides(b_batch, batch_shape);

        // Sampling: pick ~sqrt(max_samples) per axis (for the matmul dims)
        let (row_stride, col_stride) = if self.max_output_samples > 0 {
            let per_axis = (self.max_output_samples as f64).sqrt().ceil() as usize;
            let rs = if m > per_axis { m / per_axis } else { 1 };
            let cs = if n > per_axis { n / per_axis } else { 1 };
            (rs.max(1), cs.max(1))
        } else {
            (1, 1)
        };

        for batch_idx in 0..batch_total {
            // Compute batch multi-index
            let batch_multi = flat_to_multi(batch_idx, batch_shape);

            // Flat offset into A and B for this batch
            let a_batch_off: usize = batch_multi
                .iter()
                .zip(a_batch_strides.iter())
                .map(|(&mi, &s)| mi * s)
                .sum();
            let b_batch_off: usize = batch_multi
                .iter()
                .zip(b_batch_strides.iter())
                .map(|(&mi, &s)| mi * s)
                .sum();
            let out_batch_off = batch_idx * out_mat_size;

            let mut row = 0;
            while row < m {
                let mut col = 0;
                while col < n {
                    let mut expr = ScalarExpr::Literal { value: 0.0 };
                    for kk in 0..k {
                        let a_elem = ScalarExpr::Element {
                            tensor: a_id,
                            flat_index: a_batch_off + row * k + kk,
                        };
                        let b_elem = ScalarExpr::Element {
                            tensor: b_id,
                            flat_index: b_batch_off + kk * n + col,
                        };
                        let prod = ScalarExpr::Binary {
                            op: ScalarBinOp::Mul,
                            a: Box::new(a_elem),
                            b: Box::new(b_elem),
                        };
                        expr = ScalarExpr::Binary {
                            op: ScalarBinOp::Add,
                            a: Box::new(expr),
                            b: Box::new(prod),
                        };
                    }
                    out.push(OutputBinding {
                        output_tensor: out_id,
                        flat_index: out_batch_off + row * n + col,
                        expr,
                    });
                    col += col_stride;
                }
                row += row_stride;
            }
        }
        Ok(())
    }

    /// Identity view: output[flat_i] = input[flat_i].
    /// Used for Reshape, Squeeze, Unsqueeze where the flat layout is preserved.
    fn emit_identity_view(
        &self,
        input_id: GlobalId,
        out_id: GlobalId,
        out: &mut Vec<OutputBinding>,
    ) -> Result<(), ExpandError> {
        let out_shape = self.get_shape(&out_id)?.to_vec();

        for flat_idx in self.sample_indices_nd(&out_shape) {
            out.push(OutputBinding {
                output_tensor: out_id,
                flat_index: flat_idx,
                expr: ScalarExpr::Element {
                    tensor: input_id,
                    flat_index: flat_idx,
                },
            });
        }
        Ok(())
    }

    /// ClampMin: output[i] = max(input[i], min_val)
    fn emit_clamp_min(
        &self,
        input_id: GlobalId,
        out_id: GlobalId,
        min_val: f32,
        out: &mut Vec<OutputBinding>,
    ) -> Result<(), ExpandError> {
        let out_shape = self.get_shape(&out_id)?.to_vec();

        for flat_idx in self.sample_indices_nd(&out_shape) {
            let expr = ScalarExpr::Binary {
                op: ScalarBinOp::Max,
                a: Box::new(ScalarExpr::Element {
                    tensor: input_id,
                    flat_index: flat_idx,
                }),
                b: Box::new(ScalarExpr::Literal {
                    value: min_val as f64,
                }),
            };
            out.push(OutputBinding {
                output_tensor: out_id,
                flat_index: flat_idx,
                expr,
            });
        }
        Ok(())
    }

    /// Transpose: output[out_multi] = input[permuted_multi].
    fn emit_transpose(
        &self,
        input_id: GlobalId,
        out_id: GlobalId,
        perm: Option<&[i64]>,
        out: &mut Vec<OutputBinding>,
    ) -> Result<(), ExpandError> {
        let input_shape = self.get_shape(&input_id)?;
        let out_shape = self.get_shape(&out_id)?;
        let rank = input_shape.len();
        let _total: usize = out_shape.iter().product();
        let out_shape_owned = out_shape.to_vec();

        // Build full permutation
        let full_perm: Vec<usize> = if let Some(p) = perm {
            if p.len() < rank {
                // Partial perm: prepend identity
                let prefix_len = rank - p.len();
                let mut fp: Vec<usize> = (0..prefix_len).collect();
                fp.extend(p.iter().map(|&x| {
                    if x < 0 {
                        (x + rank as i64) as usize
                    } else {
                        x as usize
                    }
                }));
                fp
            } else {
                p.iter()
                    .map(|&x| {
                        if x < 0 {
                            (x + rank as i64) as usize
                        } else {
                            x as usize
                        }
                    })
                    .collect()
            }
        } else {
            // None = reverse all dims
            (0..rank).rev().collect()
        };

        // Compute input strides (row-major)
        let input_strides = row_major_strides(input_shape);

        for flat_out in self.sample_indices_nd(&out_shape_owned) {
            let out_multi = flat_to_multi(flat_out, &out_shape_owned);
            // Inverse: input_multi[perm[i]] = out_multi[i]
            let mut input_multi = vec![0usize; rank];
            for (out_dim, &in_dim) in full_perm.iter().enumerate() {
                input_multi[in_dim] = out_multi[out_dim];
            }
            let flat_in = multi_to_flat(&input_multi, &input_strides);

            out.push(OutputBinding {
                output_tensor: out_id,
                flat_index: flat_out,
                expr: ScalarExpr::Element {
                    tensor: input_id,
                    flat_index: flat_in,
                },
            });
        }
        Ok(())
    }

    /// Reduction with a binary accumulator op (Add for sum, Max for max, etc.)
    #[allow(clippy::too_many_arguments)]
    fn emit_reduce(
        &self,
        data_id: GlobalId,
        out_id: GlobalId,
        _axes_tensor: Option<GlobalId>,
        keepdims: bool,
        accum_op: ScalarBinOp,
        identity: f64,
        out: &mut Vec<OutputBinding>,
    ) -> Result<(), ExpandError> {
        let data_shape = self.get_shape(&data_id)?;
        let out_shape = self.get_shape(&out_id)?;
        let data_rank = data_shape.len();

        // Figure out which axes were reduced by comparing shapes.
        // With keepdims=true, reduced dims have size 1 in output.
        // Without keepdims, reduced dims are removed entirely.
        let reduced_axes = infer_reduced_axes(data_shape, out_shape, keepdims);
        let data_shape_owned = data_shape.to_vec();
        let out_shape_owned = out_shape.to_vec();

        let _total_out: usize = out_shape.iter().product();

        for flat_out in self.sample_indices_nd(&out_shape_owned) {
            let out_multi = flat_to_multi(flat_out, &out_shape_owned);

            // Map output multi-index back to data multi-index template.
            // Non-reduced dims come from the output index; reduced dims iterate.
            let mut data_multi_template = vec![0usize; data_rank];
            if keepdims {
                // keepdims=true: output has same rank as data, reduced dims are size 1
                for d in 0..data_rank {
                    if !reduced_axes.contains(&d) {
                        data_multi_template[d] = out_multi[d];
                    }
                }
            } else {
                // keepdims=false: reduced dims are removed, output has fewer dims
                let mut out_dim = 0;
                for (d, slot) in data_multi_template.iter_mut().enumerate().take(data_rank) {
                    if !reduced_axes.contains(&d) {
                        *slot = out_multi[out_dim];
                        out_dim += 1;
                    }
                }
            }

            // Build the reduction expression tree
            let data_strides = row_major_strides(&data_shape_owned);
            let mut expr = ScalarExpr::Literal { value: identity };

            // Iterate over all combinations of reduced dims
            let reduced_dims: Vec<usize> =
                reduced_axes.iter().map(|&d| data_shape_owned[d]).collect();
            let reduced_total: usize = reduced_dims.iter().product();

            for ri in 0..reduced_total {
                let mut data_multi = data_multi_template.clone();
                let mut remaining = ri;
                for (i, &ax) in reduced_axes.iter().enumerate().rev() {
                    data_multi[ax] = remaining % reduced_dims[i];
                    remaining /= reduced_dims[i];
                }
                let flat_data = multi_to_flat(&data_multi, &data_strides);

                let elem = ScalarExpr::Element {
                    tensor: data_id,
                    flat_index: flat_data,
                };
                expr = ScalarExpr::Binary {
                    op: accum_op,
                    a: Box::new(expr),
                    b: Box::new(elem),
                };
            }

            out.push(OutputBinding {
                output_tensor: out_id,
                flat_index: flat_out,
                expr,
            });
        }
        Ok(())
    }

    /// ReduceMean = ReduceSum / count
    fn emit_reduce_mean(
        &self,
        data_id: GlobalId,
        out_id: GlobalId,
        axes_tensor: Option<GlobalId>,
        keepdims: bool,
        out: &mut Vec<OutputBinding>,
    ) -> Result<(), ExpandError> {
        let data_shape = self.get_shape(&data_id)?;
        let out_shape = self.get_shape(&out_id)?;

        let reduced_axes = infer_reduced_axes(data_shape, out_shape, keepdims);
        let count: usize = reduced_axes.iter().map(|&d| data_shape[d]).product();

        // First emit sum bindings into a temporary vec
        let start = out.len();
        self.emit_reduce(
            data_id,
            out_id,
            axes_tensor,
            keepdims,
            ScalarBinOp::Add,
            0.0,
            out,
        )?;

        // Wrap each sum expression with / count
        let count_f64 = count as f64;
        for binding in &mut out[start..] {
            let sum_expr = std::mem::replace(
                &mut binding.expr,
                ScalarExpr::Literal { value: 0.0 }, // placeholder
            );
            binding.expr = ScalarExpr::Binary {
                op: ScalarBinOp::Div,
                a: Box::new(sum_expr),
                b: Box::new(ScalarExpr::Literal { value: count_f64 }),
            };
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Sampling
    // ------------------------------------------------------------------

    /// Returns an iterator over the flat indices to sample from `0..total`.
    /// Generate sampled flat indices with guaranteed per-axis variation.
    ///
    /// Unlike flat-stride sampling, this ensures that for an N-dimensional
    /// tensor, sampled indices vary along every axis. This is critical for
    /// the affine coefficient solver which needs at least two samples with
    /// different values on each axis.
    fn sample_indices_nd(&self, shape: &[usize]) -> Vec<usize> {
        let total: usize = shape.iter().product::<usize>().max(1);
        if self.max_output_samples == 0 || total <= self.max_output_samples {
            return (0..total).collect();
        }

        let ndim = shape.len();
        if ndim == 0 {
            return vec![0];
        }

        // Compute per-axis sample count: nth root of max_output_samples
        let per_axis = (self.max_output_samples as f64)
            .powf(1.0 / ndim as f64)
            .ceil() as usize;
        let per_axis = per_axis.max(2); // minimum 2 per axis for affine recovery

        // Per-axis strides
        let axis_strides: Vec<usize> = shape
            .iter()
            .map(|&s| if s > per_axis { s / per_axis } else { 1 })
            .collect();

        let row_strides = row_major_strides(shape);
        let mut indices = Vec::new();

        // Generate multi-dimensional grid of samples
        fn recurse(
            dim: usize,
            shape: &[usize],
            axis_strides: &[usize],
            row_strides: &[usize],
            current_flat: usize,
            indices: &mut Vec<usize>,
        ) {
            if dim == shape.len() {
                indices.push(current_flat);
                return;
            }
            let mut idx = 0;
            while idx < shape[dim] {
                recurse(
                    dim + 1,
                    shape,
                    axis_strides,
                    row_strides,
                    current_flat + idx * row_strides[dim],
                    indices,
                );
                idx += axis_strides[dim];
            }
        }

        recurse(0, shape, &axis_strides, &row_strides, 0, &mut indices);
        indices
    }
}

// ---------------------------------------------------------------------------
// Index helpers
// ---------------------------------------------------------------------------

fn flat_to_multi(flat: usize, shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    if ndim == 0 {
        return vec![];
    }
    let mut indices = vec![0usize; ndim];
    let mut remaining = flat;
    for i in (0..ndim).rev() {
        indices[i] = remaining % shape[i];
        remaining /= shape[i];
    }
    indices
}

fn multi_to_flat(indices: &[usize], strides: &[usize]) -> usize {
    indices.iter().zip(strides).map(|(&i, &s)| i * s).sum()
}

fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Compute broadcast strides for `source` broadcast into `target`.
fn broadcast_strides(source: &[usize], target: &[usize]) -> Vec<usize> {
    let t_ndim = target.len();
    let s_ndim = source.len();
    let offset = t_ndim.saturating_sub(s_ndim);
    let mut strides = vec![0usize; t_ndim];

    let mut stride = 1usize;
    for i in (0..s_ndim).rev() {
        let ti = i + offset;
        if source[i] == target[ti] {
            strides[ti] = stride;
            stride *= source[i];
        } else if source[i] == 1 {
            strides[ti] = 0;
        } else {
            panic!(
                "Incompatible shapes for broadcasting: {:?} vs {:?}",
                source, target
            );
        }
    }
    strides
}

/// Infer which data axes were reduced by comparing data_shape to out_shape.
fn infer_reduced_axes(data_shape: &[usize], out_shape: &[usize], keepdims: bool) -> Vec<usize> {
    if keepdims {
        // Same rank, reduced dims have size 1 in output
        assert_eq!(data_shape.len(), out_shape.len());
        (0..data_shape.len())
            .filter(|&d| data_shape[d] != out_shape[d] && out_shape[d] == 1)
            .collect()
    } else {
        // Output has fewer dims. Try to match non-reduced dims.
        // Walk data dims, match against output dims in order.
        let mut reduced = Vec::new();
        let mut out_dim = 0;
        for (d, &ds) in data_shape.iter().enumerate() {
            if out_dim < out_shape.len() && ds == out_shape[out_dim] {
                out_dim += 1;
            } else {
                reduced.push(d);
            }
        }
        reduced
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::graph::GlobalId;
    use crate::milli_graph::MilliOpGraph;
    use crate::milli_graph::ops::{MatMul, SimpleBinary, SimpleUnaryOp};

    fn make_id(rng: &mut impl rand::Rng) -> GlobalId {
        GlobalId::new(rng)
    }

    #[test]
    fn test_expand_add() {
        let mut rng = wyrand::WyRand::new(42);
        let ext_a = make_id(&mut rng);
        let ext_b = make_id(&mut rng);
        let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
        let int_a = input_map[&ext_a];
        let int_b = input_map[&ext_b];
        let int_out = SimpleBinary::add(&mut graph, int_a, int_b, &mut rng);

        let mut shapes = HashMap::new();
        shapes.insert(int_a, vec![2, 3]);
        shapes.insert(int_b, vec![2, 3]);
        shapes.insert(int_out, vec![2, 3]);

        let expander = ExprExpander::new(shapes);
        let bindings = expander.expand(&graph).unwrap();

        // 6 output elements, each with a Binary::Add expression
        assert_eq!(bindings.len(), 6);
        for (i, b) in bindings.iter().enumerate() {
            assert_eq!(b.output_tensor, int_out);
            assert_eq!(b.flat_index, i);
            match &b.expr {
                ScalarExpr::Binary {
                    op: ScalarBinOp::Add,
                    a,
                    b,
                } => {
                    match a.as_ref() {
                        ScalarExpr::Element { tensor, flat_index } => {
                            assert_eq!(*tensor, int_a);
                            assert_eq!(*flat_index, i);
                        }
                        _ => panic!("Expected Element"),
                    }
                    match b.as_ref() {
                        ScalarExpr::Element { tensor, flat_index } => {
                            assert_eq!(*tensor, int_b);
                            assert_eq!(*flat_index, i);
                        }
                        _ => panic!("Expected Element"),
                    }
                }
                _ => panic!("Expected Binary::Add"),
            }
        }
    }

    #[test]
    fn test_expand_broadcast() {
        let mut rng = wyrand::WyRand::new(42);
        let ext_a = make_id(&mut rng);
        let ext_b = make_id(&mut rng);
        let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
        let int_a = input_map[&ext_a];
        let int_b = input_map[&ext_b];
        let int_out = SimpleBinary::add(&mut graph, int_a, int_b, &mut rng);

        // a=[2,3], b=[1,3] -> broadcast b across dim 0
        let mut shapes = HashMap::new();
        shapes.insert(int_a, vec![2, 3]);
        shapes.insert(int_b, vec![1, 3]);
        shapes.insert(int_out, vec![2, 3]);

        let expander = ExprExpander::new(shapes);
        let bindings = expander.expand(&graph).unwrap();
        assert_eq!(bindings.len(), 6);

        // Check that b element indices wrap (stride 0 on dim 0)
        let b_indices: Vec<usize> = bindings
            .iter()
            .map(|bind| match &bind.expr {
                ScalarExpr::Binary { b, .. } => match b.as_ref() {
                    ScalarExpr::Element { flat_index, .. } => *flat_index,
                    _ => panic!(),
                },
                _ => panic!(),
            })
            .collect();
        // Row 0: b[0,0], b[0,1], b[0,2]; Row 1: b[0,0], b[0,1], b[0,2]
        assert_eq!(b_indices, vec![0, 1, 2, 0, 1, 2]);
    }

    #[test]
    fn test_expand_unary() {
        let mut rng = wyrand::WyRand::new(42);
        let ext_a = make_id(&mut rng);
        let (mut graph, input_map) = MilliOpGraph::new([ext_a], &mut rng);
        let int_a = input_map[&ext_a];
        let int_out = SimpleUnaryOp::neg(&mut graph, int_a, &mut rng);

        let mut shapes = HashMap::new();
        shapes.insert(int_a, vec![4]);
        shapes.insert(int_out, vec![4]);

        let expander = ExprExpander::new(shapes);
        let bindings = expander.expand(&graph).unwrap();
        assert_eq!(bindings.len(), 4);
        for (i, b) in bindings.iter().enumerate() {
            assert_eq!(b.flat_index, i);
            assert!(matches!(
                &b.expr,
                ScalarExpr::Unary {
                    op: ScalarUnaryOp::Neg,
                    ..
                }
            ));
        }
    }

    #[test]
    fn test_expand_matmul() {
        let mut rng = wyrand::WyRand::new(123);
        let ext_a = make_id(&mut rng);
        let ext_b = make_id(&mut rng);
        let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
        let int_a = input_map[&ext_a];
        let int_b = input_map[&ext_b];
        let int_out =
            MatMul::push_new_default_precision(&mut graph, int_a, int_b, DType::F32, &mut rng);

        let mut shapes = HashMap::new();
        shapes.insert(int_a, vec![2, 3]);
        shapes.insert(int_b, vec![3, 4]);
        shapes.insert(int_out, vec![2, 4]);

        let expander = ExprExpander::new(shapes);
        let bindings = expander.expand(&graph).unwrap();
        assert_eq!(bindings.len(), 8); // 2×4 output elements
    }

    #[test]
    fn test_expand_sampled() {
        let mut rng = wyrand::WyRand::new(42);
        let ext_a = make_id(&mut rng);
        let ext_b = make_id(&mut rng);
        let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
        let int_a = input_map[&ext_a];
        let int_b = input_map[&ext_b];
        let int_out = SimpleBinary::add(&mut graph, int_a, int_b, &mut rng);

        let mut shapes = HashMap::new();
        shapes.insert(int_a, vec![100]);
        shapes.insert(int_b, vec![100]);
        shapes.insert(int_out, vec![100]);

        let expander = ExprExpander::new_sampled(shapes, 10);
        let bindings = expander.expand(&graph).unwrap();
        assert!(bindings.len() <= 11); // ~10 samples
        assert!(bindings.len() >= 5);
    }

    #[test]
    fn test_sampled_nd_covers_all_axes() {
        // Regression test: flat-stride sampling of a [768, 768] tensor with
        // 64 samples produced stride=9216=12*768, placing ALL samples at
        // column 0. The affine solver could never recover the j-axis
        // coefficient. Multi-dimensional sampling must avoid this.
        let mut rng = wyrand::WyRand::new(42);
        let ext_a = make_id(&mut rng);
        let ext_b = make_id(&mut rng);
        let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
        let int_a = input_map[&ext_a];
        let int_b = input_map[&ext_b];
        let int_out = SimpleBinary::add(&mut graph, int_a, int_b, &mut rng);

        let mut shapes = HashMap::new();
        shapes.insert(int_a, vec![768, 768]);
        shapes.insert(int_b, vec![768, 768]);
        shapes.insert(int_out, vec![768, 768]);

        let expander = ExprExpander::new_sampled(shapes, 64);
        let bindings = expander.expand(&graph).unwrap();
        assert!(
            bindings.len() >= 4,
            "need at least 4 bindings, got {}",
            bindings.len()
        );

        // Verify: sampled positions must vary along BOTH axes
        let out_shape = vec![768usize, 768];
        let coords: Vec<Vec<usize>> = bindings
            .iter()
            .map(|b| flat_to_multi(b.flat_index, &out_shape))
            .collect();

        let rows: std::collections::HashSet<usize> = coords.iter().map(|c| c[0]).collect();
        let cols: std::collections::HashSet<usize> = coords.iter().map(|c| c[1]).collect();
        assert!(
            rows.len() >= 2,
            "samples must vary along axis 0, got {} unique rows",
            rows.len()
        );
        assert!(
            cols.len() >= 2,
            "samples must vary along axis 1, got {} unique cols",
            cols.len()
        );
    }

    #[test]
    fn test_expand_chain() {
        // mul then neg — two ops, both produce bindings
        let mut rng = wyrand::WyRand::new(42);
        let ext_a = make_id(&mut rng);
        let ext_b = make_id(&mut rng);
        let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
        let int_a = input_map[&ext_a];
        let int_b = input_map[&ext_b];
        let mul_out = SimpleBinary::mul(&mut graph, int_a, int_b, &mut rng);
        let neg_out = SimpleUnaryOp::neg(&mut graph, mul_out, &mut rng);

        let mut shapes = HashMap::new();
        shapes.insert(int_a, vec![8]);
        shapes.insert(int_b, vec![8]);
        shapes.insert(mul_out, vec![8]);
        shapes.insert(neg_out, vec![8]);

        let expander = ExprExpander::new(shapes);
        let bindings = expander.expand(&graph).unwrap();
        // 8 bindings for mul + 8 bindings for neg = 16
        assert_eq!(bindings.len(), 16);

        // First 8 are mul, next 8 are neg
        assert_eq!(bindings[0].output_tensor, mul_out);
        assert_eq!(bindings[8].output_tensor, neg_out);
    }
}
