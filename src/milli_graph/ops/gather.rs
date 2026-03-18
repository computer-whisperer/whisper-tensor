use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::dtype::DType;
use crate::graph::{GlobalId, Node};
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::nano_graph::lower::{DimKind, TensorAtomMap};
use crate::nano_graph::ops::{ScalarBinOp, ScalarOp};
use crate::nano_graph::pattern::InputRef;
use crate::numeric_scalar::NumericScalar;
use crate::numeric_tensor::NumericTensor;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gather {
    global_id: GlobalId,
    pub(crate) label: Option<String>,
    output: GlobalId,
    data: GlobalId,
    indices: GlobalId,
    axis: i64,
}

impl Gather {
    pub fn push_new(
        graph: &mut MilliOpGraph,
        data: GlobalId,
        indices: GlobalId,
        axis: i64,
        rng: &mut impl Rng,
    ) -> GlobalId {
        Self::push_new_with_label(graph, data, indices, axis, None, rng)
    }

    pub fn push_new_with_label(
        graph: &mut MilliOpGraph,
        data: GlobalId,
        indices: GlobalId,
        axis: i64,
        label: Option<String>,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            global_id: GlobalId::new(rng),
            label,
            output,
            data,
            indices,
            axis,
        };
        graph.push_op(AnyMilliOp::Gather(node));
        output
    }
}

impl Gather {
    pub fn axis(&self) -> i64 {
        self.axis
    }
    pub fn data_id(&self) -> GlobalId {
        self.data
    }
    pub fn indices_id(&self) -> GlobalId {
        self.indices
    }
    pub fn output_id(&self) -> GlobalId {
        self.output
    }

    pub fn lower_to_nano(&self, ctx: &mut crate::nano_graph::NanoLoweringContext) {
        let all_infos = ctx.all_infos;
        let data_id = self.data_id();
        let indices_id = self.indices_id();
        let out_id = self.output_id();

        // If both inputs are fully numeric (constant-folded), treat as constant.
        let all_numeric = [data_id, indices_id]
            .iter()
            .all(|id| all_infos.get(id).is_some_and(|i| i.as_numeric().is_some()));
        if all_numeric && let Some(out_info) = all_infos.get(&out_id) {
            ctx.register_constant(out_id, out_info);
            return;
        }

        let Some(data_map) = ctx.tensor_map.get(&data_id).cloned() else {
            ctx.lower_as_boundary_named(self, "Gather");
            return;
        };
        let Some(indices_map) = ctx.tensor_map.get(&indices_id).cloned() else {
            ctx.lower_as_boundary_named(self, "Gather");
            return;
        };
        let Some(out_info) = all_infos.get(&out_id) else {
            ctx.lower_as_boundary_named(self, "Gather");
            return;
        };
        let Some(data_info) = all_infos.get(&data_id) else {
            ctx.lower_as_boundary_named(self, "Gather");
            return;
        };
        let Some(_indices_info) = all_infos.get(&indices_id) else {
            ctx.lower_as_boundary_named(self, "Gather");
            return;
        };

        // Normalize axis.
        let data_rank = match data_info.rank_if_known() {
            Some(r) => r,
            None => {
                ctx.lower_as_boundary_named(self, "Gather");
                return;
            }
        };
        let axis = if self.axis() < 0 {
            (self.axis() + data_rank as i64) as usize
        } else {
            self.axis() as usize
        };

        // Only handle axis=0 for now.
        if axis != 0 {
            ctx.lower_as_boundary_named(self, "Gather");
            return;
        }

        // Indices can be any rank — we treat them as a flat list of index values.
        // The output shape is indices_shape + data_shape[1:] for axis=0.

        // data shape must be fully known (it's an embedding table).
        let data_known: Vec<u64> = data_map
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
        if data_known.len() != data_rank {
            // Some data dims are symbolic — can't lower.
            ctx.lower_as_boundary_named(self, "Gather");
            return;
        }

        // For axis=0: data=[V, D, ...], D_total = product of data_known[1..]
        let d_total: u64 = data_known[1..].iter().product();
        let d_total = d_total.max(1); // handle scalar gather (data_rank == 1)

        // Check if indices are symbolic (runtime) or known.
        let indices_sym = !indices_map.sym_dims.is_empty();

        // Output classification.
        let Some((out_layout, out_known_dims, out_sym_dims, out_count)) =
            ctx.classify_dims(out_info)
        else {
            ctx.lower_as_boundary_named(self, "Gather");
            return;
        };
        let out_count = out_count.max(1);

        let out_dt = out_info.dtype();

        // The output shape for axis=0, 1D indices is [N, D, ...] where N = indices count.
        // N may be symbolic (runtime token IDs) or known.
        //
        // For each output element (i, j) where j indexes the D_total trailing dims:
        //   flat_index_into_data = indices[i] * D_total + j
        //   output[i, j] = data[flat_index_into_data]
        //
        // We emit:
        //   1. A Literal group for the stride constant (D_total)
        //   2. A Mul group: indices[i] * D_total
        //   3. An Add group: (indices[i] * D_total) + j  (j is the column offset per atom)
        //   4. An IndirectLoad group: load from data table at the computed index

        // Step 1: stride literal (single atom)
        let stride_lit = ctx.nano.push_atom(
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(d_total as f32)),
            vec![],
            vec![],
        );

        if indices_sym {
            // Indices are symbolic (runtime). The indices_map has N atoms with sym_dims.
            // Output has N*D_total known atoms with the same sym_dims.
            // For each output atom j (0..D_total), it reads indices[0] (the single atom
            // in the indices group), since the sym dim handles the N dimension.

            // Simple case: indices_map.count == 1, sym_dims present
            // Output should have count == D_total, same sym_dims
            if indices_map.count != 1 || out_count != d_total {
                ctx.lower_as_boundary_named(self, "Gather");
                return;
            }

            // Mul group: 1 atom * broadcast stride → 1 atom (sym_dims from indices)
            let mul_id = ctx.nano.push_atom(
                DType::F32,
                ScalarOp::Binary {
                    op: ScalarBinOp::Mul,
                    compute_dtype: DType::F32,
                },
                indices_map.sym_dims.clone(),
                vec![
                    InputRef::Broadcast(indices_map.base_id),
                    InputRef::Broadcast(stride_lit),
                ],
            );

            // Step 3: Add group — D_total atoms, each adds its column offset j
            let col_offsets_base = ctx.nano.push_atom(
                DType::F32,
                ScalarOp::Literal(NumericScalar::F32(0.0)),
                vec![],
                vec![],
            );
            for j in 1..d_total {
                ctx.nano.push_atom(
                    DType::F32,
                    ScalarOp::Literal(NumericScalar::F32(j as f32)),
                    vec![],
                    vec![],
                );
            }

            let add_id = ctx.nano.push_group(
                d_total,
                DType::F32,
                ScalarOp::Binary {
                    op: ScalarBinOp::Add,
                    compute_dtype: DType::F32,
                },
                indices_map.sym_dims.clone(),
                vec![
                    InputRef::Broadcast(mul_id),
                    InputRef::Affine {
                        base: col_offsets_base,
                        stride: 1,
                    },
                ],
            );

            // Step 4: IndirectLoad group — D_total atoms, each loads from data table
            let base_id = ctx.nano.push_group(
                d_total,
                out_dt,
                ScalarOp::IndirectLoad {
                    table_base: data_map.base_id,
                },
                indices_map.sym_dims.clone(),
                vec![InputRef::Affine {
                    base: add_id,
                    stride: 1,
                }],
            );

            let out_strides = TensorAtomMap::compute_strides(&out_known_dims);
            ctx.tensor_map.insert(
                out_id,
                TensorAtomMap::simple(
                    base_id,
                    d_total,
                    out_dt,
                    out_layout,
                    out_strides,
                    out_sym_dims,
                ),
            );
        } else {
            // Indices are fully known (constant). out_count = indices_count * D_total.
            let _indices_count = indices_map.count;

            // Build the indices InputRef: for output atom `flat`, row = flat / D_total
            let indices_ref = if d_total == 1 {
                // 1:1 mapping
                InputRef::Affine {
                    base: indices_map.base_id,
                    stride: 1,
                }
            } else {
                // For each output flat index, the row is flat / D_total
                let mut ids = Vec::with_capacity(out_count as usize);
                for flat in 0..out_count {
                    let row = flat / d_total;
                    ids.push(indices_map.base_id.offset(row));
                }
                InputRef::Explicit(ids)
            };

            // Mul group: out_count atoms, each computes indices[row] * D_total
            let mul_base = ctx.nano.push_group(
                out_count,
                DType::F32,
                ScalarOp::Binary {
                    op: ScalarBinOp::Mul,
                    compute_dtype: DType::F32,
                },
                vec![],
                vec![indices_ref, InputRef::Broadcast(stride_lit)],
            );

            // Column offset literals: d_total singletons with values [0, 1, ..., d_total-1].
            let col_lit_base = ctx.nano.push_atom(
                DType::F32,
                ScalarOp::Literal(NumericScalar::F32(0.0)),
                vec![],
                vec![],
            );
            for j in 1..d_total {
                ctx.nano.push_atom(
                    DType::F32,
                    ScalarOp::Literal(NumericScalar::F32(j as f32)),
                    vec![],
                    vec![],
                );
            }

            // Add group: mul_result + column_offset (via Modular over d_total literals)
            let add_base = ctx.nano.push_group(
                out_count,
                DType::F32,
                ScalarOp::Binary {
                    op: ScalarBinOp::Add,
                    compute_dtype: DType::F32,
                },
                vec![],
                vec![
                    InputRef::Affine {
                        base: mul_base,
                        stride: 1,
                    },
                    InputRef::Modular {
                        base: col_lit_base,
                        stride: 1,
                        modulus: d_total,
                    },
                ],
            );

            // IndirectLoad group
            let base_id = ctx.nano.push_group(
                out_count,
                out_dt,
                ScalarOp::IndirectLoad {
                    table_base: data_map.base_id,
                },
                vec![],
                vec![InputRef::Affine {
                    base: add_base,
                    stride: 1,
                }],
            );

            let out_strides = TensorAtomMap::compute_strides(&out_known_dims);
            ctx.tensor_map.insert(
                out_id,
                TensorAtomMap::simple(
                    base_id,
                    out_count,
                    out_dt,
                    out_layout,
                    out_strides,
                    out_sym_dims,
                ),
            );
        }
    }

    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl rand::Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.data, map);
        super::remap(&mut self.indices, map);
    }
}

impl Node for Gather {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Gather".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.data, self.indices].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.output].into_iter())
    }
}

impl MilliOp for Gather {
    fn infer(
        &self,
        known_inputs: &HashMap<GlobalId, crate::tensor_info::TensorInfo>,
        _symbolic_resolver: &mut crate::symbolic_scalar::SymbolicResolver,
        backend: &mut EvalBackend,
    ) -> Result<
        Box<dyn Iterator<Item = (GlobalId, crate::tensor_info::TensorInfo)>>,
        crate::milli_graph::MilliOpGraphError,
    > {
        use crate::tensor_info::TensorInfo;

        let data_info = known_inputs
            .get(&self.data)
            .ok_or(MilliOpGraphError::UnableToInfer)?;
        let indices_info = known_inputs
            .get(&self.indices)
            .ok_or(MilliOpGraphError::UnableToInfer)?;

        // If both inputs are concrete, fall back to eval for full precision.
        if data_info.as_numeric().is_some() && indices_info.as_numeric().is_some() {
            let mut resolved = HashMap::new();
            resolved.insert(self.data, data_info.as_numeric().unwrap().clone());
            resolved.insert(self.indices, indices_info.as_numeric().unwrap().clone());
            let collected: Vec<(GlobalId, TensorInfo)> = self
                .eval(&resolved, backend)?
                .map(|(a, b)| (a, TensorInfo::from(b)))
                .collect();
            return Ok(Box::new(collected.into_iter()));
        }

        // Shape-only inference: output_shape = data[:axis] + indices.shape + data[axis+1:]
        let data_ranked = data_info
            .as_ranked()
            .ok_or(MilliOpGraphError::UnableToInfer)?;
        let indices_ranked = indices_info
            .as_ranked()
            .ok_or(MilliOpGraphError::UnableToInfer)?;

        let data_shape = data_ranked.shape();
        let indices_shape = indices_ranked.shape();
        let data_rank = data_shape.len();

        let axis = if self.axis < 0 {
            (self.axis + data_rank as i64) as usize
        } else {
            self.axis as usize
        };

        // Build output dims: data[:axis] + indices_shape + data[axis+1:]
        let mut out_dims: Vec<crate::scalar_info::ScalarInfoTyped<u64>> = Vec::new();
        out_dims.extend_from_slice(&data_shape[..axis]);
        out_dims.extend_from_slice(&indices_shape);
        out_dims.extend_from_slice(&data_shape[axis + 1..]);

        let out_dtype = data_info.dtype();
        let out_info = TensorInfo::from_dtype_and_shape_scalars(out_dtype, &out_dims);
        Ok(Box::new([(self.output, out_info)].into_iter()))
    }

    fn backward(
        &self,
        output_grads: &HashMap<GlobalId, GlobalId>,
        graph: &mut MilliOpGraph,
        rng: &mut impl Rng,
    ) -> Option<HashMap<GlobalId, GlobalId>> {
        let grad_output = *output_grads.get(&self.output)?;
        let mut result = HashMap::new();
        // Only the data input is differentiable (not indices)
        let grad_data =
            GatherGrad::push_new(graph, grad_output, self.indices, self.data, self.axis, rng);
        result.insert(self.data, grad_data);
        Some(result)
    }

    fn eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, MilliOpGraphError>
    {
        let out = NumericTensor::<DynRank>::gather(
            &inputs[&self.data],
            &inputs[&self.indices],
            self.axis,
            backend,
        )?;
        Ok(Box::new([(self.output, out)].into_iter()))
    }
}

// ---------------------------------------------------------------------------
// GatherGrad: scatter-add grad_output back into data shape
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatherGrad {
    global_id: GlobalId,
    pub(crate) label: Option<String>,
    output: GlobalId,
    grad_output: GlobalId,
    indices: GlobalId,
    /// Original data tensor — needed for its shape
    data: GlobalId,
    axis: i64,
}

impl GatherGrad {
    pub fn push_new(
        graph: &mut MilliOpGraph,
        grad_output: GlobalId,
        indices: GlobalId,
        data: GlobalId,
        axis: i64,
        rng: &mut impl Rng,
    ) -> GlobalId {
        Self::push_new_with_label(graph, grad_output, indices, data, axis, None, rng)
    }

    pub fn push_new_with_label(
        graph: &mut MilliOpGraph,
        grad_output: GlobalId,
        indices: GlobalId,
        data: GlobalId,
        axis: i64,
        label: Option<String>,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            global_id: GlobalId::new(rng),
            label,
            output,
            grad_output,
            indices,
            data,
            axis,
        };
        graph.push_op(AnyMilliOp::GatherGrad(node));
        output
    }

    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.grad_output, map);
        super::remap(&mut self.indices, map);
        super::remap(&mut self.data, map);
    }
}

impl Node for GatherGrad {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> String {
        "GatherGrad".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.grad_output, self.indices, self.data].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl MilliOp for GatherGrad {
    fn eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, MilliOpGraphError>
    {
        let grad_out = &inputs[&self.grad_output];
        let indices = &inputs[&self.indices];
        let data = &inputs[&self.data];

        let data_shape: Vec<usize> = data.shape().iter().map(|&x| x as usize).collect();
        let rank = data_shape.len();
        let axis = if self.axis < 0 {
            (self.axis + rank as i64) as usize
        } else {
            self.axis as usize
        };

        let grad_f32 = grad_out.cast(DType::F32, backend)?;
        let grad_data: Vec<f32> = grad_f32.to_ndarray()?.flatten().try_into()?;

        let indices_i64 = indices.cast(DType::I64, backend)?;
        let idx_data: Vec<i64> = indices_i64.to_ndarray()?.flatten().try_into()?;
        let idx_shape: Vec<usize> = indices.shape().iter().map(|&x| x as usize).collect();

        let out_size: usize = data_shape.iter().product();
        let mut result = vec![0.0f32; out_size];

        // Compute strides for the data tensor
        let mut data_strides = vec![1usize; rank];
        for i in (0..rank - 1).rev() {
            data_strides[i] = data_strides[i + 1] * data_shape[i + 1];
        }

        // The grad_output shape is: data_shape[..axis] ++ idx_shape ++ data_shape[axis+1..]
        // We iterate over every element of grad_output, compute which data element it
        // came from, and scatter-add.
        let grad_shape: Vec<usize> = grad_out.shape().iter().map(|&x| x as usize).collect();
        let grad_rank = grad_shape.len();
        let mut grad_strides = vec![1usize; grad_rank];
        for i in (0..grad_rank.saturating_sub(1)).rev() {
            grad_strides[i] = grad_strides[i + 1] * grad_shape[i + 1];
        }

        let prefix_dims = axis;
        let suffix_dims = rank - axis - 1;
        let idx_ndim = idx_shape.len();
        let axis_len = data_shape[axis] as i64;

        for (flat_g, &val) in grad_data.iter().enumerate() {
            if val == 0.0 {
                continue;
            }

            // Decompose flat_g into (prefix_coords, idx_coords, suffix_coords)
            let mut rem = flat_g;

            // Prefix coords (dims 0..axis of data)
            let mut data_flat = 0usize;
            for d in 0..prefix_dims {
                let coord = rem / grad_strides[d];
                rem %= grad_strides[d];
                data_flat += coord * data_strides[d];
            }

            // Index coords (middle dims from indices shape)
            let mut idx_flat = 0usize;
            let mut idx_strides = vec![1usize; idx_ndim];
            for i in (0..idx_ndim.saturating_sub(1)).rev() {
                idx_strides[i] = idx_strides[i + 1] * idx_shape[i + 1];
            }
            for (d, &stride) in idx_strides.iter().enumerate() {
                let grad_dim = prefix_dims + d;
                let coord = rem / grad_strides[grad_dim];
                rem %= grad_strides[grad_dim];
                idx_flat += coord * stride;
            }

            // Suffix coords (dims axis+1.. of data)
            for d in 0..suffix_dims {
                let grad_dim = prefix_dims + idx_ndim + d;
                let coord = rem / grad_strides[grad_dim];
                rem %= grad_strides[grad_dim];
                data_flat += coord * data_strides[axis + 1 + d];
            }

            // Look up the actual index along the gather axis
            let mut idx_val = idx_data[idx_flat];
            if idx_val < 0 {
                idx_val += axis_len;
            }
            data_flat += idx_val as usize * data_strides[axis];

            result[data_flat] += val;
        }

        let result_tensor = NumericTensor::<DynRank>::from_vec_shape(result, data_shape)?;
        Ok(Box::new(std::iter::once((self.output, result_tensor))))
    }
}
