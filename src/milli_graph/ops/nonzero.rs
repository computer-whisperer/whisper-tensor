use crate::graph::{GlobalId, Node};
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::pool::Pool;
use crate::scalar_info::ScalarInfo;
use crate::symbolic_scalar::{SymbolicScalar, SymbolicScalarTyped};
use crate::tensor_info::{MinimalTensor, TensorInfo};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NonZero {
    global_id: GlobalId,
    pub(crate) label: Option<String>,
    output: GlobalId,
    input: GlobalId,
}

impl NonZero {
    pub fn push_new(
        graph: &mut MilliOpGraph,
        input: GlobalId,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        Self::push_new_with_label(graph, input, None, rng)
    }

    pub fn push_new_with_label(
        graph: &mut MilliOpGraph,
        input: GlobalId,
        label: Option<String>,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            output,
            input,
            global_id: GlobalId::new(rng),
            label,
        };
        graph.push_op(AnyMilliOp::NonZero(node));
        output
    }
}

impl NonZero {
    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl rand::Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.input, map);
    }
}

impl Node for NonZero {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "NonZero".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.input].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.output].into_iter())
    }
}

impl MilliOp for NonZero {
    fn infer<'a, 'p, P: Pool + 'p>(
        &self,
        known_inputs: &HashMap<GlobalId, TensorInfo<'a, 'p, P>>,
        rng: &mut impl Rng,
        pool: &'p P,
    ) -> Result<Vec<(GlobalId, TensorInfo<'a, 'p, P>)>, MilliOpGraphError>
    where
        'p: 'a,
    {
        let input_info = known_inputs
            .get(&self.input)
            .ok_or(MilliOpGraphError::UnableToInfer)?;

        let out_dtype = crate::numeric_dtype::NumericDType::I64;

        // If input is concrete, compute nonzero indices directly.
        if let Some(tensor) = input_info.as_concrete() {
            let shape = tensor.shape().clone();
            let rank = shape.len();
            let numel = tensor.numel();

            // Find all nonzero element flat indices.
            let mut nz_flat: Vec<usize> = Vec::new();
            for i in 0..numel {
                if tensor.read_element(i).is_nonzero() {
                    nz_flat.push(i);
                }
            }
            let nnz = nz_flat.len();

            // Build output: [rank, nnz] I64 tensor.
            // Column j contains the multi-index of the j-th nonzero element.
            let out_shape = vec![rank as u64, nnz as u64];
            let layout = crate::numeric_tensor::TensorLayout::row_major(out_shape, out_dtype);
            if let Ok(buf) = pool.allocate(layout.buffer_size_bytes()) {
                let mut out_tensor = crate::numeric_tensor::NumericTensor::from_parts(buf, layout);

                // Compute row-major strides for decomposing flat index → multi-index.
                let mut strides = vec![1usize; rank];
                for i in (0..rank.saturating_sub(1)).rev() {
                    strides[i] = strides[i + 1] * shape[i + 1] as usize;
                }

                for (col, &flat) in nz_flat.iter().enumerate() {
                    let mut rem = flat;
                    #[allow(clippy::needless_range_loop)]
                    for row in 0..rank {
                        let idx = rem / strides[row];
                        rem %= strides[row];
                        // Output layout is row-major [rank, nnz]: element at (row, col) = row * nnz + col
                        out_tensor.write_element(
                            row * nnz + col,
                            crate::numeric_scalar::NumericScalar::from_i64(idx as i64),
                        );
                    }
                }

                return Ok(vec![(self.output, TensorInfo::from(out_tensor))]);
            }
        }

        // Fallback: dtype I64, unknown shape.
        let minimal = TensorInfo::Minimal(MinimalTensor::new(
            ScalarInfo::Symbolic(SymbolicScalar::new(out_dtype, rng)),
            SymbolicScalarTyped::new(rng),
        ));
        Ok(vec![(self.output, minimal)])
    }

    fn eval_new<'p, P2: crate::pool::Pool + 'p>(
        &self,
        inputs: &[crate::numeric_tensor::NumericTensorView<'_, crate::tensor_rank::DynRank>],
        pool: &'p P2,
    ) -> Result<
        Vec<crate::numeric_tensor::NumericTensor<'p, crate::tensor_rank::DynRank, P2>>,
        crate::nano_graph::pool_eval::PoolEvalError,
    > {
        use crate::numeric_dtype::NumericDType;
        use crate::numeric_scalar::NumericScalar;
        use crate::numeric_tensor::NumericTensor;
        use crate::tensor_rank::DynRank;

        let data = &inputs[0];
        let rank = data.shape().len();
        let data_layout = data.layout().clone();

        // Find all nonzero element indices and decompose to coordinates.
        let nz_coords: Vec<Vec<usize>> = (0..data.numel())
            .filter(|&i| data.read_element(i).is_nonzero())
            .map(|i| data_layout.flat_to_coords(i))
            .collect();
        let nnz = nz_coords.len();

        // Output: [rank, nnz] I64.
        let out = NumericTensor::<DynRank, P2>::from_fn(
            vec![rank as u64, nnz as u64],
            NumericDType::I64,
            pool,
            |flat| {
                let row = flat / nnz;
                let col = flat % nnz;
                NumericScalar::from_i64(nz_coords[col][row] as i64)
            },
        )
        .map_err(crate::nano_graph::pool_eval::PoolEvalError::Allocation)?;

        Ok(vec![out])
    }
}
