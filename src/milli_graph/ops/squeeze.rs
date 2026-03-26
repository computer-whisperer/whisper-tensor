use crate::pool::Pool;
use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::dtype::DType;
use crate::graph::{GlobalId, Node};
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::migration::numeric_tensor::NumericTensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use typenum::P1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Squeeze {
    global_id: GlobalId,
    pub(crate) label: Option<String>,
    output: GlobalId,
    data: GlobalId,
    axes: GlobalId,
}

impl Squeeze {
    pub fn push_new(
        graph: &mut MilliOpGraph,
        data: GlobalId,
        axes: GlobalId,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        Self::push_new_with_label(graph, data, axes, None, rng)
    }

    pub fn push_new_with_label(
        graph: &mut MilliOpGraph,
        data: GlobalId,
        axes: GlobalId,
        label: Option<String>,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            output,
            data,
            axes,
            global_id: GlobalId::new(rng),
            label,
        };
        graph.push_op(AnyMilliOp::Squeeze(node));
        output
    }
}

impl Squeeze {
    pub fn lower_to_nano(&self, ctx: &mut crate::nano_graph::NanoLoweringContext) -> crate::milli_graph::ops::LowerResult {
        ctx.lower_view_op(self)
    }

    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl rand::Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.data, map);
        super::remap(&mut self.axes, map);
    }
}

impl Node for Squeeze {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> String {
        "Squeeze".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new([self.data, self.axes].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new([self.output].into_iter())
    }
}

impl MilliOp for Squeeze {
    fn infer<'p, P: Pool + 'p>(
        &self,
        known_inputs: &HashMap<GlobalId, crate::tensor_info::TensorInfo<'p, P>>,
        symbolic_resolver: &mut crate::symbolic_scalar::SymbolicResolver,
        pool: &'p P,
    ) -> Result<
        Vec<(GlobalId, crate::tensor_info::TensorInfo<'p, P>)>,
        MilliOpGraphError,
    > {
        use crate::scalar_info::ScalarInfoTyped;
        use crate::symbolic_scalar::SymbolicScalarTyped;
        use crate::tensor_info::TensorInfo;

        let data_info = known_inputs
            .get(&self.data)
            .ok_or(MilliOpGraphError::UnableToInfer)?;
        let axes_info = known_inputs
            .get(&self.axes)
            .ok_or(MilliOpGraphError::UnableToInfer)?;

        // Build output hint: same dtype as data, shape = data shape with axes removed.
        let out_dtype = data_info.dtype();
        let output_hint = if let (Some(data_ranked), Some(axes_values)) =
            (data_info.as_ranked(), axes_info.to_i64_vec())
        {
            let input_shape = data_ranked.shape();
            let input_rank = input_shape.len();
            let normalized_axes: Vec<usize> = axes_values
                .iter()
                .map(|&a| {
                    if a < 0 {
                        (input_rank as i64 + a) as usize
                    } else {
                        a as usize
                    }
                })
                .collect();
            let output_shape: Vec<ScalarInfoTyped<u64>> = input_shape
                .iter()
                .enumerate()
                .filter(|(i, _)| !normalized_axes.contains(i))
                .map(|(_, d)| d.clone())
                .collect();
            TensorInfo::from_dtype_and_shape_scalars(out_dtype, &output_shape)
        } else {
            TensorInfo::from_dtype_and_shape_scalars(out_dtype, &[])
        };

        // If both inputs are concrete, try constant fold via nano+pool_eval path.
        if let Some(results) = super::constant_fold(self, known_inputs, &[(self.output, output_hint)], pool) {
            return Ok(results);
        }

        let first_elem = data_info.first_element();

        // If we know the input shape and the axes are concrete, compute output shape
        if let (Some(data_ranked), Some(axes_values)) =
            (data_info.as_ranked(), axes_info.to_i64_vec())
        {
            let input_shape = data_ranked.shape();
            let input_rank = input_shape.len();

            let normalized_axes: Vec<usize> = axes_values
                .iter()
                .map(|&a| {
                    if a < 0 {
                        (input_rank as i64 + a) as usize
                    } else {
                        a as usize
                    }
                })
                .collect();

            let output_shape: Vec<ScalarInfoTyped<u64>> = input_shape
                .iter()
                .enumerate()
                .filter(|(i, _)| !normalized_axes.contains(i))
                .map(|(_, d)| d.clone())
                .collect();

            let out = TensorInfo::Ranked(crate::tensor_info::TensorInfoRanked::new(
                first_elem,
                output_shape,
                symbolic_resolver,
            ));
            return Ok(vec![((self.output, out))]);
        }

        // If axes are concrete, we can at least compute output rank
        if let Some(axes_values) = axes_info.to_i64_vec() {
            let num_axes = axes_values.len() as u32;

            if let ScalarInfoTyped::Numeric(input_rank) = data_info.rank() {
                let output_rank = input_rank - num_axes;
                let out = TensorInfo::new_from_first_element_and_rank(
                    first_elem,
                    ScalarInfoTyped::Numeric(output_rank),
                    symbolic_resolver,
                );
                return Ok(vec![((self.output, out))]);
            }
        }

        // Fallback: propagate dtype only
        let out = TensorInfo::new_from_first_element_and_rank(
            first_elem,
            ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(symbolic_resolver)),
            symbolic_resolver,
        );
        Ok(vec![((self.output, out))])
    }

    fn backward(
        &self,
        output_grads: &HashMap<GlobalId, GlobalId>,
        graph: &mut MilliOpGraph,
        rng: &mut impl rand::Rng,
    ) -> Option<HashMap<GlobalId, GlobalId>> {
        let grad_output = *output_grads.get(&self.output)?;
        // Squeeze backward = Unsqueeze along the same axes
        let grad_input = super::Unsqueeze::push_new(graph, grad_output, self.axes, rng);
        let mut result = HashMap::new();
        result.insert(self.data, grad_input);
        Some(result)
    }

    fn eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        _config: &super::MilliEvalConfig,
        backend: &mut EvalBackend,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, MilliOpGraphError>
    {
        let axes_ndarray = NDArrayNumericTensor::<DynRank>::try_from(
            inputs[&self.axes].cast(DType::I64, backend)?,
        )?;
        let axes = Vec::<i64>::try_from(axes_ndarray.try_to_rank::<P1>()?)?;
        if axes.len() == 1 {
            let axis = axes[0];
            let input_shape = inputs[&self.data].shape();
            let axis = if axis >= 0 {
                axis as usize
            } else {
                (input_shape.len() as i64 + axis) as usize
            };
            let output = inputs[&self.data].squeeze(axis)?;
            Ok(Box::new([(self.output, output)].into_iter()))
        } else {
            // Multiple axes (use reshape)
            let input_shape = inputs[&self.data].shape();
            let normalized_axes: Vec<usize> = axes
                .iter()
                .map(|&a| {
                    if a < 0 {
                        (input_shape.len() as i64 + a) as usize
                    } else {
                        a as usize
                    }
                })
                .collect();
            let mut output_shape = Vec::new();
            for (i, &dim) in input_shape.iter().enumerate() {
                if !normalized_axes.contains(&i) {
                    output_shape.push(dim);
                }
            }
            let output = inputs[&self.data].reshape(output_shape, backend)?;
            Ok(Box::new([(self.output, output)].into_iter()))
        }
    }

    fn eval_new<'p, P2: crate::pool::Pool + 'p>(
        &self,
        inputs: &[crate::numeric_tensor::NumericTensorView<'_, crate::tensor_rank::DynRank>],
        pool: &'p P2,
    ) -> Result<Vec<crate::numeric_tensor::NumericTensor<'p, crate::tensor_rank::DynRank, P2>>, crate::nano_graph::pool_eval::PoolEvalError> {
        use crate::numeric_tensor::{NumericTensor, TensorLayout};
        use crate::tensor_rank::DynRank;

        let data = &inputs[0];
        let axes_tensor = &inputs[1];
        let input_shape = data.shape();
        let input_rank = input_shape.len();
        let dtype = data.dtype();
        let numel = data.numel();

        let axes: Vec<usize> = (0..axes_tensor.numel())
            .map(|i| {
                let a = axes_tensor.read_element(i).to_i64();
                if a < 0 { (a + input_rank as i64) as usize } else { a as usize }
            })
            .collect();

        let output_shape: Vec<u64> = input_shape.iter().enumerate()
            .filter(|(i, _)| !axes.contains(i))
            .map(|(_, &d)| d)
            .collect();

        let layout = TensorLayout::<DynRank>::row_major(output_shape, dtype);
        let buf = pool.allocate(layout.buffer_size_bytes())
            .map_err(crate::nano_graph::pool_eval::PoolEvalError::Allocation)?;
        let mut out = NumericTensor::from_parts(buf, layout);
        for i in 0..numel {
            out.write_element(i, data.read_element(i));
        }
        Ok(vec![out])
    }

    fn lower_to_nano(&self, ctx: &mut crate::nano_graph::NanoLoweringContext) -> crate::milli_graph::ops::LowerResult {
        Squeeze::lower_to_nano(self, ctx)
    }
}
