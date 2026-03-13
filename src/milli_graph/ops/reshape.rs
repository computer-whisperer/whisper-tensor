use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::dtype::DType;
use crate::graph::{GlobalId, Node};
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::numeric_tensor::NumericTensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use typenum::P1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reshape {
    global_id: GlobalId,
    output: GlobalId,
    data: GlobalId,
    shape: GlobalId,
    allowzero: bool,
}

impl Reshape {
    pub fn push_new(
        graph: &mut MilliOpGraph,
        data: GlobalId,
        shape: GlobalId,
        allowzero: bool,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            global_id: GlobalId::new(rng),
            output,
            data,
            shape,
            allowzero,
        };
        graph.push_op(AnyMilliOp::Reshape(node));
        output
    }

    fn calculate_new_shape(
        &self,
        data_input_shape: &[u64],
        shape_input_value: &[i64],
    ) -> Result<Vec<u64>, MilliOpGraphError> {
        let mut new_shape_dims = vec![];
        let mut backfill_dim: Option<usize> = None;
        for i in 0..shape_input_value.len() {
            new_shape_dims.push(if shape_input_value[i] == 0 {
                data_input_shape[i]
            } else if shape_input_value[i] == -1 {
                if backfill_dim.is_some() {
                    // Only one dimension can be inferred
                    Err(MilliOpGraphError::InvalidInput("Reshape".to_string()))?
                }
                backfill_dim = Some(i);
                1
            } else if shape_input_value[i] < -1 {
                Err(MilliOpGraphError::InvalidInput("Reshape".to_string()))?
            } else {
                shape_input_value[i] as u64
            });
        }

        // Backfill the inferred dimension
        if let Some(i) = backfill_dim {
            let total_input_size = data_input_shape.iter().product::<u64>();

            // Calculate the current product of the dimensions
            let mut current_product = 1;
            for (j, dim) in new_shape_dims.iter().enumerate() {
                if j != i {
                    current_product *= dim;
                }
            }
            // Calculate the inferred dimension size
            let inferred_size = total_input_size / current_product;
            new_shape_dims[i] = inferred_size;
        }
        let output_shape = new_shape_dims;

        // Verify that the dimensions are compatible
        if output_shape.iter().product::<u64>() != data_input_shape.iter().product::<u64>() {
            Err(MilliOpGraphError::InvalidInput("Reshape".to_string()))?
        }

        Ok(output_shape)
    }
}

impl Reshape {
    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl rand::Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.data, map);
        super::remap(&mut self.shape, map);
    }
}

impl Node for Reshape {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Reshape".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.data, self.shape].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.output].into_iter())
    }
}

impl MilliOp for Reshape {
    fn infer(
        &self,
        known_inputs: &HashMap<GlobalId, crate::tensor_info::TensorInfo>,
        symbolic_resolver: &mut crate::symbolic_scalar::SymbolicResolver,
        backend: &mut EvalBackend,
    ) -> Result<
        Box<dyn Iterator<Item = (GlobalId, crate::tensor_info::TensorInfo)>>,
        MilliOpGraphError,
    > {
        use crate::scalar_info::ScalarInfoTyped;
        use crate::symbolic_scalar::SymbolicScalarTyped;
        use crate::tensor_info::TensorInfo;

        let data_info = known_inputs
            .get(&self.data)
            .ok_or(MilliOpGraphError::UnableToInfer)?;
        let shape_info = known_inputs
            .get(&self.shape)
            .ok_or(MilliOpGraphError::UnableToInfer)?;

        // If both inputs are concrete, delegate to eval
        if let (Some(data_num), Some(shape_num)) = (data_info.as_numeric(), shape_info.as_numeric())
        {
            let inputs = HashMap::from([
                (self.data, data_num.clone()),
                (self.shape, shape_num.clone()),
            ]);
            let out: Vec<_> = self
                .eval(&inputs, backend)?
                .map(|(id, t)| (id, TensorInfo::from(t)))
                .collect();
            return Ok(Box::new(out.into_iter()));
        }

        let first_elem = data_info.first_element();

        // If shape tensor is concrete, we can determine the output shape
        if let Some(shape_num) = shape_info.as_numeric() {
            let shape_values: Vec<i64> = shape_num
                .cast(DType::I64, backend)?
                .try_to_rank::<P1>()?
                .try_into()?;

            // Try to compute output shape, handling -1 and 0 dims
            let data_shape_known = data_info.as_ranked().map(|r| {
                r.shape()
                    .iter()
                    .map(|d| d.as_numeric().copied())
                    .collect::<Vec<_>>()
            });

            let mut output_dims: Vec<ScalarInfoTyped<u64>> = Vec::new();
            let mut has_minus_one = false;
            for (i, &sv) in shape_values.iter().enumerate() {
                if sv == 0 {
                    // Copy from input shape if known
                    if let Some(ref ds) = data_shape_known {
                        if let Some(Some(d)) = ds.get(i) {
                            output_dims.push(ScalarInfoTyped::Numeric(*d));
                        } else {
                            output_dims.push(ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(
                                symbolic_resolver,
                            )));
                        }
                    } else {
                        output_dims.push(ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(
                            symbolic_resolver,
                        )));
                    }
                } else if sv == -1 {
                    has_minus_one = true;
                    // Placeholder, will try to resolve below
                    output_dims.push(ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(
                        symbolic_resolver,
                    )));
                } else if sv > 0 {
                    output_dims.push(ScalarInfoTyped::Numeric(sv as u64));
                } else {
                    // Invalid shape value
                    return Err(MilliOpGraphError::InvalidInput("Reshape".to_string()));
                }
            }

            // Try to resolve -1 dimension if input shape is fully known
            if has_minus_one
                && let Some(ref ds) = data_shape_known
                && ds.iter().all(|d| d.is_some())
            {
                let total: u64 = ds.iter().map(|d| d.unwrap()).product();
                let known_product: Option<u64> = output_dims
                    .iter()
                    .enumerate()
                    .filter(|(idx, _)| shape_values[*idx] != -1)
                    .try_fold(1u64, |acc, (_, d)| d.as_numeric().map(|v| acc * v));
                if let Some(kp) = known_product
                    && kp > 0
                {
                    let inferred = total / kp;
                    let minus_one_idx = shape_values.iter().position(|&v| v == -1).unwrap();
                    output_dims[minus_one_idx] = ScalarInfoTyped::Numeric(inferred);
                }
            }

            let out = TensorInfo::Ranked(crate::tensor_info::TensorInfoRanked::new(
                first_elem,
                output_dims,
                symbolic_resolver,
            ));
            return Ok(Box::new([(self.output, out)].into_iter()));
        }

        // Shape tensor is not concrete. Try to get output rank from shape tensor's
        // rank (shape tensor is 1-D, so its length = output rank)
        if let Some(shape_ranked) = shape_info.as_ranked() {
            let shape_shape = shape_ranked.shape();
            if shape_shape.len() == 1
                && let Some(output_rank) = shape_shape[0].as_numeric()
            {
                let out = TensorInfo::new_from_first_element_and_rank(
                    first_elem,
                    ScalarInfoTyped::Numeric(*output_rank as u32),
                    symbolic_resolver,
                );
                return Ok(Box::new([(self.output, out)].into_iter()));
            }
        }

        // Fallback: propagate dtype only
        let out = TensorInfo::new_from_first_element_and_rank(
            first_elem,
            ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(symbolic_resolver)),
            symbolic_resolver,
        );
        Ok(Box::new([(self.output, out)].into_iter()))
    }

    fn backward(
        &self,
        output_grads: &HashMap<GlobalId, GlobalId>,
        graph: &mut MilliOpGraph,
        rng: &mut impl rand::Rng,
    ) -> Option<HashMap<GlobalId, GlobalId>> {
        let grad_output = *output_grads.get(&self.output)?;
        // Reshape grad back to input shape
        let input_shape = super::Shape::push_new(graph, self.data, rng);
        let grad_input = Reshape::push_new(graph, grad_output, input_shape, false, rng);
        let mut result = HashMap::new();
        result.insert(self.data, grad_input);
        Some(result)
    }

    fn eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, MilliOpGraphError>
    {
        let data_input = &inputs[&self.data];
        let shape_input = &inputs[&self.shape];
        let shape_input_value: Vec<i64> = shape_input
            .cast(DType::I64, backend)?
            .try_to_rank::<P1>()?
            .try_into()?;

        let output_shape = self.calculate_new_shape(&data_input.shape(), &shape_input_value)?;

        let output_value = data_input.reshape(output_shape, backend)?;

        Ok(Box::new([(self.output, output_value)].into_iter()))
    }
}
