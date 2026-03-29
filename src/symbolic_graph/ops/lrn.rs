use crate::backends::eval_backend::EvalBackend;
use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::graph::{GlobalId, Node, Property, PropertyValue};
use crate::migration::numeric_tensor::NumericTensor;
use crate::milli_graph::{MilliLoweringContext, MilliOpGraph};
use crate::onnx::AttributeProto;
use crate::symbolic_graph::ops::{EvalError, Operation};
use crate::symbolic_graph::{ONNXDecodingError, query_attribute_float, query_attribute_int};
use crate::tensor_rank::DynRank;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// ONNX LRN (Local Response Normalization) operator.
///
/// Y[n,c,d1,...,dk] = X[n,c,d1,...,dk] / (bias + alpha/size * sum(X[n,j,d1,...,dk]^2))^beta
/// where j ranges over max(0, c - floor((size-1)/2)) to min(C-1, c + ceil((size-1)/2))
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LrnOperation {
    global_id: GlobalId,
    input: GlobalId,
    output: GlobalId,
    alpha: f32,
    beta: f32,
    bias: f32,
    size: i64,
}

impl LrnOperation {
    pub fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.is_empty() {
            return Err(ONNXDecodingError::InvalidOperatorInputs("LRN"));
        }
        if outputs.is_empty() {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("LRN"));
        }

        let size = query_attribute_int(attributes, "size")
            .ok_or(ONNXDecodingError::MissingField("size"))?;

        Ok(Self {
            global_id: GlobalId::new(rng),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("LRN"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("LRN"))?,
            alpha: query_attribute_float(attributes, "alpha").unwrap_or(0.0001),
            beta: query_attribute_float(attributes, "beta").unwrap_or(0.75),
            bias: query_attribute_float(attributes, "bias").unwrap_or(1.0),
            size,
        })
    }
}

impl Node for LrnOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "LRN".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.input))
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

fn scalar_f32(v: f32) -> Result<NumericTensor<DynRank>, EvalError> {
    Ok(NumericTensor::NDArray(
        NDArrayNumericTensor::from_vec_shape(vec![v], &vec![1u64])?,
    ))
}

fn zeros_f32(shape: &[u64]) -> Result<NumericTensor<DynRank>, EvalError> {
    let n: u64 = shape.iter().product();
    Ok(NumericTensor::NDArray(
        NDArrayNumericTensor::from_vec_shape(vec![0.0f32; n as usize], &shape.to_vec())?,
    ))
}

impl Operation for LrnOperation {
    fn parameters(&self) -> Vec<Property> {
        vec![
            Property::new("alpha", PropertyValue::Float(self.alpha.into())),
            Property::new("beta", PropertyValue::Float(self.beta.into())),
            Property::new("bias", PropertyValue::Float(self.bias.into())),
            Property::new("size", PropertyValue::Int(self.size)),
        ]
    }

    fn is_differentiable(&self) -> bool {
        false
    }

    fn eval(
        &self,
        backend: &mut EvalBackend,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, EvalError> {
        let x = &inputs[&self.input];
        let shape = x.shape();
        let c_dim = shape[1] as usize;
        let size = self.size as usize;
        let half = (size - 1) / 2;

        // x_sq = x^2
        let x_sq = NumericTensor::mul(x, x, backend)?;

        // Pad channel dimension with zeros via concat:
        // x_sq [N, C, d...] -> x_sq_padded [N, C + size - 1, d...]
        let mut pad_shape = shape.clone();
        pad_shape[1] = half as u64;
        let zeros_left = zeros_f32(&pad_shape)?;
        pad_shape[1] = (size - 1 - half) as u64;
        let zeros_right = zeros_f32(&pad_shape)?;
        let x_sq_padded = NumericTensor::concat(&[&zeros_left, &x_sq, &zeros_right], 1, backend)?;

        // Sliding window sum over channel dimension
        // For k in 0..size: accumulate x_sq_padded[:, k:k+C, ...]
        let r = |s: usize, e: usize| s as u64..e as u64;
        let ndim = shape.len();

        let mut sq_sum: Option<NumericTensor<DynRank>> = None;
        for k in 0..size {
            let mut ranges: Vec<std::ops::Range<u64>> = Vec::with_capacity(ndim);
            ranges.push(r(0, shape[0] as usize)); // N
            ranges.push(r(k, k + c_dim)); // C window
            for &dim in &shape[2..] {
                ranges.push(r(0, dim as usize));
            }
            let tap = x_sq_padded.slice(&ranges, backend)?;
            sq_sum = Some(match sq_sum {
                None => tap,
                Some(acc) => NumericTensor::add(&acc, &tap, backend)?,
            });
        }

        let sq_sum = sq_sum.unwrap();

        // norm_factor = (bias + alpha/size * sq_sum)^beta
        let alpha_over_size = scalar_f32(self.alpha / self.size as f32)?;
        let bias_val = scalar_f32(self.bias)?;
        let beta_val = scalar_f32(self.beta)?;

        let scaled = NumericTensor::mul(&sq_sum, &alpha_over_size, backend)?;
        let biased = NumericTensor::add(&scaled, &bias_val, backend)?;
        let norm_factor = biased.pow(&beta_val, backend)?;

        // Y = X / norm_factor
        let y = NumericTensor::div(x, &norm_factor, backend)?;

        let mut result = HashMap::new();
        result.insert(self.output, y);
        Ok(Box::new(result.into_iter()))
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, _rng: &mut impl Rng) -> MilliOpGraph {
        panic!("LRN uses custom eval, not milli-op decomposition")
    }
}
