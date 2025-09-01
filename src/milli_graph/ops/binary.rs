use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::dtype::DType;
use crate::milli_graph::ops::{
    MilliOp, infer_multidirectional_broadcasting_rank, infer_multidirectional_broadcasting_shape,
};
use crate::milli_graph::{MilliOpGraphError, MilliOpGraphTensorId};
use crate::numeric_tensor::NumericTensor;
use crate::scalar_info::{ScalarInfo, ScalarInfoTyped};
use crate::symbolic_scalar::{SymbolicResolver, SymbolicScalar, SymbolicScalarTyped};
use crate::tensor_info::{MinimalTensor, TensorInfo, TensorInfoRanked};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) enum SimpleBinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Modulo(Option<bool>),
    And,
    Or,
    Xor,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    Equal,
    Greater,
    GreaterOrEqual,
    Less,
    LessOrEqual,
    Max,
    Min,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpSimpleBinary {
    which_op: SimpleBinaryOp,
    a: MilliOpGraphTensorId,
    b: MilliOpGraphTensorId,
}

impl MilliOpSimpleBinary {
    pub fn add(a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> Self {
        Self {
            a,
            b,
            which_op: SimpleBinaryOp::Add,
        }
    }

    pub fn sub(a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> Self {
        Self {
            a,
            b,
            which_op: SimpleBinaryOp::Sub,
        }
    }

    pub fn mul(a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> Self {
        Self {
            a,
            b,
            which_op: SimpleBinaryOp::Mul,
        }
    }

    pub fn div(a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> Self {
        Self {
            a,
            b,
            which_op: SimpleBinaryOp::Div,
        }
    }

    pub fn modulo(a: MilliOpGraphTensorId, b: MilliOpGraphTensorId, fmod: Option<bool>) -> Self {
        Self {
            a,
            b,
            which_op: SimpleBinaryOp::Modulo(fmod),
        }
    }
    pub fn and(a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> Self {
        Self {
            a,
            b,
            which_op: SimpleBinaryOp::And,
        }
    }

    pub fn or(a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> Self {
        Self {
            a,
            b,
            which_op: SimpleBinaryOp::Or,
        }
    }

    pub fn xor(a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> Self {
        Self {
            a,
            b,
            which_op: SimpleBinaryOp::Xor,
        }
    }

    pub fn bitwise_and(a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> Self {
        Self {
            a,
            b,
            which_op: SimpleBinaryOp::BitwiseAnd,
        }
    }

    pub fn bitwise_or(a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> Self {
        Self {
            a,
            b,
            which_op: SimpleBinaryOp::BitwiseOr,
        }
    }

    pub fn bitwise_xor(a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> Self {
        Self {
            a,
            b,
            which_op: SimpleBinaryOp::BitwiseXor,
        }
    }

    pub fn equal(a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> Self {
        Self {
            a,
            b,
            which_op: SimpleBinaryOp::Equal,
        }
    }

    pub fn greater(a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> Self {
        Self {
            a,
            b,
            which_op: SimpleBinaryOp::Greater,
        }
    }

    pub fn greater_or_equal(a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> Self {
        Self {
            a,
            b,
            which_op: SimpleBinaryOp::GreaterOrEqual,
        }
    }

    pub fn less(a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> Self {
        Self {
            a,
            b,
            which_op: SimpleBinaryOp::Less,
        }
    }

    pub fn less_or_equal(a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> Self {
        Self {
            a,
            b,
            which_op: SimpleBinaryOp::LessOrEqual,
        }
    }

    pub fn max(a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> Self {
        Self {
            a,
            b,
            which_op: SimpleBinaryOp::Max,
        }
    }

    pub fn min(a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> Self {
        Self {
            a,
            b,
            which_op: SimpleBinaryOp::Min,
        }
    }
}

impl MilliOp for MilliOpSimpleBinary {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        vec![self.a, self.b]
    }

    fn infer(
        &self,
        known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>,
        symbolic_resolver: &mut SymbolicResolver,
        backend: &mut EvalBackend,
    ) -> Result<TensorInfo, MilliOpGraphError> {
        let a = &known_inputs[&self.a];
        let b = &known_inputs[&self.b];

        let a_shape = a.shape(symbolic_resolver);
        let b_shape = b.shape(symbolic_resolver);
        let output_rank = infer_multidirectional_broadcasting_rank(
            &[a_shape.clone(), b_shape.clone()],
            symbolic_resolver,
        )?;

        if let (Some(a), Some(b)) = (a.as_numeric(), b.as_numeric()) {
            let inputs = HashMap::from([(self.a, a.clone()), (self.b, b.clone())]);
            Ok(TensorInfo::from(self.eval(&inputs, backend)?))
        } else if let (Some(a), Some(b)) = (a.as_ranked(), b.as_ranked()) {
            let output_shape = infer_multidirectional_broadcasting_shape(
                &[a.shape(), b.shape()],
                symbolic_resolver,
            )?;
            Ok(TensorInfo::from(TensorInfoRanked::<DynRank>::new(
                ScalarInfo::Symbolic(SymbolicScalar::new(a.dtype(), symbolic_resolver)),
                output_shape,
                symbolic_resolver,
            )))
        } else {
            Ok(TensorInfo::new_from_first_element_and_rank(
                ScalarInfo::Symbolic(SymbolicScalar::new(a.dtype(), symbolic_resolver)),
                output_rank,
                symbolic_resolver,
            ))
        }
    }

    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        let a = &inputs[&self.a];
        let b = &inputs[&self.b];
        Ok(match self.which_op {
            SimpleBinaryOp::Add => NumericTensor::<DynRank>::add(a, b, backend)?,
            SimpleBinaryOp::Sub => NumericTensor::<DynRank>::sub(a, b, backend)?,
            SimpleBinaryOp::Mul => NumericTensor::<DynRank>::mul(a, b, backend)?,
            SimpleBinaryOp::Div => NumericTensor::<DynRank>::div(a, b, backend)?,
            SimpleBinaryOp::Modulo(fmod) => {
                let is_float =
                    [DType::F64, DType::F32, DType::BF16, DType::F16].contains(&a.dtype());
                let fmod = if is_float {
                    true
                } else {
                    fmod.unwrap_or(false)
                };
                if fmod {
                    NumericTensor::<DynRank>::fmod(a, b, backend)?
                } else {
                    NumericTensor::<DynRank>::imod(a, b, backend)?
                }
            }
            SimpleBinaryOp::And => NumericTensor::<DynRank>::and(a, b, backend)?,
            SimpleBinaryOp::Or => NumericTensor::<DynRank>::or(a, b, backend)?,
            SimpleBinaryOp::Xor => NumericTensor::<DynRank>::xor(a, b, backend)?,
            SimpleBinaryOp::BitwiseAnd => NumericTensor::<DynRank>::bitwise_and(a, b, backend)?,
            SimpleBinaryOp::BitwiseOr => NumericTensor::<DynRank>::bitwise_or(a, b, backend)?,
            SimpleBinaryOp::BitwiseXor => NumericTensor::<DynRank>::bitwise_xor(a, b, backend)?,
            SimpleBinaryOp::Equal => NumericTensor::<DynRank>::equal(a, b, backend)?,
            SimpleBinaryOp::Greater => NumericTensor::<DynRank>::greater(a, b, backend)?,
            SimpleBinaryOp::GreaterOrEqual => {
                NumericTensor::<DynRank>::greater_or_equal(a, b, backend)?
            }
            SimpleBinaryOp::Less => NumericTensor::<DynRank>::less(a, b, backend)?,
            SimpleBinaryOp::LessOrEqual => NumericTensor::<DynRank>::less_or_equal(a, b, backend)?,
            SimpleBinaryOp::Max => NumericTensor::<DynRank>::max(a, b, backend)?,
            SimpleBinaryOp::Min => NumericTensor::<DynRank>::min(a, b, backend)?,
        })
    }

    fn get_name(&self) -> String {
        match self.which_op {
            SimpleBinaryOp::Add => "Add",
            SimpleBinaryOp::Sub => "Sub",
            SimpleBinaryOp::Mul => "Mul",
            SimpleBinaryOp::Div => "Div",
            SimpleBinaryOp::Modulo(_) => "Modulo",
            SimpleBinaryOp::And => "And",
            SimpleBinaryOp::Or => "Or",
            SimpleBinaryOp::Xor => "Xor",
            SimpleBinaryOp::BitwiseAnd => "Bitwise And",
            SimpleBinaryOp::BitwiseOr => "Bitwise Or",
            SimpleBinaryOp::BitwiseXor => "Bitwise Xor",
            SimpleBinaryOp::Equal => "Equal",
            SimpleBinaryOp::Greater => "Greater",
            SimpleBinaryOp::GreaterOrEqual => "Greater or Equal",
            SimpleBinaryOp::Less => "Less",
            SimpleBinaryOp::LessOrEqual => "Less or Equal",
            SimpleBinaryOp::Max => "Max",
            SimpleBinaryOp::Min => "Min",
        }
        .to_string()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpPow {
    a: MilliOpGraphTensorId,
    b: MilliOpGraphTensorId,
}

impl MilliOpPow {
    pub(crate) fn new(a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> Self {
        Self { a, b }
    }
}

impl MilliOp for MilliOpPow {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        vec![self.a, self.b]
    }

    fn infer(
        &self,
        known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>,
        symbolic_resolver: &mut SymbolicResolver,
        backend: &mut EvalBackend,
    ) -> Result<TensorInfo, MilliOpGraphError> {
        let a = &known_inputs[&self.a];
        let b = &known_inputs[&self.b];

        let a_shape = a.shape(symbolic_resolver);
        let b_shape = b.shape(symbolic_resolver);
        let output_rank = infer_multidirectional_broadcasting_rank(
            &[a_shape.clone(), b_shape.clone()],
            symbolic_resolver,
        )?;

        if let (Some(a), Some(b)) = (a.as_numeric(), b.as_numeric()) {
            let inputs = HashMap::from([(self.a, a.clone()), (self.b, b.clone())]);
            Ok(TensorInfo::from(self.eval(&inputs, backend)?))
        } else if let (Some(a), Some(b)) = (a.as_ranked(), b.as_ranked()) {
            let output_shape = infer_multidirectional_broadcasting_shape(
                &[a.shape(), b.shape()],
                symbolic_resolver,
            )?;
            Ok(TensorInfo::from(TensorInfoRanked::<DynRank>::new(
                ScalarInfo::Symbolic(SymbolicScalar::new(a.dtype(), symbolic_resolver)),
                output_shape,
                symbolic_resolver,
            )))
        } else {
            Ok(TensorInfo::new_from_first_element_and_rank(
                ScalarInfo::Symbolic(SymbolicScalar::new(a.dtype(), symbolic_resolver)),
                output_rank,
                symbolic_resolver,
            ))
        }
    }

    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        Ok(NumericTensor::<DynRank>::pow(
            &inputs[&self.a],
            &inputs[&self.b],
            backend,
        )?)
    }

    fn get_name(&self) -> String {
        "Pow".to_string()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpMatMul {
    a: MilliOpGraphTensorId,
    b: MilliOpGraphTensorId,
}

impl MilliOpMatMul {
    pub fn new(a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> Self {
        Self { a, b }
    }
}

impl MilliOp for MilliOpMatMul {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        vec![self.a, self.b]
    }

    fn infer(
        &self,
        known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>,
        symbolic_resolver: &mut SymbolicResolver,
        backend: &mut EvalBackend,
    ) -> Result<TensorInfo, MilliOpGraphError> {
        let a = &known_inputs[&self.a];
        let b = &known_inputs[&self.b];
        if let (Some(a), Some(b)) = (a.as_numeric(), b.as_numeric()) {
            let inputs = HashMap::from([(self.a, a.clone()), (self.b, b.clone())]);
            Ok(TensorInfo::from(self.eval(&inputs, backend)?))
        } else {
            let dtype = a.dtype();
            assert_eq!(b.dtype(), dtype);

            if let (Some(a), Some(b)) = (a.as_ranked(), b.as_ranked()) {
                let shape_a = a.shape().to_vec();
                let shape_b = b.shape().to_vec();
                // Prepend to a if rank 1
                let (mut shape_a, prune_first_after) = if shape_a.len() == 1 {
                    (vec![ScalarInfoTyped::Numeric(1), shape_a[0].clone()], true)
                } else {
                    (shape_a, false)
                };

                // Append to b if rank 1
                let (mut shape_b, prune_last_after) = if shape_b.len() == 1 {
                    (vec![shape_b[0].clone(), ScalarInfoTyped::Numeric(1)], true)
                } else {
                    (shape_b, false)
                };

                // Broadcast both shapes
                while shape_a.len() < shape_b.len() {
                    shape_a.insert(0, ScalarInfoTyped::Numeric(1))
                }
                while shape_b.len() < shape_a.len() {
                    shape_b.insert(0, ScalarInfoTyped::Numeric(1))
                }

                let mut dims_out = vec![];
                for i in 0..shape_a.len() - 2 {
                    let dim = match shape_a[i].clone() {
                        ScalarInfoTyped::Numeric(x) => {
                            if x == 1 {
                                // Use the other one
                                shape_b[i].clone()
                            } else {
                                match shape_b[i] {
                                    ScalarInfoTyped::Numeric(y) => {
                                        ScalarInfoTyped::Numeric(x.max(y))
                                    }
                                    _ => {
                                        // Assume it's the known one
                                        ScalarInfoTyped::Numeric(x)
                                    }
                                }
                            }
                        }
                        ScalarInfoTyped::Symbolic(x) => {
                            match shape_b[i].clone() {
                                ScalarInfoTyped::Numeric(y) => {
                                    // Assume it's the known one
                                    ScalarInfoTyped::Numeric(y)
                                }
                                ScalarInfoTyped::Symbolic(y) => {
                                    match x.try_eq(&y) {
                                        None => {
                                            // Can't compare them, must use a new unknown
                                            ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(
                                                symbolic_resolver,
                                            ))
                                        }
                                        Some(is_same) => {
                                            if is_same {
                                                // They are the same dimension, so we can use it
                                                ScalarInfoTyped::Symbolic(x)
                                            } else {
                                                // They are different dimensions, so we can't use it
                                                ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(
                                                    symbolic_resolver,
                                                ))
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    };
                    dims_out.push(dim);
                }
                dims_out.push(shape_a[shape_a.len() - 2].clone());
                dims_out.push(shape_b[shape_b.len() - 1].clone());

                if prune_first_after {
                    dims_out.remove(0);
                }
                if prune_last_after {
                    dims_out.pop();
                }

                Ok(TensorInfo::from(TensorInfoRanked::<DynRank>::new(
                    ScalarInfo::Symbolic(SymbolicScalar::new(dtype, symbolic_resolver)),
                    dims_out,
                    symbolic_resolver,
                )))
            } else {
                // One of the input ranks was unknown, must simply pass on the confusion
                Ok(TensorInfo::Minimal(MinimalTensor::new(
                    ScalarInfo::Symbolic(SymbolicScalar::new(dtype, symbolic_resolver)),
                    SymbolicScalarTyped::new(symbolic_resolver),
                )))
            }
        }
    }
    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        let a_input = &inputs[&self.a];
        let b_input = &inputs[&self.b];
        let accumulate_dtype = match a_input.dtype() {
            DType::BF16 | DType::F16 => Some(DType::F32),
            _ => None,
        };
        Ok(NumericTensor::<DynRank>::matmul(
            a_input,
            b_input,
            accumulate_dtype,
            backend,
        )?)
    }

    fn get_name(&self) -> String {
        "MatMul".to_string()
    }
}
