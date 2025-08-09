use crate::TrigOp;
use crate::backends::eval_backend::EvalBackend;
use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::backends::ndarray_backend::conversions::NDArrayNumericTensorType;
use crate::dtype::DType;
use crate::milli_graph::{MilliOpGraphError, MilliOpGraphTensorId};
use crate::numeric_scalar::NumericScalar;
use crate::numeric_tensor::NumericTensor;
use crate::scalar_info::{ScalarInfo, ScalarInfoTyped};
use crate::symbolic_scalar::{SymbolicResolver, SymbolicScalar, SymbolicScalarTyped};
use crate::tensor_info::{
    MinimalTensor, TensorInfo, TensorInfoRanked, TensorInfoShaped, TensorInfoTypedRanked,
    TensorInfoTypedShaped,
};
use crate::tensor_rank::DynRank;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use typenum::P1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MilliOpTensorIDOrLiteral {
    TensorID(MilliOpGraphTensorId),
    Literal(NDArrayNumericTensor<DynRank>),
}

pub trait MilliOp {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId>;

    fn infer(
        &self,
        known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>,
        _symbolic_resolver: &mut SymbolicResolver,
        backend: &mut EvalBackend,
    ) -> Result<TensorInfo, MilliOpGraphError> {
        let mut resolved_inputs = HashMap::new();
        for input in self.get_inputs() {
            if let Some(tensor_info) = known_inputs.get(&input) {
                if let Some(tensor) = tensor_info.as_numeric() {
                    resolved_inputs.insert(input, tensor.clone());
                } else {
                    return Err(MilliOpGraphError::UnableToInfer);
                }
            } else {
                return Err(MilliOpGraphError::UnableToInfer);
            }
        }

        Ok(TensorInfo::from(self.eval(&resolved_inputs, backend)?))
    }
    //fn infer(&self, known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>, _symbolic_resolver: &mut SymbolicResolver, backend: &mut EvalBackend) -> Result<TensorInfo, MilliOpGraphError>;
    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        _backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, MilliOpGraphError>;

    fn get_name(&self) -> String;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpConstant {
    data: NDArrayNumericTensor<DynRank>,
}

impl MilliOpConstant {
    pub fn new(a: NDArrayNumericTensor<DynRank>) -> Self {
        Self { data: a }
    }

    pub(crate) fn new_scalar<T>(v: T) -> Self
    where
        T: NDArrayNumericTensorType,
    {
        Self {
            data: NDArrayNumericTensor::<DynRank>::from_vec_shape(vec![v], &vec![1]).unwrap(),
        }
    }
}

impl MilliOp for MilliOpConstant {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        vec![]
    }

    fn infer(
        &self,
        _known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>,
        _symbolic_resolver: &mut SymbolicResolver,
        _backend: &mut EvalBackend,
    ) -> Result<TensorInfo, MilliOpGraphError> {
        Ok(TensorInfo::from(NumericTensor::NDArray(self.data.clone())))
    }
    fn eval(
        &self,
        _inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        _backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        Ok(self.data.clone().into())
    }

    fn get_name(&self) -> String {
        "Constant".to_string()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpRange {
    start: MilliOpGraphTensorId,
    end: MilliOpGraphTensorId,
    delta: MilliOpGraphTensorId,
}

impl MilliOpRange {
    pub fn new(
        start: MilliOpGraphTensorId,
        end: MilliOpGraphTensorId,
        delta: MilliOpGraphTensorId,
    ) -> Self {
        Self { start, end, delta }
    }
}

impl MilliOp for MilliOpRange {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        vec![]
    }

    fn infer(
        &self,
        known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>,
        symbolic_resolver: &mut SymbolicResolver,
        backend: &mut EvalBackend,
    ) -> Result<TensorInfo, MilliOpGraphError> {
        let start = &known_inputs[&self.start].first_element();
        let end = &known_inputs[&self.end].first_element();
        let delta = &known_inputs[&self.delta].first_element();
        assert_eq!(start.dtype(), end.dtype());
        assert_eq!(start.dtype(), end.dtype());
        Ok(
            if let (
                ScalarInfo::Numeric(start),
                ScalarInfo::Numeric(end),
                ScalarInfo::Numeric(delta),
            ) = (start, end, delta)
            {
                // We have enough info, so just resolve it
                TensorInfo::from(NumericTensor::<P1>::range(
                    start.clone(),
                    end.clone(),
                    delta.clone(),
                    backend,
                )?)
            } else {
                TensorInfo::from(TensorInfoRanked::<DynRank>::new(
                    ScalarInfo::Symbolic(SymbolicScalar::new(DType::I64, symbolic_resolver)),
                    vec![ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(
                        symbolic_resolver,
                    ))],
                    symbolic_resolver,
                ))
            },
        )
    }

    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        Ok(NumericTensor::<P1>::range(
            inputs[&self.start].first_element(),
            inputs[&self.end].first_element(),
            inputs[&self.delta].first_element(),
            backend,
        )?
        .to_dyn_rank())
    }

    fn get_name(&self) -> String {
        "Range".to_string()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpExpand {
    input: MilliOpGraphTensorId,
    shape: MilliOpGraphTensorId,
}

impl MilliOpExpand {
    pub fn new(input: MilliOpGraphTensorId, shape: MilliOpGraphTensorId) -> Self {
        Self { input, shape }
    }
}

impl MilliOp for MilliOpExpand {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        vec![self.input, self.shape]
    }

    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        _backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        let shape: Vec<i64> = inputs[&self.shape].try_to_rank::<P1>()?.try_into()?;
        let shape_u = shape.iter().map(|x| *x as u64).collect::<Vec<u64>>();
        let mut x = inputs[&self.input].clone();
        while x.rank() < shape_u.len() {
            x = x.unsqueeze(0)?;
        }
        let shape_u = shape_u
            .iter()
            .zip(x.shape().iter())
            .map(|(a, b)| std::cmp::max(*a, *b))
            .collect::<Vec<u64>>();
        Ok(x.expand(&shape_u)?)
    }

    fn get_name(&self) -> String {
        "Expand".to_string()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpConstantOfShape {
    value: NumericScalar,
    shape: MilliOpGraphTensorId,
}

impl MilliOpConstantOfShape {
    pub fn new(value: NumericScalar, shape: MilliOpGraphTensorId) -> Self {
        Self { value, shape }
    }
}

impl MilliOp for MilliOpConstantOfShape {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        vec![]
    }

    fn infer(
        &self,
        known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>,
        symbolic_resolver: &mut SymbolicResolver,
        backend: &mut EvalBackend,
    ) -> Result<TensorInfo, MilliOpGraphError> {
        let input = &known_inputs[&self.shape];
        let input = input
            .try_to_rank::<P1>(symbolic_resolver)?
            .try_to_type::<u64>()?;

        match input {
            TensorInfoTypedRanked::Shaped(tensor) => match tensor {
                TensorInfoTypedShaped::Numeric(tensor) => {
                    let inputs = HashMap::from([(self.shape, tensor.to_dyn_rank().to_dyn_type())]);
                    Ok(TensorInfo::from(self.eval(&inputs, backend)?))
                }
                TensorInfoTypedShaped::Shaped(tensor) => {
                    let mut new_shape = vec![];
                    for i in 0..tensor.shape()[0] {
                        new_shape.push(tensor.get(&[i]).unwrap().clone());
                    }
                    Ok(TensorInfo::from(TensorInfoRanked::<DynRank>::new(
                        ScalarInfo::Numeric(self.value.clone()),
                        new_shape,
                        symbolic_resolver,
                    )))
                }
            },
            TensorInfoTypedRanked::Ranked(tensor) => {
                Ok(TensorInfo::new_from_first_element_and_rank(
                    ScalarInfo::Numeric(self.value.clone()),
                    tensor.shape()[0].cast(),
                    symbolic_resolver,
                ))
            }
        }
    }

    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        _backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        let shape: Vec<i64> = inputs[&self.shape].try_to_rank::<P1>()?.try_into()?;
        let shape_usize = shape.iter().map(|x| *x as u64).collect::<Vec<_>>();
        Ok(NDArrayNumericTensor::<DynRank>::fill(self.value.clone(), &shape_usize)?.into())
    }

    fn get_name(&self) -> String {
        "Constant of Shape".to_string()
    }
}

fn infer_multidirectional_broadcasting_rank(
    shapes: &[TensorInfoTypedRanked<u64, P1>],
    symbolic_resolver: &mut SymbolicResolver,
) -> Result<ScalarInfoTyped<u32>, MilliOpGraphError> {
    let mut output_rank: Option<usize> = None;
    for shape in shapes {
        match shape {
            TensorInfoTypedRanked::Shaped(x) => {
                let this_rank = x.shape()[0] as usize;
                if let Some(o) = output_rank {
                    output_rank = Some(o.min(this_rank));
                } else {
                    output_rank = Some(this_rank)
                }
            }
            TensorInfoTypedRanked::Ranked(_x) => {
                output_rank = None;
                break;
            }
        }
    }
    match output_rank {
        Some(x) => Ok(ScalarInfoTyped::Numeric(x as u32)),
        None => Ok(ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(
            symbolic_resolver,
        ))),
    }
}

fn infer_multidirectional_broadcasting_shape(
    shapes: &[Vec<ScalarInfoTyped<u64>>],
    symbolic_resolver: &mut SymbolicResolver,
) -> Result<Vec<ScalarInfoTyped<u64>>, MilliOpGraphError> {
    if shapes.is_empty() {
        return Err(MilliOpGraphError::InvalidInput(
            "Cannot broadcast empty input".to_string(),
        ));
    }

    let output_rank = shapes.iter().map(|x| x.len()).max().unwrap();

    let mut output_shape = vec![];
    for i in 0..output_rank {
        let mut dim = ScalarInfoTyped::<u64>::Numeric(1);
        for shape in shapes {
            let rank = shape.len();
            let local_i = (i as i64 - output_rank as i64) + rank as i64;
            if local_i < 0 {
                // Infer dim of size 1, and pass
            } else {
                let local_dim = shape[local_i as usize].clone();
                match local_dim {
                    ScalarInfoTyped::Numeric(x) => {
                        if x == 1 {
                            // Do not modify the dimension, pass it through.
                        } else {
                            match dim {
                                ScalarInfoTyped::Numeric(y) => {
                                    if y == 1 || x == y {
                                        dim = ScalarInfoTyped::Numeric(y.max(x));
                                    } else {
                                        return Err(MilliOpGraphError::InvalidInput(
                                            "Cannot broadcast input shape".to_string(),
                                        ));
                                    }
                                }
                                _ => {
                                    // The only way this is valid is if the unknown dim matches the known one, so be optimistic here
                                    dim = local_dim.clone();
                                }
                            }
                        }
                    }
                    _ => {
                        // Incoming dim is unknown
                        match dim {
                            ScalarInfoTyped::Numeric(y) => {
                                if y == 1 {
                                    // Use the existing unknown dim
                                    dim = local_dim.clone();
                                } else {
                                    // The only way this is valid is if the unknown dim matches the known one, so be optimistic here
                                }
                            }
                            _ => {
                                // Two unknown dimensions
                                match dim.try_eq(&local_dim) {
                                    None => {
                                        // Must use new unknown dimension
                                        dim = ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(
                                            symbolic_resolver,
                                        ))
                                    }
                                    Some(is_same) => {
                                        if is_same {
                                            // Ok, use the unknown dim already in there
                                        } else {
                                            // Must use new unknown dimension
                                            dim = ScalarInfoTyped::Symbolic(
                                                SymbolicScalarTyped::new(symbolic_resolver),
                                            )
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        output_shape.push(dim);
    }
    Ok(output_shape)
}

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
        Ok(NumericTensor::<DynRank>::matmul(
            &inputs[&self.a],
            &inputs[&self.b],
            backend,
        )?)
    }

    fn get_name(&self) -> String {
        "MatMul".to_string()
    }
}

trait SimpleUnaryMilliOp {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId>;

    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, MilliOpGraphError>;

    #[allow(dead_code)]
    fn eval_scalar(&self, input: &NumericScalar) -> Result<NumericScalar, MilliOpGraphError>;

    fn get_name(&self) -> String;
}

impl<T: SimpleUnaryMilliOp> MilliOp for T {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        <T as SimpleUnaryMilliOp>::get_inputs(self)
    }

    fn infer(
        &self,
        known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>,
        symbolic_resolver: &mut SymbolicResolver,
        backend: &mut EvalBackend,
    ) -> Result<TensorInfo, MilliOpGraphError> {
        let input_id = self.get_inputs()[0];
        let input = &known_inputs[&input_id];

        if let Some(input) = input.as_shaped() {
            match input {
                TensorInfoShaped::Numeric(input) => {
                    let inputs = HashMap::from([(input_id, input.clone())]);
                    Ok(TensorInfo::from(self.eval(&inputs, backend)?))
                }
                TensorInfoShaped::Symbolic(input) => {
                    // For now, just decay this to a ranked tensor
                    Ok(TensorInfo::from(TensorInfoRanked::<DynRank>::new(
                        ScalarInfo::Symbolic(SymbolicScalar::new(input.dtype(), symbolic_resolver)),
                        input
                            .shape()
                            .iter()
                            .map(|x| ScalarInfoTyped::Numeric(*x))
                            .collect::<Vec<_>>(),
                        symbolic_resolver,
                    )))
                }
            }
        } else {
            // For now, just decay this to a ranked tensor
            Ok(TensorInfo::new_from_first_element_and_shape(
                ScalarInfo::Symbolic(SymbolicScalar::new(input.dtype(), symbolic_resolver)),
                input.shape(symbolic_resolver),
                symbolic_resolver,
            ))
        }
    }

    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        <T as SimpleUnaryMilliOp>::eval(self, inputs, backend)
    }

    fn get_name(&self) -> String {
        <T as SimpleUnaryMilliOp>::get_name(self)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) enum SimpleUnaryOp {
    Neg,
    Abs,
    Exp,
    Ln,
    Sqrt,
    Not,
    Sign,
    BitwiseNot,
    Reciprocal,
    Trig(TrigOp),
    Floor,
    Ceil,
    Round,
    IsNan,
    IsInf {
        detect_positive: bool,
        detect_negative: bool,
    },
    Erf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpSimpleUnary {
    input: MilliOpGraphTensorId,
    op: SimpleUnaryOp,
}

impl MilliOpSimpleUnary {
    pub(crate) fn new(input: MilliOpGraphTensorId, op: SimpleUnaryOp) -> Self {
        Self { input, op }
    }

    pub fn neg(input: MilliOpGraphTensorId) -> Self {
        Self::new(input, SimpleUnaryOp::Neg)
    }

    pub fn abs(input: MilliOpGraphTensorId) -> Self {
        Self::new(input, SimpleUnaryOp::Abs)
    }

    pub fn exp(input: MilliOpGraphTensorId) -> Self {
        Self::new(input, SimpleUnaryOp::Exp)
    }

    pub fn ln(input: MilliOpGraphTensorId) -> Self {
        Self::new(input, SimpleUnaryOp::Ln)
    }

    pub fn sqrt(input: MilliOpGraphTensorId) -> Self {
        Self::new(input, SimpleUnaryOp::Sqrt)
    }

    pub fn not(input: MilliOpGraphTensorId) -> Self {
        Self::new(input, SimpleUnaryOp::Not)
    }

    pub fn sign(input: MilliOpGraphTensorId) -> Self {
        Self::new(input, SimpleUnaryOp::Sign)
    }

    pub fn bitwise_not(input: MilliOpGraphTensorId) -> Self {
        Self::new(input, SimpleUnaryOp::BitwiseNot)
    }

    pub fn reciprocal(input: MilliOpGraphTensorId) -> Self {
        Self::new(input, SimpleUnaryOp::Reciprocal)
    }

    pub fn trig(input: MilliOpGraphTensorId, trig_op: TrigOp) -> Self {
        Self::new(input, SimpleUnaryOp::Trig(trig_op))
    }

    pub fn floor(input: MilliOpGraphTensorId) -> Self {
        Self::new(input, SimpleUnaryOp::Floor)
    }

    pub fn ceil(input: MilliOpGraphTensorId) -> Self {
        Self::new(input, SimpleUnaryOp::Ceil)
    }

    pub fn round(input: MilliOpGraphTensorId) -> Self {
        Self::new(input, SimpleUnaryOp::Round)
    }

    pub fn is_inf(
        input: MilliOpGraphTensorId,
        detect_positive: bool,
        detect_negative: bool,
    ) -> Self {
        Self::new(
            input,
            SimpleUnaryOp::IsInf {
                detect_positive,
                detect_negative,
            },
        )
    }

    pub fn is_nan(input: MilliOpGraphTensorId) -> Self {
        Self::new(input, SimpleUnaryOp::IsNan)
    }

    pub fn erf(input: MilliOpGraphTensorId) -> Self {
        Self::new(input, SimpleUnaryOp::Erf)
    }
}

impl SimpleUnaryMilliOp for MilliOpSimpleUnary {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        vec![self.input]
    }

    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        let input = &inputs[&self.input];
        match self.op {
            SimpleUnaryOp::Neg => Ok(input.neg(backend)?),
            SimpleUnaryOp::Abs => Ok(input.abs(backend)?),
            SimpleUnaryOp::Exp => Ok(input.exp(backend)?),
            SimpleUnaryOp::Ln => Ok(input.ln(backend)?),
            SimpleUnaryOp::Sqrt => Ok(input.sqrt(backend)?),
            SimpleUnaryOp::Not => Ok(input.not(backend)?),
            SimpleUnaryOp::Sign => Ok(input.sign(backend)?),
            SimpleUnaryOp::BitwiseNot => Ok(input.bitwise_not(backend)?),
            SimpleUnaryOp::Reciprocal => Ok(input.reciprocal(backend)?),
            SimpleUnaryOp::Trig(trig_op) => Ok(input.trig(trig_op, backend)?),
            SimpleUnaryOp::Floor => Ok(input.floor(backend)?),
            SimpleUnaryOp::Ceil => Ok(input.ceil(backend)?),
            SimpleUnaryOp::Round => Ok(input.round(backend)?),
            SimpleUnaryOp::IsInf {
                detect_positive,
                detect_negative,
            } => Ok(input.is_inf(detect_positive, detect_negative)?),
            SimpleUnaryOp::IsNan => Ok(input.is_nan(backend)?),
            SimpleUnaryOp::Erf => Ok(input.erf(backend)?),
        }
    }

    fn eval_scalar(&self, input: &NumericScalar) -> Result<NumericScalar, MilliOpGraphError> {
        match self.op {
            SimpleUnaryOp::Neg => Ok(input.neg()),
            SimpleUnaryOp::Abs => Ok(input.abs()),
            SimpleUnaryOp::Exp => Ok(input.exp()),
            SimpleUnaryOp::Ln => Ok(input.ln()),
            SimpleUnaryOp::Sqrt => Ok(input.sqrt()),
            SimpleUnaryOp::Not => Ok(input.not()),
            SimpleUnaryOp::Sign => Ok(input.sign()),
            SimpleUnaryOp::BitwiseNot => Ok(input.bitwise_not()),
            SimpleUnaryOp::Reciprocal => Ok(input.recip()),
            SimpleUnaryOp::Trig(trig_op) => Ok(input.trig(trig_op)),
            SimpleUnaryOp::Floor => Ok(input.floor()),
            SimpleUnaryOp::Ceil => Ok(input.ceil()),
            SimpleUnaryOp::Round => Ok(input.round()),
            SimpleUnaryOp::IsInf {
                detect_positive,
                detect_negative,
            } => Ok(input.is_inf(detect_positive, detect_negative)),
            SimpleUnaryOp::IsNan => Ok(input.is_nan()),
            SimpleUnaryOp::Erf => Ok(input.erf()),
        }
    }

    fn get_name(&self) -> String {
        match self.op {
            SimpleUnaryOp::Neg => "Neg",
            SimpleUnaryOp::Abs => "Abs",
            SimpleUnaryOp::Exp => "Exp",
            SimpleUnaryOp::Ln => "Ln",
            SimpleUnaryOp::Sqrt => "Sqrt",
            SimpleUnaryOp::Not => "Not",
            SimpleUnaryOp::Sign => "Sign",
            SimpleUnaryOp::BitwiseNot => "Bitwise Not",
            SimpleUnaryOp::Reciprocal => "Reciprocal",
            SimpleUnaryOp::Trig(trig_op) => trig_op.get_name(),
            SimpleUnaryOp::Floor => "Floor",
            SimpleUnaryOp::Ceil => "Ceil",
            SimpleUnaryOp::Round => "Round",
            SimpleUnaryOp::IsNan => "IsNan",
            SimpleUnaryOp::IsInf { .. } => "IsInf",
            SimpleUnaryOp::Erf => "Erf",
        }
        .to_string()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpClampMin {
    input: MilliOpGraphTensorId,
    value: f32,
}

impl MilliOpClampMin {
    pub fn new(a: MilliOpGraphTensorId, value: f32) -> Self {
        Self { input: a, value }
    }
}

impl SimpleUnaryMilliOp for MilliOpClampMin {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        vec![self.input]
    }

    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        Ok(inputs[&self.input].clamp_min(self.value, backend)?)
    }

    fn eval_scalar(&self, input: &NumericScalar) -> Result<NumericScalar, MilliOpGraphError> {
        Ok(input.clamp_min(self.value))
    }

    fn get_name(&self) -> String {
        "Clamp Min".to_string()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpNonZero {
    input: MilliOpGraphTensorId,
}

impl MilliOpNonZero {
    pub fn new(input: MilliOpGraphTensorId) -> Self {
        Self { input }
    }
}

impl MilliOp for MilliOpNonZero {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        vec![self.input]
    }

    fn infer(
        &self,
        known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>,
        symbolic_resolver: &mut SymbolicResolver,
        backend: &mut EvalBackend,
    ) -> Result<TensorInfo, MilliOpGraphError> {
        let input = &known_inputs[&self.input];
        if let Some(x) = input.as_numeric() {
            let inputs = HashMap::from([(self.input, x.clone())]);
            Ok(TensorInfo::from(self.eval(&inputs, backend)?))
        } else {
            // Don't even try shape inference for now
            Ok(TensorInfo::Minimal(MinimalTensor::new(
                ScalarInfo::Symbolic(SymbolicScalar::new(DType::I64, symbolic_resolver)),
                SymbolicScalarTyped::new(symbolic_resolver),
            )))
        }
    }

    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        Ok(inputs[&self.input].nonzero(backend)?)
    }

    fn get_name(&self) -> String {
        "NonZero".to_string()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpCumSum {
    a: MilliOpGraphTensorId,
    axis: MilliOpGraphTensorId,
    exclusive: bool,
    reverse: bool,
}

impl MilliOpCumSum {
    pub(crate) fn new(
        a: MilliOpGraphTensorId,
        axis: MilliOpGraphTensorId,
        exclusive: bool,
        reverse: bool,
    ) -> Self {
        Self {
            a,
            axis,
            exclusive,
            reverse,
        }
    }
}

impl MilliOp for MilliOpCumSum {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        vec![self.a, self.axis]
    }

    fn eval(
        &self,
        _inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        _backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        Err(MilliOpGraphError::UnimplementedOperatorError(
            "CumSum".to_string(),
        ))?;
        todo!()
    }

    fn get_name(&self) -> String {
        "CumSum".to_string()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpShape {
    input: MilliOpGraphTensorId,
}

impl MilliOpShape {
    pub fn new(input: MilliOpGraphTensorId) -> Self {
        Self { input }
    }
}

impl MilliOp for MilliOpShape {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        vec![self.input]
    }

    fn infer(
        &self,
        known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>,
        symbolic_resolver: &mut SymbolicResolver,
        _backend: &mut EvalBackend,
    ) -> Result<TensorInfo, MilliOpGraphError> {
        let input = &known_inputs[&self.input];
        Ok(TensorInfo::from(input.shape(symbolic_resolver)))
    }

    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        _backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        let output_shape = inputs[&self.input]
            .shape()
            .into_iter()
            .map(|x| x as i64)
            .collect::<Vec<_>>();
        Ok(NDArrayNumericTensor::<P1>::from(output_shape)
            .to_dyn()
            .into())
    }

    fn get_name(&self) -> String {
        "Shape".to_string()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpReduceSum {
    data: MilliOpGraphTensorId,
    axes: Option<MilliOpGraphTensorId>,
    keepdims: bool,
    noop_with_empty_axes: bool,
}

impl MilliOpReduceSum {
    pub fn new(
        data: MilliOpGraphTensorId,
        axes: Option<MilliOpGraphTensorId>,
        keepdims: bool,
        noop_with_empty_axes: bool,
    ) -> Self {
        Self {
            data,
            axes,
            keepdims,
            noop_with_empty_axes,
        }
    }
}

impl MilliOp for MilliOpReduceSum {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        vec![self.data]
    }

    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        let data = &inputs[&self.data];
        let axes = if let Some(axes) = self.axes {
            Vec::<i64>::try_from(inputs[&axes].try_to_rank::<P1>()?)?
        } else {
            (0i64..(data.shape().len() as i64)).collect()
        };
        let axes = if axes.is_empty() {
            if self.noop_with_empty_axes {
                return Ok(data.clone());
            } else {
                (0i64..(data.shape().len() as i64)).collect::<Vec<_>>()
            }
        } else {
            axes
        };
        let axes = axes
            .into_iter()
            .map(|x| {
                (if x < 0 {
                    x + data.shape().len() as i64
                } else {
                    x
                }) as usize
            })
            .collect::<Vec<_>>();
        let out = data.reduce_sum(axes, self.keepdims, backend)?;
        Ok(out)
    }

    fn get_name(&self) -> String {
        "ReduceSum".to_string()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpReduceMin {
    data: MilliOpGraphTensorId,
    axes: Option<MilliOpGraphTensorId>,
    keepdims: bool,
    noop_with_empty_axes: bool,
}

impl MilliOpReduceMin {
    pub fn new(
        data: MilliOpGraphTensorId,
        axes: Option<MilliOpGraphTensorId>,
        keepdims: bool,
        noop_with_empty_axes: bool,
    ) -> Self {
        Self {
            data,
            axes,
            keepdims,
            noop_with_empty_axes,
        }
    }
}

impl MilliOp for MilliOpReduceMin {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        vec![self.data]
    }

    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        let data = &inputs[&self.data];
        let axes = if let Some(axes) = self.axes {
            Vec::<i64>::try_from(inputs[&axes].try_to_rank::<P1>()?)?
        } else {
            (0i64..(data.shape().len() as i64)).collect()
        };
        let axes = if axes.is_empty() {
            if self.noop_with_empty_axes {
                return Ok(data.clone());
            } else {
                (0i64..(data.shape().len() as i64)).collect::<Vec<_>>()
            }
        } else {
            axes
        };
        let axes = axes
            .into_iter()
            .map(|x| {
                (if x < 0 {
                    x + data.shape().len() as i64
                } else {
                    x
                }) as usize
            })
            .collect::<Vec<_>>();
        let out = data.reduce_min(axes, self.keepdims, backend)?;
        Ok(out)
    }

    fn get_name(&self) -> String {
        "ReduceMin".to_string()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpReduceMax {
    data: MilliOpGraphTensorId,
    axes: Option<MilliOpGraphTensorId>,
    keepdims: bool,
    noop_with_empty_axes: bool,
}

impl MilliOpReduceMax {
    pub(crate) fn new(
        data: MilliOpGraphTensorId,
        axes: Option<MilliOpGraphTensorId>,
        keepdims: bool,
        noop_with_empty_axes: bool,
    ) -> Self {
        Self {
            data,
            axes,
            keepdims,
            noop_with_empty_axes,
        }
    }
}

impl MilliOp for MilliOpReduceMax {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        vec![self.data]
    }

    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        let data = &inputs[&self.data];
        let axes = if let Some(axes) = self.axes {
            Vec::<i64>::try_from(inputs[&axes].try_to_rank::<P1>()?)?
        } else {
            (0i64..(data.shape().len() as i64)).collect()
        };
        let axes = if axes.is_empty() {
            if self.noop_with_empty_axes {
                return Ok(data.clone());
            } else {
                (0i64..(data.shape().len() as i64)).collect::<Vec<_>>()
            }
        } else {
            axes
        };
        let axes = axes
            .into_iter()
            .map(|x| {
                (if x < 0 {
                    x + data.shape().len() as i64
                } else {
                    x
                }) as usize
            })
            .collect::<Vec<_>>();
        let out = data.reduce_max(axes, self.keepdims, backend)?;
        Ok(out)
    }

    fn get_name(&self) -> String {
        "ReduceMax".to_string()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpReduceProd {
    data: MilliOpGraphTensorId,
    axes: Option<MilliOpGraphTensorId>,
    keepdims: bool,
    noop_with_empty_axes: bool,
}

impl MilliOpReduceProd {
    pub fn new(
        data: MilliOpGraphTensorId,
        axes: Option<MilliOpGraphTensorId>,
        keepdims: bool,
        noop_with_empty_axes: bool,
    ) -> Self {
        Self {
            data,
            axes,
            keepdims,
            noop_with_empty_axes,
        }
    }
}

impl MilliOp for MilliOpReduceProd {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        vec![self.data]
    }

    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        let data = &inputs[&self.data];
        let axes = if let Some(axes) = self.axes {
            Vec::<i64>::try_from(inputs[&axes].try_to_rank::<P1>()?)?
        } else {
            (0i64..(data.shape().len() as i64)).collect()
        };
        let axes = if axes.is_empty() {
            if self.noop_with_empty_axes {
                return Ok(data.clone());
            } else {
                (0i64..(data.shape().len() as i64)).collect::<Vec<_>>()
            }
        } else {
            axes
        };
        let axes = axes
            .into_iter()
            .map(|x| {
                (if x < 0 {
                    x + data.shape().len() as i64
                } else {
                    x
                }) as usize
            })
            .collect::<Vec<_>>();
        let out = data.reduce_prod(axes, self.keepdims, backend)?;
        Ok(out)
    }

    fn get_name(&self) -> String {
        "ReduceProd".to_string()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpReduceMean {
    data: MilliOpGraphTensorId,
    axes: Option<MilliOpGraphTensorId>,
    keepdims: bool,
    noop_with_empty_axes: bool,
}

impl MilliOpReduceMean {
    pub fn new(
        data: MilliOpGraphTensorId,
        axes: Option<MilliOpGraphTensorId>,
        keepdims: bool,
        noop_with_empty_axes: bool,
    ) -> Self {
        Self {
            data,
            axes,
            keepdims,
            noop_with_empty_axes,
        }
    }
}

impl MilliOp for MilliOpReduceMean {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        vec![self.data]
    }

    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        let data = &inputs[&self.data];
        let axes = if let Some(axes) = self.axes {
            Vec::<i64>::try_from(inputs[&axes].try_to_rank::<P1>()?)?
        } else {
            (0i64..(data.rank() as i64)).collect()
        };
        let axes = if axes.is_empty() {
            if self.noop_with_empty_axes {
                return Ok(data.clone());
            } else {
                (0i64..(data.rank() as i64)).collect::<Vec<_>>()
            }
        } else {
            axes
        };
        let axes = axes
            .into_iter()
            .map(|x| (if x < 0 { x + data.rank() as i64 } else { x }) as usize)
            .collect::<Vec<_>>();
        let out = data.reduce_mean(axes, self.keepdims, backend)?;
        Ok(out)
    }

    fn get_name(&self) -> String {
        "ReduceMean".to_string()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpSlice {
    data: MilliOpGraphTensorId,
    starts: MilliOpGraphTensorId,
    ends: MilliOpGraphTensorId,
    steps: Option<MilliOpGraphTensorId>,
    axes: Option<MilliOpGraphTensorId>,
}

impl MilliOpSlice {
    pub fn new(
        data: MilliOpGraphTensorId,
        starts: MilliOpGraphTensorId,
        ends: MilliOpGraphTensorId,
        steps: Option<MilliOpGraphTensorId>,
        axes: Option<MilliOpGraphTensorId>,
    ) -> Self {
        Self {
            data,
            starts,
            ends,
            steps,
            axes,
        }
    }
}

impl MilliOp for MilliOpSlice {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        let mut res = vec![self.data, self.starts, self.ends];
        if let Some(steps) = &self.steps {
            res.push(*steps);
        }
        if let Some(axes) = &self.axes {
            res.push(*axes);
        }
        res
    }

    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        let data_input = &inputs[&self.data];
        let input_shape = data_input.shape();
        let input_rank = data_input.rank();
        let axes: Vec<i64> = if let Some(axes) = &self.axes {
            inputs[axes]
                .cast(DType::I64, backend)?
                .try_to_rank::<P1>()?
                .try_into()?
        } else {
            (0i64..(input_rank as i64)).collect()
        };
        let steps: Vec<i64> = if let Some(steps) = &self.steps {
            inputs[steps]
                .cast(DType::I64, backend)?
                .try_to_rank::<P1>()?
                .try_into()?
        } else {
            axes.iter().map(|_| 1).collect()
        };
        let starts: Vec<i64> = inputs[&self.starts]
            .cast(DType::I64, backend)?
            .try_to_rank::<P1>()?
            .try_into()?;
        let ends: Vec<i64> = inputs[&self.ends]
            .cast(DType::I64, backend)?
            .try_to_rank::<P1>()?
            .try_into()?;
        let mut output_slice = vec![];
        for &dim in &input_shape {
            output_slice.push(0..dim);
        }
        for (i, axis) in axes.into_iter().enumerate() {
            let axis = if axis < 0 {
                (input_rank as i64 + axis) as usize
            } else {
                axis as usize
            };
            let step = steps[i];
            if step != 1 {
                return Err(MilliOpGraphError::InvalidInput(format!(
                    "Step {step} is not supported"
                )));
            }

            let start = (if starts[i] < 0 {
                input_shape[axis] as i64 + starts[i]
            } else {
                starts[i]
            })
            .min(input_shape[axis] as i64)
            .max(0) as u64;

            let end = (if ends[i] < 0 {
                input_shape[axis] as i64 + ends[i]
            } else {
                ends[i]
            })
            .min(input_shape[axis] as i64)
            .max(0) as u64;
            output_slice[axis] = start..end;
        }
        let output = data_input.slice(&output_slice, backend)?;
        Ok(output)
    }

    fn get_name(&self) -> String {
        "Slice".to_string()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpReshape {
    data: MilliOpGraphTensorId,
    shape: MilliOpGraphTensorId,
    allowzero: bool,
}

impl MilliOpReshape {
    pub fn new(data: MilliOpGraphTensorId, shape: MilliOpGraphTensorId, allowzero: bool) -> Self {
        Self {
            data,
            shape,
            allowzero,
        }
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

impl MilliOp for MilliOpReshape {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        vec![self.data, self.shape]
    }

    fn infer(
        &self,
        known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>,
        symbolic_resolver: &mut SymbolicResolver,
        backend: &mut EvalBackend,
    ) -> Result<TensorInfo, MilliOpGraphError> {
        let data_input = &known_inputs[&self.data];
        let shape_input = known_inputs[&self.shape]
            .try_to_rank::<P1>(symbolic_resolver)?
            .try_to_type::<i64>()?;

        if let Some(shape) = shape_input.as_numeric() {
            if let Some(data) = data_input.as_shaped() {
                match data {
                    TensorInfoShaped::Numeric(data) => {
                        let inputs = HashMap::from([
                            (self.shape, shape.to_dyn_rank().to_dyn_type()),
                            (self.data, data.clone()),
                        ]);
                        Ok(TensorInfo::from(self.eval(&inputs, backend)?))
                    }
                    TensorInfoShaped::Symbolic(data) => {
                        let new_shape = self.calculate_new_shape(data.shape(), &shape.to_vec())?;
                        Ok(TensorInfo::from(data.reshape(new_shape)))
                    }
                }
            } else {
                let mut new_shape = vec![];
                for (i, dim) in shape.to_vec().into_iter().enumerate() {
                    if dim > 0 {
                        new_shape.push(ScalarInfoTyped::Numeric(dim as u64));
                    } else if dim == 0 {
                        new_shape.push(
                            data_input
                                .shape(symbolic_resolver)
                                .get(&[i as u64], symbolic_resolver)
                                .unwrap(),
                        );
                    } else {
                        new_shape.push(ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(
                            symbolic_resolver,
                        )));
                    }
                }
                Ok(TensorInfo::from(TensorInfoRanked::<DynRank>::new(
                    data_input.first_element(),
                    new_shape,
                    symbolic_resolver,
                )))
            }
        } else if let Some(shape) = shape_input.as_shaped() {
            let output_rank = shape.shape()[0];
            let mut new_shape = vec![];
            for i in 0..output_rank {
                let dim = shape.get(&[i]).unwrap();
                match dim {
                    ScalarInfoTyped::Numeric(dim) => {
                        if dim > 0 {
                            new_shape.push(ScalarInfoTyped::Numeric(dim as u64));
                        } else if dim == 0 {
                            new_shape.push(
                                data_input
                                    .shape(symbolic_resolver)
                                    .get(&[i], symbolic_resolver)
                                    .unwrap(),
                            );
                        } else {
                            new_shape.push(ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(
                                symbolic_resolver,
                            )));
                        }
                    }
                    ScalarInfoTyped::Symbolic(_x) => {
                        // Could be negative or zero, so have to use new symbol
                        new_shape.push(ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(
                            symbolic_resolver,
                        )));
                    }
                }
            }
            Ok(TensorInfo::from(TensorInfoRanked::<DynRank>::new(
                data_input.first_element(),
                new_shape,
                symbolic_resolver,
            )))
        } else {
            // We don't even know the rank of the output
            Ok(TensorInfo::new_from_first_element_and_rank(
                data_input.first_element(),
                shape_input.shape()[0].cast(),
                symbolic_resolver,
            ))
        }
    }

    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        let data_input = &inputs[&self.data];
        let shape_input = &inputs[&self.shape];
        let shape_input_value: Vec<i64> = shape_input
            .cast(DType::I64, backend)?
            .try_to_rank::<P1>()?
            .try_into()?;

        let output_shape = self.calculate_new_shape(&data_input.shape(), &shape_input_value)?;

        let output_value = data_input.reshape(output_shape, backend)?;

        Ok(output_value)
    }

    fn get_name(&self) -> String {
        "Reshape".to_string()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpSqueeze {
    data: MilliOpGraphTensorId,
    axes: MilliOpGraphTensorId,
}

impl MilliOpSqueeze {
    pub fn new(data: MilliOpGraphTensorId, axes: MilliOpGraphTensorId) -> Self {
        Self { data, axes }
    }
}

impl MilliOp for MilliOpSqueeze {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        vec![self.data, self.axes]
    }

    fn infer(
        &self,
        known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>,
        symbolic_resolver: &mut SymbolicResolver,
        backend: &mut EvalBackend,
    ) -> Result<TensorInfo, MilliOpGraphError> {
        let data_input = &known_inputs[&self.data];
        let axes_input = &known_inputs[&self.axes]
            .try_to_rank::<P1>(symbolic_resolver)?
            .try_to_type::<i64>()?;

        if let Some(axes) = axes_input.as_numeric() {
            if let Some(data) = data_input.as_numeric() {
                let inputs = HashMap::from([
                    (self.data, data.clone()),
                    (self.axes, axes.to_dyn_type().to_dyn_rank()),
                ]);
                Ok(TensorInfo::from(self.eval(&inputs, backend)?))
            } else {
                let axes = axes.to_vec();
                if let Some(data) = data_input.as_ranked() {
                    let mut new_shape = vec![];
                    for i in 0..data.rank() {
                        let mut found = false;
                        for axis in &axes {
                            if (axis >= &0 && i == *axis as usize)
                                || (axis < &0 && i == (data.rank() as i64 + axis) as usize)
                            {
                                // Skip dim
                                found = true;
                                break;
                            } else {
                                // keep dim
                            }
                        }
                        if !found {
                            new_shape.push(data.shape()[i].clone());
                        }
                    }
                    Ok(TensorInfo::from(data.reshape(
                        new_shape,
                        symbolic_resolver,
                        backend,
                    )?))
                } else {
                    // Data has no defined rank, just decrease the rank by the number of axes.
                    let new_rank = data_input.rank().add_offset(-(axes.len() as i64));
                    Ok(TensorInfo::new_from_first_element_and_rank(
                        data_input.first_element(),
                        new_rank,
                        symbolic_resolver,
                    ))
                }
            }
        } else if let Some(axes) = axes_input.as_shaped() {
            // Data has no defined rank, just decrease the rank by the number of axes.
            let new_rank = data_input.rank().add_offset(-(axes.shape()[0] as i64));
            Ok(TensorInfo::new_from_first_element_and_rank(
                data_input.first_element(),
                new_rank,
                symbolic_resolver,
            ))
        } else {
            // Can't infer the output shape at all
            Ok(TensorInfo::new_from_first_element_and_rank(
                data_input.first_element(),
                ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(symbolic_resolver)),
                symbolic_resolver,
            ))
        }
    }

    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
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
            Ok(output)
        } else {
            // Multiple axes (use reshape)
            let input_shape = inputs[&self.data].shape();
            let mut output_shape = Vec::new();
            for i in 0..(input_shape.len() - axes.len()) {
                let mut is_selected = false;
                for axis in &axes {
                    let axis = if *axis < 0 {
                        input_shape.len() as i64 + *axis
                    } else {
                        *axis
                    };
                    if axis == i as i64 {
                        is_selected = true;
                        break;
                    }
                }
                if is_selected {
                    // Skip it
                } else {
                    output_shape.push(input_shape[i]);
                }
            }
            let output = inputs[&self.data].reshape(output_shape, backend)?;
            Ok(output)
        }
    }

    fn get_name(&self) -> String {
        "Squeeze".to_string()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpUnsqueeze {
    data: MilliOpGraphTensorId,
    axes: MilliOpGraphTensorId,
}

impl MilliOpUnsqueeze {
    pub fn new(data: MilliOpGraphTensorId, axes: MilliOpGraphTensorId) -> Self {
        Self { data, axes }
    }
}

impl MilliOp for MilliOpUnsqueeze {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        vec![self.data, self.axes]
    }

    fn infer(
        &self,
        known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>,
        symbolic_resolver: &mut SymbolicResolver,
        backend: &mut EvalBackend,
    ) -> Result<TensorInfo, MilliOpGraphError> {
        let data_input = &known_inputs[&self.data];
        let axes_input = &known_inputs[&self.axes]
            .try_to_rank::<P1>(symbolic_resolver)?
            .try_to_type::<i64>()?;

        if let Some(axes) = axes_input.as_numeric() {
            if let Some(data) = data_input.as_numeric() {
                let inputs = HashMap::from([
                    (self.data, data.clone()),
                    (self.axes, axes.to_dyn_type().to_dyn_rank()),
                ]);
                Ok(TensorInfo::from(self.eval(&inputs, backend)?))
            } else {
                let axes = axes.to_vec();
                if let Some(data) = data_input.as_ranked() {
                    let mut new_shape = vec![];
                    let new_rank = data.rank() + axes.len();
                    let mut input_i = 0;
                    for i in 0..new_rank {
                        let mut found = false;
                        for axis in &axes {
                            if (axis >= &0 && i == *axis as usize)
                                || (axis < &0 && i == (new_rank as i64 + axis) as usize)
                            {
                                // Skip dim
                                found = true;
                                break;
                            } else {
                                // keep dim
                            }
                        }
                        if found {
                            new_shape.push(ScalarInfoTyped::Numeric(1));
                        } else {
                            new_shape.push(data.shape()[input_i].clone());
                            input_i += 1;
                        }
                    }
                    Ok(TensorInfo::from(data.reshape(
                        new_shape,
                        symbolic_resolver,
                        backend,
                    )?))
                } else {
                    // Data has no defined rank, just decrease the rank by the number of axes.
                    let new_rank = data_input.rank().add_offset(axes.len() as i64);
                    Ok(TensorInfo::new_from_first_element_and_rank(
                        data_input.first_element(),
                        new_rank,
                        symbolic_resolver,
                    ))
                }
            }
        } else if let Some(axes) = axes_input.as_shaped() {
            // Data has no defined rank, just decrease the rank by the number of axes.
            let new_rank = data_input.rank().add_offset(axes.shape()[0] as i64);
            Ok(TensorInfo::new_from_first_element_and_rank(
                data_input.first_element(),
                new_rank,
                symbolic_resolver,
            ))
        } else {
            // Can't infer the output shape at all
            Ok(TensorInfo::new_from_first_element_and_rank(
                data_input.first_element(),
                ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(symbolic_resolver)),
                symbolic_resolver,
            ))
        }
    }

    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
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
            let output = inputs[&self.data].unsqueeze(axis)?;
            Ok(output)
        } else {
            // Multiple axes (use reshape)
            let input_shape = inputs[&self.data].shape();
            let mut output_shape = Vec::new();
            let mut input_index = 0;
            for i in 0..(input_shape.len() + axes.len()) {
                let mut is_selected = false;
                for axis in &axes {
                    let axis = if *axis < 0 {
                        input_shape.len() as i64 + *axis
                    } else {
                        *axis
                    };
                    if axis == i as i64 {
                        is_selected = true;
                        break;
                    }
                }
                if is_selected {
                    output_shape.push(1);
                } else {
                    output_shape.push(input_shape[input_index]);
                    input_index += 1;
                }
            }
            let output = inputs[&self.data].reshape(output_shape, backend)?;
            Ok(output)
        }
    }

    fn get_name(&self) -> String {
        "Unsqueeze".to_string()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpCast {
    data: MilliOpGraphTensorId,
    dtype: DType,
}

impl MilliOpCast {
    pub fn new(data: MilliOpGraphTensorId, dtype: DType) -> Self {
        Self { data, dtype }
    }
}

impl MilliOp for MilliOpCast {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        vec![self.data]
    }

    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        Ok(inputs[&self.data].cast(self.dtype, backend)?)
    }

    fn get_name(&self) -> String {
        "Cast".to_string()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpCastLike {
    data: MilliOpGraphTensorId,
    target_type: MilliOpGraphTensorId,
}

impl MilliOpCastLike {
    pub fn new(data: MilliOpGraphTensorId, target_type: MilliOpGraphTensorId) -> Self {
        Self { data, target_type }
    }
}

impl MilliOp for MilliOpCastLike {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        vec![self.data]
    }

    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        Ok(inputs[&self.data].cast(inputs[&self.target_type].dtype(), backend)?)
    }

    fn get_name(&self) -> String {
        "CastLike".to_string()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpTranspose {
    data: MilliOpGraphTensorId,
    perm: Option<Vec<i64>>,
}

impl MilliOpTranspose {
    pub fn new(data: MilliOpGraphTensorId, perm: Option<Vec<i64>>) -> Self {
        Self { data, perm }
    }
}

impl MilliOp for MilliOpTranspose {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        vec![self.data]
    }

    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        Ok(inputs[&self.data].transpose(self.perm.clone(), backend)?)
    }

    fn get_name(&self) -> String {
        "Transpose".to_string()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpGather {
    data: MilliOpGraphTensorId,
    indices: MilliOpGraphTensorId,
    axis: i64,
}

impl MilliOpGather {
    pub fn new(data: MilliOpGraphTensorId, indices: MilliOpGraphTensorId, axis: i64) -> Self {
        Self {
            data,
            indices,
            axis,
        }
    }
}

impl MilliOp for MilliOpGather {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        vec![self.data]
    }

    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        Ok(NumericTensor::<DynRank>::gather(
            &inputs[&self.data],
            &inputs[&self.indices],
            self.axis,
            backend,
        )?)
    }

    fn get_name(&self) -> String {
        "Gather".to_string()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpConcat {
    inputs: Vec<MilliOpGraphTensorId>,
    axis: i64,
}

impl MilliOpConcat {
    pub fn new(inputs: Vec<MilliOpGraphTensorId>, axis: i64) -> Self {
        Self { inputs, axis }
    }
}

impl MilliOp for MilliOpConcat {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        self.inputs.clone()
    }

    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        let mut resolved_inputs = vec![];
        for input in &self.inputs {
            resolved_inputs.push(&inputs[input]);
        }
        let axis = if self.axis < 0 {
            resolved_inputs[0].shape().len() as i64 + self.axis
        } else {
            self.axis
        } as usize;
        Ok(NumericTensor::<DynRank>::concat(
            resolved_inputs.as_slice(),
            axis,
            backend,
        )?)
    }

    fn get_name(&self) -> String {
        "Concat".to_string()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpArgMax {
    input: MilliOpGraphTensorId,
    axis: i64,
    keepdims: bool,
    select_last_index: bool,
}

impl MilliOpArgMax {
    pub fn new(
        input: MilliOpGraphTensorId,
        axis: i64,
        keepdims: bool,
        select_last_index: bool,
    ) -> Self {
        Self {
            input,
            axis,
            keepdims,
            select_last_index,
        }
    }
}

impl MilliOp for MilliOpArgMax {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        vec![self.input]
    }

    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        let input = inputs[&self.input].clone();
        let axis = if self.axis < 0 {
            input.shape().len() as i64 + self.axis
        } else {
            self.axis
        } as usize;
        let max = input.argmax(axis, self.keepdims, self.select_last_index, backend)?;
        Ok(max)
    }

    fn get_name(&self) -> String {
        "ArgMax".to_string()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpArgMin {
    input: MilliOpGraphTensorId,
    axis: i64,
    keepdims: bool,
    select_last_index: bool,
}

impl MilliOpArgMin {
    pub fn new(
        input: MilliOpGraphTensorId,
        axis: i64,
        keepdims: bool,
        select_last_index: bool,
    ) -> Self {
        Self {
            input,
            axis,
            keepdims,
            select_last_index,
        }
    }
}

impl MilliOp for MilliOpArgMin {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        vec![self.input]
    }

    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        let input = inputs[&self.input].clone();
        let axis = if self.axis < 0 {
            input.shape().len() as i64 + self.axis
        } else {
            self.axis
        } as usize;
        let max = input.argmin(axis, self.keepdims, self.select_last_index, backend)?;
        Ok(max)
    }

    fn get_name(&self) -> String {
        "ArgMin".to_string()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpSplit {
    data: MilliOpGraphTensorId,
    split: Option<MilliOpTensorIDOrLiteral>,
    axis: i64,
    num_outputs: Option<usize>,
    output_id: usize,
}

impl MilliOpSplit {
    pub fn new(
        data: MilliOpGraphTensorId,
        split: Option<MilliOpTensorIDOrLiteral>,
        axis: i64,
        num_outputs: Option<usize>,
        output_id: usize,
    ) -> Self {
        Self {
            data,
            split,
            axis,
            num_outputs,
            output_id,
        }
    }
}

impl MilliOp for MilliOpSplit {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        vec![self.data]
    }

    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        let split: Vec<i64> = if let Some(split) = &self.split {
            match split {
                MilliOpTensorIDOrLiteral::TensorID(split) => {
                    inputs[split].clone().try_to_rank::<P1>()?.try_into()?
                }
                MilliOpTensorIDOrLiteral::Literal(split) => {
                    split.try_to_rank::<P1>()?.try_into()?
                }
            }
        } else {
            Err(MilliOpGraphError::InvalidInput(
                "Split attribute is not set, and we do not support num_outputs yet".to_string(),
            ))?
        };
        let outs = inputs[&self.data].split(&split, self.axis, backend)?;
        Ok(outs[self.output_id].clone())
    }

    fn get_name(&self) -> String {
        "Split".to_string()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpWhere {
    condition: MilliOpGraphTensorId,
    x: MilliOpGraphTensorId,
    y: MilliOpGraphTensorId,
}

impl MilliOpWhere {
    pub(crate) fn new(
        condition: MilliOpGraphTensorId,
        x: MilliOpGraphTensorId,
        y: MilliOpGraphTensorId,
    ) -> Self {
        Self { condition, x, y }
    }
}

impl MilliOp for MilliOpWhere {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        vec![self.condition, self.x, self.y]
    }

    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        Ok(inputs[&self.condition].where_op(&inputs[&self.x], &inputs[&self.y], backend)?)
    }

    fn get_name(&self) -> String {
        "Where".to_string()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnyMilliOp {
    Constant(MilliOpConstant),
    ConstantOfShape(MilliOpConstantOfShape),
    SimpleBinary(MilliOpSimpleBinary),
    MatMul(MilliOpMatMul),
    Pow(MilliOpPow),
    SimpleUnary(MilliOpSimpleUnary),
    ClampMin(MilliOpClampMin),
    NonZero(MilliOpNonZero),
    CumSum(MilliOpCumSum),
    Shape(MilliOpShape),
    Reshape(MilliOpReshape),
    Slice(MilliOpSlice),
    ReduceSum(MilliOpReduceSum),
    ReduceMin(MilliOpReduceMin),
    ReduceMax(MilliOpReduceMax),
    ReduceProd(MilliOpReduceProd),
    ReduceMean(MilliOpReduceMean),
    Cast(MilliOpCast),
    CastLike(MilliOpCastLike),
    Transpose(MilliOpTranspose),
    Squeeze(MilliOpSqueeze),
    Unsqueeze(MilliOpUnsqueeze),
    Gather(MilliOpGather),
    Concat(MilliOpConcat),
    Split(MilliOpSplit),
    Where(MilliOpWhere),
    Range(MilliOpRange),
    Expand(MilliOpExpand),
    ArgMax(MilliOpArgMax),
    ArgMin(MilliOpArgMin),
}

impl MilliOp for AnyMilliOp {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        match self {
            AnyMilliOp::Constant(x) => x.get_inputs(),
            AnyMilliOp::ConstantOfShape(x) => x.get_inputs(),
            AnyMilliOp::SimpleBinary(x) => x.get_inputs(),
            AnyMilliOp::Pow(x) => x.get_inputs(),
            AnyMilliOp::SimpleUnary(x) => <_ as MilliOp>::get_inputs(x),
            AnyMilliOp::MatMul(x) => x.get_inputs(),
            AnyMilliOp::ClampMin(x) => <_ as MilliOp>::get_inputs(x),
            AnyMilliOp::NonZero(x) => x.get_inputs(),
            AnyMilliOp::CumSum(x) => x.get_inputs(),
            AnyMilliOp::Shape(x) => x.get_inputs(),
            AnyMilliOp::Reshape(x) => x.get_inputs(),
            AnyMilliOp::Slice(x) => x.get_inputs(),
            AnyMilliOp::ReduceSum(x) => x.get_inputs(),
            AnyMilliOp::ReduceMin(x) => x.get_inputs(),
            AnyMilliOp::ReduceMax(x) => x.get_inputs(),
            AnyMilliOp::ReduceProd(x) => x.get_inputs(),
            AnyMilliOp::ReduceMean(x) => x.get_inputs(),
            AnyMilliOp::Cast(x) => x.get_inputs(),
            AnyMilliOp::CastLike(x) => x.get_inputs(),
            AnyMilliOp::Transpose(x) => x.get_inputs(),
            AnyMilliOp::Squeeze(x) => x.get_inputs(),
            AnyMilliOp::Unsqueeze(x) => x.get_inputs(),
            AnyMilliOp::Gather(x) => x.get_inputs(),
            AnyMilliOp::Concat(x) => x.get_inputs(),
            AnyMilliOp::Split(x) => x.get_inputs(),
            AnyMilliOp::Where(x) => x.get_inputs(),
            AnyMilliOp::Range(x) => x.get_inputs(),
            AnyMilliOp::Expand(x) => x.get_inputs(),
            AnyMilliOp::ArgMax(x) => x.get_inputs(),
            AnyMilliOp::ArgMin(x) => x.get_inputs(),
        }
    }

    fn infer(
        &self,
        known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>,
        symbolic_resolver: &mut SymbolicResolver,
        backend: &mut EvalBackend,
    ) -> Result<TensorInfo, MilliOpGraphError> {
        match self {
            AnyMilliOp::Constant(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::ConstantOfShape(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::SimpleBinary(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::Pow(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::SimpleUnary(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::MatMul(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::ClampMin(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::NonZero(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::CumSum(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::Shape(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::Reshape(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::Slice(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::ReduceSum(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::ReduceMin(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::ReduceMax(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::ReduceProd(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::ReduceMean(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::Cast(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::CastLike(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::Transpose(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::Squeeze(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::Unsqueeze(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::Gather(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::Concat(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::Split(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::Where(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::Range(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::Expand(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::ArgMax(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::ArgMin(x) => x.infer(known_inputs, symbolic_resolver, backend),
        }
    }

    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        match self {
            AnyMilliOp::Constant(x) => x.eval(inputs, backend),
            AnyMilliOp::ConstantOfShape(x) => x.eval(inputs, backend),
            AnyMilliOp::SimpleBinary(x) => x.eval(inputs, backend),
            AnyMilliOp::Pow(x) => x.eval(inputs, backend),
            AnyMilliOp::SimpleUnary(x) => <_ as MilliOp>::eval(x, inputs, backend),
            AnyMilliOp::MatMul(x) => x.eval(inputs, backend),
            AnyMilliOp::ClampMin(x) => <_ as MilliOp>::eval(x, inputs, backend),
            AnyMilliOp::NonZero(x) => x.eval(inputs, backend),
            AnyMilliOp::CumSum(x) => x.eval(inputs, backend),
            AnyMilliOp::Shape(x) => x.eval(inputs, backend),
            AnyMilliOp::Reshape(x) => x.eval(inputs, backend),
            AnyMilliOp::Slice(x) => x.eval(inputs, backend),
            AnyMilliOp::ReduceSum(x) => x.eval(inputs, backend),
            AnyMilliOp::ReduceMin(x) => x.eval(inputs, backend),
            AnyMilliOp::ReduceMax(x) => x.eval(inputs, backend),
            AnyMilliOp::ReduceProd(x) => x.eval(inputs, backend),
            AnyMilliOp::ReduceMean(x) => x.eval(inputs, backend),
            AnyMilliOp::Cast(x) => x.eval(inputs, backend),
            AnyMilliOp::CastLike(x) => x.eval(inputs, backend),
            AnyMilliOp::Transpose(x) => x.eval(inputs, backend),
            AnyMilliOp::Squeeze(x) => x.eval(inputs, backend),
            AnyMilliOp::Unsqueeze(x) => x.eval(inputs, backend),
            AnyMilliOp::Gather(x) => x.eval(inputs, backend),
            AnyMilliOp::Concat(x) => x.eval(inputs, backend),
            AnyMilliOp::Split(x) => x.eval(inputs, backend),
            AnyMilliOp::Where(x) => x.eval(inputs, backend),
            AnyMilliOp::Range(x) => x.eval(inputs, backend),
            AnyMilliOp::Expand(x) => x.eval(inputs, backend),
            AnyMilliOp::ArgMax(x) => x.eval(inputs, backend),
            AnyMilliOp::ArgMin(x) => x.eval(inputs, backend),
        }
    }

    fn get_name(&self) -> String {
        match self {
            AnyMilliOp::Constant(x) => x.get_name(),
            AnyMilliOp::ConstantOfShape(x) => x.get_name(),
            AnyMilliOp::SimpleBinary(x) => x.get_name(),
            AnyMilliOp::MatMul(x) => x.get_name(),
            AnyMilliOp::Pow(x) => x.get_name(),
            AnyMilliOp::SimpleUnary(x) => <_ as MilliOp>::get_name(x),
            AnyMilliOp::ClampMin(x) => <_ as MilliOp>::get_name(x),
            AnyMilliOp::NonZero(x) => <_ as MilliOp>::get_name(x),
            AnyMilliOp::CumSum(x) => x.get_name(),
            AnyMilliOp::Shape(x) => x.get_name(),
            AnyMilliOp::Reshape(x) => x.get_name(),
            AnyMilliOp::Slice(x) => x.get_name(),
            AnyMilliOp::ReduceSum(x) => x.get_name(),
            AnyMilliOp::ReduceMin(x) => x.get_name(),
            AnyMilliOp::ReduceMax(x) => x.get_name(),
            AnyMilliOp::ReduceProd(x) => x.get_name(),
            AnyMilliOp::ReduceMean(x) => x.get_name(),
            AnyMilliOp::Cast(x) => x.get_name(),
            AnyMilliOp::CastLike(x) => x.get_name(),
            AnyMilliOp::Transpose(x) => x.get_name(),
            AnyMilliOp::Squeeze(x) => x.get_name(),
            AnyMilliOp::Unsqueeze(x) => x.get_name(),
            AnyMilliOp::Gather(x) => x.get_name(),
            AnyMilliOp::Concat(x) => x.get_name(),
            AnyMilliOp::Split(x) => x.get_name(),
            AnyMilliOp::Where(x) => x.get_name(),
            AnyMilliOp::Range(x) => x.get_name(),
            AnyMilliOp::Expand(x) => x.get_name(),
            AnyMilliOp::ArgMax(x) => x.get_name(),
            AnyMilliOp::ArgMin(x) => x.get_name(),
        }
    }
}
