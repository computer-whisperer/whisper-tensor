use std::sync::Arc;
use super::{operators, Error};
use super::operators::{Cast, Constant, CumSum, Div, Expand, GroupNormalization, LayerNormalization, Mul, RMSNormalization, ReduceSum, Reshape, Sigmoid, Slice, Squeeze, TopK, Transpose, Unsqueeze};
use super::tensor::{DType, Dimension, Shape, Tensor, TensorData, TensorDataValue};
use super::weights::WeightManager;

pub fn linear(weight_manager: &impl WeightManager, input: Arc<dyn Tensor>) -> Result<Arc<dyn Tensor>, Error> {

    let weight = weight_manager.get_tensor("weight")?;
    let weight = transpose(weight);
    
    let input_shape = input.shape();
    let weight_shape = weight.shape();
    
    let bias = weight_manager.get_tensor("bias").ok();
    let input_rank = input.rank();
    let input = unsqueeze(input, (input_rank as i64) - 1)?;

    let input_shape_2 = input.shape();
    
    let weight_rank = weight.rank();
    //let weight = unsqueeze(weight, weight_rank as i64)?;
    let mat_out = operators::MatMul::new(
        weight_manager.get_prefix().map(|x| x.to_string()),
        input,
        weight
    )?;
    let mat_out_shape = mat_out.shape();
    let mat_out = squeeze(mat_out, (input_rank as i64) - 1)?;
    let mat_out_shape_1 = mat_out.shape();
    if let Some(bias) = bias {
        Ok(operators::Add::new(Some(format!("{}.bias", weight_manager.get_prefix().unwrap())), mat_out, bias)?)
    } else {
        Ok(mat_out)
    }
}

pub fn reshape(input: Arc<dyn Tensor>, dims: Vec<i64>) -> Result<Arc<Reshape>, Error> {
    let shape = Shape::from(&[dims.len()][..]);
    let c = Constant::new(None, TensorData::new(dims.into(), shape)?);
    Ok(Reshape::new(None, input, c)?)
}

pub fn unsqueeze(input: Arc<dyn Tensor>, dim: i64) -> Result<Arc<Unsqueeze>, Error> {
    let c = Constant::new(None, TensorData::new(vec![dim].into(), Shape::from(&[1][..]))?);
    Ok(Unsqueeze::new(None, input, c)?)
}

pub fn squeeze(input: Arc<dyn Tensor>, dim: i64) -> Result<Arc<Squeeze>, Error> {
    let c = Constant::new(None, TensorData::new(vec![dim].into(), Shape::from(&[1][..]))?);
    Ok(Squeeze::new(None, input, c)?)
}

pub fn slice(input: Arc<dyn Tensor>, start: Vec<i64>, end: Vec<i64>) -> Result<Arc<Slice>, Error>  {
    let const_shape = Shape::new(vec![Dimension::new(Some(start.len()), None, None)]);
    let start = Constant::new(None, TensorData::new(start.into(), const_shape.clone())?);
    let end = Constant::new(None, TensorData::new(end.into(), const_shape)?);
    Ok(Slice::new(None, input, start, end, None, None)?)
}

pub fn cast(input: Arc<dyn Tensor>, dtype: DType) -> Arc<dyn Tensor>  {
    if input.dtype() != dtype {
        Cast::new(None, input, dtype)
    } else {
        input
    }
}

pub fn transpose(input: Arc<dyn Tensor>) -> Arc<Transpose> {
    let mut dims: Vec<_> = (0..input.rank() as i64).collect();
    let (a, b) = (dims[input.rank()-2], dims[input.rank()-1]);
    dims[input.rank()-2] = b;
    dims[input.rank()-1] = a;
    Transpose::new(None, input, Some(dims))
}

pub fn layer_norm(weight_manager: &impl WeightManager, input: Arc<dyn Tensor>, epsilon: f32) -> Result<Arc<LayerNormalization>, Error> {
    LayerNormalization::new(
        weight_manager.get_prefix().map(|x| x.to_string()),
        input,
        weight_manager.get_tensor("weight")?,
        weight_manager.get_tensor("bias").ok(),
        -1,
        epsilon,
        1
    )
}

pub fn group_norm(weight_manager: &impl WeightManager, input: Arc<dyn Tensor>, epsilon: f32, num_groups: i64) -> Result<Arc<GroupNormalization>, Error> {
    GroupNormalization::new(
        weight_manager.get_prefix().map(|x| x.to_string()),
        input,
        weight_manager.get_tensor("weight")?,
        weight_manager.get_tensor("bias")?,
        num_groups,
        epsilon
    )
}

pub fn cumsum(input: Arc<dyn Tensor>, axis: i32) -> Result<Arc<CumSum>, Error> {
    let shape = Shape::new(vec![Dimension::new(Some(1), None, None)]);
    let axis = Constant::new(None, TensorData::fill(shape, axis)?);
    CumSum::new(None, input, axis)
}

pub fn sum_dim(input: Arc<dyn Tensor>, axis: i64, keepdims: Option<bool>) -> Result<Arc<ReduceSum>, Error> {
    let shape = Shape::new(vec![Dimension::new(Some(1), None, None)]);
    let constant = Constant::new(None, TensorData::fill(shape, axis)?);
    ReduceSum::new(None, input, constant, keepdims)
}

pub fn rms_norm(weight_manager: &impl WeightManager, input: Arc<dyn Tensor>, epsilon: Option<f32>) -> Result<Arc<RMSNormalization>, Error> {
    RMSNormalization::new(
        weight_manager.get_prefix().map(|x| x.to_string()),
        input,
        weight_manager.get_tensor("weight")?,
        epsilon,
        -1
    )
}

pub fn silu(input: Arc<dyn Tensor>) -> Result<Arc<dyn Tensor>, Error> {
    let x = Sigmoid::new(None, input.clone());
    Ok(Mul::new(None, input.clone(), x)?)
}

pub fn swiglu(weight_manager: &impl WeightManager, input: Arc<dyn Tensor>) -> Result<Arc<dyn Tensor>, Error> {
    let x = linear(&weight_manager.prefix("linear_inner"), input.clone())?;
    let x = silu(x)?;
    let x2 = linear(&weight_manager.prefix("linear_outer"), input.clone())?;
    let out = Mul::new(None, x, x2)?;
    Ok(out)
}

pub fn div_scalar<T>(input: Arc<dyn Tensor>, scalar: T) -> Result<Arc<Div>, Error>
where
    T: Copy,
    TensorDataValue: From<Vec<T>>
{
    let shape = Shape::new(vec![Dimension::new(Some(1), None, None)]);
    let constant = Constant::new(None, TensorData::fill(shape, scalar)?);
    
    Ok(Div::new(None, input, constant)?)
}

pub fn expand(input: Arc<dyn Tensor>, dims: Vec<i64>) -> Result<Arc<Expand>, Error> {
    let shape = Shape::from(&[dims.len()][..]);
    let c = Constant::new(None, TensorData::new(dims.into(), shape)?);
    Ok(Expand::new(None, input, c)?)
}

pub fn topk(input: Arc<dyn Tensor>, k: i64, axis: i64) -> Result<(Arc<dyn Tensor>, Arc<dyn Tensor>), Error> {
    let shape = Shape::from(vec![1]);
    let c = Constant::new(None, TensorData::fill(shape, k)?);
    let (a, b) = TopK::new(None, input, c, axis, false, false)?;
    Ok((a, b))
}