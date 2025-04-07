use std::sync::Arc;
use crate::{validate_elementwise_inputs, Error};
use crate::node::{Node, NodeElementwise, SingleOutputNode};
use crate::tensor::{DType, Dimension, Shape, Tensor, TensorData};

fn validate_index_dtype(dtype: DType) -> Result<(), Error> {
    if dtype != DType::I32 {
        Err(Error::InvalidDTypeError)?;
    }
    Ok(())
}

pub struct Add {
    name: Option<String>,
    a: Arc<dyn Tensor>,
    b: Arc<dyn Tensor>
}
impl NodeElementwise for Add {}

impl Add {
    pub fn new(name: Option<String>, a: Arc<dyn Tensor>, b: Arc<dyn Tensor>) -> Result<Arc<Self>, Error> {
        validate_elementwise_inputs(&[a.clone(), b.clone()])?;
        Ok(Arc::new(Self {name, a, b }))
    }
}

impl Node for Add {
    fn get_input_tensors(&self) -> Vec<&dyn Tensor> {
        vec![self.a.as_ref(), self.b.as_ref()]
    }
    fn get_output_tensors(&self) -> Vec<&dyn Tensor> { vec![self] }
    fn get_name(&self) -> Option<&str> {
       self.name.as_ref().map(|x| x.as_str())
   }
    fn get_onnx_type(&self) -> &str {
        "Add"
    }
}

pub struct Sub {
    name: Option<String>,
    a: Arc<dyn Tensor>,
    b: Arc<dyn Tensor>
}

impl Sub {
    pub fn new(name: Option<String>, a: Arc<dyn Tensor>, b: Arc<dyn Tensor>) -> Result<Arc<Self>, Error> {
        validate_elementwise_inputs(&[a.clone(), b.clone()])?;
        Ok(Arc::new(Self {name, a, b }))
    }
}
impl Node for Sub {
    fn get_input_tensors(&self) -> Vec<&dyn Tensor> {
        vec![self.a.as_ref(), self.b.as_ref()]
    }
    fn get_output_tensors(&self) -> Vec<&dyn Tensor> { vec![self] }

    fn get_name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn get_onnx_type(&self) -> &str {
        "Sub"
    }
}

impl NodeElementwise for Sub {}

pub struct Mul {
    name: Option<String>,
    a: Arc<dyn Tensor>,
    b: Arc<dyn Tensor>
}

impl Mul {
    pub fn new(name: Option<String>, a: Arc<dyn Tensor>, b: Arc<dyn Tensor>) -> Result<Arc<Self>, Error> {
        validate_elementwise_inputs(&[a.clone(), b.clone()])?;
        Ok(Arc::new(Self {name, a, b }))
    }
}

impl Node for Mul {
    fn get_input_tensors(&self) -> Vec<&dyn Tensor> {
        vec![self.a.as_ref(), self.b.as_ref()]
    }
    fn get_output_tensors(&self) -> Vec<&dyn Tensor> { vec![self] }
    fn get_name(&self) -> Option<&str> {
        self.name.as_deref()
    }
    fn get_onnx_type(&self) -> &str {
        "Mul"
    }
}

impl NodeElementwise for Mul {}

pub struct Gather {
    name: Option<String>,
    data: Arc<dyn Tensor>,
    indices: Arc<dyn Tensor>,
    axis: isize,
    output_shape: Shape,
    dtype: DType
}

impl Gather {
    pub fn new(name: Option<String>, data: Arc<dyn Tensor>, indices: Arc<dyn Tensor>, axis: usize) -> Result<Arc<Self>, Error> {
        let data_shape = data.shape();
        let data_dtype = data.dtype();
        let indices_shape = indices.shape();
        let indices_dtype = indices.dtype();

        validate_index_dtype(indices_dtype)?;
        
        if data_shape.rank() < 2 {
            return Err(Error::InvalidInputError);
        }
        let mut output_shape = vec![];
        let mut data_i = 0usize;
        loop {
            if data_i != axis {
                output_shape.push(data_shape[data_i].clone())
            }
            else {
                if indices_shape.rank() > 1 || indices_shape[0].resolve()? > 1 {
                    output_shape.extend_from_slice(&indices_shape.dims)
                }
            }
            data_i += 1;
            if data_i >= data_shape.rank() {
                break;
            }
        }

        Ok(Arc::new(Self {
            name,
            data,
            indices,
            axis: axis as isize,
            output_shape: Shape{dims: output_shape},
            dtype: data_dtype
        }))
    }
}

impl Node for Gather {
    fn get_input_tensors(&self) -> Vec<&dyn Tensor> {
        vec![self.data.as_ref(), self.indices.as_ref()]
    }

    fn get_output_tensors(&self) -> Vec<&dyn Tensor> {
        vec![self]
    }
    fn get_name(&self) -> Option<&str> {
        self.name.as_ref().map(|x| x.as_str())
    }
    fn get_onnx_type(&self) -> &str {
        "Gather"
    }
}

impl SingleOutputNode for Gather {
    fn get_output_shape(&self) -> &Shape {
        &self.output_shape
    }

    fn get_output_dtype(&self) -> DType {
        self.dtype
    }
}

pub struct LayerNormalization {
    name: Option<String>,
    input: Arc<dyn Tensor>,
    scale: Arc<dyn Tensor>,
    bias: Option<Arc<dyn Tensor>>,
    axis: i32,
    epsilon: f32,
    stash_type: i32
}

impl SingleOutputNode for LayerNormalization {
    fn get_output_shape(&self) -> &Shape {
        self.input.shape()
    }

    fn get_output_dtype(&self) -> DType {
        self.input.dtype()
    }
}

impl LayerNormalization {
    pub fn new(name: Option<String>, input: Arc<dyn Tensor>, scale: Arc<dyn Tensor>, bias: Option<Arc<dyn Tensor>>, axis: i32, epsilon: f32, stash_type: i32) -> Result<Arc<Self>, Error> {
        if input.dtype() != scale.dtype() {
            return Err(Error::DTypeMismatchError);
        }
        Ok(Arc::new(Self {
            name,
            input,
            scale,
            bias,
            axis,
            epsilon,
            stash_type
        }))
    }
}

impl Node for LayerNormalization {
    fn get_input_tensors(&self) -> Vec<&dyn Tensor> {
        let mut inputs = vec![self.input.as_ref(), self.scale.as_ref()];
        if let Some(bias) = &self.bias {
            inputs.push(bias.as_ref());
        }
        inputs
    }

    fn get_output_tensors(&self) -> Vec<&dyn Tensor> {
        vec![self]
    }
    fn get_name(&self) -> Option<&str> {
        self.name.as_ref().map(|x| x.as_str())
    }
    fn get_onnx_type(&self) -> &str {
        "LayerNormalization"
    }
}

pub struct MatMul {
    name: Option<String>,
    a: Arc<dyn Tensor>,
    b: Arc<dyn Tensor>,
    output_dtype: DType,
    output_shape: Shape
}

impl MatMul { 
    pub fn new(name: Option<String>, a: Arc<dyn Tensor>, b: Arc<dyn Tensor>) -> Result<Arc<Self>, Error> {
        if a.dtype() != b.dtype() {
            Err(Error::DTypeMismatchError)?
        }
        // Handle broadcasting
        let mut a_shape = if a.rank() == 1 {
            a.shape().unsqueeze(0)
        } else {
            a.shape().clone()
        };
        
        let mut b_shape = if b.rank() == 1 {
            b.shape().unsqueeze(1)
        } else {
            b.shape().clone()
        };
        
        while a_shape.rank() > b_shape.rank() {
            b_shape = b_shape.unsqueeze(0)
        }
        
        while b_shape.rank() > a_shape.rank() {
            a_shape = a_shape.unsqueeze(0)
        }
        
        // Validate shapes
        if a_shape.dim(-1).as_ref() != b_shape.dim(-2).as_ref() {
            Err(Error::InputShapeError)?
        }
        
        
        let mut output_dims = vec![];
        // Start with batch dims
        for i in 0..a_shape.rank()-2 {
            output_dims.push(
                if a_shape[i].as_ref() != b_shape[i].as_ref() {
                    a_shape[i].clone()
                }
                else {
                    if let Some(a_value) = a_shape[i].value {
                        if a_value == 1 {
                            b_shape[i].clone()
                        }
                        else {
                            if let Some(b_value) = b_shape[i].value {
                                if b_value == 1 {
                                    a_shape[i].clone()
                                }
                                else {
                                    Err(Error::InputShapeError)?
                                }
                            }
                            else {
                                Err(Error::InputShapeError)?
                            }
                        }
                    }
                    else if let Some(b_value) = b_shape[i].value {
                        if b_value == 1 {
                            a_shape[i].clone()
                        }
                        else {
                            Err(Error::InputShapeError)?
                        }
                    }
                    else {
                        Err(Error::InputShapeError)?
                    }
                }
            )
        }
        // Add output dims
        if a.rank() != 1 {
            output_dims.push(a_shape.dim(-2).clone());
        }
        if b.rank() != 1 {
            output_dims.push(b_shape.dim(-1).clone());
        }
        
        let output_dtype = a.dtype();
        let output_shape = Shape::new(output_dims);
        Ok(Arc::new(MatMul {
            name,
            a,
            b,
            output_dtype,
            output_shape
        }))
    }
}


impl Node for MatMul {
    fn get_input_tensors(&self) -> Vec<&dyn Tensor> {
        vec![self.a.as_ref(), self.b.as_ref()]
    }

    fn get_output_tensors(&self) -> Vec<&dyn Tensor> {
        vec![self]
    }

    fn get_name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn get_onnx_type(&self) -> &str {
        "MatMul"
    }
}

impl SingleOutputNode for MatMul {
    fn get_output_shape(&self) -> &Shape {
        &self.output_shape
    }

    fn get_output_dtype(&self) -> DType {
        self.output_dtype
    }
}

pub struct Gemm {
    name: Option<String>,
    a: Arc<dyn Tensor>,
    b: Arc<dyn Tensor>,
    c: Option<Arc<dyn Tensor>>,
    trans_a: bool,
    trans_b: bool,
    alpha: f32,
    beta: f32,
    output_dtype: DType,
    output_shape: Shape
}

impl Gemm {
    pub fn new(name: Option<String>, a: Arc<dyn Tensor>, b: Arc<dyn Tensor>, c: Option<Arc<dyn Tensor>>, trans_a: bool, trans_b: bool, alpha: f32, beta: f32) -> Result<Arc<Self>, Error> {
        if a.dtype() != b.dtype() {
            return Err(Error::DTypeMismatchError)
        }
        if let Some(c) = &c {
            if c.dtype() != a.dtype() {
                return Err(Error::DTypeMismatchError)
            }
        }
        let a_shape = if trans_a {
            a.shape().transpose()
        } else {
            a.shape().clone()
        };
        let b_shape = if trans_b {
            b.shape().transpose()
        } else {
            b.shape().clone()
        };
        if a_shape.rank() != 2 || b_shape.rank() != 2 {
            return Err(Error::InputShapeError)
        }
        if a_shape[1].as_ref() != b_shape[0].as_ref() {
            return Err(Error::InputShapeError)
        }
        let output_shape = Shape::new(vec![a_shape[0].clone(), b_shape[1].clone()]);
        let output_dtype = a.dtype();
        Ok(Arc::new(Self {
            name,
            a,
            b,
            c,
            trans_a,
            trans_b,
            alpha,
            beta,
            output_dtype,
            output_shape
        }))
    }
}



impl Node for Gemm {
    fn get_input_tensors(&self) -> Vec<&dyn Tensor> {
        if let Some(c) = &self.c {
            vec![&*self.a, &*self.b, c.as_ref()]
        } else {
            vec![&*self.a, &*self.b]
        }
    }

    fn get_output_tensors(&self) -> Vec<&dyn Tensor> {
        vec![self]
    }

    fn get_name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn get_onnx_type(&self) -> &str {
        "Gemm"
    }
}

impl SingleOutputNode for Gemm {
    fn get_output_shape(&self) -> &Shape {
        &self.output_shape
    }

    fn get_output_dtype(&self) -> DType {
        self.output_dtype
    }
}

pub struct Concat {
    name: Option<String>,
    inputs: Vec<Arc<dyn Tensor>>,
    axis: i64,
    output_shape: Shape,
    output_dtype: DType
}

impl Concat {
    pub fn new(name: Option<String>, inputs: Vec<Arc<dyn Tensor>>, axis: i64) -> Result<Arc<Self>, Error> {
        if inputs.is_empty() {
            Err(Error::InvalidInputError)?;
        }
        for input in &inputs {
            if input.dtype() != inputs[0].dtype() {
                Err(Error::DTypeMismatchError)?;
            }
        }
        let u_axis = if axis < 0 { (inputs[0].rank() as i64 + axis) as usize } else { axis as usize };
        if u_axis >= inputs[0].rank() {
            Err(Error::InvalidInputError)?;
        }
        
        let output_dtype = inputs[0].dtype();
        let mut output_dims = vec![];
        for i in 0..inputs[0].rank() {
            if i == u_axis {
                let mut v = 0;
                for input in &inputs {
                    v += input.shape()[i].resolve()?;
                }
                output_dims.push(Dimension::new(Some(v), None, None));
            }
            else {
                for input in &inputs {
                    if input.shape()[i].as_ref() != output_dims[i].as_ref() {
                        Err(Error::InputShapeError)?;
                    }
                }
                output_dims.push(inputs[0].shape()[i].clone());
            }
        }
        let output_shape = Shape::new(output_dims);
        Ok(Arc::new(Self {
            name,
            inputs,
            axis,
            output_dtype,
            output_shape
        }))
    }
}

impl Node for Concat {
    fn get_input_tensors(&self) -> Vec<&dyn Tensor> {
        self.inputs.iter().map(|x| x.as_ref()).collect()
    }

    fn get_output_tensors(&self) -> Vec<&dyn Tensor> {
        vec![]
    }

    fn get_name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn get_onnx_type(&self) -> &str {
        "Concat"
    }
}

/*
pub struct Slice {
    name: Option<String>,
    input: Arc<dyn Tensor>,
    starts: Arc<dyn Tensor>,
    ends: Arc<dyn Tensor>,
    axes: Option<Arc<dyn Tensor>>,
    steps: Option<Arc<dyn Tensor>>,
    output_dtype: DType,
    output_shape: Shape
}

impl Slice {
    
    pub fn new(name: Option<String>, input: Arc<dyn Tensor>, starts: Arc<dyn Tensor>, ends: Arc<dyn Tensor>, axes: Option<Arc<dyn Tensor>>, steps: Option<Arc<dyn Tensor>>) -> Result<Arc<Self>> {
        validate_index_dtype(starts.dtype())?;
        validate_index_dtype(ends.dtype())?;
        if let Some(axes) = &axes {
            validate_index_dtype(axes.dtype())?;
        }
        if let Some(steps) = &steps {
            validate_index_dtype(steps.dtype())?;
        }

        let output_dtype = input.dtype().clone();
        Ok(Arc::new(Self {
            name,
            input,
            starts,
            ends,
            axes,
            steps,
            output_dtype,
            output_shape
        }))
    }
}*/



pub struct Sigmoid {
    name: Option<String>,
    input: Arc<dyn Tensor>
}

impl Sigmoid {
    pub fn new(name: Option<String>, input: Arc<dyn Tensor>) -> Arc<Sigmoid> {
        Arc::new(Sigmoid {
            name,
            input
        })
    }
}

impl Node for Sigmoid {
    fn get_input_tensors(&self) -> Vec<&dyn Tensor> {
        vec![self.input.as_ref()]
    }

    fn get_output_tensors(&self) -> Vec<&dyn Tensor> {
        vec![self]
    }

    fn get_name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn get_onnx_type(&self) -> &str {
        "Sigmoid"
    }
}

impl SingleOutputNode for Sigmoid {
    fn get_output_shape(&self) -> &Shape {
        self.input.shape()
    }

    fn get_output_dtype(&self) -> DType {
        self.input.dtype()
    }
}

pub struct Softplus {
    name: Option<String>,
    input: Arc<dyn Tensor>
}

impl Softplus {
    pub fn new(name: Option<String>, input: Arc<dyn Tensor>) -> Arc<Softplus> {
        Arc::new(Softplus {
            name,
            input
        })
    }
}

impl Node for Softplus {
    fn get_input_tensors(&self) -> Vec<&dyn Tensor> {
        vec![self.input.as_ref()]
    }

    fn get_output_tensors(&self) -> Vec<&dyn Tensor> {
        vec![self]
    }

    fn get_name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn get_onnx_type(&self) -> &str {
        "Softplus"
    }
}

impl SingleOutputNode for Softplus {
    fn get_output_shape(&self) -> &Shape {
        self.input.shape()
    }

    fn get_output_dtype(&self) -> DType {
        self.input.dtype()
    }
}

pub struct Neg {
    name: Option<String>,
    input: Arc<dyn Tensor>
}

impl Neg {
    pub fn new(name: Option<String>, input: Arc<dyn Tensor>) -> Arc<Neg> {
        Arc::new(Neg {
            name,
            input
        })
    }
}
impl Node for Neg {
    fn get_input_tensors(&self) -> Vec<&dyn Tensor> {
        vec![self.input.as_ref()]
    }
    fn get_output_tensors(&self) -> Vec<&dyn Tensor> {
        vec![self]
    }

    fn get_name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn get_onnx_type(&self) -> &str {
        "Neg"
    }
}

impl SingleOutputNode for Neg {
    fn get_output_shape(&self) -> &Shape {
        self.input.shape()
    }

    fn get_output_dtype(&self) -> DType {
        self.input.dtype()
    }
}


pub struct Tanh {
    name: Option<String>,
    input: Arc<dyn Tensor>
}

impl Tanh {
    pub fn new(name: Option<String>, input: Arc<dyn Tensor>) -> Arc<Tanh> {
        Arc::new(Tanh {
            name,
            input
        })
    }
}

impl Node for Tanh {
    fn get_input_tensors(&self) -> Vec<&dyn Tensor> {
        vec![self.input.as_ref()]
    }

    fn get_output_tensors(&self) -> Vec<&dyn Tensor> {
        vec![self]
    }

    fn get_name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn get_onnx_type(&self) -> &str {
        "Tanh"
    }
}

impl SingleOutputNode for Tanh {
    fn get_output_shape(&self) -> &Shape {
        self.input.shape()
    }

    fn get_output_dtype(&self) -> DType {
        self.input.dtype()
    }
}

pub struct Constant {
    name: Option<String>,
    data: TensorData
}

impl Constant {
    pub fn new(name: Option<String>, data: TensorData) -> Arc<Constant> {
        Arc::new(Constant {
            name,
            data
        })
    }
}

impl Node for Constant {
    fn get_input_tensors(&self) -> Vec<&dyn Tensor> {
        vec![]
    }
    
    fn get_output_tensors(&self) -> Vec<&dyn Tensor> {
        vec![self]
    }

    fn get_name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn get_onnx_type(&self) -> &str {
        "Constant"
    }
}

impl SingleOutputNode for Constant {
    fn get_output_shape(&self) -> &Shape {
        &self.data.shape()
    }
    fn get_output_dtype(&self) -> DType {
        self.data.dtype()
    }
    fn resolve_output_data(&self) -> Option<TensorData> {
        Some(self.data.clone())
    }
}

pub struct Cast {
    name: Option<String>,
    input: Arc<dyn Tensor>,
    to: DType
}

impl Cast {
    pub fn new(name: Option<String>, input: Arc<dyn Tensor>, to: DType) -> Arc<Cast> {
        Arc::new(Cast {
            name,
            input,
            to
        })
    }
}

impl Node for Cast {
    fn get_input_tensors(&self) -> Vec<&dyn Tensor> {
        vec![self.input.as_ref()]
    }
    
    fn get_output_tensors(&self) -> Vec<&dyn Tensor> {
        vec![self]
    }
    
    fn get_name(&self) -> Option<&str> {
        self.name.as_deref()
    }
    
    fn get_onnx_type(&self) -> &str {
        "Cast"
    }
}

impl SingleOutputNode for Cast {
    fn get_output_shape(&self) -> &Shape {
        self.input.shape()
    }
    
    fn get_output_dtype(&self) -> DType {
        self.to
    }
}

pub struct Exp {
    name: Option<String>,
    input: Arc<dyn Tensor>
}

impl Exp {
    pub fn new(name: Option<String>, input: Arc<dyn Tensor>) -> Arc<Exp> {
        Arc::new(Exp {
            name,
            input 
        })
    }
}

impl Node for Exp {
    fn get_input_tensors(&self) -> Vec<&dyn Tensor> {
        vec![self.input.as_ref()]
    }
    
    fn get_output_tensors(&self) -> Vec<&dyn Tensor> {
        vec![self]
    }
    
    fn get_name(&self) -> Option<&str> {
        self.name.as_deref()
    }
    
    fn get_onnx_type(&self) -> &str {
        "Exp"
    }
}

impl SingleOutputNode for Exp {
    fn get_output_shape(&self) -> &Shape {
        self.input.shape()
    }

    fn get_output_dtype(&self) -> DType {
        self.input.dtype()
    }
}

pub struct LpNormalization {
    name: Option<String>,
    input: Arc<dyn Tensor>,
    p: i64,
    axis: i64
}

impl LpNormalization {
    pub fn new(name: Option<String>, input: Arc<dyn Tensor>, p: i64, axis: i64) -> Arc<LpNormalization> {
        Arc::new(LpNormalization {
            name,
            input,
            p,
            axis
        })
    }
}

impl Node for LpNormalization {
    fn get_input_tensors(&self) -> Vec<&dyn Tensor> {
        vec![self.input.as_ref()]
    }
    
    fn get_output_tensors(&self) -> Vec<&dyn Tensor> {
        vec![self]
    }
    
    fn get_name(&self) -> Option<&str> {
        self.name.as_deref()
    }
    
    fn get_onnx_type(&self) -> &str {
        "LpNormalization"
    }
}
impl SingleOutputNode for LpNormalization {
    fn get_output_shape(&self) -> &Shape {
        self.input.shape()
    }
    fn get_output_dtype(&self) -> DType {
        self.input.dtype()
    }
}

/*
pub struct Reshape {
    name: Option<String>,
    input: Arc<dyn Tensor>,
    shape: Arc<dyn Tensor>,
    output_shape: Shape
}

impl Reshape {
    pub fn new(name: Option<String>, input: Arc<dyn Tensor>, shape: Arc<dyn Tensor>) -> Result<Arc<Reshape>, Error> {
        if shape.dtype() != DType::I64 {
            Err(Error::InvalidDTypeError)?
        }
        let shape_data = shape.resolve_data().ok_or(Error::CannotResolveDataError)?;
        let old_shape_dims = input.shape().clone().dims;
        let mut new_shape_dims = vec![];
        let mut backfill_dim: Option<usize> = None;
        loop {
            if new_shape_dims.len() < old_shape_dims.len() {
                
            }
        }
        
        Arc::new(Reshape {
            name,
            input,
            shape
        }
    }
}*/