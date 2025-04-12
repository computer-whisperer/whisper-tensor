use std::collections::HashSet;
use std::sync::Arc;
use crate::{onnx, Error};
use crate::node::{MultiOutputNode, MultiOutputNodeOutput, Node, SingleOutputNode};
use crate::onnx::{AttributeProto};
use crate::tensor::{DType, Dimension, StubTensor, Shape, Tensor, TensorData};

fn validate_index_dtype(dtype: DType) -> Result<(), Error> {
    if dtype != DType::I32 && dtype != DType::I64 {
        Err(Error::InvalidDTypeError)?;
    }
    Ok(())
}

fn try_multidirectional_broadcasting(a: &Shape, b: &Shape) -> Result<Shape, Error> {
    let mut a = a.clone();
    let mut b = b.clone();

    // Prepend dimensions to match
    while a.rank() < b.rank() {
        a = a.unsqueeze(0);
    }
    while b.rank() < a.rank() {
        b = b.unsqueeze(0);
    }
    let mut output_dims = vec![];
    for i in 0..a.rank() {
        output_dims.push(
            if a[i].as_ref() == b[i].as_ref() {
                a[i].clone()
            }
            else {
                if let Some(a_val) = a[i].resolve().ok() {
                    if a_val == 1 {
                        b[i].clone()
                    }
                    else {
                        if let Some(b_val) = b[i].resolve().ok() {
                            if b_val == 1 {
                                a[i].clone()
                            }
                            else {
                                Err(Error::ShapeMismatchError(a.clone(), b.clone()))?
                            }
                        }
                        else {
                            Err(Error::ShapeMismatchError(a.clone(), b.clone()))?
                        }
                    }
                }
                else {
                    if let Some(b_val) = b[i].resolve().ok() {
                        if b_val == 1 {
                            a[i].clone()
                        }
                        else {
                            Err(Error::ShapeMismatchError(a.clone(), b.clone()))?
                        }
                    }
                    else {
                        Err(Error::ShapeMismatchError(a.clone(), b.clone()))?
                    }
                }
            }
        );
    }

    Ok(Shape::new(output_dims))
}

fn make_int_attribute(name: &str, value: i64) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: onnx::attribute_proto::AttributeType::Int.into(),
        i: value,
        ..Default::default()
    }
}

fn make_int_list_attribute(name: &str, values: &[i64]) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: onnx::attribute_proto::AttributeType::Ints.into(),
        ints: values.to_vec(),
        ..Default::default()
    }
}

fn make_float_attribute(name: &str, value: f32) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: onnx::attribute_proto::AttributeType::Float.into(),
        f: value,
        ..Default::default()
    }
}

#[allow(dead_code)]
fn make_float_list_attribute(name: &str, values: &[f32]) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: onnx::attribute_proto::AttributeType::Floats.into(),
        floats: values.to_vec(),
        ..Default::default()
    }
}

fn make_tensor_attribute(name: &str, tensor: &TensorData) -> Result<AttributeProto, Error> {
    Ok(AttributeProto {
        name: name.to_string(),
        r#type: onnx::attribute_proto::AttributeType::Tensor.into(),
        t: Some(tensor.to_tensor_data_proto(Some(name.to_string()))?),
        ..Default::default()
    })
}

pub struct Add {
    name: Option<String>,
    a: Arc<dyn Tensor>,
    b: Arc<dyn Tensor>,
    output_shape: Shape
}
impl SingleOutputNode for Add {
    fn get_output_shape(&self) -> &Shape {
        &self.output_shape
    }

    fn get_output_dtype(&self) -> DType {
        self.a.dtype()
    }
}

impl Add {
    pub fn new(name: Option<String>, a: Arc<dyn Tensor>, b: Arc<dyn Tensor>) -> Result<Arc<Self>, Error> {
        if a.dtype() != b.dtype() {
            return Err(Error::DTypeMismatchError(a.dtype(), b.dtype()));
        }
        let output_shape = try_multidirectional_broadcasting(a.shape(), b.shape())?;
        Ok(Arc::new(Self {name, a, b, output_shape}))
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
    fn get_onnx_attributes(&self) -> Vec<AttributeProto> {
        vec![]
    }
}

pub struct Sub {
    name: Option<String>,
    a: Arc<dyn Tensor>,
    b: Arc<dyn Tensor>,
    output_shape: Shape
}

impl Sub {
    pub fn new(name: Option<String>, a: Arc<dyn Tensor>, b: Arc<dyn Tensor>) -> Result<Arc<Self>, Error> {
        if a.dtype() != b.dtype() {
            return Err(Error::DTypeMismatchError(a.dtype(), b.dtype()));
        }
        let output_shape = try_multidirectional_broadcasting(a.shape(), b.shape())?;
        Ok(Arc::new(Self {name, a, b, output_shape}))
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

    fn get_onnx_attributes(&self) -> Vec<AttributeProto> {
        vec![]
    }
}

impl SingleOutputNode for Sub {
    fn get_output_shape(&self) -> &Shape {
        &self.output_shape
    }

    fn get_output_dtype(&self) -> DType {
        self.a.dtype()
    }
}

pub struct Mul {
    name: Option<String>,
    a: Arc<dyn Tensor>,
    b: Arc<dyn Tensor>,
    output_shape: Shape
}

impl Mul {
    pub fn new(name: Option<String>, a: Arc<dyn Tensor>, b: Arc<dyn Tensor>) -> Result<Arc<Self>, Error> {
        if a.dtype() != b.dtype() {
            return Err(Error::DTypeMismatchError(a.dtype(), b.dtype()));
        }
        let output_shape = try_multidirectional_broadcasting(a.shape(), b.shape())?;
        Ok(Arc::new(Self {name, a, b, output_shape}))
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
    fn get_onnx_attributes(&self) -> Vec<AttributeProto> {
        vec![]
    }
}

impl SingleOutputNode for Mul {
    fn get_output_shape(&self) -> &Shape {
        &self.output_shape
    }

    fn get_output_dtype(&self) -> DType {
        self.a.dtype()
    }
}

pub struct Div {
    name: Option<String>,
    a: Arc<dyn Tensor>,
    b: Arc<dyn Tensor>,
    output_shape: Shape
}

impl Div {
    pub fn new(name: Option<String>, a: Arc<dyn Tensor>, b: Arc<dyn Tensor>) -> Result<Arc<Self>, Error> {
        if a.dtype() != b.dtype() {
            return Err(Error::DTypeMismatchError(a.dtype(), b.dtype()));
        }
        let output_shape = try_multidirectional_broadcasting(a.shape(), b.shape())?;
        Ok(Arc::new(Self {name, a, b, output_shape}))
    }
}

impl Node for Div {
    fn get_input_tensors(&self) -> Vec<&dyn Tensor> {
        vec![self.a.as_ref(), self.b.as_ref()]
    }
    fn get_output_tensors(&self) -> Vec<&dyn Tensor> { vec![self] }
    fn get_name(&self) -> Option<&str> {
        self.name.as_deref()
    }
    fn get_onnx_type(&self) -> &str {
        "Div"
    }
    fn get_onnx_attributes(&self) -> Vec<AttributeProto> {
        vec![]
    }
}

impl SingleOutputNode for Div {
    fn get_output_shape(&self) -> &Shape {
        &self.output_shape
    }

    fn get_output_dtype(&self) -> DType {
        self.a.dtype()
    }
}

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
                    output_shape.extend_from_slice(&indices_shape.dims[0..indices_shape.rank()-1])
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

    fn get_onnx_attributes(&self) -> Vec<AttributeProto> {
        vec![
            make_int_attribute("axis", self.axis as i64)
        ]
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
    axis: i64,
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
    pub fn new(name: Option<String>, input: Arc<dyn Tensor>, scale: Arc<dyn Tensor>, bias: Option<Arc<dyn Tensor>>, axis: i64, epsilon: f32, stash_type: i32) -> Result<Arc<Self>, Error> {
        if input.dtype() != scale.dtype() {
            return Err(Error::DTypeMismatchError(input.dtype(), scale.dtype()));
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
    fn get_onnx_attributes(&self) -> Vec<AttributeProto> {
        vec![
            make_int_attribute("axis", self.axis),
            make_float_attribute("epsilon", self.epsilon),
            make_int_attribute("stash_type", self.stash_type as i64)
        ]
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
            Err(Error::DTypeMismatchError(a.dtype(), b.dtype()))?
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
            Err(Error::ShapeMismatchError(a_shape.clone(), b_shape.clone()))?
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
                                else if a_value == b_value {
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

    fn get_onnx_attributes(&self) -> Vec<AttributeProto> {
        vec![]
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
            return Err(Error::DTypeMismatchError(a.dtype(), b.dtype()))
        }
        if let Some(c) = &c {
            if c.dtype() != a.dtype() {
                return Err(Error::DTypeMismatchError(c.dtype(), a.dtype()))
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

    fn get_onnx_attributes(&self) -> Vec<AttributeProto> {
        vec![
            make_float_attribute("alpha", self.alpha),
            make_float_attribute("beta", self.beta),
            make_int_attribute("transA", self.trans_a as i64),
            make_int_attribute("transB", self.trans_b as i64)
        ]
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
                Err(Error::DTypeMismatchError(inputs[0].dtype(), input.dtype()))?;
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
                let mut v = Some(0);
                for input in &inputs {
                    let t = input.shape()[i].resolve();
                    v = match t {
                        Ok(t) => match v {
                            None => None,
                            Some(v) => Some(v+t)
                        },
                        Err(_) => None
                    };
                }
                output_dims.push(Dimension::new(v, None, None));
            }
            else {
                output_dims.push(inputs[0].shape()[i].clone());
                for input in &inputs {
                    if input.shape()[i].as_ref() != output_dims[i].as_ref() {
                        Err(Error::ShapeMismatchError(inputs[0].shape().clone(), input.shape().clone()))?;
                    }
                }
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
    
    pub fn new_with_output_shape(name: Option<String>, inputs: Vec<Arc<dyn Tensor>>, axis: i64, output_shape: Shape) -> Result<Arc<Self>, Error> {
        if inputs.is_empty() {
            Err(Error::InvalidInputError)?;
        }
        for input in &inputs {
            if input.dtype() != inputs[0].dtype() {
                Err(Error::DTypeMismatchError(inputs[0].dtype(), input.dtype()))?;
            }
        }
        let u_axis = if axis < 0 { (inputs[0].rank() as i64 + axis) as usize } else { axis as usize };
        if u_axis >= inputs[0].rank() {
            Err(Error::InvalidInputError)?;
        }
        
        let output_dtype = inputs[0].dtype();
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
        vec![self]
    }

    fn get_name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn get_onnx_type(&self) -> &str {
        "Concat"
    }

    fn get_onnx_attributes(&self) -> Vec<AttributeProto> {
        vec![
            make_int_attribute("axis", self.axis)
        ]
    }
}

impl SingleOutputNode for Concat {
    fn get_output_shape(&self) -> &Shape {
        &self.output_shape
    }

    fn get_output_dtype(&self) -> DType {
        self.output_dtype
    }
}

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

    fn get_onnx_attributes(&self) -> Vec<AttributeProto> {
        vec![]
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

    fn get_onnx_attributes(&self) -> Vec<AttributeProto> {
        vec![]
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

    fn get_onnx_attributes(&self) -> Vec<AttributeProto> {
        vec![]
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

    fn get_onnx_attributes(&self) -> Vec<AttributeProto> {
        vec![]
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

    fn get_onnx_attributes(&self) -> Vec<AttributeProto> {
        vec![
            make_tensor_attribute("value", &self.data).unwrap()
        ]
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

    fn get_onnx_attributes(&self) -> Vec<AttributeProto> {
        vec![
            make_int_attribute("to", onnx::tensor_proto::DataType::from(self.to) as i64)
        ]
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

    fn get_onnx_attributes(&self) -> Vec<AttributeProto> {
        vec![]
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

    fn get_onnx_attributes(&self) -> Vec<AttributeProto> {
        vec![
            make_int_attribute("p", self.p),
            make_int_attribute("axis", self.axis)
        ]
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


pub struct Reshape {
    name: Option<String>,
    data_input: Arc<dyn Tensor>,
    shape_input: Arc<dyn Tensor>,
    output_shape: Shape
}

impl Reshape {
    pub fn new(name: Option<String>, data_input: Arc<dyn Tensor>, shape_input: Arc<dyn Tensor>) -> Result<Arc<Reshape>, Error> {
        if shape_input.dtype() != DType::I64 {
            Err(Error::InvalidDTypeError)?
        }
        let shape_data = shape_input.resolve_data().ok_or(Error::CannotResolveDataError)?.to_int_vec()?;
        let old_shape_dims = data_input.shape().clone().dims;
        let mut new_shape_dims = vec![];
        let mut backfill_dim: Option<usize> = None;
        for i in 0..shape_data.len() {
            new_shape_dims.push(if shape_data[i] == 0 {
                old_shape_dims[i].clone()
            } else if shape_data[i] == -1 {
                if backfill_dim.is_some() {
                    // Only one dimension can be inferred
                    Err(Error::InvalidInputError)?
                }
                backfill_dim = Some(i);
                Dimension::new(Some(1), None, None)
            }
            else if shape_data[i] < -1 {
                Err(Error::InvalidInputError)?
            } else {
                Dimension::new(Some(shape_data[i] as usize), None, None)
            });
        }
        // Backfill the inferred dimension
        if let Some(i) = backfill_dim {
            let total_input_size = data_input.shape().num_elements()?;

            // Calculate the current product of the dimensions
            let mut current_product = 1;
            for (j, dim) in new_shape_dims.iter().enumerate() {
                if j != i {
                    current_product *= dim.resolve()?;
                }
            }
            // Calculate the inferred dimension size
            let inferred_size = total_input_size / current_product;
            new_shape_dims[i] = Dimension::new(Some(inferred_size), None, None);
        }
        let output_shape = Shape::new(new_shape_dims);
        
        // Verify that the dimensions are compatible
        if output_shape.num_elements()? != data_input.shape().num_elements()? {
            Err(Error::InvalidInputError)?
        }
        
        Ok(Arc::new(Reshape {
            name,
            data_input,
            shape_input,
            output_shape
        }))
    }

    pub fn new_with_forced_output(name: Option<String>, data_input: Arc<dyn Tensor>, shape_input: Arc<dyn Tensor>, output_shape: Shape) -> Result<Arc<Self>, Error>{
        if shape_input.dtype() != DType::I64 {
            return Err(Error::InvalidDTypeError);
        }
        if shape_input.rank() != 1 {
            return Err(Error::InvalidInputError);
        }

        Ok(Arc::new(Self {
            name,
            data_input,
            shape_input,
            output_shape
        }))
    }
}

impl Node for Reshape {
    fn get_input_tensors(&self) -> Vec<&dyn Tensor> {
        vec![self.data_input.as_ref(), self.shape_input.as_ref()]
    }

    fn get_output_tensors(&self) -> Vec<&dyn Tensor> {
        vec![self]
    }

    fn get_name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn get_onnx_type(&self) -> &str {
        "Reshape"
    }

    fn get_onnx_attributes(&self) -> Vec<AttributeProto> {
        vec![

        ]
    }
}

impl SingleOutputNode for Reshape {
    fn get_output_shape(&self) -> &Shape {
        &self.output_shape
    }

    fn get_output_dtype(&self) -> DType {
        self.data_input.dtype()
    }
}

pub struct Slice {
    name: Option<String>,
    data_input: Arc<dyn Tensor>,
    starts_input: Arc<dyn Tensor>,
    ends_input: Arc<dyn Tensor>,
    axes_input: Option<Arc<dyn Tensor>>,
    steps_input: Option<Arc<dyn Tensor>>,
    output_shape: Shape,
}

impl Slice {
    pub fn new(name: Option<String>, data_input: Arc<dyn Tensor>, starts_input: Arc<dyn Tensor>, ends_input: Arc<dyn Tensor>, axes_input: Option<Arc<dyn Tensor>>, steps_input: Option<Arc<dyn Tensor>>) -> Result<Arc<Self>, Error> {
        validate_index_dtype(starts_input.dtype())?;
        validate_index_dtype(ends_input.dtype())?;
        if let Some(axes_input) = &axes_input {
            validate_index_dtype(axes_input.dtype())?;
        }
        if let Some(steps_input) = &steps_input {
            validate_index_dtype(steps_input.dtype())?;
        }

        let starts_value = starts_input.resolve_data().ok_or(Error::CannotResolveDataError)?.to_int_vec()?;
        let ends_value = ends_input.resolve_data().ok_or(Error::CannotResolveDataError)?.to_int_vec()?;

        let axes_value = if let Some(axes_input) = &axes_input {
            axes_input.resolve_data().ok_or(Error::CannotResolveDataError)?.to_int_vec()?
        } else {
            (0..data_input.rank() as i64).collect()
        };

        let steps_value = if let Some(steps_input) = &steps_input {
            steps_input.resolve_data().ok_or(Error::CannotResolveDataError)?.to_int_vec()?
        } else {
            vec![1; axes_value.len()]
        };

        if starts_value.len() != axes_value.len() || ends_value.len() != axes_value.len() || steps_value.len() != axes_value.len() {
            Err(Error::InputShapeError)?;
        }

        let mut output_dimensions = vec![];
        for i in 0..data_input.rank() {
            let mut found = false;
            for j in 0..axes_value.len() {
                if i == axes_value[j] as usize {
                    output_dimensions.push(Dimension::new(Some(((ends_value[j] - starts_value[j]) / steps_value[j]) as usize), None, None));
                    found = true;
                    break;
                }
            }
            if !found {
                output_dimensions.push(data_input.shape()[i].clone());
            }
        }

        let output_shape = Shape::new(output_dimensions);

        Ok(Arc::new(Self {
            name,
            data_input,
            starts_input,
            ends_input,
            axes_input,
            steps_input,
            output_shape
        }))
    }
}

impl SingleOutputNode for Slice {
    fn get_output_shape(&self) -> &Shape {
        &self.output_shape
    }

    fn get_output_dtype(&self) -> DType {
        self.data_input.dtype()
    }
}

impl Node for Slice {
    fn get_input_tensors(&self) -> Vec<&dyn Tensor> {
        let mut inputs = vec![self.data_input.as_ref(), self.starts_input.as_ref(), self.ends_input.as_ref()];
        if let Some(axes_input) = &self.axes_input {
            inputs.push(axes_input.as_ref());
        }
        if let Some(steps_input) = &self.steps_input {
            inputs.push(steps_input.as_ref());
        }
        inputs

    }

    fn get_output_tensors(&self) -> Vec<&dyn Tensor> {
        vec![self]
    }

    fn get_name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn get_onnx_type(&self) -> &str {
        "Slice"
    }

    fn get_onnx_attributes(&self) -> Vec<AttributeProto> {
        vec![]
    }
}

pub struct Squeeze {
    name: Option<String>,
    data_input: Arc<dyn Tensor>,
    axes_input: Arc<dyn Tensor>,
    output_shape: Shape,
}

impl Squeeze {
    pub fn new(name: Option<String>, data_input: Arc<dyn Tensor>, axes_input: Arc<dyn Tensor>) -> Result<Arc<Self>, Error> {

        if axes_input.rank() != 1 {
            Err(Error::InputShapeError)?;
        }
        if axes_input.dtype() != DType::I64 {
            Err(Error::InvalidDTypeError)?;
        }

        let axes = axes_input.resolve_data().ok_or(Error::CannotResolveDataError)?.to_int_vec()?;
        // Unsign the axes
        let axes = axes.into_iter().map(|x| {if x < 0 {((data_input.rank() as i64) + x) as usize} else {x as usize}}).collect::<Vec<_>>();
        for axis in &axes {
            if *axis > data_input.rank() {
                Err(Error::InvalidInputError)?;
            }
        }

        let mut new_dims = vec![];
        for i in 0..data_input.rank() {
            if !axes.contains(&i) {
                new_dims.push(data_input.shape()[i].clone());
            }
            else {
                if data_input.shape()[i].resolve()? != 1 {
                    Err(Error::InvalidInputError)?;
                }
            }
        }

        let output_shape = Shape::new(new_dims);

        Ok(Arc::new(
            Self {
                name,
                data_input,
                axes_input,
                output_shape
            }
        ))
    }
}

impl Node for Squeeze {
    fn get_input_tensors(&self) -> Vec<&dyn Tensor> {
        vec![self.data_input.as_ref(), self.axes_input.as_ref()]
    }

    fn get_output_tensors(&self) -> Vec<&dyn Tensor> {
        vec![self]
    }

    fn get_name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn get_onnx_type(&self) -> &str {
        "Squeeze"
    }

    fn get_onnx_attributes(&self) -> Vec<AttributeProto> {
        vec![]
    }
}

impl SingleOutputNode for Squeeze{
    fn get_output_shape(&self) -> &Shape {
        &self.output_shape
    }

    fn get_output_dtype(&self) -> DType {
        self.data_input.dtype()
    }
}

pub struct Unsqueeze {
    name: Option<String>,
    data_input: Arc<dyn Tensor>,
    axes_input: Arc<dyn Tensor>,
    output_shape: Shape
}

impl Unsqueeze {
    pub fn new(name: Option<String>, data_input: Arc<dyn Tensor>, axes_input: Arc<dyn Tensor>) -> Result<Arc<Self>, Error> {
        if axes_input.rank() != 1 {
            Err(Error::InputShapeError)?;
        }
        if axes_input.dtype() != DType::I64 {
            Err(Error::InvalidDTypeError)?;
        }

        let axes = axes_input.resolve_data().ok_or(Error::CannotResolveDataError)?.to_int_vec()?;
        // Unsign the axes
        let axes = axes.into_iter().map(|x| {if x < 0 {((data_input.rank() as i64) + x) as usize} else {x as usize}}).collect::<Vec<_>>();
        for axis in &axes {
            if *axis > data_input.rank() {
                Err(Error::InvalidInputError)?;
            }
        }

        let mut new_dims = vec![];
        let mut old_dims_i = 0;
        for i in 0..(data_input.rank() + axes.len()) {
            if axes.contains(&i) {
                new_dims.push(Dimension::new(Some(1), None, None));
            }
            else {
                if old_dims_i > data_input.rank() {
                    Err(Error::InvalidInputError)?;
                }
                new_dims.push(data_input.shape()[old_dims_i].clone());
                old_dims_i += 1;
            }
        }

        let output_shape = Shape::new(new_dims);

        Ok(Arc::new(Self {
            name,
            data_input,
            axes_input,
            output_shape
        }))
    }
}

impl Node for Unsqueeze {
    fn get_input_tensors(&self) -> Vec<&dyn Tensor> {
        vec![self.data_input.as_ref(), self.axes_input.as_ref()]
    }

    fn get_output_tensors(&self) -> Vec<&dyn Tensor> {
        vec![self]
    }

    fn get_name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn get_onnx_type(&self) -> &str {
        "Unsqueeze"
    }

    fn get_onnx_attributes(&self) -> Vec<AttributeProto> {
        vec![]
    }
}

impl SingleOutputNode for Unsqueeze{
    fn get_output_shape(&self) -> &Shape {
        &self.output_shape
    }

    fn get_output_dtype(&self) -> DType {
        self.data_input.dtype()
    }
}

pub struct Transpose {
    name: Option<String>,
    data_input: Arc<dyn Tensor>,
    perm: Option<Vec<i64>>,
    output_shape: Shape
}

impl Transpose {
    pub fn new(name: Option<String>, data_input: Arc<dyn Tensor>, perm_in: Option<Vec<i64>>) -> Arc<Self> {
        let perm = if let Some(perm) = &perm_in {
            perm.clone()
        } else {
            (0..data_input.shape().dims.len() as i64).rev().collect()
        };

        let input_dims = data_input.shape().dims.clone();
        let output_dims = perm.iter().map(|&i| input_dims[i as usize].clone()).collect();
        let output_shape = Shape::new(output_dims);
        Arc::new(Self {
            name,
            data_input,
            perm: perm_in,
            output_shape
        })
    }
}

impl Node for Transpose {
    fn get_input_tensors(&self) -> Vec<&dyn Tensor> {
        vec![self.data_input.as_ref()]
    }
    fn get_output_tensors(&self) -> Vec<&dyn Tensor> {
        vec![self]
    }
    fn get_name(&self) -> Option<&str> {
        self.name.as_deref()
    }
    fn get_onnx_type(&self) -> &str {
        "Transpose"
    }
    fn get_onnx_attributes(&self) -> Vec<AttributeProto> {
        if let Some(perm) = &self.perm {
            vec![
                make_int_list_attribute("perm", perm)
            ]
        }
        else {
            vec![]
        }
    }
}

impl SingleOutputNode for Transpose {
    fn get_output_shape(&self) -> &Shape {
        &self.output_shape
    }
    fn get_output_dtype(&self) -> DType {
        self.data_input.dtype()
    }
}

pub struct GroupNormalization {
    name: Option<String>,
    data_input: Arc<dyn Tensor>,
    scale: Arc<dyn Tensor>,
    bias: Arc<dyn Tensor>,
    num_groups: i64,
    epsilon: f32
}

impl GroupNormalization {
    pub fn new(name: Option<String>, data_input: Arc<dyn Tensor>, scale: Arc<dyn Tensor>, bias: Arc<dyn Tensor>, num_groups: i64, epsilon: f32) -> Result<Arc<Self>, Error> {
        Ok(Arc::new(Self {
            name,
            data_input,
            scale,
            bias,
            num_groups,
            epsilon
        }))
    }
}

impl Node for GroupNormalization {
    fn get_input_tensors(&self) -> Vec<&dyn Tensor> {
        vec![self.data_input.as_ref(), self.scale.as_ref(), self.bias.as_ref()]
    }
    fn get_output_tensors(&self) -> Vec<&dyn Tensor> {
        vec![self]
    }
    fn get_name(&self) -> Option<&str> {
        self.name.as_deref()
    }
    fn get_onnx_type(&self) -> &str {
        "GroupNormalization"
    }

    fn get_onnx_attributes(&self) -> Vec<AttributeProto> {
        vec![
            make_float_attribute("epsilon", self.epsilon),
            make_int_attribute("num_groups", self.num_groups)
        ]
    }
}

impl SingleOutputNode for GroupNormalization {
    fn get_output_shape(&self) -> &Shape {
        &self.data_input.shape()
    }
    fn get_output_dtype(&self) -> DType {
        self.data_input.dtype()
    }
}

pub struct CumSum {
    name: Option<String>,
    data_input: Arc<dyn Tensor>,
    axis_input: Arc<dyn Tensor>,
    output_shape: Shape
}

impl CumSum {
    pub fn new(name: Option<String>, data_input: Arc<dyn Tensor>, axis_input: Arc<dyn Tensor>) -> Result<Arc<Self>, Error> {
        validate_index_dtype(axis_input.dtype())?;

        if axis_input.rank() > 1 {
            return Err(Error::InputShapeError);
        }

        let axis_data = axis_input.resolve_data().ok_or(Error::CannotResolveDataError)?.to_int_vec()?;
        if axis_data.len() > 1 {
            return Err(Error::InputShapeError);
        }
        let axis = axis_data[0];
        let axis = if axis < 0 {
            (axis + data_input.rank() as i64) as usize
        } else {
            axis as usize
        };
        if axis >= data_input.rank() {
            return Err(Error::InvalidInputError);
        }

        let mut output_dims = data_input.shape().dims.clone();
        output_dims[axis] = Dimension::new(Some(1), None, None);
        let output_shape = Shape::new(output_dims);

        Ok(Arc::new(
            Self {
                name,
                data_input,
                axis_input,
                output_shape
            }
        ))
    }
}

impl Node for CumSum {
    fn get_input_tensors(&self) -> Vec<&dyn Tensor> {
        vec![self.data_input.as_ref(), self.axis_input.as_ref()]
    }

    fn get_output_tensors(&self) -> Vec<&dyn Tensor> {
        vec![self]
    }

    fn get_name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn get_onnx_type(&self) -> &str {
        "CumSum"
    }

    fn get_onnx_attributes(&self) -> Vec<AttributeProto> {
        vec![]
    }
}

impl SingleOutputNode for CumSum {
    fn get_output_shape(&self) -> &Shape {
        &self.output_shape
    }

    fn get_output_dtype(&self) -> DType {
        self.data_input.dtype()
    }
}

pub struct Relu {
    name: Option<String>,
    input: Arc<dyn Tensor>
}

impl Relu {
    pub fn new(name: Option<String>, input: Arc<dyn Tensor>) -> Result<Arc<Self>, Error> {
        Ok(Arc::new(
            Self {
                name,
                input
            }
        ))
    }
}

impl Node for Relu {
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
        "Relu"
    }
    fn get_onnx_attributes(&self) -> Vec<AttributeProto> {
        vec![]
    }
}

impl SingleOutputNode for Relu {
    fn get_output_shape(&self) -> &Shape {
        self.input.shape()
    }
    fn get_output_dtype(&self) -> DType {
        self.input.dtype()
    }
}

pub struct RMSNormalization {
    name: Option<String>,
    input: Arc<dyn Tensor>,
    scale: Arc<dyn Tensor>,
    epsilon: Option<f32>,
    axis: i64
}

impl RMSNormalization {
    pub fn new(name: Option<String>, input: Arc<dyn Tensor>, scale: Arc<dyn Tensor>, epsilon: Option<f32>, axis: i64) -> Result<Arc<Self>, Error> {
        if input.dtype() != scale.dtype() {
            return Err(Error::DTypeMismatchError(input.dtype(), scale.dtype()));
        }
        Ok(Arc::new(
            Self {
                name,
                input,
                scale,
                epsilon,
                axis
            }
        ))
    }
}

impl Node for RMSNormalization {
    fn get_input_tensors(&self) -> Vec<&dyn Tensor> {
        vec![self.input.as_ref(), self.scale.as_ref()]
    }

    fn get_output_tensors(&self) -> Vec<&dyn Tensor> {
        vec![self]
    }

    fn get_name(&self) -> Option<&str> {
        self.name.as_deref()
    }
    fn get_onnx_type(&self) -> &str {
        "RMSNormalization"
    }

    fn get_onnx_attributes(&self) -> Vec<AttributeProto> {
        let mut out = vec![
            make_int_attribute("axis", self.axis)
        ];
        if let Some(epsilon) = self.epsilon {
            out.push(make_float_attribute("epsilon", epsilon))
        };
        out
    }
}

impl SingleOutputNode for RMSNormalization {
    fn get_output_shape(&self) -> &Shape {
        self.input.shape()
    }
    fn get_output_dtype(&self) -> DType {
        self.input.dtype()
    }
}


pub struct ShapeOp {
    name: Option<String>,
    input: Arc<dyn Tensor>,
    start: Option<i64>,
    end: Option<i64>,
    output_shape: Shape
}

impl ShapeOp {
    pub fn new(name: Option<String>, input: Arc<dyn Tensor>, start: Option<i64>, end: Option<i64>) -> Result<Arc<Self>, Error> {

        let start_v = if let Some(start) = start {
            start
        } else {
            0
        };

        let end_v = if let Some(end) = end {
            end
        } else {
            input.rank() as i64
        };

        let output_dims = vec![Dimension::new(Some((end_v - start_v) as usize), None, None)];

        let output_shape = Shape::new(output_dims);

        Ok(Arc::new(Self {
            name,
            input,
            start,
            end,
            output_shape
        }))
    }
}

impl Node for ShapeOp {
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
        "Shape"
    }

    fn get_onnx_attributes(&self) -> Vec<AttributeProto> {
        let mut out = vec![];

        if let Some(start) = self.start{
            out.push(make_int_attribute("start", start))
        }

        if let Some(end) = self.end {
            out.push(make_int_attribute("end", end))
        }

        out
    }
}

impl SingleOutputNode for ShapeOp {
    fn get_output_shape(&self) -> &Shape {
        &self.output_shape
    }

    fn get_output_dtype(&self) -> DType {
        DType::I64
    }
}

pub struct RotaryEmbedding {
    name: Option<String>,
    input_data: Arc<dyn Tensor>,
    cos_cache: Arc<dyn Tensor>,
    sin_cache: Arc<dyn Tensor>,
    position_ids: Option<Arc<dyn Tensor>>,
    interleaved: Option<i64>,
    num_heads: Option<i64>,
    rotary_embedding_dim: Option<i64>
}

impl RotaryEmbedding {
    pub fn new(name: Option<String>,
               input_data: Arc<dyn Tensor>,
               cos_cache: Arc<dyn Tensor>,
               sin_cache: Arc<dyn Tensor>,
               position_ids: Option<Arc<dyn Tensor>>,
               interleaved: Option<i64>,
               num_heads: Option<i64>,
               rotary_embedding_dim: Option<i64>
    ) -> Result<Arc<RotaryEmbedding>, Error> {
        Ok(Arc::new(Self{
            name,
            input_data,
            cos_cache,
            sin_cache,
            position_ids,
            interleaved,
            rotary_embedding_dim,
            num_heads
        }))
    }
}

impl Node for RotaryEmbedding {
    fn get_input_tensors(&self) -> Vec<&dyn Tensor> {
        let mut out = vec![self.input_data.as_ref(), self.cos_cache.as_ref(), self.sin_cache.as_ref()];
        if let Some(position_ids) = &self.position_ids {
            out.push(position_ids.as_ref())
        }
        out
    }

    fn get_output_tensors(&self) -> Vec<&dyn Tensor> {
        vec![self]
    }

    fn get_name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn get_onnx_type(&self) -> &str {
        "RotaryEmbedding"
    }

    fn get_onnx_attributes(&self) -> Vec<AttributeProto> {
        let mut out = vec![];

        if let Some(interleaved) = self.interleaved {
            out.push(make_int_attribute("interleaved", interleaved))
        }

        if let Some(num_heads) = self.num_heads {
            out.push(make_int_attribute("num_heads", num_heads))
        }

        if let Some(rotary_embedding_dim) = self.rotary_embedding_dim {
            out.push(make_int_attribute("rotary_embedding_dim", rotary_embedding_dim))
        }

        out
    }
}

impl SingleOutputNode for RotaryEmbedding {
    fn get_output_shape(&self) -> &Shape {
        self.input_data.shape()
    }

    fn get_output_dtype(&self) -> DType {
        self.input_data.dtype()
    }
}


pub struct Softmax {
    name: Option<String>,
    input: Arc<dyn Tensor>,
    axis: Option<i64>
}

impl Softmax {
    pub fn new(name: Option<String>, input: Arc<dyn Tensor>, axis: Option<i64>) -> Arc<Softmax> {
        Arc::new(Self {
            name,
            input,
            axis
        })
    }
}

impl Node for Softmax {
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
        "Softmax"
    }

    fn get_onnx_attributes(&self) -> Vec<AttributeProto> {
        if let Some(axis) = self.axis {
            vec![
                make_int_attribute("axis", axis)
            ]
        }
        else {
            vec![]
        }
    }
}

impl SingleOutputNode for Softmax {
    fn get_output_shape(&self) -> &Shape {
        self.input.shape()
    }

    fn get_output_dtype(&self) -> DType {
        self.input.dtype()
    }
}

pub struct Expand {
    name: Option<String>,
    input: Arc<dyn Tensor>,
    shape: Arc<dyn Tensor>,
    output_shape: Shape
}

impl Expand {
    pub fn new(name: Option<String>, input: Arc<dyn Tensor>, shape: Arc<dyn Tensor>) -> Result<Arc<Self>, Error> {
        if shape.dtype() != DType::I64 {
            return Err(Error::InvalidDTypeError);
        }
        if input.rank() != 1 {
            return Err(Error::InvalidInputError);
        }
        
        let shape_dims = shape.resolve_data().ok_or(Error::CannotResolveDataError)?.to_int_vec()?;
        let mut output_dims = vec![];
        for i in 0..shape_dims.len() {
            output_dims.push(Dimension::new(Some(shape_dims[i] as usize), None, None));
        }
        
        let output_shape = Shape::new(output_dims);
        
        Ok(Arc::new(Self {
            name,
            input,
            shape,
            output_shape
        }))
    }
    
    pub fn new_with_forced_output(name: Option<String>, input: Arc<dyn Tensor>, shape: Arc<dyn Tensor>, output_shape: Shape) -> Result<Arc<Self>, Error>{
        if shape.dtype() != DType::I64 {
            return Err(Error::InvalidDTypeError);
        }
        if shape.rank() != 1 {
            return Err(Error::InvalidInputError);
        }
        
        Ok(Arc::new(Self {
            name,
            input,
            shape,
            output_shape
        }))
    }
}

impl Node for Expand {
    fn get_input_tensors(&self) -> Vec<&dyn Tensor> {
        vec![self.input.as_ref(), self.shape.as_ref()]
    }

    fn get_output_tensors(&self) -> Vec<&dyn Tensor> {
        vec![self]
    }

    fn get_name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn get_onnx_type(&self) -> &str {
        "Expand"
    }

    fn get_onnx_attributes(&self) -> Vec<AttributeProto> {
        vec![]
    }
}

impl SingleOutputNode for Expand {
    fn get_output_shape(&self) -> &Shape {
        &self.output_shape
    }

    fn get_output_dtype(&self) -> DType {
        self.input.dtype()
    }
}

pub struct TopK {
    name: Option<String>,
    input: Arc<dyn Tensor>,
    k: Arc<dyn Tensor>,
    axis: i64,
    largest: bool,
    sorted: bool,
    output_shape: Shape,
    value_output: Arc<StubTensor>
}

impl TopK {
    pub fn new(
        name: Option<String>,
        input: Arc<dyn Tensor>,
        k: Arc<dyn Tensor>,
        axis: i64,
        largest: bool,
        sorted: bool,
    ) -> Result<(Arc<StubTensor>, Arc<Self>), Error> {
        if k.rank() != 1 {
            Err(Error::InputShapeError)?;
        }
        if k.dtype() != DType::I64 {
            Err(Error::InvalidDTypeError)?;
        }
        
        let k_val = k.resolve_data().ok_or(Error::CannotResolveDataError)?.to_int_vec()?[0];
        if k_val < 0 {
            Err(Error::InvalidInputError)?;
        }
        
        let mut output_dims = input.shape().dims.clone();
        let axis_u = if axis < 0 { (axis + input.rank() as i64) as usize } else { axis as usize };
        output_dims[axis_u] = Dimension::new(Some(k_val as usize), None, None);
        let output_shape = Shape::new(output_dims);
        
        let value_output = StubTensor::new(output_shape.clone(), input.dtype());
        
        Ok((value_output.clone(), Arc::new(Self {
            name,
            input,
            k,
            axis,
            largest,
            sorted,
            output_shape,
            value_output: value_output.clone()
        })))
    }
}

impl Node for TopK {
    fn get_input_tensors(&self) -> Vec<&dyn Tensor> {
        vec![self.input.as_ref()]
    }

    fn get_output_tensors(&self) -> Vec<&dyn Tensor> {
        vec![self.value_output.as_ref(), self]
    }

    fn get_name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn get_onnx_type(&self) -> &str {
        "TopK"
    }

    fn get_tensors<'a>(&'a self, table: &mut HashSet<&'a dyn Tensor>) {
        for input in self.get_input_tensors() {
            if !table.contains(&input) {
                table.insert(input);
                input.get_sub_tensors(table);
            }
        }
        table.insert(self.value_output.as_ref());
    }

    fn get_onnx_attributes(&self) -> Vec<AttributeProto> {
        vec![
            make_int_attribute("axis", self.axis),
            make_int_attribute("largest", self.largest as i64),
            make_int_attribute("sorted", self.sorted as i64)
        ]
    }
}

impl SingleOutputNode for TopK {
    fn get_output_shape(&self) -> &Shape {
        &self.output_shape
    }

    fn get_output_dtype(&self) -> DType {
        DType::I64
    }
}