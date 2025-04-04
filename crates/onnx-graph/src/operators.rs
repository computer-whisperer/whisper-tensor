use std::sync::Arc;
use crate::{validate_elementwise_inputs, Error};
use crate::node::{Node, NodeElementwise, SingleOutputNode};
use crate::tensor::{DType, Tensor};

pub struct Add {
    a: Arc<dyn Tensor>,
    b: Arc<dyn Tensor>
}
impl NodeElementwise for Add {}

impl Add {
    pub fn new(a: Arc<dyn Tensor>, b: Arc<dyn Tensor>) -> Result<Arc<Self>, Error> {
        validate_elementwise_inputs(&[a.clone(), b.clone()])?;
        Ok(Arc::new(Self { a, b }))
    }
}

impl Node for Add {
    fn get_input_tensors(&self) -> Vec<&dyn Tensor> {
        vec![self.a.as_ref(), self.b.as_ref()]
    }
    fn get_output_tensors(&self) -> Vec<&dyn Tensor> { vec![self] }
    fn get_onnx_type(&self) -> &str {
        "Add"
    }
}

pub struct Gather {
    data: Arc<dyn Tensor>,
    indices: Arc<dyn Tensor>,
    axis: isize,
    output_shape: Vec<usize>,
    dtype: DType
}

impl Gather {
    pub fn new(data: Arc<dyn Tensor>, indices: Arc<dyn Tensor>, axis: usize) -> Result<Arc<Self>, Error> {
        let data_shape = data.shape();
        let data_dtype = data.dtype();
        let indices_shape = indices.shape();
        let indices_dtype = indices.dtype();
        
        if data_shape.len() < 2 {
            return Err(Error::InvalidInputError);
        }
        let mut output_shape = vec![];
        let mut data_i = 0usize;
        loop {
            if data_i != axis {
                output_shape.push(data_shape[data_i])
            }
            else {
                if indices_shape.len() > 1 || indices_shape[0] > 1 {
                    output_shape.extend_from_slice(&indices_shape)
                }
            }
            data_i += 1;
            if data_i >= data_shape.len() {
                break;
            }
        }
        
        Ok(Arc::new(Self { 
            data, 
            indices, 
            axis: axis as isize,
            output_shape,
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

    fn get_onnx_type(&self) -> &str {
        "Gather"
    }
}

impl SingleOutputNode for Gather {
    fn get_output_shape(&self) -> Vec<usize> {
        self.output_shape.clone()
    }

    fn get_output_dtype(&self) -> DType {
        self.dtype
    }
}

pub struct LayerNormalization {
    input: Arc<dyn Tensor>,
    scale: Arc<dyn Tensor>,
    bias: Option<Arc<dyn Tensor>>,
    axis: i32,
    epsilon: f32,
    stash_type: i32
}

impl SingleOutputNode for LayerNormalization {
    fn get_output_shape(&self) -> Vec<usize> {
        self.input.shape()
    }

    fn get_output_dtype(&self) -> DType {
        self.input.dtype()
    }
}

impl LayerNormalization {
    pub fn new(input: Arc<dyn Tensor>, scale: Arc<dyn Tensor>, bias: Option<Arc<dyn Tensor>>, axis: i32, epsilon: f32, stash_type: i32) -> Result<Arc<Self>, Error> {
        if input.dtype() != scale.dtype() {
            return Err(Error::DTypeMismatchError);
        }
        Ok(Arc::new(Self {
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

    fn get_onnx_type(&self) -> &str {
        "LayerNormalization"
    }
}