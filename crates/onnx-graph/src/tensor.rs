use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use crate::{onnx};
use crate::node::{Node, SingleOutputNode};
use crate::onnx::ValueInfoProto;

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum DType {
    F32,
    F16,
    BF16,
    U16
}

impl From<DType> for onnx::tensor_proto::DataType {
    fn from(value: DType) -> Self {
        match value {
            DType::F32 => onnx::tensor_proto::DataType::Float,
            DType::F16 => onnx::tensor_proto::DataType::Float16,
            DType::BF16 => onnx::tensor_proto::DataType::Bfloat16,
            DType::U16 => onnx::tensor_proto::DataType::Uint16
        }
    }
}

pub fn build_value_info_type(data_type: onnx::tensor_proto::DataType, shape: &[usize]) -> onnx::TypeProto {
    onnx::TypeProto{
        value: Some(
            onnx::type_proto::Value::TensorType(onnx::type_proto::Tensor {
                elem_type: data_type as i32,
                shape: Some(onnx::TensorShapeProto {
                    dim: shape.iter().map(|x| {
                        onnx::tensor_shape_proto::Dimension{
                            value: Some(onnx::tensor_shape_proto::dimension::Value::DimValue(*x as i64)),
                            .. Default::default()
                        }
                    }).collect()
                })
            })
        ),
        .. Default::default()
    }
}

pub trait Tensor  {
    fn dtype(&self) -> DType;
    fn shape(&self) -> Vec<usize>;
    fn to_value_info_proto(&self, name: String) -> ValueInfoProto {
        ValueInfoProto{
            name,
            r#type: Some(build_value_info_type(self.dtype().into(), &self.shape())),
            .. Default::default()
        }
    }
    
    fn get_nodes(&self) -> HashSet<&dyn Node> {
        HashSet::new()
    }

    fn get_tensors(&self) -> HashSet<&dyn Tensor> where Self: Sized {
        let mut tensors = self.get_sub_tensors();
        tensors.insert(self as &dyn Tensor);
        tensors
    }

    fn get_sub_tensors(&self) -> HashSet<&dyn Tensor> {
        HashSet::new()
    }
    
    fn get_initializer(&self, name: String) -> Option<onnx::TensorProto> {
        None
    }
}

impl<'a> PartialEq for &'a dyn Tensor{
    fn eq(&self, other:&Self) -> bool{
        std::ptr::addr_eq(*self, *other)
    }
}

impl<'a> Eq for &'a dyn Tensor{}

impl<'a> Hash for &'a dyn Tensor {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let a: *const _ = *self;
        let address: *const u8 = a.cast();
        state.write_usize(address.addr());
    }
}


impl Tensor for InputTensor {
    fn dtype(&self) -> DType {
        self.data_type
    }
    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }
}

impl <T: SingleOutputNode> Tensor for T {
    fn dtype(&self) -> DType {
        self.get_output_dtype()
    }

    fn shape(&self)  -> Vec<usize> {
        self.get_output_shape()
    }

    fn get_nodes(&self) -> HashSet<&dyn Node> {
        <Self as Node>::get_nodes(self)
    }
    
    fn get_sub_tensors(&self) -> HashSet<&dyn Tensor> {
        <Self as Node>::get_tensors(self)
    }
}

pub struct InputTensor {
    data_type: DType,
    shape: Vec<usize>
}

impl InputTensor {
    pub fn new(data_type: DType, shape: Vec<usize>) -> Arc<Self> {
        Arc::new(Self { data_type, shape })
    }
}