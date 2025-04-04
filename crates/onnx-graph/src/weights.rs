use std::sync::Arc;
use crate::{onnx, Error};
use crate::tensor::{DType, Tensor};

pub struct PthTensor {
    tensor_info: candle_core::pickle::TensorInfo,
    tensors: Arc::<candle_core::pickle::PthTensors>,
    data_type: DType,
}

impl PthTensor {
    pub fn new(tensor_info: candle_core::pickle::TensorInfo, tensors: Arc::<candle_core::pickle::PthTensors>) -> Result<Arc<Self>, Error> {
        let data_type = match tensor_info.dtype {
            candle_core::DType::F32 => DType::F32,
            candle_core::DType::F16 => DType::F16,
            candle_core::DType::BF16 => DType::BF16,
            _ => return Err(Error::UnsupportedDtypeError)
        };
        Ok(Arc::new(Self {
            tensors,
            tensor_info,
            data_type
        }))
    }
}

impl Tensor for PthTensor {
    fn dtype(&self) -> DType {
        self.data_type
    }

    fn shape(&self) -> Vec<usize> {
        self.tensor_info.layout.shape().dims().to_vec()
    }
    
    fn get_initializer(&self, name: String) -> Option<onnx::TensorProto> {
        Some(onnx::TensorProto {
            name,
            data_type: onnx::tensor_proto::DataType::from(self.dtype()) as i32,
            dims: self.shape().iter().map(|x| *x as i64).collect(),
            .. Default::default()
        })
    }
}