use crate::dtype::DType;
use crate::numeric_tensor::NumericTensor;
use crate::tensor_rank::DynRank;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Hash, Eq, PartialEq, Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TensorStoreTensorId(u64);

pub enum StoredTensor {
    Numeric(NumericTensor<DynRank>),
}

impl StoredTensor {
    pub fn to_numeric(&self) -> NumericTensor<DynRank> {
        match self {
            StoredTensor::Numeric(tensor) => tensor.clone(),
        }
    }

    pub fn shape(&self) -> Vec<u64> {
        match self {
            StoredTensor::Numeric(tensor) => tensor.shape(),
        }
    }

    pub fn dtype(&self) -> DType {
        match self {
            StoredTensor::Numeric(tensor) => tensor.dtype(),
        }
    }
}

pub struct TensorStore {
    next_tensor_id: TensorStoreTensorId,
    tensors: HashMap<TensorStoreTensorId, StoredTensor>,
}

impl TensorStore {
    pub fn new() -> TensorStore {
        TensorStore {
            next_tensor_id: TensorStoreTensorId(0),
            tensors: HashMap::new(),
        }
    }

    pub fn get_tensor(&self, id: TensorStoreTensorId) -> Option<&StoredTensor> {
        self.tensors.get(&id)
    }

    pub fn add_tensor(&mut self, tensor: StoredTensor) -> TensorStoreTensorId {
        let tensor_id = self.next_tensor_id;
        self.tensors.insert(tensor_id, tensor);
        self.next_tensor_id.0 += 1;
        tensor_id
    }
}
