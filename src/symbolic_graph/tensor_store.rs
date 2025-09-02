use crate::dtype::DType;
use crate::numeric_tensor::NumericTensor;
use crate::tensor_rank::DynRank;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Hash, Eq, PartialEq, Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TensorStoreTensorId(u64);

pub enum StoredTensor {
    Numeric(NumericTensor<DynRank>),
    ExternalBinary {
        path: String,
        offset: usize,
        length: usize,
        dtype: DType,
        shape: Vec<u64>,
    },
    ExternalPth {
        path: String,
        tensor_name: String,
        dtype: DType,
        shape: Vec<u64>,
    },
}

impl StoredTensor {
    pub fn to_numeric(&self) -> NumericTensor<DynRank> {
        match self {
            StoredTensor::Numeric(tensor) => tensor.clone(),
            StoredTensor::ExternalBinary {
                path,
                offset,
                length,
                dtype,
                shape,
            } => {
                // Load on demand from external binary file
                let mut file =
                    std::fs::File::open(path).expect("Failed to open external tensor file");
                use std::io::{Read, Seek, SeekFrom};
                file.seek(SeekFrom::Start(*offset as u64))
                    .expect("seek failed");
                let mut buf = vec![0u8; *length];
                file.read_exact(&mut buf).expect("read failed");
                let nd = crate::backends::ndarray_backend::NDArrayNumericTensor::from_raw_data(
                    &buf,
                    *dtype,
                    shape.clone(),
                )
                .expect("decode external tensor");
                NumericTensor::NDArray(nd)
            }
            StoredTensor::ExternalPth {
                path,
                tensor_name,
                dtype,
                shape,
            } => {
                // Load specific tensor by name from a .pth file via candle (if enabled)
                #[cfg(feature = "candle")]
                {
                    use std::path::Path;
                    let tensors = std::sync::Arc::new(
                        candle_core::pickle::PthTensors::new(Path::new(path), None)
                            .expect("open .pth"),
                    );
                    let info = tensors
                        .tensor_infos()
                        .get(tensor_name)
                        .expect("tensor name in pth")
                        .clone();
                    let ct = tensors
                        .get(&info.name)
                        .expect("get tensor")
                        .expect("tensor present");
                    let bytes: Vec<u8> = match dtype {
                        crate::dtype::DType::F32 => {
                            let v: Vec<f32> =
                                ct.flatten_all().unwrap().to_vec1().expect("to_vec f32");
                            v.iter().flat_map(|x| x.to_le_bytes()).collect()
                        }
                        crate::dtype::DType::BF16 => {
                            let v: Vec<half::bf16> =
                                ct.flatten_all().unwrap().to_vec1().expect("to_vec bf16");
                            v.iter().flat_map(|x| x.to_le_bytes()).collect()
                        }
                        crate::dtype::DType::F16 => {
                            let v: Vec<half::f16> =
                                ct.flatten_all().unwrap().to_vec1().expect("to_vec f16");
                            v.iter().flat_map(|x| x.to_le_bytes()).collect()
                        }
                        crate::dtype::DType::I64 => {
                            let v: Vec<i64> =
                                ct.flatten_all().unwrap().to_vec1().expect("to_vec i64");
                            v.iter().flat_map(|x| x.to_le_bytes()).collect()
                        }
                        other => panic!("Unsupported dtype for PTH external tensor: {:?}", other),
                    };
                    let nd = crate::backends::ndarray_backend::NDArrayNumericTensor::from_raw_data(
                        &bytes,
                        *dtype,
                        shape.clone(),
                    )
                    .expect("decode external pth tensor");
                    NumericTensor::NDArray(nd)
                }
                #[cfg(not(feature = "candle"))]
                {
                    let _ = (&path, &tensor_name, &dtype, &shape);
                    panic!(
                        "OriginReference PTH tensors require the 'candle' feature. Rebuild with --features candle"
                    );
                }
            }
        }
    }

    pub fn shape(&self) -> Vec<u64> {
        match self {
            StoredTensor::Numeric(tensor) => tensor.shape(),
            StoredTensor::ExternalBinary { shape, .. } => shape.clone(),
            StoredTensor::ExternalPth { shape, .. } => shape.clone(),
        }
    }

    pub fn dtype(&self) -> DType {
        match self {
            StoredTensor::Numeric(tensor) => tensor.dtype(),
            StoredTensor::ExternalBinary { dtype, .. } => *dtype,
            StoredTensor::ExternalPth { dtype, .. } => *dtype,
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

impl Default for TensorStore {
    fn default() -> Self {
        Self::new()
    }
}
