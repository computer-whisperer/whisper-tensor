use crate::dtype::DType;
use crate::numeric_tensor::NumericTensor;
use crate::packed_tensor::PackedTensor;
use crate::tensor_rank::DynRank;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

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
    ExternalSafetensors {
        path: String,
        tensor_name: String,
        dtype: DType,
        shape: Vec<u64>,
    },
    /// A tensor stored in a GGUF file, addressed by name.
    /// For packed (quantized) tensors, `dtype` will be `DType::Packed(format)`.
    /// For non-packed tensors (e.g. F32 norm weights), `dtype` is the element type.
    ExternalGGUF {
        path: String,
        tensor_name: String,
        offset: usize,
        length: usize,
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
                // Load specific tensor by name from a .pth file via local parser.
                let pth_path = std::path::Path::new(path);
                let tensors = crate::pth::PthTensors::new(pth_path, None).expect("open .pth");
                let bytes = tensors
                    .get_raw_bytes(tensor_name)
                    .expect("read tensor bytes")
                    .expect("tensor present");
                let nd = crate::backends::ndarray_backend::NDArrayNumericTensor::from_raw_data(
                    &bytes,
                    *dtype,
                    shape.clone(),
                )
                .expect("decode external pth tensor");
                NumericTensor::NDArray(nd)
            }
            StoredTensor::ExternalSafetensors {
                path,
                tensor_name,
                dtype,
                shape,
            } => {
                #[cfg(feature = "safetensors")]
                {
                    use memmap2::Mmap;
                    use safetensors::SafeTensors;
                    use std::fs::File;

                    let file = File::open(path).expect("Failed to open safetensors file");
                    let mmap =
                        unsafe { Mmap::map(&file) }.expect("Failed to mmap safetensors file");
                    let st = SafeTensors::deserialize(&mmap).expect("Failed to parse safetensors");
                    let view = st
                        .tensor(tensor_name)
                        .expect("tensor not found in safetensors");
                    let bytes = view.data();
                    let nd = crate::backends::ndarray_backend::NDArrayNumericTensor::from_raw_data(
                        bytes,
                        *dtype,
                        shape.clone(),
                    )
                    .expect("decode external safetensors tensor");
                    NumericTensor::NDArray(nd)
                }
                #[cfg(not(feature = "safetensors"))]
                {
                    let _ = (&path, &tensor_name, &dtype, &shape);
                    panic!(
                        "ExternalSafetensors tensors require the 'safetensors' feature. Rebuild with --features safetensors"
                    );
                }
            }
            StoredTensor::ExternalGGUF {
                path,
                offset,
                length,
                dtype,
                shape,
                ..
            } => {
                use std::io::{Read, Seek, SeekFrom};
                let mut file = std::fs::File::open(path).expect("Failed to open GGUF tensor file");
                file.seek(SeekFrom::Start(*offset as u64))
                    .expect("seek failed");
                let mut buf = vec![0u8; *length];
                file.read_exact(&mut buf).expect("read failed");

                if let Some(packed_format) = dtype.packed_format() {
                    // Return as a PackedTensor — stays quantized until explicitly dequantized
                    let packed = PackedTensor::new(Arc::from(buf), shape.clone(), packed_format);
                    NumericTensor::Packed(packed)
                } else {
                    // Non-packed tensor (e.g. F32 norm weights) — decode as NDArray
                    let nd = crate::backends::ndarray_backend::NDArrayNumericTensor::from_raw_data(
                        &buf,
                        *dtype,
                        shape.clone(),
                    )
                    .expect("decode external GGUF tensor");
                    NumericTensor::NDArray(nd)
                }
            }
        }
    }

    pub fn shape(&self) -> Vec<u64> {
        match self {
            StoredTensor::Numeric(tensor) => tensor.shape(),
            StoredTensor::ExternalBinary { shape, .. } => shape.clone(),
            StoredTensor::ExternalPth { shape, .. } => shape.clone(),
            StoredTensor::ExternalSafetensors { shape, .. } => shape.clone(),
            StoredTensor::ExternalGGUF { shape, .. } => shape.clone(),
        }
    }

    pub fn dtype(&self) -> DType {
        match self {
            StoredTensor::Numeric(tensor) => tensor.dtype(),
            StoredTensor::ExternalBinary { dtype, .. } => *dtype,
            StoredTensor::ExternalPth { dtype, .. } => *dtype,
            StoredTensor::ExternalSafetensors { dtype, .. } => *dtype,
            StoredTensor::ExternalGGUF { dtype, .. } => *dtype,
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
