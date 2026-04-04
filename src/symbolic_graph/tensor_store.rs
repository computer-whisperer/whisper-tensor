use crate::numeric_dtype::NumericDType;
use crate::numeric_tensor::TensorFormat;
use crate::tensor_rank::DynRank;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Hash, Eq, PartialEq, Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TensorStoreTensorId(u64);

pub enum StoredTensor {
    /// Pool-backed tensor (new type system).
    Inline(crate::numeric_tensor::NumericTensor<'static, DynRank, crate::pool::SystemPool>),
    ExternalBinary {
        path: String,
        offset: usize,
        length: usize,
        format: TensorFormat,
        shape: Vec<u64>,
    },
    ExternalPth {
        path: String,
        tensor_name: String,
        format: TensorFormat,
        shape: Vec<u64>,
    },
    ExternalSafetensors {
        path: String,
        tensor_name: String,
        format: TensorFormat,
        shape: Vec<u64>,
    },
    /// A tensor stored in a GGUF file, addressed by name.
    /// For quantized tensors, `format` describes the block quantization scheme.
    /// For non-quantized tensors (e.g. F32 norm weights), `format` is Element(dtype).
    ExternalGGUF {
        path: String,
        tensor_name: String,
        offset: usize,
        length: usize,
        format: TensorFormat,
        shape: Vec<u64>,
    },
}

impl StoredTensor {
    /// Load this tensor into a pool-backed new-type NumericTensor.
    ///
    /// For external file formats, reads bytes and copies directly into the pool
    /// buffer — no legacy NDArrayNumericTensor intermediary. For the Numeric
    /// variant (in-memory legacy tensor), bridges via TensorInfo.
    ///
    /// Returns None for packed/quantized tensors or unsupported dtypes.
    pub fn to_pool_tensor<'p, P: crate::pool::Pool + 'p>(
        &self,
        pool: &'p P,
    ) -> Option<crate::numeric_tensor::NumericTensor<'p, DynRank, P>> {
        use crate::numeric_tensor::{NumericTensor as NewTensor, TensorLayout};

        match self {
            StoredTensor::Inline(src) => {
                // Copy from SystemPool tensor into the target pool.
                let ndt = src.dtype();
                let shape = src.shape().clone();
                let layout = TensorLayout::<DynRank>::row_major(shape, ndt);
                let buf = pool.allocate(layout.buffer_size_bytes()).ok()?;
                let mut tensor = NewTensor::from_parts(buf, layout);
                for i in 0..src.numel() {
                    tensor.write_element(i, src.read_element(i));
                }
                Some(tensor)
            }
            StoredTensor::ExternalBinary { format, shape, .. }
            | StoredTensor::ExternalPth { format, shape, .. }
            | StoredTensor::ExternalSafetensors { format, shape, .. }
            | StoredTensor::ExternalGGUF { format, shape, .. } => {
                self.load_raw_to_pool(*format, shape, pool)
            }
        }
    }

    /// Read raw bytes from disk for this stored tensor.
    fn load_raw_bytes(&self) -> Option<Vec<u8>> {
        use std::io::{Read, Seek, SeekFrom};
        match self {
            StoredTensor::ExternalBinary {
                path,
                offset,
                length,
                ..
            } => {
                let mut file = std::fs::File::open(path).ok()?;
                file.seek(SeekFrom::Start(*offset as u64)).ok()?;
                let mut buf = vec![0u8; *length];
                file.read_exact(&mut buf).ok()?;
                Some(buf)
            }
            StoredTensor::ExternalPth {
                path, tensor_name, ..
            } => {
                let pth_path = std::path::Path::new(path);
                let tensors = crate::pth::PthTensors::new(pth_path, None).ok()?;
                tensors.get_raw_bytes(tensor_name).ok()?
            }
            StoredTensor::ExternalSafetensors {
                path, tensor_name, ..
            } => {
                #[cfg(feature = "safetensors")]
                {
                    use memmap2::Mmap;
                    use safetensors::SafeTensors;
                    let file = std::fs::File::open(path).ok()?;
                    let mmap = unsafe { Mmap::map(&file) }.ok()?;
                    let st = SafeTensors::deserialize(&mmap).ok()?;
                    let view = st.tensor(tensor_name).ok()?;
                    Some(view.data().to_vec())
                }
                #[cfg(not(feature = "safetensors"))]
                {
                    let _ = (path, tensor_name);
                    None
                }
            }
            StoredTensor::ExternalGGUF {
                path,
                offset,
                length,
                ..
            } => {
                let mut file = std::fs::File::open(path).ok()?;
                file.seek(SeekFrom::Start(*offset as u64)).ok()?;
                let mut buf = vec![0u8; *length];
                file.read_exact(&mut buf).ok()?;
                Some(buf)
            }
            _ => None,
        }
    }

    /// Load raw file bytes directly into a pool-backed tensor.
    /// The raw bytes are copied directly into the appropriate layout —
    /// no legacy intermediate. Handles both element-strided and quantized formats.
    fn load_raw_to_pool<'p, P: crate::pool::Pool + 'p>(
        &self,
        format: TensorFormat,
        shape: &[u64],
        pool: &'p P,
    ) -> Option<crate::numeric_tensor::NumericTensor<'p, DynRank, P>> {
        use crate::numeric_tensor::NumericTensor as NewTensor;

        let raw = self.load_raw_bytes()?;
        let layout = format.to_layout(shape.to_vec());
        let buf_size = layout.buffer_size_bytes();

        // For all formats, the raw file bytes ARE the buffer contents
        // (little-endian, matching the layout's expected byte packing).
        if raw.len() < buf_size {
            return None;
        }
        let mut buf = pool.allocate(buf_size).ok()?;
        buf.as_mut()[..buf_size].copy_from_slice(&raw[..buf_size]);
        Some(NewTensor::from_parts(buf, layout))
    }

    pub fn shape(&self) -> Vec<u64> {
        match self {
            StoredTensor::Inline(t) => t.shape().clone(),
            StoredTensor::ExternalBinary { shape, .. } => shape.clone(),
            StoredTensor::ExternalPth { shape, .. } => shape.clone(),
            StoredTensor::ExternalSafetensors { shape, .. } => shape.clone(),
            StoredTensor::ExternalGGUF { shape, .. } => shape.clone(),
        }
    }

    /// The tensor format (dtype for element-strided, quant format for quantized).
    pub fn format(&self) -> TensorFormat {
        match self {
            StoredTensor::Inline(t) => TensorFormat::Element(t.dtype()),
            StoredTensor::ExternalBinary { format, .. }
            | StoredTensor::ExternalPth { format, .. }
            | StoredTensor::ExternalSafetensors { format, .. }
            | StoredTensor::ExternalGGUF { format, .. } => *format,
        }
    }

    pub fn numeric_dtype(&self) -> Option<NumericDType> {
        match self.format() {
            TensorFormat::Element(ndt) => Some(ndt),
            _ => None,
        }
    }

    pub fn loading_label(&self) -> Option<String> {
        match self {
            StoredTensor::ExternalPth { tensor_name, .. } => Some(tensor_name.clone()),
            StoredTensor::ExternalSafetensors { tensor_name, .. } => Some(tensor_name.clone()),
            StoredTensor::ExternalGGUF { tensor_name, .. } => Some(tensor_name.clone()),
            _ => None,
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
