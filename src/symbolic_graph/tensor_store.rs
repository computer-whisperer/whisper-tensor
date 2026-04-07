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

    /// Load raw file bytes directly into a pool-backed tensor.
    ///
    /// Allocates the pool buffer first, then streams file bytes straight
    /// into it. For `ExternalBinary`/`ExternalGGUF` this is a single
    /// `read_exact`; for `ExternalPth` we delegate to
    /// [`PthTensors::read_raw_bytes_into`]; for `ExternalSafetensors` we
    /// `copy_from_slice` off the mmap view. No intermediate `Vec<u8>`
    /// lives on the heap during a weight load — every byte goes directly
    /// from the kernel (or mmap page) into the pool's accounting.
    fn load_raw_to_pool<'p, P: crate::pool::Pool + 'p>(
        &self,
        format: TensorFormat,
        shape: &[u64],
        pool: &'p P,
    ) -> Option<crate::numeric_tensor::NumericTensor<'p, DynRank, P>> {
        use crate::numeric_tensor::NumericTensor as NewTensor;
        use std::io::{Read, Seek, SeekFrom};

        let layout = format.to_layout(shape.to_vec());
        let buf_size = layout.buffer_size_bytes();
        let mut buf = pool.allocate(buf_size).ok()?;
        let dst = &mut buf.as_mut()[..buf_size];

        match self {
            StoredTensor::ExternalBinary {
                path,
                offset,
                length,
                ..
            }
            | StoredTensor::ExternalGGUF {
                path,
                offset,
                length,
                ..
            } => {
                if *length < buf_size {
                    return None;
                }
                let mut file = std::fs::File::open(path).ok()?;
                file.seek(SeekFrom::Start(*offset as u64)).ok()?;
                file.read_exact(dst).ok()?;
            }
            StoredTensor::ExternalPth {
                path, tensor_name, ..
            } => {
                // The metadata cache is what makes repeated weight loads
                // cheap — parsing the pickle stream is the dominant cost
                // when resolving hundreds of small constants per iter.
                let pth_path = std::path::Path::new(path);
                let tensors = crate::pth::PthTensors::cached(pth_path).ok()?;
                let byte_len = tensors.byte_len(tensor_name)?;
                if byte_len != buf_size {
                    return None;
                }
                tensors.read_raw_bytes_into(tensor_name, dst).ok()?;
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
                    let src = view.data();
                    if src.len() < buf_size {
                        return None;
                    }
                    dst.copy_from_slice(&src[..buf_size]);
                }
                #[cfg(not(feature = "safetensors"))]
                {
                    let _ = (path, tensor_name);
                    return None;
                }
            }
            StoredTensor::Inline(_) => return None,
        }

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
