use crate::dtype::DType;
use crate::migration::numeric_tensor::NumericTensor;
use crate::numeric_dtype::NumericDType;
use crate::packed_tensor::PackedTensor;
use crate::tensor_rank::DynRank;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Hash, Eq, PartialEq, Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TensorStoreTensorId(u64);

pub enum StoredTensor {
    /// Pool-backed tensor (new type system).
    Inline(crate::numeric_tensor::NumericTensor<'static, DynRank, crate::pool::SystemPool>),
    /// Legacy in-memory tensor — kept for backward compatibility during migration.
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
        use crate::numeric_dtype::NumericDType;
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
            StoredTensor::Numeric(legacy) => {
                // Bridge: legacy → TensorInfo → concrete → copy
                let sys_pool = crate::pool::SystemPool;
                let info = crate::tensor_info::TensorInfo::from_legacy(legacy, &sys_pool);
                let concrete = info.as_concrete()?;
                let ndt = concrete.dtype();
                let shape = concrete.shape().clone();
                let layout = TensorLayout::<DynRank>::row_major(shape, ndt);
                let buf = pool.allocate(layout.buffer_size_bytes()).ok()?;
                let mut tensor = NewTensor::from_parts(buf, layout);
                for i in 0..concrete.numel() {
                    tensor.write_element(i, concrete.read_element(i));
                }
                Some(tensor)
            }
            StoredTensor::ExternalBinary { dtype, shape, .. }
            | StoredTensor::ExternalPth { dtype, shape, .. }
            | StoredTensor::ExternalSafetensors { dtype, shape, .. } => {
                let ndt = NumericDType::from_legacy(*dtype)?;
                self.load_raw_to_pool(ndt, shape, pool)
            }
            StoredTensor::ExternalGGUF { dtype, shape, .. } => {
                if dtype.packed_format().is_some() {
                    return None;
                }
                let ndt = NumericDType::from_legacy(*dtype)?;
                self.load_raw_to_pool(ndt, shape, pool)
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
    /// The raw bytes are little-endian and copied directly — no legacy intermediate.
    fn load_raw_to_pool<'p, P: crate::pool::Pool + 'p>(
        &self,
        ndt: crate::numeric_dtype::NumericDType,
        shape: &[u64],
        pool: &'p P,
    ) -> Option<crate::numeric_tensor::NumericTensor<'p, DynRank, P>> {
        use crate::numeric_scalar::NumericScalar;
        use crate::numeric_tensor::{NumericTensor as NewTensor, TensorLayout};

        let raw = self.load_raw_bytes()?;
        let layout = TensorLayout::<DynRank>::row_major(shape.to_vec(), ndt);
        let numel = shape.iter().product::<u64>() as usize;
        let buf = pool.allocate(layout.buffer_size_bytes()).ok()?;
        let mut tensor = NewTensor::from_parts(buf, layout);

        let bits = ndt.total_bits() as usize;
        for i in 0..numel {
            let bit_offset = i * bits;
            let raw_val =
                crate::numeric_scalar::conversions::read_raw_bits(&raw, bit_offset, bits as u8);
            tensor.write_element(i, NumericScalar::from_raw_bits(raw_val, ndt));
        }

        Some(tensor)
    }

    pub fn to_numeric(&self) -> NumericTensor<DynRank> {
        match self {
            StoredTensor::Inline(src) => {
                // Bridge to legacy: read elements from new-type tensor.
                let legacy_dt = src.dtype().to_legacy();
                let shape = src.shape().clone();
                let numel = src.numel();
                // Build legacy via element extraction.
                let mut vals = Vec::with_capacity(numel);
                for i in 0..numel {
                    vals.push(src.read_element(i).to_f64());
                }
                // Create f64 ndarray then cast to target dtype.
                let nd = crate::backends::ndarray_backend::NDArrayNumericTensor::from_vec_shape(
                    vals, &shape,
                )
                .expect("build legacy from inline");
                let mut backend = crate::backends::eval_backend::EvalBackend::NDArray;
                let cast = NumericTensor::NDArray(nd)
                    .cast(legacy_dt, &mut backend)
                    .expect("cast inline to legacy dtype");
                cast
            }
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
            StoredTensor::Inline(t) => t.shape().clone(),
            StoredTensor::Numeric(tensor) => tensor.shape(),
            StoredTensor::ExternalBinary { shape, .. } => shape.clone(),
            StoredTensor::ExternalPth { shape, .. } => shape.clone(),
            StoredTensor::ExternalSafetensors { shape, .. } => shape.clone(),
            StoredTensor::ExternalGGUF { shape, .. } => shape.clone(),
        }
    }

    pub fn dtype(&self) -> DType {
        match self {
            StoredTensor::Inline(t) => t.dtype().to_legacy(),
            StoredTensor::Numeric(tensor) => tensor.dtype(),
            StoredTensor::ExternalBinary { dtype, .. } => *dtype,
            StoredTensor::ExternalPth { dtype, .. } => *dtype,
            StoredTensor::ExternalSafetensors { dtype, .. } => *dtype,
            StoredTensor::ExternalGGUF { dtype, .. } => *dtype,
        }
    }

    pub fn numeric_dtype(&self) -> Option<NumericDType> {
        match self {
            StoredTensor::Inline(t) => Some(t.dtype()),
            _ => NumericDType::from_legacy(self.dtype()),
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
