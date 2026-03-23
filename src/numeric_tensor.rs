//! Pool-managed numeric tensor with dtype-erased storage.
//!
//! [`NumericTensor`] is a struct (not an enum) — bytes in a managed pool.
//! [`NumericTensorView`] borrows `&[u8]` — no pool, no lifetime coupling.
//! All element access returns owned [`NumericScalar`].

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::numeric_dtype::NumericDType;
use crate::numeric_scalar::{NumericScalar, NumericScalarView, NumericScalarViewMut};
use crate::packed_format::PackedFormat;
use crate::pool::Pool;
use crate::tensor_rank::{DimContainer, Rank};

// ---------------------------------------------------------------------------
// StorageFormat
// ---------------------------------------------------------------------------

/// How elements are stored in a buffer.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StorageFormat {
    /// Uniform-stride elements of a single numeric type.
    /// Strides in `BufferLayout` fully describe the element spacing.
    Element { dtype: NumericDType },
    /// Block-quantized with per-block scales/offsets (GGUF Q4_K, etc.)
    BlockQuantized(PackedFormat),
}

impl StorageFormat {
    /// The numeric dtype for element-wise access. Block-quantized formats
    /// dequantize to their output dtype (typically F32).
    pub fn element_dtype(&self) -> NumericDType {
        match self {
            StorageFormat::Element { dtype } => *dtype,
            StorageFormat::BlockQuantized(_) => NumericDType::F32, // dequantized output
        }
    }
}

// ---------------------------------------------------------------------------
// BufferLayout
// ---------------------------------------------------------------------------

/// How elements are packed into a buffer — separate from what the values mean.
///
/// Strides are measured in **bits**. This unifies byte-aligned and sub-byte
/// layouts under a single indexing model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BufferLayout<R: Rank> {
    pub format: StorageFormat,
    /// Strides in bits, one per dimension. Always positive (no negative stride support).
    pub strides: R::KnownDims,
}

impl<R: Rank> BufferLayout<R> {
    /// Create a standard row-major (C-contiguous) layout for the given shape and dtype.
    pub fn row_major(shape: &R::KnownDims, dtype: NumericDType) -> Self {
        let dims = shape.as_slice();
        let element_bits = dtype.total_bits() as u64;
        let mut strides = vec![0u64; dims.len()];

        if !dims.is_empty() {
            // Last dimension has stride = element_bits
            strides[dims.len() - 1] = element_bits;
            // Work backwards: stride[i] = stride[i+1] * shape[i+1]
            for i in (0..dims.len() - 1).rev() {
                strides[i] = strides[i + 1] * dims[i + 1];
            }
        }

        BufferLayout {
            format: StorageFormat::Element { dtype },
            strides: R::KnownDims::try_from_slice(&strides)
                .expect("stride length matches shape length"),
        }
    }

    /// Total buffer size in bytes for this layout + shape.
    pub fn buffer_size_bytes(&self, shape: &R::KnownDims) -> usize {
        match &self.format {
            StorageFormat::Element { .. } => {
                let dims = shape.as_slice();
                if dims.is_empty() {
                    return 0;
                }
                // Total bits = max over all dimensions of (stride[i] * shape[i])
                // For row-major, this equals stride[0] * shape[0].
                let total_bits: u64 = dims
                    .iter()
                    .zip(self.strides.as_slice().iter())
                    .map(|(&dim, &stride)| dim * stride)
                    .max()
                    .unwrap_or(0);
                ((total_bits + 7) / 8) as usize
            }
            StorageFormat::BlockQuantized(fmt) => {
                let numel: u64 = shape.as_slice().iter().product();
                fmt.storage_bytes(numel as usize)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// NumericTensorView
// ---------------------------------------------------------------------------

/// View into tensor data — borrows a byte slice, no pool involvement.
///
/// Erases the pool type at the computation boundary. Ops take views, not
/// owned tensors. Enables zero-copy reshape/transpose via stride manipulation.
pub struct NumericTensorView<'a, R: Rank> {
    data: &'a [u8],
    shape: R::KnownDims,
    layout: BufferLayout<R>,
}

impl<'a, R: Rank> NumericTensorView<'a, R> {
    /// Create a view from raw parts.
    pub fn new(data: &'a [u8], shape: R::KnownDims, layout: BufferLayout<R>) -> Self {
        Self {
            data,
            shape,
            layout,
        }
    }

    pub fn shape(&self) -> &R::KnownDims {
        &self.shape
    }

    pub fn layout(&self) -> &BufferLayout<R> {
        &self.layout
    }

    pub fn data(&self) -> &[u8] {
        self.data
    }

    /// The dtype of elements (after dequantization for block-quantized formats).
    pub fn dtype(&self) -> NumericDType {
        self.layout.format.element_dtype()
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.as_slice().iter().product::<u64>() as usize
    }

    /// Read a single element by flat index, returning an owned scalar.
    ///
    /// For `StorageFormat::Element`, computes the bit offset via strides.
    /// For `StorageFormat::BlockQuantized`, delegates to the packed format's
    /// dequantization logic.
    pub fn read_element(&self, flat_index: usize) -> NumericScalar {
        match &self.layout.format {
            StorageFormat::Element { dtype } => {
                let bit_offset = self.flat_index_to_bit_offset(flat_index);
                NumericScalarView {
                    data: self.data,
                    bit_offset,
                    dtype: *dtype,
                }
                .to_owned_scalar()
            }
            StorageFormat::BlockQuantized(_fmt) => {
                // TODO: per-element dequantization for block-quantized formats.
                // For v1, block-quantized tensors are dequantized in bulk elsewhere.
                panic!(
                    "Per-element read_element not yet implemented for block-quantized formats"
                );
            }
        }
    }

    /// Convert a flat index to a bit offset in the buffer.
    fn flat_index_to_bit_offset(&self, flat_index: usize) -> usize {
        let dims = self.shape.as_slice();
        let strides = self.layout.strides.as_slice();
        if dims.is_empty() {
            return 0;
        }

        // Decompose flat index into per-dimension indices, then dot with strides
        let mut remaining = flat_index as u64;
        let mut bit_offset: u64 = 0;
        for i in 0..dims.len() {
            let dim_size: u64 = dims[i + 1..].iter().product();
            let dim_size = if dim_size == 0 { 1 } else { dim_size };
            let idx = remaining / dim_size;
            remaining %= dim_size;
            bit_offset += idx * strides[i];
        }
        bit_offset as usize
    }
}

impl<R: Rank> fmt::Debug for NumericTensorView<'_, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NumericTensorView")
            .field("shape", &self.shape)
            .field("dtype", &self.layout.format.element_dtype())
            .field("data_len", &self.data.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// NumericTensor
// ---------------------------------------------------------------------------

/// Owned tensor — pool-managed buffer + shape + layout.
///
/// Tensors are pure data — no operation methods. All computation goes through
/// `MilliOpGraph`. The tensor is the noun; it has no verbs.
pub struct NumericTensor<'a, R: Rank, P: Pool + 'a> {
    buffer: P::Buffer<'a>,
    shape: R::KnownDims,
    layout: BufferLayout<R>,
}

impl<'a, R: Rank, P: Pool + 'a> NumericTensor<'a, R, P> {
    /// Create a new zero-initialized tensor with row-major layout.
    pub fn zeros(shape: R::KnownDims, dtype: NumericDType, pool: &'a P) -> Result<Self, crate::pool::AllocationError> {
        let layout = BufferLayout::row_major(&shape, dtype);
        let size = layout.buffer_size_bytes(&shape);
        let buffer = pool.allocate(size)?;
        Ok(Self {
            buffer,
            shape,
            layout,
        })
    }

    /// Create a tensor from an existing pool buffer + metadata.
    pub fn from_parts(
        buffer: P::Buffer<'a>,
        shape: R::KnownDims,
        layout: BufferLayout<R>,
    ) -> Self {
        Self {
            buffer,
            shape,
            layout,
        }
    }

    /// Borrow as a view — erases the pool type.
    pub fn view(&self) -> NumericTensorView<'_, R> {
        NumericTensorView {
            data: &self.buffer,
            shape: self.shape.clone(),
            layout: self.layout.clone(),
        }
    }

    pub fn shape(&self) -> &R::KnownDims {
        &self.shape
    }

    pub fn layout(&self) -> &BufferLayout<R> {
        &self.layout
    }

    pub fn dtype(&self) -> NumericDType {
        self.layout.format.element_dtype()
    }

    pub fn numel(&self) -> usize {
        self.shape.as_slice().iter().product::<u64>() as usize
    }

    /// Read a single element by flat index.
    pub fn read_element(&self, flat_index: usize) -> NumericScalar {
        self.view().read_element(flat_index)
    }

    /// Write a single element by flat index.
    pub fn write_element(&mut self, flat_index: usize, value: NumericScalar) {
        match &self.layout.format {
            StorageFormat::Element { dtype } => {
                let bit_offset = self.view().flat_index_to_bit_offset(flat_index);
                let mut view = NumericScalarViewMut {
                    data: &mut self.buffer,
                    bit_offset,
                    dtype: *dtype,
                };
                view.write_scalar(&value);
            }
            StorageFormat::BlockQuantized(_) => {
                panic!("write_element not supported for block-quantized formats");
            }
        }
    }

    /// Access the raw buffer bytes.
    pub fn buffer(&self) -> &[u8] {
        &self.buffer
    }

    /// Access the raw buffer bytes mutably.
    pub fn buffer_mut(&mut self) -> &mut [u8] {
        &mut self.buffer
    }
}

impl<'a, R: Rank, P: Pool + 'a> fmt::Debug for NumericTensor<'a, R, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NumericTensor")
            .field("shape", &self.shape)
            .field("dtype", &self.layout.format.element_dtype())
            .field("buffer_len", &self.buffer.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pool::SystemPool;
    use crate::tensor_rank::DynRank;
    use half::bf16;

    #[test]
    fn row_major_layout_f32() {
        let shape: Vec<u64> = vec![2, 3];
        let layout = BufferLayout::<DynRank>::row_major(&shape, NumericDType::F32);
        // F32 = 32 bits. Row-major [2,3]: strides = [96, 32]
        assert_eq!(layout.strides, vec![96, 32]);
        assert_eq!(layout.buffer_size_bytes(&shape), 24); // 2*3*4 bytes
    }

    #[test]
    fn row_major_layout_bool() {
        let shape: Vec<u64> = vec![8];
        let layout = BufferLayout::<DynRank>::row_major(&shape, NumericDType::BOOL);
        assert_eq!(layout.strides, vec![1]); // 1 bit per element
        assert_eq!(layout.buffer_size_bytes(&shape), 1); // 8 bits = 1 byte
    }

    #[test]
    fn tensor_zeros() {
        let pool = SystemPool;
        let shape: Vec<u64> = vec![2, 3];
        let t = NumericTensor::<DynRank, SystemPool>::zeros(shape.clone(), NumericDType::F32, &pool)
            .unwrap();
        assert_eq!(t.shape(), &shape);
        assert_eq!(t.dtype(), NumericDType::F32);
        assert_eq!(t.numel(), 6);
        assert_eq!(t.buffer().len(), 24);
        // All zeros
        assert!(t.buffer().iter().all(|&b| b == 0));
    }

    #[test]
    fn write_read_elements_f32() {
        let pool = SystemPool;
        let shape: Vec<u64> = vec![2, 3];
        let mut t =
            NumericTensor::<DynRank, SystemPool>::zeros(shape, NumericDType::F32, &pool).unwrap();

        // Write values
        for i in 0..6 {
            t.write_element(i, NumericScalar::from_f64(i as f64 * 1.5).cast_to(NumericDType::F32));
        }

        // Read back — assert bit-equality with the expected f32 value
        for i in 0..6 {
            let expected = NumericScalar::from_f32((i as f64 * 1.5) as f32);
            assert_eq!(
                t.read_element(i), expected,
                "element {i} mismatch",
            );
        }
    }

    #[test]
    fn write_read_elements_i32() {
        let pool = SystemPool;
        let shape: Vec<u64> = vec![4];
        let mut t =
            NumericTensor::<DynRank, SystemPool>::zeros(shape, NumericDType::I32, &pool).unwrap();

        t.write_element(0, NumericScalar::from_f64(-100.0).cast_to(NumericDType::I32));
        t.write_element(1, NumericScalar::from_f64(0.0).cast_to(NumericDType::I32));
        t.write_element(2, NumericScalar::from_f64(42.0).cast_to(NumericDType::I32));
        t.write_element(3, NumericScalar::from_f64(1000.0).cast_to(NumericDType::I32));

        assert_eq!(t.read_element(0), NumericScalar::from_i32(-100));
        assert_eq!(t.read_element(1), NumericScalar::from_i32(0));
        assert_eq!(t.read_element(2), NumericScalar::from_i32(42));
        assert_eq!(t.read_element(3), NumericScalar::from_i32(1000));
    }

    #[test]
    fn view_erases_pool() {
        let pool = SystemPool;
        let shape: Vec<u64> = vec![3];
        let mut t =
            NumericTensor::<DynRank, SystemPool>::zeros(shape, NumericDType::F32, &pool).unwrap();
        t.write_element(1, NumericScalar::from_f64(7.0).cast_to(NumericDType::F32));

        // View has no pool type parameter
        let view: NumericTensorView<'_, DynRank> = t.view();
        assert_eq!(view.numel(), 3);
        assert_eq!(view.read_element(1), NumericScalar::from_f32(7.0));
    }

    #[test]
    fn tracked_pool_tensor() {
        use crate::pool::TrackedPool;

        let pool = TrackedPool::new(Some(1024));
        let shape: Vec<u64> = vec![4, 4];
        let t = NumericTensor::<DynRank, TrackedPool>::zeros(shape, NumericDType::F32, &pool)
            .unwrap();
        assert_eq!(pool.bytes_in_use(), 64); // 4*4*4 = 64 bytes

        drop(t);
        assert_eq!(pool.bytes_in_use(), 0);
    }

    #[test]
    fn tracked_pool_budget_prevents_allocation() {
        use crate::pool::{AllocationError, TrackedPool};

        let pool = TrackedPool::new(Some(32)); // 32 byte budget
        let shape: Vec<u64> = vec![100]; // 100 * 4 = 400 bytes needed
        let result =
            NumericTensor::<DynRank, TrackedPool>::zeros(shape, NumericDType::F32, &pool);
        assert!(matches!(result, Err(AllocationError::BudgetExceeded { .. })));
    }

    #[test]
    fn bf16_tensor() {
        let pool = SystemPool;
        let shape: Vec<u64> = vec![2];
        let mut t =
            NumericTensor::<DynRank, SystemPool>::zeros(shape, NumericDType::BF16, &pool).unwrap();
        assert_eq!(t.buffer().len(), 4); // 2 * 2 bytes

        t.write_element(0, NumericScalar::from_f64(1.5).cast_to(NumericDType::BF16));
        t.write_element(1, NumericScalar::from_f64(-2.0).cast_to(NumericDType::BF16));

        assert_eq!(t.read_element(0), NumericScalar::from_bf16(bf16::from_f32(1.5)));
        assert_eq!(t.read_element(1), NumericScalar::from_bf16(bf16::from_f32(-2.0)));
    }

    #[test]
    fn scalar_tensor() {
        let pool = SystemPool;
        let shape: Vec<u64> = vec![];
        let layout = BufferLayout::<DynRank>::row_major(&shape, NumericDType::F32);
        assert_eq!(layout.strides, Vec::<u64>::new());
        assert_eq!(layout.buffer_size_bytes(&shape), 0);
        // Scalar tensors have 0 dimensions, 0 buffer size — the value lives inline
        // in the graph as a constant, not in a pool buffer.
    }
}
