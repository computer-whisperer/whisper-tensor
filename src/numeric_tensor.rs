//! Pool-managed numeric tensor with dtype-erased storage.
//!
//! [`TensorLayout`] is the core abstraction: a complete description of how a
//! tensor is packed into a contiguous memory region. It knows the shape, dtype,
//! and access pattern, and can read/write elements given a `&[u8]` span.
//!
//! [`NumericTensor`] and [`NumericTensorView`] are thin wrappers pairing a
//! `TensorLayout` with actual memory (pool buffer or byte slice).

use std::fmt;

use crate::numeric_dtype::NumericDType;
use crate::numeric_scalar::{NumericScalar, NumericScalarView, NumericScalarViewMut};
use crate::packed_format::PackedFormat;
use crate::pool::Pool;
use crate::tensor_rank::{DimContainer, Rank};

// ---------------------------------------------------------------------------
// TensorLayout
// ---------------------------------------------------------------------------

/// Complete description of how a tensor is packed into a contiguous memory region.
///
/// Each arm is fully self-contained — it carries shape, dtype, and all format-
/// specific metadata needed to compute buffer sizes, element offsets, and
/// read/write elements from a `&[u8]` span. No back-references needed.
///
/// All core access methods live here. [`NumericTensor`] and [`NumericTensorView`]
/// simply delegate to these methods, passing their data pointer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TensorLayout<R: Rank> {
    /// Uniform-stride elements of a single numeric type.
    /// Strides (in bits, per dimension) fully describe element spacing.
    ElementStrided {
        shape: R::KnownDims,
        dtype: NumericDType,
        /// Per-dimension strides in bits. Always positive (no negative stride support).
        strides: R::KnownDims,
    },
    /// Block-quantized with per-block scales/offsets (GGUF Q4_K, etc.)
    /// Access is by block index + element within block, not by strides.
    BlockQuantized {
        shape: R::KnownDims,
        format: PackedFormat,
    },
}

// -- Constructors --

impl<R: Rank> TensorLayout<R> {
    /// Create a row-major (C-contiguous) element-strided layout.
    pub fn row_major(shape: R::KnownDims, dtype: NumericDType) -> Self {
        let dims = shape.as_slice();
        let element_bits = dtype.total_bits() as u64;
        let mut strides_vec = vec![0u64; dims.len()];

        if !dims.is_empty() {
            strides_vec[dims.len() - 1] = element_bits;
            for i in (0..dims.len() - 1).rev() {
                strides_vec[i] = strides_vec[i + 1] * dims[i + 1];
            }
        }

        TensorLayout::ElementStrided {
            shape,
            dtype,
            strides: R::KnownDims::try_from_slice(&strides_vec)
                .expect("stride length matches shape length"),
        }
    }

    /// Create a block-quantized layout.
    pub fn block_quantized(shape: R::KnownDims, format: PackedFormat) -> Self {
        TensorLayout::BlockQuantized { shape, format }
    }
}

// -- Universal accessors (dispatch into arms) --

impl<R: Rank> TensorLayout<R> {
    pub fn shape(&self) -> &R::KnownDims {
        match self {
            TensorLayout::ElementStrided { shape, .. } => shape,
            TensorLayout::BlockQuantized { shape, .. } => shape,
        }
    }

    /// The numeric dtype for element-wise access.
    /// Block-quantized formats dequantize to F32.
    pub fn element_dtype(&self) -> NumericDType {
        match self {
            TensorLayout::ElementStrided { dtype, .. } => *dtype,
            TensorLayout::BlockQuantized { .. } => NumericDType::F32,
        }
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.shape().as_slice().iter().product::<u64>() as usize
    }

    /// Total buffer size in bytes needed to hold this tensor.
    pub fn buffer_size_bytes(&self) -> usize {
        match self {
            TensorLayout::ElementStrided {
                shape,
                dtype,
                strides,
            } => {
                let dims = shape.as_slice();
                if dims.is_empty() {
                    // Scalar tensor: one element, no dimensions
                    return dtype.bytes_per_element();
                }
                // For row-major contiguous: stride[0] * shape[0] gives total bits.
                // General case: sum of (dim_i - 1) * stride_i + element_bits
                let element_bits = dtype.total_bits() as u64;
                let extent: u64 = dims
                    .iter()
                    .zip(strides.as_slice().iter())
                    .map(|(&dim, &stride)| if dim > 0 { (dim - 1) * stride } else { 0 })
                    .sum::<u64>()
                    + element_bits;
                ((extent + 7) / 8) as usize
            }
            TensorLayout::BlockQuantized { shape, format } => {
                let numel: u64 = shape.as_slice().iter().product();
                format.storage_bytes(numel as usize)
            }
        }
    }
}

// -- Element access (takes a byte span as argument) --

impl<R: Rank> TensorLayout<R> {
    /// Read a single element by flat index from a byte buffer.
    pub fn read_element(&self, data: &[u8], flat_index: usize) -> NumericScalar {
        match self {
            TensorLayout::ElementStrided {
                shape,
                dtype,
                strides,
            } => {
                let bit_offset = flat_to_bit_offset(flat_index, shape.as_slice(), strides.as_slice());
                NumericScalarView {
                    data,
                    bit_offset,
                    dtype: *dtype,
                }
                .to_owned_scalar()
            }
            TensorLayout::BlockQuantized { .. } => {
                panic!(
                    "Per-element read_element not yet implemented for block-quantized formats"
                );
            }
        }
    }

    /// Write a single element by flat index into a byte buffer.
    pub fn write_element(&self, data: &mut [u8], flat_index: usize, value: NumericScalar) {
        match self {
            TensorLayout::ElementStrided {
                shape,
                dtype,
                strides,
            } => {
                assert_eq!(
                    value.dtype(),
                    *dtype,
                    "write_element: scalar dtype {:?} != layout dtype {:?}",
                    value.dtype(),
                    dtype
                );
                let bit_offset = flat_to_bit_offset(flat_index, shape.as_slice(), strides.as_slice());
                let mut view = NumericScalarViewMut {
                    data,
                    bit_offset,
                    dtype: *dtype,
                };
                view.write_scalar(&value);
            }
            TensorLayout::BlockQuantized { .. } => {
                panic!("write_element not supported for block-quantized formats");
            }
        }
    }
}

/// Convert a flat element index to a bit offset using shape and strides.
fn flat_to_bit_offset(flat_index: usize, dims: &[u64], strides: &[u64]) -> usize {
    if dims.is_empty() {
        return 0;
    }
    let mut remaining = flat_index as u64;
    let mut bit_offset: u64 = 0;
    for i in 0..dims.len() {
        let tail_size: u64 = dims[i + 1..].iter().product();
        let tail_size = if tail_size == 0 { 1 } else { tail_size };
        let idx = remaining / tail_size;
        remaining %= tail_size;
        bit_offset += idx * strides[i];
    }
    bit_offset as usize
}

// ---------------------------------------------------------------------------
// NumericTensorView
// ---------------------------------------------------------------------------

/// View into tensor data — borrows a byte slice, no pool involvement.
///
/// Erases the pool type at the computation boundary. Ops take views, not
/// owned tensors. Works with mmap'd data, pool buffers, or any `&[u8]`.
pub struct NumericTensorView<'a, R: Rank> {
    data: &'a [u8],
    layout: TensorLayout<R>,
}

impl<'a, R: Rank> NumericTensorView<'a, R> {
    pub fn new(data: &'a [u8], layout: TensorLayout<R>) -> Self {
        Self { data, layout }
    }

    pub fn layout(&self) -> &TensorLayout<R> {
        &self.layout
    }

    pub fn data(&self) -> &[u8] {
        self.data
    }

    pub fn shape(&self) -> &R::KnownDims {
        self.layout.shape()
    }

    pub fn dtype(&self) -> NumericDType {
        self.layout.element_dtype()
    }

    pub fn numel(&self) -> usize {
        self.layout.numel()
    }

    pub fn read_element(&self, flat_index: usize) -> NumericScalar {
        self.layout.read_element(self.data, flat_index)
    }
}

impl<R: Rank> fmt::Debug for NumericTensorView<'_, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NumericTensorView")
            .field("shape", self.layout.shape())
            .field("dtype", &self.layout.element_dtype())
            .field("data_len", &self.data.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// NumericTensor
// ---------------------------------------------------------------------------

/// Owned tensor — pool-managed buffer + layout.
///
/// Tensors are pure data — no operation methods. All computation goes through
/// `MilliOpGraph`. The tensor is the noun; it has no verbs.
pub struct NumericTensor<'a, R: Rank, P: Pool + 'a> {
    buffer: P::Buffer<'a>,
    layout: TensorLayout<R>,
}

impl<'a, R: Rank, P: Pool + 'a> NumericTensor<'a, R, P> {
    /// Create a new zero-initialized tensor with row-major layout.
    pub fn zeros(
        shape: R::KnownDims,
        dtype: NumericDType,
        pool: &'a P,
    ) -> Result<Self, crate::pool::AllocationError> {
        let layout = TensorLayout::row_major(shape, dtype);
        let size = layout.buffer_size_bytes();
        let buffer = pool.allocate(size)?;
        Ok(Self { buffer, layout })
    }

    /// Create a tensor from an existing pool buffer + layout.
    pub fn from_parts(buffer: P::Buffer<'a>, layout: TensorLayout<R>) -> Self {
        Self { buffer, layout }
    }

    /// Borrow as a view — erases the pool type.
    pub fn view(&self) -> NumericTensorView<'_, R> {
        NumericTensorView {
            data: &self.buffer,
            layout: self.layout.clone(),
        }
    }

    pub fn layout(&self) -> &TensorLayout<R> {
        &self.layout
    }

    pub fn shape(&self) -> &R::KnownDims {
        self.layout.shape()
    }

    pub fn dtype(&self) -> NumericDType {
        self.layout.element_dtype()
    }

    pub fn numel(&self) -> usize {
        self.layout.numel()
    }

    pub fn read_element(&self, flat_index: usize) -> NumericScalar {
        self.layout.read_element(&self.buffer, flat_index)
    }

    pub fn write_element(&mut self, flat_index: usize, value: NumericScalar) {
        self.layout
            .write_element(&mut self.buffer, flat_index, value);
    }

    pub fn buffer(&self) -> &[u8] {
        &self.buffer
    }

    pub fn buffer_mut(&mut self) -> &mut [u8] {
        &mut self.buffer
    }
}

impl<'a, R: Rank, P: Pool + 'a> fmt::Debug for NumericTensor<'a, R, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NumericTensor")
            .field("shape", self.layout.shape())
            .field("dtype", &self.layout.element_dtype())
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
    fn row_major_strides_f32() {
        let layout = TensorLayout::<DynRank>::row_major(vec![2, 3], NumericDType::F32);
        match &layout {
            TensorLayout::ElementStrided { strides, .. } => {
                assert_eq!(*strides, vec![96, 32]);
            }
            _ => panic!("expected ElementStrided"),
        }
        assert_eq!(layout.buffer_size_bytes(), 24); // 2*3*4 bytes
    }

    #[test]
    fn row_major_strides_bool() {
        let layout = TensorLayout::<DynRank>::row_major(vec![8], NumericDType::BOOL);
        match &layout {
            TensorLayout::ElementStrided { strides, .. } => {
                assert_eq!(*strides, vec![1]);
            }
            _ => panic!("expected ElementStrided"),
        }
        assert_eq!(layout.buffer_size_bytes(), 1); // 8 bits = 1 byte
    }

    #[test]
    fn scalar_tensor_has_nonzero_size() {
        let layout = TensorLayout::<DynRank>::row_major(vec![], NumericDType::F32);
        assert_eq!(layout.numel(), 1); // product of empty dims = 1... actually no
        // Product of empty slice is 1 by convention for iter().product()
        // But for a scalar tensor, numel should be 1.
        // Let's verify buffer size:
        assert_eq!(layout.buffer_size_bytes(), 4); // one f32 = 4 bytes
    }

    #[test]
    fn tensor_zeros() {
        let pool = SystemPool;
        let t = NumericTensor::<DynRank, SystemPool>::zeros(vec![2, 3], NumericDType::F32, &pool)
            .unwrap();
        assert_eq!(t.shape(), &vec![2, 3]);
        assert_eq!(t.dtype(), NumericDType::F32);
        assert_eq!(t.numel(), 6);
        assert_eq!(t.buffer().len(), 24);
        assert!(t.buffer().iter().all(|&b| b == 0));
    }

    #[test]
    fn write_read_elements_f32() {
        let pool = SystemPool;
        let mut t =
            NumericTensor::<DynRank, SystemPool>::zeros(vec![2, 3], NumericDType::F32, &pool)
                .unwrap();

        for i in 0..6 {
            t.write_element(
                i,
                NumericScalar::from_f64(i as f64 * 1.5).cast_to(NumericDType::F32),
            );
        }

        for i in 0..6 {
            let expected = NumericScalar::from_f32((i as f64 * 1.5) as f32);
            assert_eq!(t.read_element(i), expected, "element {i} mismatch");
        }
    }

    #[test]
    fn write_read_elements_i32() {
        let pool = SystemPool;
        let mut t =
            NumericTensor::<DynRank, SystemPool>::zeros(vec![4], NumericDType::I32, &pool).unwrap();

        t.write_element(0, NumericScalar::from_i32(-100));
        t.write_element(1, NumericScalar::from_i32(0));
        t.write_element(2, NumericScalar::from_i32(42));
        t.write_element(3, NumericScalar::from_i32(1000));

        assert_eq!(t.read_element(0), NumericScalar::from_i32(-100));
        assert_eq!(t.read_element(1), NumericScalar::from_i32(0));
        assert_eq!(t.read_element(2), NumericScalar::from_i32(42));
        assert_eq!(t.read_element(3), NumericScalar::from_i32(1000));
    }

    #[test]
    fn view_erases_pool() {
        let pool = SystemPool;
        let mut t =
            NumericTensor::<DynRank, SystemPool>::zeros(vec![3], NumericDType::F32, &pool).unwrap();
        t.write_element(1, NumericScalar::from_f32(7.0));

        let view: NumericTensorView<'_, DynRank> = t.view();
        assert_eq!(view.numel(), 3);
        assert_eq!(view.read_element(1), NumericScalar::from_f32(7.0));
    }

    #[test]
    fn tracked_pool_tensor() {
        use crate::pool::TrackedPool;

        let pool = TrackedPool::new(Some(1024));
        let t =
            NumericTensor::<DynRank, TrackedPool>::zeros(vec![4, 4], NumericDType::F32, &pool)
                .unwrap();
        assert_eq!(pool.bytes_in_use(), 64);

        drop(t);
        assert_eq!(pool.bytes_in_use(), 0);
    }

    #[test]
    fn tracked_pool_budget_prevents_allocation() {
        use crate::pool::{AllocationError, TrackedPool};

        let pool = TrackedPool::new(Some(32));
        let result =
            NumericTensor::<DynRank, TrackedPool>::zeros(vec![100], NumericDType::F32, &pool);
        assert!(matches!(result, Err(AllocationError::BudgetExceeded { .. })));
    }

    #[test]
    fn bf16_tensor() {
        let pool = SystemPool;
        let mut t =
            NumericTensor::<DynRank, SystemPool>::zeros(vec![2], NumericDType::BF16, &pool)
                .unwrap();
        assert_eq!(t.buffer().len(), 4);

        t.write_element(0, NumericScalar::from_bf16(bf16::from_f32(1.5)));
        t.write_element(1, NumericScalar::from_bf16(bf16::from_f32(-2.0)));

        assert_eq!(
            t.read_element(0),
            NumericScalar::from_bf16(bf16::from_f32(1.5))
        );
        assert_eq!(
            t.read_element(1),
            NumericScalar::from_bf16(bf16::from_f32(-2.0))
        );
    }

    #[test]
    fn layout_read_write_without_tensor() {
        // TensorLayout can read/write from raw byte slices directly
        let layout = TensorLayout::<DynRank>::row_major(vec![3], NumericDType::F32);
        let mut buf = vec![0u8; layout.buffer_size_bytes()];

        layout.write_element(&mut buf, 0, NumericScalar::from_f32(1.0));
        layout.write_element(&mut buf, 1, NumericScalar::from_f32(2.0));
        layout.write_element(&mut buf, 2, NumericScalar::from_f32(3.0));

        assert_eq!(layout.read_element(&buf, 0), NumericScalar::from_f32(1.0));
        assert_eq!(layout.read_element(&buf, 1), NumericScalar::from_f32(2.0));
        assert_eq!(layout.read_element(&buf, 2), NumericScalar::from_f32(3.0));
    }

    #[test]
    fn layout_accessors() {
        let layout = TensorLayout::<DynRank>::row_major(vec![2, 3], NumericDType::BF16);
        assert_eq!(layout.shape(), &vec![2, 3]);
        assert_eq!(layout.element_dtype(), NumericDType::BF16);
        assert_eq!(layout.numel(), 6);
        assert_eq!(layout.buffer_size_bytes(), 12); // 6 * 2 bytes
    }

    #[test]
    #[should_panic(expected = "write_element: scalar dtype")]
    fn write_element_dtype_mismatch_panics() {
        let layout = TensorLayout::<DynRank>::row_major(vec![2], NumericDType::F32);
        let mut buf = vec![0u8; layout.buffer_size_bytes()];
        // Writing an I32 scalar into an F32 layout should panic
        layout.write_element(&mut buf, 0, NumericScalar::from_i32(42));
    }
}
