//! Pool-managed numeric tensor with dtype-erased storage.
//!
//! [`TensorLayout`] is the core abstraction: a complete description of how a
//! tensor is packed into a contiguous memory region. It knows the shape, dtype,
//! and access pattern, and can read/write elements given a `&[u8]` span.
//!
//! [`NumericTensor`] and [`NumericTensorView`] are thin wrappers pairing a
//! `TensorLayout` with actual memory (pool buffer or byte slice).

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::numeric_dtype::NumericDType;
use crate::numeric_scalar::{NumericScalar, NumericScalarView, NumericScalarViewMut};
use crate::pool::Pool;
use crate::tensor_rank::{DimContainer, Rank};

// ---------------------------------------------------------------------------
// KQuantVariant
// ---------------------------------------------------------------------------

/// K-quant format variant. Each has a unique internal block structure
/// (hierarchical sub-blocks with format-specific bit packing).
///
/// All variants use 256-element super-blocks.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KQuantVariant {
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
}

impl KQuantVariant {
    /// Super-block size (always 256 for K-quants).
    pub const BLOCK_SIZE: usize = 256;

    /// Bytes per 256-element super-block.
    pub fn block_bytes(self) -> usize {
        match self {
            KQuantVariant::Q2_K => 84,
            KQuantVariant::Q3_K => 110,
            KQuantVariant::Q4_K => 144,
            KQuantVariant::Q5_K => 176,
            KQuantVariant::Q6_K => 210,
            KQuantVariant::Q8_K => 292,
        }
    }

    /// Nominal bits per weight (for display/comparison, not exact).
    pub fn weight_bits(self) -> u8 {
        match self {
            KQuantVariant::Q2_K => 2,
            KQuantVariant::Q3_K => 3,
            KQuantVariant::Q4_K => 4,
            KQuantVariant::Q5_K => 5,
            KQuantVariant::Q6_K => 6,
            KQuantVariant::Q8_K => 8,
        }
    }
}

impl fmt::Display for KQuantVariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KQuantVariant::Q2_K => write!(f, "Q2_K"),
            KQuantVariant::Q3_K => write!(f, "Q3_K"),
            KQuantVariant::Q4_K => write!(f, "Q4_K"),
            KQuantVariant::Q5_K => write!(f, "Q5_K"),
            KQuantVariant::Q6_K => write!(f, "Q6_K"),
            KQuantVariant::Q8_K => write!(f, "Q8_K"),
        }
    }
}

// ---------------------------------------------------------------------------
// TensorLayout
// ---------------------------------------------------------------------------

/// Complete description of how a tensor is packed into a contiguous memory region.
///
/// Each arm is fully self-contained — it carries shape, dtype/format info, and
/// all metadata needed to compute buffer sizes, element offsets, and read/write
/// elements from a `&[u8]` span. No back-references needed.
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
        /// Per-dimension strides in bits. Always positive.
        strides: R::KnownDims,
    },

    /// Simple block quantization (GGUF legacy Q-types).
    ///
    /// Block structure: `[f16 scale][optional f16 min][weight_bits × 32 packed weights]`
    /// - Symmetric (`has_min=false`): `value = scale * (weight - 2^(weight_bits-1))`
    /// - Asymmetric (`has_min=true`): `value = scale * weight + min`
    ///
    /// Covers Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1.
    SimpleBlockQuant {
        shape: R::KnownDims,
        /// Bits per quantized weight (4, 5, or 8).
        weight_bits: u8,
        /// Whether each block has a per-block minimum (f16).
        /// False = symmetric quantization, true = asymmetric.
        has_min: bool,
    },

    /// K-quant hierarchical block format (GGUF K-types).
    ///
    /// 256-element super-blocks with format-specific internal structure
    /// (sub-block scales, mixed bit packing). Each variant has unique
    /// internals that aren't parametrizable.
    KQuant {
        shape: R::KnownDims,
        variant: KQuantVariant,
    },
}

// -- SimpleBlockQuant constants --

impl<R: Rank> TensorLayout<R> {
    /// Block size for simple block quantization (always 32 elements).
    const SIMPLE_BLOCK_SIZE: usize = 32;
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

    /// Create a simple block quantization layout.
    pub fn simple_block_quant(shape: R::KnownDims, weight_bits: u8, has_min: bool) -> Self {
        assert!(
            matches!(weight_bits, 4 | 5 | 8),
            "SimpleBlockQuant weight_bits must be 4, 5, or 8, got {weight_bits}"
        );
        TensorLayout::SimpleBlockQuant {
            shape,
            weight_bits,
            has_min,
        }
    }

    /// Create a K-quant layout.
    pub fn k_quant(shape: R::KnownDims, variant: KQuantVariant) -> Self {
        TensorLayout::KQuant { shape, variant }
    }
}

// -- Universal accessors --

impl<R: Rank> TensorLayout<R> {
    pub fn shape(&self) -> &R::KnownDims {
        match self {
            TensorLayout::ElementStrided { shape, .. }
            | TensorLayout::SimpleBlockQuant { shape, .. }
            | TensorLayout::KQuant { shape, .. } => shape,
        }
    }

    /// The numeric dtype for element-wise access.
    /// Quantized formats dequantize to F32.
    pub fn element_dtype(&self) -> NumericDType {
        match self {
            TensorLayout::ElementStrided { dtype, .. } => *dtype,
            TensorLayout::SimpleBlockQuant { .. } | TensorLayout::KQuant { .. } => {
                NumericDType::F32
            }
        }
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.shape().as_slice().iter().product::<u64>() as usize
    }

    /// Block size for quantized formats, or 1 for element-strided.
    pub fn block_size(&self) -> usize {
        match self {
            TensorLayout::ElementStrided { .. } => 1,
            TensorLayout::SimpleBlockQuant { .. } => Self::SIMPLE_BLOCK_SIZE,
            TensorLayout::KQuant { .. } => KQuantVariant::BLOCK_SIZE,
        }
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
                    return dtype.bytes_per_element();
                }
                let element_bits = dtype.total_bits() as u64;
                let extent: u64 = dims
                    .iter()
                    .zip(strides.as_slice().iter())
                    .map(|(&dim, &stride)| if dim > 0 { (dim - 1) * stride } else { 0 })
                    .sum::<u64>()
                    + element_bits;
                ((extent + 7) / 8) as usize
            }
            TensorLayout::SimpleBlockQuant {
                shape,
                weight_bits,
                has_min,
            } => {
                let numel: usize = shape.as_slice().iter().product::<u64>() as usize;
                let block_bytes = simple_block_bytes(*weight_bits, *has_min);
                (numel / Self::SIMPLE_BLOCK_SIZE) * block_bytes
            }
            TensorLayout::KQuant { shape, variant } => {
                let numel: usize = shape.as_slice().iter().product::<u64>() as usize;
                (numel / KQuantVariant::BLOCK_SIZE) * variant.block_bytes()
            }
        }
    }
}

/// Bytes per 32-element block for simple block quantization.
/// Layout: `[f16 scale (2B)] [optional f16 min (2B)] [weight_bits * 32 / 8 bytes]`
fn simple_block_bytes(weight_bits: u8, has_min: bool) -> usize {
    let scale_bytes = 2; // f16
    let min_bytes = if has_min { 2 } else { 0 }; // optional f16
    let weight_bytes = (weight_bits as usize * 32) / 8;
    scale_bytes + min_bytes + weight_bytes
}

// -- Element access --

impl<R: Rank> TensorLayout<R> {
    /// Read a single element by flat index from a byte buffer.
    pub fn read_element(&self, data: &[u8], flat_index: usize) -> NumericScalar {
        match self {
            TensorLayout::ElementStrided {
                shape,
                dtype,
                strides,
            } => {
                let bit_offset =
                    flat_to_bit_offset(flat_index, shape.as_slice(), strides.as_slice());
                NumericScalarView {
                    data,
                    bit_offset,
                    dtype: *dtype,
                }
                .to_owned_scalar()
            }
            TensorLayout::SimpleBlockQuant { .. } | TensorLayout::KQuant { .. } => {
                // TODO: per-element dequantization.
                // For v1, quantized tensors are dequantized in bulk via PackedTensor.
                panic!("Per-element read_element not yet implemented for quantized formats");
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
                let bit_offset =
                    flat_to_bit_offset(flat_index, shape.as_slice(), strides.as_slice());
                let mut view = NumericScalarViewMut {
                    data,
                    bit_offset,
                    dtype: *dtype,
                };
                view.write_scalar(&value);
            }
            TensorLayout::SimpleBlockQuant { .. } | TensorLayout::KQuant { .. } => {
                panic!("write_element not supported for quantized formats");
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
// Conversion from legacy PackedFormat
// ---------------------------------------------------------------------------

use crate::migration::packed_format::PackedFormat;

impl<R: Rank> TensorLayout<R> {
    /// Convert a legacy PackedFormat + shape into the appropriate TensorLayout arm.
    pub fn from_legacy_packed(shape: R::KnownDims, format: PackedFormat) -> Self {
        match format {
            PackedFormat::Q4_0 => Self::simple_block_quant(shape, 4, false),
            PackedFormat::Q4_1 => Self::simple_block_quant(shape, 4, true),
            PackedFormat::Q5_0 => Self::simple_block_quant(shape, 5, false),
            PackedFormat::Q5_1 => Self::simple_block_quant(shape, 5, true),
            PackedFormat::Q8_0 => Self::simple_block_quant(shape, 8, false),
            PackedFormat::Q8_1 => Self::simple_block_quant(shape, 8, true),
            PackedFormat::Q2_K => Self::k_quant(shape, KQuantVariant::Q2_K),
            PackedFormat::Q3_K => Self::k_quant(shape, KQuantVariant::Q3_K),
            PackedFormat::Q4_K => Self::k_quant(shape, KQuantVariant::Q4_K),
            PackedFormat::Q5_K => Self::k_quant(shape, KQuantVariant::Q5_K),
            PackedFormat::Q6_K => Self::k_quant(shape, KQuantVariant::Q6_K),
            PackedFormat::Q8_K => Self::k_quant(shape, KQuantVariant::Q8_K),
        }
    }

    /// Convert back to a legacy PackedFormat (for quantized arms only).
    pub fn to_legacy_packed(&self) -> Option<PackedFormat> {
        match self {
            TensorLayout::ElementStrided { .. } => None,
            TensorLayout::SimpleBlockQuant {
                weight_bits,
                has_min,
                ..
            } => match (*weight_bits, *has_min) {
                (4, false) => Some(PackedFormat::Q4_0),
                (4, true) => Some(PackedFormat::Q4_1),
                (5, false) => Some(PackedFormat::Q5_0),
                (5, true) => Some(PackedFormat::Q5_1),
                (8, false) => Some(PackedFormat::Q8_0),
                (8, true) => Some(PackedFormat::Q8_1),
                _ => None,
            },
            TensorLayout::KQuant { variant, .. } => Some(match variant {
                KQuantVariant::Q2_K => PackedFormat::Q2_K,
                KQuantVariant::Q3_K => PackedFormat::Q3_K,
                KQuantVariant::Q4_K => PackedFormat::Q4_K,
                KQuantVariant::Q5_K => PackedFormat::Q5_K,
                KQuantVariant::Q6_K => PackedFormat::Q6_K,
                KQuantVariant::Q8_K => PackedFormat::Q8_K,
            }),
        }
    }
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

impl<'a, R: Rank, P: Pool + 'a> Clone for NumericTensor<'a, R, P>
where
    P::Buffer<'a>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            buffer: self.buffer.clone(),
            layout: self.layout.clone(),
        }
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

    // -- ElementStrided --

    #[test]
    fn row_major_strides_f32() {
        let layout = TensorLayout::<DynRank>::row_major(vec![2, 3], NumericDType::F32);
        match &layout {
            TensorLayout::ElementStrided { strides, .. } => {
                assert_eq!(*strides, vec![96, 32]);
            }
            _ => panic!("expected ElementStrided"),
        }
        assert_eq!(layout.buffer_size_bytes(), 24);
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
        assert_eq!(layout.buffer_size_bytes(), 1);
    }

    #[test]
    fn scalar_tensor_has_nonzero_size() {
        let layout = TensorLayout::<DynRank>::row_major(vec![], NumericDType::F32);
        assert_eq!(layout.buffer_size_bytes(), 4);
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
        assert_eq!(t.read_element(0), NumericScalar::from_bf16(bf16::from_f32(1.5)));
        assert_eq!(t.read_element(1), NumericScalar::from_bf16(bf16::from_f32(-2.0)));
    }

    #[test]
    fn layout_read_write_without_tensor() {
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
        assert_eq!(layout.buffer_size_bytes(), 12);
    }

    #[test]
    #[should_panic(expected = "write_element: scalar dtype")]
    fn write_element_dtype_mismatch_panics() {
        let layout = TensorLayout::<DynRank>::row_major(vec![2], NumericDType::F32);
        let mut buf = vec![0u8; layout.buffer_size_bytes()];
        layout.write_element(&mut buf, 0, NumericScalar::from_i32(42));
    }

    // -- SimpleBlockQuant --

    #[test]
    fn simple_block_quant_buffer_sizes() {
        // Verify parametric block_bytes matches the legacy PackedFormat values
        assert_eq!(simple_block_bytes(4, false), 18);  // Q4_0
        assert_eq!(simple_block_bytes(4, true), 20);   // Q4_1
        assert_eq!(simple_block_bytes(5, false), 22);  // Q5_0
        assert_eq!(simple_block_bytes(5, true), 24);   // Q5_1
        assert_eq!(simple_block_bytes(8, false), 34);  // Q8_0
        assert_eq!(simple_block_bytes(8, true), 36);   // Q8_1
    }

    #[test]
    fn simple_block_quant_layout() {
        // 1024 elements in Q4_0: 1024/32 = 32 blocks × 18 bytes = 576
        let layout =
            TensorLayout::<DynRank>::simple_block_quant(vec![1024], 4, false);
        assert_eq!(layout.buffer_size_bytes(), 576);
        assert_eq!(layout.element_dtype(), NumericDType::F32);
        assert_eq!(layout.numel(), 1024);
        assert_eq!(layout.block_size(), 32);
    }

    // -- KQuant --

    #[test]
    fn k_quant_buffer_sizes() {
        // 256 elements = 1 block
        let cases: &[(KQuantVariant, usize)] = &[
            (KQuantVariant::Q2_K, 84),
            (KQuantVariant::Q3_K, 110),
            (KQuantVariant::Q4_K, 144),
            (KQuantVariant::Q5_K, 176),
            (KQuantVariant::Q6_K, 210),
            (KQuantVariant::Q8_K, 292),
        ];
        for &(variant, expected_bytes) in cases {
            let layout = TensorLayout::<DynRank>::k_quant(vec![256], variant);
            assert_eq!(
                layout.buffer_size_bytes(),
                expected_bytes,
                "{variant} buffer size"
            );
            assert_eq!(layout.block_size(), 256);
        }
    }

    #[test]
    fn k_quant_multi_block() {
        // 512 elements = 2 blocks of Q4_K (144 bytes each)
        let layout = TensorLayout::<DynRank>::k_quant(vec![512], KQuantVariant::Q4_K);
        assert_eq!(layout.buffer_size_bytes(), 288);
    }

    // -- Legacy conversion --

    #[test]
    fn legacy_packed_format_roundtrip() {
        let legacy_formats = [
            PackedFormat::Q4_0,
            PackedFormat::Q4_1,
            PackedFormat::Q5_0,
            PackedFormat::Q5_1,
            PackedFormat::Q8_0,
            PackedFormat::Q8_1,
            PackedFormat::Q2_K,
            PackedFormat::Q3_K,
            PackedFormat::Q4_K,
            PackedFormat::Q5_K,
            PackedFormat::Q6_K,
            PackedFormat::Q8_K,
        ];
        for fmt in legacy_formats {
            let shape: Vec<u64> = vec![256]; // minimum for K-quants
            let layout = TensorLayout::<DynRank>::from_legacy_packed(shape.clone(), fmt);
            let back = layout.to_legacy_packed().unwrap();
            assert_eq!(fmt, back, "roundtrip failed for {fmt}");
        }
    }

    #[test]
    fn legacy_buffer_size_matches() {
        // Verify our buffer_size_bytes matches the legacy PackedFormat::storage_bytes
        let cases: &[(PackedFormat, usize)] = &[
            (PackedFormat::Q4_0, 1024),
            (PackedFormat::Q4_K, 256),
            (PackedFormat::Q6_K, 512),
        ];
        for &(fmt, numel) in cases {
            let shape: Vec<u64> = vec![numel as u64];
            let legacy_bytes = fmt.storage_bytes(numel);
            let layout = TensorLayout::<DynRank>::from_legacy_packed(shape, fmt);
            assert_eq!(
                layout.buffer_size_bytes(),
                legacy_bytes,
                "buffer size mismatch for {fmt} with {numel} elements"
            );
        }
    }
}
