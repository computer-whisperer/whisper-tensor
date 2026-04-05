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
#[allow(non_camel_case_types)]
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
// TensorFormat
// ---------------------------------------------------------------------------

/// What kind of data a tensor contains — enough to construct a [`TensorLayout`]
/// given a shape.
///
/// This is the shape-independent part of a tensor's storage description.
/// Combined with a shape, it fully determines the memory layout.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TensorFormat {
    /// Standard element-strided numeric tensor.
    Element(NumericDType),
    /// Simple block quantization (GGUF Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1).
    SimpleBlockQuant {
        /// Bits per quantized weight (4, 5, or 8).
        weight_bits: u8,
        /// Whether each block has a per-block minimum (f16).
        has_min: bool,
    },
    /// K-quant hierarchical block format (GGUF Q2_K through Q8_K).
    KQuant(KQuantVariant),
}

impl TensorFormat {
    /// The dtype you get when reading individual elements.
    /// Quantized formats dequantize to F32.
    pub fn element_dtype(&self) -> NumericDType {
        match self {
            TensorFormat::Element(dt) => *dt,
            TensorFormat::SimpleBlockQuant { .. } | TensorFormat::KQuant(_) => NumericDType::F32,
        }
    }

    /// Whether this is a quantized (block-packed) format.
    pub fn is_quantized(&self) -> bool {
        !matches!(self, TensorFormat::Element(_))
    }

    /// Construct a row-major [`TensorLayout`] for this format and shape.
    pub fn to_layout<R: Rank>(&self, shape: R::KnownDims) -> TensorLayout<R> {
        match self {
            TensorFormat::Element(dt) => TensorLayout::row_major(shape, *dt),
            TensorFormat::SimpleBlockQuant {
                weight_bits,
                has_min,
            } => TensorLayout::simple_block_quant(shape, *weight_bits, *has_min),
            TensorFormat::KQuant(variant) => TensorLayout::k_quant(shape, *variant),
        }
    }
}

impl From<NumericDType> for TensorFormat {
    fn from(dt: NumericDType) -> Self {
        TensorFormat::Element(dt)
    }
}

impl fmt::Display for TensorFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorFormat::Element(dt) => write!(f, "{dt}"),
            TensorFormat::SimpleBlockQuant {
                weight_bits,
                has_min,
            } => {
                let suffix = if *has_min { "1" } else { "0" };
                write!(f, "Q{weight_bits}_{suffix}")
            }
            TensorFormat::KQuant(v) => write!(f, "{v}"),
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
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TensorLayout<R: Rank> {
    /// Uniform-stride elements of a single numeric type.
    /// Strides (in bits, per dimension) fully describe element spacing.
    ElementStrided {
        shape: R::KnownDims,
        dtype: NumericDType,
        /// Per-dimension strides in bits. Always positive.
        strides: R::KnownDims,
        /// Bit offset of the first element from the start of the buffer.
        /// Non-zero after slicing. Element access adds this before computing
        /// the final position.
        offset_bits: u64,
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
            offset_bits: 0,
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

    /// Returns true if the layout is contiguous from the start of the buffer:
    /// row-major strides, zero offset. A memcpy of the buffer reproduces the tensor.
    ///
    /// Always true for quantized formats (they have no offset/stride concept).
    pub fn is_contiguous(&self) -> bool {
        match self {
            TensorLayout::ElementStrided {
                shape,
                dtype,
                strides,
                offset_bits,
            } => {
                if *offset_bits != 0 {
                    return false;
                }
                let dims = shape.as_slice();
                let stride_slice = strides.as_slice();
                let element_bits = dtype.total_bits() as u64;
                // Check row-major: last stride == element_bits,
                // each earlier stride == next_stride * next_dim.
                if dims.is_empty() {
                    return true;
                }
                if stride_slice[dims.len() - 1] != element_bits {
                    return false;
                }
                for i in (0..dims.len() - 1).rev() {
                    if stride_slice[i] != stride_slice[i + 1] * dims[i + 1] {
                        return false;
                    }
                }
                true
            }
            TensorLayout::SimpleBlockQuant { .. } | TensorLayout::KQuant { .. } => true,
        }
    }

    /// Total buffer size in bytes needed to hold this tensor.
    pub fn buffer_size_bytes(&self) -> usize {
        match self {
            TensorLayout::ElementStrided {
                shape,
                dtype,
                strides,
                offset_bits,
            } => {
                let dims = shape.as_slice();
                if dims.is_empty() {
                    return (*offset_bits + dtype.total_bits() as u64).div_ceil(8) as usize;
                }
                let element_bits = dtype.total_bits() as u64;
                let extent: u64 = offset_bits
                    + dims
                        .iter()
                        .zip(strides.as_slice().iter())
                        .map(|(&dim, &stride)| if dim > 0 { (dim - 1) * stride } else { 0 })
                        .sum::<u64>()
                    + element_bits;
                extent.div_ceil(8) as usize
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

// -- Index coordinate helpers --

impl<R: Rank> TensorLayout<R> {
    /// Decompose a flat (row-major) element index into multi-dimensional coordinates.
    ///
    /// For a tensor with shape `[d0, d1, d2]`, flat index `k` maps to
    /// `[k / (d1*d2), (k / d2) % d1, k % d2]`.
    pub fn flat_to_coords(&self, flat_index: usize) -> Vec<usize> {
        let dims = self.shape().as_slice();
        let rank = dims.len();
        let mut coords = vec![0usize; rank];
        let mut remaining = flat_index;
        for i in (0..rank).rev() {
            let d = dims[i] as usize;
            coords[i] = remaining % d;
            remaining /= d;
        }
        coords
    }

    /// Convert multi-dimensional coordinates to a flat (row-major) element index.
    ///
    /// Inverse of [`flat_to_coords`].
    pub fn coords_to_flat(&self, coords: &[usize]) -> usize {
        let dims = self.shape().as_slice();
        let mut flat = 0usize;
        let mut stride = 1usize;
        for i in (0..dims.len()).rev() {
            flat += coords[i] * stride;
            stride *= dims[i] as usize;
        }
        flat
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
                offset_bits,
            } => {
                let bit_offset = *offset_bits as usize
                    + flat_to_bit_offset(flat_index, shape.as_slice(), strides.as_slice());
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
                offset_bits,
            } => {
                assert_eq!(
                    value.dtype(),
                    *dtype,
                    "write_element: scalar dtype {:?} != layout dtype {:?}",
                    value.dtype(),
                    dtype
                );
                let bit_offset = *offset_bits as usize
                    + flat_to_bit_offset(flat_index, shape.as_slice(), strides.as_slice());
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

    /// Write a single element without dtype checking.
    ///
    /// # Safety
    /// Caller must ensure `value.dtype() == self.element_dtype()`.
    #[inline(always)]
    pub unsafe fn write_element_unchecked(
        &self,
        data: &mut [u8],
        flat_index: usize,
        value: NumericScalar,
    ) {
        match self {
            TensorLayout::ElementStrided {
                shape,
                strides,
                offset_bits,
                dtype,
            } => {
                let bit_offset = *offset_bits as usize
                    + flat_to_bit_offset(flat_index, shape.as_slice(), strides.as_slice());
                let mut view = NumericScalarViewMut {
                    data,
                    bit_offset,
                    dtype: *dtype,
                };
                view.write_scalar(&value);
            }
            _ => unsafe { std::hint::unreachable_unchecked() },
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
// TensorLayoutError
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum TensorLayoutError {
    #[error("slice not supported for quantized layouts")]
    QuantizedSliceUnsupported,
    #[error("slice ranges length {got} does not match rank {expected}")]
    SliceRankMismatch { got: usize, expected: usize },
    #[error("slice range {start}..{end} out of bounds for dimension {dim} (size {dim_size})")]
    SliceOutOfBounds {
        dim: usize,
        start: u64,
        end: u64,
        dim_size: u64,
    },
    #[error("slice range {start}..{end} is empty or inverted on dimension {dim}")]
    SliceEmpty { dim: usize, start: u64, end: u64 },
    #[error("transpose not supported for quantized layouts")]
    QuantizedTransposeUnsupported,
    #[error("transpose permutation length {got} does not match rank {expected}")]
    TransposeRankMismatch { got: usize, expected: usize },
    #[error("transpose permutation is not a valid permutation of 0..{rank}")]
    TransposeInvalidPerm { rank: usize },
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

    /// Zero-copy flatten to a 1D view. The resulting view has the same data
    /// and element count, just reinterpreted as a single dimension.
    ///
    /// Returns `None` for non-contiguous ElementStrided layouts (e.g. after
    /// slicing) since the flat index mapping would be incorrect.
    /// Quantized layouts are always contiguous and always succeed.
    pub fn flatten(&self) -> Option<NumericTensorView<'a, crate::tensor_rank::DynRank>> {
        let numel = self.numel() as u64;
        let layout = match &self.layout {
            TensorLayout::ElementStrided {
                dtype, offset_bits, ..
            } => {
                if !self.layout.is_contiguous() || *offset_bits != 0 {
                    return None;
                }
                TensorLayout::row_major(vec![numel], *dtype)
            }
            TensorLayout::SimpleBlockQuant {
                weight_bits,
                has_min,
                ..
            } => TensorLayout::SimpleBlockQuant {
                shape: vec![numel],
                weight_bits: *weight_bits,
                has_min: *has_min,
            },
            TensorLayout::KQuant { variant, .. } => TensorLayout::KQuant {
                shape: vec![numel],
                variant: *variant,
            },
        };
        Some(NumericTensorView {
            data: self.data,
            layout,
        })
    }

    /// Zero-copy slice: returns a view into a sub-region.
    ///
    /// `ranges` has one `(start, end)` pair per dimension (end is exclusive).
    /// Only supported for `ElementStrided` layouts.
    pub fn slice(
        &self,
        ranges: &[(u64, u64)],
    ) -> Result<NumericTensorView<'a, R>, TensorLayoutError> {
        match &self.layout {
            TensorLayout::ElementStrided {
                shape,
                dtype,
                strides,
                offset_bits,
            } => {
                let dims = shape.as_slice();
                if ranges.len() != dims.len() {
                    return Err(TensorLayoutError::SliceRankMismatch {
                        got: ranges.len(),
                        expected: dims.len(),
                    });
                }

                let stride_slice = strides.as_slice();
                let mut new_offset = *offset_bits;
                let mut new_shape_vec = Vec::with_capacity(dims.len());

                for (i, &(start, end)) in ranges.iter().enumerate() {
                    if start >= end {
                        return Err(TensorLayoutError::SliceEmpty { dim: i, start, end });
                    }
                    if end > dims[i] {
                        return Err(TensorLayoutError::SliceOutOfBounds {
                            dim: i,
                            start,
                            end,
                            dim_size: dims[i],
                        });
                    }
                    new_offset += start * stride_slice[i];
                    new_shape_vec.push(end - start);
                }

                let new_shape = R::KnownDims::try_from_slice(&new_shape_vec)
                    .expect("slice output has same rank as input");

                Ok(NumericTensorView {
                    data: self.data,
                    layout: TensorLayout::ElementStrided {
                        shape: new_shape,
                        dtype: *dtype,
                        strides: strides.clone(),
                        offset_bits: new_offset,
                    },
                })
            }
            TensorLayout::SimpleBlockQuant { .. } | TensorLayout::KQuant { .. } => {
                Err(TensorLayoutError::QuantizedSliceUnsupported)
            }
        }
    }

    /// Zero-copy transpose: permutes dimensions according to `perm`.
    ///
    /// `perm` maps output dimension → input dimension. For example,
    /// `perm = &[1, 0]` swaps the two dimensions of a 2D tensor.
    /// Only supported for `ElementStrided` layouts.
    pub fn transpose(&self, perm: &[usize]) -> Result<NumericTensorView<'a, R>, TensorLayoutError> {
        match &self.layout {
            TensorLayout::ElementStrided {
                shape,
                dtype,
                strides,
                offset_bits,
            } => {
                let dims = shape.as_slice();
                let rank = dims.len();
                if perm.len() != rank {
                    return Err(TensorLayoutError::TransposeRankMismatch {
                        got: perm.len(),
                        expected: rank,
                    });
                }
                // Validate permutation
                let mut seen = vec![false; rank];
                for &p in perm {
                    if p >= rank || seen[p] {
                        return Err(TensorLayoutError::TransposeInvalidPerm { rank });
                    }
                    seen[p] = true;
                }

                let stride_slice = strides.as_slice();
                let new_shape_vec: Vec<u64> = perm.iter().map(|&p| dims[p]).collect();
                let new_stride_vec: Vec<u64> = perm.iter().map(|&p| stride_slice[p]).collect();

                Ok(NumericTensorView {
                    data: self.data,
                    layout: TensorLayout::ElementStrided {
                        shape: R::KnownDims::try_from_slice(&new_shape_vec)
                            .expect("transpose preserves rank"),
                        dtype: *dtype,
                        strides: R::KnownDims::try_from_slice(&new_stride_vec)
                            .expect("transpose preserves rank"),
                        offset_bits: *offset_bits,
                    },
                })
            }
            TensorLayout::SimpleBlockQuant { .. } | TensorLayout::KQuant { .. } => {
                Err(TensorLayoutError::QuantizedTransposeUnsupported)
            }
        }
    }

    /// Materialize this view into a new pool-owned tensor.
    ///
    /// Fast path: if the layout is contiguous (row-major, zero offset), the
    /// buffer is memcpy'd directly. Otherwise elements are copied one by one
    /// into a fresh row-major tensor.
    pub fn to_tensor<'p, P: Pool>(
        &self,
        pool: &'p P,
    ) -> Result<NumericTensor<'p, R, P>, crate::pool::AllocationError> {
        if self.layout.is_contiguous() {
            // Memcpy path — layout already describes a dense row-major buffer.
            let size = self.layout.buffer_size_bytes();
            let mut buffer = pool.allocate(size)?;
            buffer[..size].copy_from_slice(&self.data[..size]);
            Ok(NumericTensor {
                buffer,
                layout: self.layout.clone(),
            })
        } else {
            // Element-wise copy into a fresh row-major layout.
            let shape = self.layout.shape().clone();
            let dtype = self.layout.element_dtype();
            let out_layout = TensorLayout::row_major(shape, dtype);
            let mut buffer = pool.allocate(out_layout.buffer_size_bytes())?;
            let numel = self.layout.numel();
            for i in 0..numel {
                let scalar = self.layout.read_element(self.data, i);
                out_layout.write_element(&mut buffer, i, scalar);
            }
            Ok(NumericTensor {
                buffer,
                layout: out_layout,
            })
        }
    }

    /// Convert this view to match a target layout, borrowing if the byte layout
    /// already matches or copying into a new pool-allocated buffer if it doesn't.
    ///
    /// "Matches" means the view's bytes are already arranged according to `target`:
    /// same dtype, same shape, same strides, and zero offset.
    pub fn relayout<'p, P: Pool>(
        &self,
        target: TensorLayout<crate::tensor_rank::DynRank>,
        pool: &'p P,
    ) -> Result<
        NumericTensorCOW<'a, 'p, crate::tensor_rank::DynRank, P>,
        crate::pool::AllocationError,
    > {
        // Check if the view's current layout matches the target byte-for-byte.
        let matches = match (&self.layout, &target) {
            (
                TensorLayout::ElementStrided {
                    shape: s1,
                    dtype: d1,
                    strides: st1,
                    offset_bits: o1,
                },
                TensorLayout::ElementStrided {
                    shape: s2,
                    dtype: d2,
                    strides: st2,
                    offset_bits: o2,
                },
            ) => {
                d1 == d2
                    && o1 == o2
                    && s1.as_slice() == s2.as_slice()
                    && st1.as_slice() == st2.as_slice()
            }
            _ => false,
        };

        if matches {
            // Bytes are already in the right order — borrow with flat 1D layout.
            let numel = target.numel() as u64;
            let flat = TensorLayout::row_major(vec![numel], target.element_dtype());
            Ok(NumericTensorCOW::Borrowed(NumericTensorView {
                data: self.data,
                layout: flat,
            }))
        } else {
            // Must copy — rearrange elements from source layout to target layout.
            let numel = target.numel();
            let mut buffer = pool.allocate(target.buffer_size_bytes())?;

            // Byte-aligned fast path: both ElementStrided, same dtype, zero offset,
            // all strides byte-aligned. Copy raw bytes without scalar abstraction.
            if let (
                TensorLayout::ElementStrided {
                    shape: src_shape,
                    dtype: src_dtype,
                    strides: src_strides,
                    offset_bits: 0,
                },
                TensorLayout::ElementStrided {
                    dtype: dst_dtype,
                    strides: dst_strides,
                    offset_bits: 0,
                    ..
                },
            ) = (&self.layout, &target)
                && src_dtype == dst_dtype
                && src_strides.as_slice().iter().all(|s| s % 8 == 0)
                && dst_strides.as_slice().iter().all(|s| s % 8 == 0)
            {
                let bpe = src_dtype.bytes_per_element();
                let src_dims = src_shape.as_slice();
                let src_byte_strides: Vec<usize> = src_strides
                    .as_slice()
                    .iter()
                    .map(|s| (*s / 8) as usize)
                    .collect();
                let dst_byte_strides: Vec<usize> = dst_strides
                    .as_slice()
                    .iter()
                    .map(|s| (*s / 8) as usize)
                    .collect();
                // Precompute row-major tail products for flat→coords decomposition.
                let ndim = src_dims.len();
                let mut tail_products = vec![1usize; ndim + 1];
                for i in (0..ndim).rev() {
                    tail_products[i] = tail_products[i + 1] * src_dims[i] as usize;
                }
                for flat in 0..numel {
                    let mut src_off = 0usize;
                    let mut dst_off = 0usize;
                    let mut rem = flat;
                    for d in 0..ndim {
                        let idx = rem / tail_products[d + 1];
                        rem %= tail_products[d + 1];
                        src_off += idx * src_byte_strides[d];
                        dst_off += idx * dst_byte_strides[d];
                    }
                    buffer[dst_off..dst_off + bpe]
                        .copy_from_slice(&self.data[src_off..src_off + bpe]);
                }

                let flat = TensorLayout::row_major(vec![numel as u64], *src_dtype);
                return Ok(NumericTensorCOW::Owned(NumericTensor {
                    buffer,
                    layout: flat,
                }));
            }

            // General fallback via scalar read/write.
            for i in 0..numel {
                let scalar = self.layout.read_element(self.data, i);
                // Safety: relayout preserves dtype — source and target have the same dtype
                // by construction (TAMI describes the same tensor data).
                unsafe {
                    target.write_element_unchecked(&mut buffer, i, scalar);
                }
            }
            let flat = TensorLayout::row_major(vec![numel as u64], target.element_dtype());
            Ok(NumericTensorCOW::Owned(NumericTensor {
                buffer,
                layout: flat,
            }))
        }
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
// NumericTensorCOW — borrow-or-own result from relayout
// ---------------------------------------------------------------------------

/// Copy-on-write tensor — either a borrowed view or a pool-allocated tensor.
/// Returned by [`NumericTensorView::relayout`].
#[derive(Debug)]
pub enum NumericTensorCOW<'a, 'p, R: Rank, P: Pool + 'p> {
    Borrowed(NumericTensorView<'a, R>),
    Owned(NumericTensor<'p, R, P>),
}

impl<'a, 'p, R: Rank, P: Pool + 'p> NumericTensorCOW<'a, 'p, R, P> {
    pub fn read_element(&self, index: usize) -> NumericScalar {
        match self {
            Self::Borrowed(v) => v.read_element(index),
            Self::Owned(t) => t.read_element(index),
        }
    }

    pub fn dtype(&self) -> NumericDType {
        match self {
            Self::Borrowed(v) => v.dtype(),
            Self::Owned(t) => t.dtype(),
        }
    }

    pub fn numel(&self) -> usize {
        match self {
            Self::Borrowed(v) => v.numel(),
            Self::Owned(t) => t.numel(),
        }
    }

    pub fn view(&self) -> NumericTensorView<'_, R> {
        match self {
            Self::Borrowed(v) => NumericTensorView::new(v.data(), v.layout().clone()),
            Self::Owned(t) => t.view(),
        }
    }

    /// Return a view with the COW's borrow lifetime `'a`.
    ///
    /// For `Borrowed`: returns the original view (lifetime `'a`).
    /// For `Owned`: the pool buffer outlives `'a` (requires `'p: 'a`),
    /// so the view is valid for `'a`.
    /// Return a view with the COW's borrow lifetime `'a`.
    ///
    /// For `Borrowed`: returns a view over the original `'a`-lifetime data.
    /// For `Owned`: the pool buffer outlives `'a` (requires `'p: 'a`),
    /// so the view is valid for `'a`.
    pub fn long_view(&self) -> NumericTensorView<'a, R>
    where
        'p: 'a,
    {
        match self {
            Self::Borrowed(v) => NumericTensorView::new(v.data, v.layout().clone()),
            Self::Owned(t) => {
                let data: &[u8] = t.buffer();
                // SAFETY: The pool buffer lives for 'p and 'p: 'a, so 'a is valid.
                let data: &'a [u8] = unsafe { &*(data as *const [u8]) };
                NumericTensorView::new(data, t.layout().clone())
            }
        }
    }

    pub fn layout(&self) -> &TensorLayout<R> {
        match self {
            Self::Borrowed(v) => v.layout(),
            Self::Owned(t) => t.layout(),
        }
    }

    pub fn shape(&self) -> &R::KnownDims {
        match self {
            Self::Borrowed(v) => v.shape(),
            Self::Owned(t) => t.shape(),
        }
    }

    pub fn to_i64_vec(&self) -> Vec<i64> {
        (0..self.numel())
            .map(|i| self.read_element(i).to_i64())
            .collect()
    }

    pub fn buffer(&self) -> &[u8] {
        match self {
            Self::Borrowed(v) => v.data(),
            Self::Owned(t) => t.buffer(),
        }
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

    /// Create a tensor by computing each element from its flat index.
    pub fn from_fn(
        shape: R::KnownDims,
        dtype: NumericDType,
        pool: &'a P,
        f: impl Fn(usize) -> NumericScalar,
    ) -> Result<Self, crate::pool::AllocationError> {
        let layout = TensorLayout::row_major(shape, dtype);
        let numel = layout.numel();
        let mut buffer = pool.allocate(layout.buffer_size_bytes())?;
        for i in 0..numel {
            layout.write_element(&mut buffer, i, f(i));
        }
        Ok(Self { buffer, layout })
    }

    /// Create a tensor from a slice of pre-computed scalars.
    ///
    /// Panics if `scalars.len() != shape.product()`.
    pub fn from_scalars(
        shape: R::KnownDims,
        dtype: NumericDType,
        pool: &'a P,
        scalars: &[NumericScalar],
    ) -> Result<Self, crate::pool::AllocationError> {
        let layout = TensorLayout::row_major(shape, dtype);
        assert_eq!(
            scalars.len(),
            layout.numel(),
            "from_scalars: {} scalars provided but shape has {} elements",
            scalars.len(),
            layout.numel()
        );
        let mut buffer = pool.allocate(layout.buffer_size_bytes())?;
        for (i, scalar) in scalars.iter().enumerate() {
            layout.write_element(&mut buffer, i, *scalar);
        }
        Ok(Self { buffer, layout })
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

    /// Reinterpret this tensor with a new layout of a different rank.
    ///
    /// The new layout must describe the same buffer size. This is a free
    /// operation — no data is copied, only the layout metadata changes.
    pub fn into_layout<R2: Rank>(self, layout: TensorLayout<R2>) -> NumericTensor<'a, R2, P> {
        debug_assert_eq!(
            self.layout.buffer_size_bytes(),
            layout.buffer_size_bytes(),
            "into_layout: buffer size mismatch ({} vs {})",
            self.layout.buffer_size_bytes(),
            layout.buffer_size_bytes(),
        );
        NumericTensor {
            buffer: self.buffer,
            layout,
        }
    }

    /// Zero-copy slice: returns a view into a sub-region of this tensor.
    ///
    /// See [`NumericTensorView::slice`] for details.
    pub fn slice(
        &self,
        ranges: &[(u64, u64)],
    ) -> Result<NumericTensorView<'_, R>, TensorLayoutError> {
        self.view().slice(ranges)
    }

    /// Zero-copy transpose: returns a view with permuted dimensions.
    ///
    /// See [`NumericTensorView::transpose`] for details.
    pub fn transpose(&self, perm: &[usize]) -> Result<NumericTensorView<'_, R>, TensorLayoutError> {
        self.view().transpose(perm)
    }

    /// Clone this tensor into a new pool-allocated tensor.
    ///
    /// See [`NumericTensorView::to_tensor`] for details.
    pub fn to_tensor<'p, P2: Pool>(
        &self,
        pool: &'p P2,
    ) -> Result<NumericTensor<'p, R, P2>, crate::pool::AllocationError> {
        self.view().to_tensor(pool)
    }

    /// Read all elements as i64 values. Useful for extracting shape/axes parameters.
    pub fn to_i64_vec(&self) -> Vec<i64> {
        (0..self.numel())
            .map(|i| self.read_element(i).to_i64())
            .collect()
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

impl<'a, R: Rank, P: Pool + 'a> Serialize for NumericTensor<'a, R, P>
where
    R::KnownDims: Serialize,
{
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut s = serializer.serialize_struct("NumericTensor", 2)?;
        s.serialize_field("layout", &self.layout)?;
        s.serialize_field("data", &*self.buffer)?;
        s.end()
    }
}

impl<'de> Deserialize<'de>
    for NumericTensor<'static, crate::tensor_rank::DynRank, crate::pool::SystemPool>
{
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::{self, MapAccess, SeqAccess, Visitor};

        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field {
            Layout,
            Data,
        }

        struct NumericTensorVisitor;

        impl<'de> Visitor<'de> for NumericTensorVisitor {
            type Value =
                NumericTensor<'static, crate::tensor_rank::DynRank, crate::pool::SystemPool>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a NumericTensor with layout and data fields")
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let layout: TensorLayout<crate::tensor_rank::DynRank> = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                let data: Vec<u8> = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(1, &self))?;
                let expected = layout.buffer_size_bytes();
                if data.len() != expected {
                    return Err(de::Error::custom(format!(
                        "buffer length {} does not match layout expected {}",
                        data.len(),
                        expected
                    )));
                }
                let mut buffer = crate::pool::SystemPool
                    .allocate(expected)
                    .map_err(de::Error::custom)?;
                (*buffer).copy_from_slice(&data);
                Ok(NumericTensor::from_parts(buffer, layout))
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut layout: Option<TensorLayout<crate::tensor_rank::DynRank>> = None;
                let mut data: Option<Vec<u8>> = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Layout => {
                            if layout.is_some() {
                                return Err(de::Error::duplicate_field("layout"));
                            }
                            layout = Some(map.next_value()?);
                        }
                        Field::Data => {
                            if data.is_some() {
                                return Err(de::Error::duplicate_field("data"));
                            }
                            data = Some(map.next_value()?);
                        }
                    }
                }
                let layout = layout.ok_or_else(|| de::Error::missing_field("layout"))?;
                let data = data.ok_or_else(|| de::Error::missing_field("data"))?;
                let expected = layout.buffer_size_bytes();
                if data.len() != expected {
                    return Err(de::Error::custom(format!(
                        "buffer length {} does not match layout expected {}",
                        data.len(),
                        expected
                    )));
                }
                let mut buffer = crate::pool::SystemPool
                    .allocate(expected)
                    .map_err(de::Error::custom)?;
                (*buffer).copy_from_slice(&data);
                Ok(NumericTensor::from_parts(buffer, layout))
            }
        }

        deserializer.deserialize_struct("NumericTensor", &["layout", "data"], NumericTensorVisitor)
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
        let t = NumericTensor::<DynRank, TrackedPool>::zeros(vec![4, 4], NumericDType::F32, &pool)
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
        assert!(matches!(
            result,
            Err(AllocationError::BudgetExceeded { .. })
        ));
    }

    #[test]
    fn bf16_tensor() {
        let pool = SystemPool;
        let mut t = NumericTensor::<DynRank, SystemPool>::zeros(vec![2], NumericDType::BF16, &pool)
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
        assert_eq!(simple_block_bytes(4, false), 18); // Q4_0
        assert_eq!(simple_block_bytes(4, true), 20); // Q4_1
        assert_eq!(simple_block_bytes(5, false), 22); // Q5_0
        assert_eq!(simple_block_bytes(5, true), 24); // Q5_1
        assert_eq!(simple_block_bytes(8, false), 34); // Q8_0
        assert_eq!(simple_block_bytes(8, true), 36); // Q8_1
    }

    #[test]
    fn simple_block_quant_layout() {
        // 1024 elements in Q4_0: 1024/32 = 32 blocks × 18 bytes = 576
        let layout = TensorLayout::<DynRank>::simple_block_quant(vec![1024], 4, false);
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

    // -- Slice --

    /// Helper: create a DynRank tensor with F32 values 0.0, 1.0, 2.0, ...
    fn make_f32_iota(shape: Vec<u64>) -> NumericTensor<'static, DynRank, SystemPool> {
        static POOL: SystemPool = SystemPool;
        let mut t =
            NumericTensor::<DynRank, SystemPool>::zeros(shape, NumericDType::F32, &POOL).unwrap();
        for i in 0..t.numel() {
            t.write_element(i, NumericScalar::from_f32(i as f32));
        }
        t
    }

    #[test]
    fn slice_1d_f32() {
        let t = make_f32_iota(vec![8]);
        let view = t.view();
        // Slice [2..5] → elements 2.0, 3.0, 4.0
        let sliced = view.slice(&[(2, 5)]).unwrap();
        assert_eq!(sliced.shape(), &vec![3u64]);
        assert_eq!(sliced.numel(), 3);
        assert_eq!(sliced.read_element(0), NumericScalar::from_f32(2.0));
        assert_eq!(sliced.read_element(1), NumericScalar::from_f32(3.0));
        assert_eq!(sliced.read_element(2), NumericScalar::from_f32(4.0));
    }

    #[test]
    fn slice_2d_f32() {
        // Shape [4, 6], row-major: element [i,j] = i*6 + j
        let t = make_f32_iota(vec![4, 6]);
        let view = t.view();
        // Slice [1..3, 2..5] → shape [2, 3]
        let sliced = view.slice(&[(1, 3), (2, 5)]).unwrap();
        assert_eq!(sliced.shape(), &vec![2u64, 3]);
        // sliced[0,0] = original[1,2] = 1*6+2 = 8
        assert_eq!(sliced.read_element(0), NumericScalar::from_f32(8.0));
        // sliced[0,2] = original[1,4] = 10
        assert_eq!(sliced.read_element(2), NumericScalar::from_f32(10.0));
        // sliced[1,0] = original[2,2] = 14
        assert_eq!(sliced.read_element(3), NumericScalar::from_f32(14.0));
        // sliced[1,2] = original[2,4] = 16
        assert_eq!(sliced.read_element(5), NumericScalar::from_f32(16.0));
    }

    #[test]
    fn slice_of_slice_composes() {
        let t = make_f32_iota(vec![10]);
        let view = t.view();
        // First slice: [2..8] → 2,3,4,5,6,7
        let s1 = view.slice(&[(2, 8)]).unwrap();
        // Second slice: [1..4] of s1 → 3,4,5
        let s2 = s1.slice(&[(1, 4)]).unwrap();
        assert_eq!(s2.shape(), &vec![3u64]);
        assert_eq!(s2.read_element(0), NumericScalar::from_f32(3.0));
        assert_eq!(s2.read_element(1), NumericScalar::from_f32(4.0));
        assert_eq!(s2.read_element(2), NumericScalar::from_f32(5.0));
    }

    #[test]
    fn slice_bool_sub_byte() {
        // 16 bool elements = 2 bytes, 1-bit strides
        let layout = TensorLayout::<DynRank>::row_major(vec![16], NumericDType::BOOL);
        let mut buf = vec![0u8; layout.buffer_size_bytes()];
        // Write alternating true/false: indices 0=F,1=T,2=F,3=T,...
        for i in 0..16 {
            layout.write_element(&mut buf, i, NumericScalar::from_bool(i % 2 == 1));
        }
        let view = NumericTensorView::<DynRank>::new(&buf, layout);

        // Slice [3..7] → indices 3,4,5,6 → T,F,T,F
        let sliced = view.slice(&[(3, 7)]).unwrap();
        assert_eq!(sliced.shape(), &vec![4u64]);
        assert_eq!(sliced.read_element(0), NumericScalar::from_bool(true));
        assert_eq!(sliced.read_element(1), NumericScalar::from_bool(false));
        assert_eq!(sliced.read_element(2), NumericScalar::from_bool(true));
        assert_eq!(sliced.read_element(3), NumericScalar::from_bool(false));
    }

    #[test]
    fn slice_full_range_is_identity() {
        let t = make_f32_iota(vec![3, 4]);
        let view = t.view();
        let sliced = view.slice(&[(0, 3), (0, 4)]).unwrap();
        assert_eq!(sliced.shape(), &vec![3u64, 4]);
        for i in 0..12 {
            assert_eq!(sliced.read_element(i), view.read_element(i));
        }
    }

    #[test]
    fn slice_single_element() {
        let t = make_f32_iota(vec![5]);
        let view = t.view();
        let sliced = view.slice(&[(3, 4)]).unwrap();
        assert_eq!(sliced.shape(), &vec![1u64]);
        assert_eq!(sliced.read_element(0), NumericScalar::from_f32(3.0));
    }

    #[test]
    fn slice_rank_mismatch_errors() {
        let t = make_f32_iota(vec![4, 4]);
        let view = t.view();
        let err = view.slice(&[(0, 2)]).unwrap_err();
        assert!(
            matches!(
                err,
                TensorLayoutError::SliceRankMismatch {
                    got: 1,
                    expected: 2
                }
            ),
            "expected rank mismatch, got: {err}"
        );
    }

    #[test]
    fn slice_out_of_bounds_errors() {
        let t = make_f32_iota(vec![4]);
        let view = t.view();
        let err = view.slice(&[(2, 5)]).unwrap_err();
        assert!(
            matches!(
                err,
                TensorLayoutError::SliceOutOfBounds {
                    dim: 0,
                    end: 5,
                    dim_size: 4,
                    ..
                }
            ),
            "expected out of bounds, got: {err}"
        );
    }

    #[test]
    fn slice_empty_range_errors() {
        let t = make_f32_iota(vec![4]);
        let view = t.view();
        let err = view.slice(&[(3, 3)]).unwrap_err();
        assert!(
            matches!(
                err,
                TensorLayoutError::SliceEmpty {
                    dim: 0,
                    start: 3,
                    end: 3
                }
            ),
            "expected empty range, got: {err}"
        );
    }

    #[test]
    fn slice_quantized_errors() {
        let layout = TensorLayout::<DynRank>::simple_block_quant(vec![32], 4, false);
        let buf = vec![0u8; layout.buffer_size_bytes()];
        let view = NumericTensorView::<DynRank>::new(&buf, layout);
        let err = view.slice(&[(0, 16)]).unwrap_err();
        assert!(matches!(err, TensorLayoutError::QuantizedSliceUnsupported));
    }

    #[test]
    fn slice_fixed_rank() {
        use typenum::P1;
        // Verify slice works on fixed-rank views too
        let layout = TensorLayout::<P1>::row_major([8], NumericDType::F32);
        let mut buf = vec![0u8; layout.buffer_size_bytes()];
        for i in 0..8 {
            layout.write_element(&mut buf, i, NumericScalar::from_f32(i as f32));
        }
        let view = NumericTensorView::<P1>::new(&buf, layout);
        let sliced = view.slice(&[(2, 6)]).unwrap();
        assert_eq!(sliced.shape(), &[4u64]);
        assert_eq!(sliced.read_element(0), NumericScalar::from_f32(2.0));
        assert_eq!(sliced.read_element(3), NumericScalar::from_f32(5.0));
    }

    // -- to_tensor --

    #[test]
    fn to_tensor_contiguous_memcpy() {
        let pool = SystemPool;
        let t = make_f32_iota(vec![3, 4]);
        let view = t.view();
        assert!(view.layout().is_contiguous());
        let cloned = view.to_tensor(&pool).unwrap();
        assert_eq!(cloned.shape(), &vec![3u64, 4]);
        for i in 0..12 {
            assert_eq!(cloned.read_element(i), NumericScalar::from_f32(i as f32));
        }
        // Verify it's a true copy — different buffer address
        assert!(!std::ptr::eq(
            view.data().as_ptr(),
            cloned.buffer().as_ptr()
        ));
    }

    #[test]
    fn to_tensor_sliced_view() {
        let pool = SystemPool;
        let t = make_f32_iota(vec![4, 6]);
        let view = t.view();
        // Slice [1..3, 2..5] → non-contiguous view
        let sliced = view.slice(&[(1, 3), (2, 5)]).unwrap();
        assert!(!sliced.layout().is_contiguous());

        let materialized = sliced.to_tensor(&pool).unwrap();
        assert_eq!(materialized.shape(), &vec![2u64, 3]);
        assert!(materialized.layout().is_contiguous());
        // materialized[0,0] = original[1,2] = 8
        assert_eq!(materialized.read_element(0), NumericScalar::from_f32(8.0));
        // materialized[1,2] = original[2,4] = 16
        assert_eq!(materialized.read_element(5), NumericScalar::from_f32(16.0));
    }

    #[test]
    fn to_tensor_sliced_1d() {
        let pool = SystemPool;
        let t = make_f32_iota(vec![10]);
        let sliced = t.view().slice(&[(3, 7)]).unwrap();
        let materialized = sliced.to_tensor(&pool).unwrap();
        assert_eq!(materialized.shape(), &vec![4u64]);
        assert!(materialized.layout().is_contiguous());
        for i in 0..4 {
            assert_eq!(
                materialized.read_element(i),
                NumericScalar::from_f32((i + 3) as f32)
            );
        }
    }

    #[test]
    fn to_tensor_bool_sliced() {
        let pool = SystemPool;
        let layout = TensorLayout::<DynRank>::row_major(vec![16], NumericDType::BOOL);
        let mut buf = vec![0u8; layout.buffer_size_bytes()];
        for i in 0..16 {
            layout.write_element(&mut buf, i, NumericScalar::from_bool(i % 2 == 1));
        }
        let view = NumericTensorView::<DynRank>::new(&buf, layout);
        // Slice at non-byte-aligned offset (bit 3)
        let sliced = view.slice(&[(3, 7)]).unwrap();
        let materialized = sliced.to_tensor(&pool).unwrap();
        assert_eq!(materialized.shape(), &vec![4u64]);
        assert!(materialized.layout().is_contiguous());
        assert_eq!(materialized.read_element(0), NumericScalar::from_bool(true));
        assert_eq!(
            materialized.read_element(1),
            NumericScalar::from_bool(false)
        );
        assert_eq!(materialized.read_element(2), NumericScalar::from_bool(true));
        assert_eq!(
            materialized.read_element(3),
            NumericScalar::from_bool(false)
        );
    }

    #[test]
    fn to_tensor_quantized_memcpy() {
        let pool = SystemPool;
        let layout = TensorLayout::<DynRank>::simple_block_quant(vec![32], 4, false);
        let size = layout.buffer_size_bytes();
        // Fill with a recognizable pattern
        let buf: Vec<u8> = (0..size).map(|i| (i & 0xFF) as u8).collect();
        let view = NumericTensorView::<DynRank>::new(&buf, layout);
        let cloned = view.to_tensor(&pool).unwrap();
        assert_eq!(cloned.buffer().len(), size);
        assert_eq!(cloned.buffer(), buf.as_slice());
    }

    #[test]
    fn to_tensor_fixed_rank() {
        use typenum::P1;
        let pool = SystemPool;
        let layout = TensorLayout::<P1>::row_major([6], NumericDType::F32);
        let mut buf = vec![0u8; layout.buffer_size_bytes()];
        for i in 0..6 {
            layout.write_element(&mut buf, i, NumericScalar::from_f32(i as f32));
        }
        let view = NumericTensorView::<P1>::new(&buf, layout);
        let sliced = view.slice(&[(2, 5)]).unwrap();
        let materialized = sliced.to_tensor(&pool).unwrap();
        assert_eq!(materialized.shape(), &[3u64]);
        assert_eq!(materialized.read_element(0), NumericScalar::from_f32(2.0));
        assert_eq!(materialized.read_element(2), NumericScalar::from_f32(4.0));
    }

    #[test]
    fn to_tensor_slice_of_slice() {
        let pool = SystemPool;
        let t = make_f32_iota(vec![10]);
        let s1 = t.view().slice(&[(2, 8)]).unwrap();
        let s2 = s1.slice(&[(1, 4)]).unwrap();
        // s2 = elements 3,4,5 — non-contiguous (offset_bits != 0)
        let materialized = s2.to_tensor(&pool).unwrap();
        assert!(materialized.layout().is_contiguous());
        assert_eq!(materialized.read_element(0), NumericScalar::from_f32(3.0));
        assert_eq!(materialized.read_element(1), NumericScalar::from_f32(4.0));
        assert_eq!(materialized.read_element(2), NumericScalar::from_f32(5.0));
    }

    // -- is_contiguous --

    #[test]
    fn is_contiguous_row_major() {
        let layout = TensorLayout::<DynRank>::row_major(vec![3, 4], NumericDType::F32);
        assert!(layout.is_contiguous());
    }

    #[test]
    fn is_contiguous_after_slice() {
        let t = make_f32_iota(vec![8]);
        let sliced = t.view().slice(&[(2, 6)]).unwrap();
        assert!(!sliced.layout().is_contiguous());
    }

    #[test]
    fn is_contiguous_quantized() {
        let layout = TensorLayout::<DynRank>::simple_block_quant(vec![32], 4, false);
        assert!(layout.is_contiguous());
        let layout = TensorLayout::<DynRank>::k_quant(vec![256], KQuantVariant::Q4_K);
        assert!(layout.is_contiguous());
    }

    // -- flat_to_coords / coords_to_flat --

    #[test]
    fn coords_roundtrip_3d() {
        let layout = TensorLayout::<DynRank>::row_major(vec![2, 3, 4], NumericDType::F32);
        for flat in 0..24 {
            let coords = layout.flat_to_coords(flat);
            assert_eq!(
                layout.coords_to_flat(&coords),
                flat,
                "roundtrip failed for {flat}"
            );
        }
    }

    #[test]
    fn coords_known_values() {
        let layout = TensorLayout::<DynRank>::row_major(vec![2, 3], NumericDType::F32);
        assert_eq!(layout.flat_to_coords(0), vec![0, 0]);
        assert_eq!(layout.flat_to_coords(1), vec![0, 1]);
        assert_eq!(layout.flat_to_coords(3), vec![1, 0]);
        assert_eq!(layout.flat_to_coords(5), vec![1, 2]);
        assert_eq!(layout.coords_to_flat(&[1, 2]), 5);
    }

    #[test]
    fn coords_scalar() {
        let layout = TensorLayout::<DynRank>::row_major(vec![], NumericDType::F32);
        assert_eq!(layout.flat_to_coords(0), Vec::<usize>::new());
        assert_eq!(layout.coords_to_flat(&[]), 0);
    }

    #[test]
    fn coords_1d() {
        let layout = TensorLayout::<DynRank>::row_major(vec![5], NumericDType::F32);
        assert_eq!(layout.flat_to_coords(3), vec![3]);
        assert_eq!(layout.coords_to_flat(&[3]), 3);
    }

    // -- from_fn --

    #[test]
    fn from_fn_1d() {
        let pool = SystemPool;
        let t =
            NumericTensor::<DynRank, SystemPool>::from_fn(vec![5], NumericDType::F32, &pool, |i| {
                NumericScalar::from_f32(i as f32 * 2.0)
            })
            .unwrap();
        assert_eq!(t.shape(), &vec![5u64]);
        assert_eq!(t.read_element(0), NumericScalar::from_f32(0.0));
        assert_eq!(t.read_element(2), NumericScalar::from_f32(4.0));
        assert_eq!(t.read_element(4), NumericScalar::from_f32(8.0));
    }

    #[test]
    fn from_fn_2d() {
        let pool = SystemPool;
        let t = NumericTensor::<DynRank, SystemPool>::from_fn(
            vec![3, 4],
            NumericDType::I32,
            &pool,
            |i| NumericScalar::from_i32(i as i32),
        )
        .unwrap();
        assert_eq!(t.numel(), 12);
        for i in 0..12 {
            assert_eq!(t.read_element(i), NumericScalar::from_i32(i as i32));
        }
    }

    #[test]
    fn from_fn_bool() {
        let pool = SystemPool;
        let t = NumericTensor::<DynRank, SystemPool>::from_fn(
            vec![8],
            NumericDType::BOOL,
            &pool,
            |i| NumericScalar::from_bool(i % 3 == 0),
        )
        .unwrap();
        assert_eq!(t.read_element(0), NumericScalar::from_bool(true));
        assert_eq!(t.read_element(1), NumericScalar::from_bool(false));
        assert_eq!(t.read_element(2), NumericScalar::from_bool(false));
        assert_eq!(t.read_element(3), NumericScalar::from_bool(true));
    }

    // -- from_scalars --

    #[test]
    fn from_scalars_basic() {
        let pool = SystemPool;
        let scalars: Vec<NumericScalar> =
            (0..6).map(|i| NumericScalar::from_f32(i as f32)).collect();
        let t = NumericTensor::<DynRank, SystemPool>::from_scalars(
            vec![2, 3],
            NumericDType::F32,
            &pool,
            &scalars,
        )
        .unwrap();
        assert_eq!(t.shape(), &vec![2u64, 3]);
        for i in 0..6 {
            assert_eq!(t.read_element(i), NumericScalar::from_f32(i as f32));
        }
    }

    #[test]
    #[should_panic(expected = "from_scalars")]
    fn from_scalars_length_mismatch_panics() {
        let pool = SystemPool;
        let scalars = vec![NumericScalar::from_f32(1.0), NumericScalar::from_f32(2.0)];
        let _ = NumericTensor::<DynRank, SystemPool>::from_scalars(
            vec![3],
            NumericDType::F32,
            &pool,
            &scalars,
        );
    }

    #[test]
    fn from_scalars_fixed_rank() {
        use typenum::P1;
        let pool = SystemPool;
        let scalars = vec![
            NumericScalar::from_i32(10),
            NumericScalar::from_i32(20),
            NumericScalar::from_i32(30),
        ];
        let t =
            NumericTensor::<P1, SystemPool>::from_scalars([3], NumericDType::I32, &pool, &scalars)
                .unwrap();
        assert_eq!(t.shape(), &[3u64]);
        assert_eq!(t.read_element(1), NumericScalar::from_i32(20));
    }

    // -- transpose --

    #[test]
    fn transpose_2d() {
        // Shape [2, 3], element [i,j] = i*3 + j
        let t = make_f32_iota(vec![2, 3]);
        let view = t.view();
        let transposed = view.transpose(&[1, 0]).unwrap();
        assert_eq!(transposed.shape(), &vec![3u64, 2]);
        // transposed[0,0] = original[0,0] = 0
        assert_eq!(transposed.read_element(0), NumericScalar::from_f32(0.0));
        // transposed[0,1] = original[1,0] = 3
        assert_eq!(transposed.read_element(1), NumericScalar::from_f32(3.0));
        // transposed[1,0] = original[0,1] = 1
        assert_eq!(transposed.read_element(2), NumericScalar::from_f32(1.0));
        // transposed[2,1] = original[1,2] = 5
        assert_eq!(transposed.read_element(5), NumericScalar::from_f32(5.0));
    }

    #[test]
    fn transpose_3d() {
        // Shape [2, 3, 4], element [i,j,k] = i*12 + j*4 + k
        let t = make_f32_iota(vec![2, 3, 4]);
        let view = t.view();
        // perm [2, 0, 1] → shape [4, 2, 3]
        let transposed = view.transpose(&[2, 0, 1]).unwrap();
        assert_eq!(transposed.shape(), &vec![4u64, 2, 3]);
        // transposed[0,0,0] = original[0,0,0] = 0
        assert_eq!(transposed.read_element(0), NumericScalar::from_f32(0.0));
        // transposed[1,0,0] = original[0,0,1] = 1
        assert_eq!(transposed.read_element(6), NumericScalar::from_f32(1.0));
        // transposed[0,1,0] = original[1,0,0] = 12
        assert_eq!(transposed.read_element(3), NumericScalar::from_f32(12.0));
    }

    #[test]
    fn transpose_materialize() {
        let pool = SystemPool;
        let t = make_f32_iota(vec![2, 3]);
        let transposed = t.view().transpose(&[1, 0]).unwrap();
        let materialized = transposed.to_tensor(&pool).unwrap();
        assert_eq!(materialized.shape(), &vec![3u64, 2]);
        assert!(materialized.layout().is_contiguous());
        // Row-major [3,2]: element [1,0] = flat 2, should be original [0,1] = 1
        assert_eq!(materialized.read_element(2), NumericScalar::from_f32(1.0));
    }

    #[test]
    fn transpose_rank_mismatch_errors() {
        let t = make_f32_iota(vec![2, 3]);
        let err = t.view().transpose(&[0, 1, 2]).unwrap_err();
        assert!(matches!(
            err,
            TensorLayoutError::TransposeRankMismatch {
                got: 3,
                expected: 2
            }
        ));
    }

    #[test]
    fn transpose_invalid_perm_errors() {
        let t = make_f32_iota(vec![2, 3]);
        let err = t.view().transpose(&[0, 0]).unwrap_err();
        assert!(matches!(
            err,
            TensorLayoutError::TransposeInvalidPerm { rank: 2 }
        ));
    }

    // -- Serde roundtrip --

    #[test]
    fn serde_roundtrip_f32() {
        let pool = SystemPool;
        let original = NumericTensor::<DynRank, SystemPool>::from_fn(
            vec![2, 3],
            NumericDType::F32,
            &pool,
            |i| crate::numeric_scalar::NumericScalar::from_f32(i as f32 * 1.5),
        )
        .unwrap();

        let mut serialized = Vec::new();
        ciborium::into_writer(&original, &mut serialized).unwrap();
        let deserialized: NumericTensor<'static, DynRank, SystemPool> =
            ciborium::from_reader(serialized.as_slice()).unwrap();

        assert_eq!(deserialized.shape(), original.shape());
        assert_eq!(deserialized.dtype(), original.dtype());
        for i in 0..original.numel() {
            assert_eq!(deserialized.read_element(i), original.read_element(i));
        }
    }

    #[test]
    fn serde_roundtrip_i32() {
        let pool = SystemPool;
        let original =
            NumericTensor::<DynRank, SystemPool>::from_fn(vec![5], NumericDType::I32, &pool, |i| {
                crate::numeric_scalar::NumericScalar::from_i32(i as i32 * -3)
            })
            .unwrap();

        let mut serialized = Vec::new();
        ciborium::into_writer(&original, &mut serialized).unwrap();
        let deserialized: NumericTensor<'static, DynRank, SystemPool> =
            ciborium::from_reader(serialized.as_slice()).unwrap();

        assert_eq!(deserialized.shape(), original.shape());
        assert_eq!(deserialized.dtype(), original.dtype());
        for i in 0..original.numel() {
            assert_eq!(deserialized.read_element(i), original.read_element(i));
        }
    }

    #[test]
    fn clone_system_pool_tensor() {
        let pool = SystemPool;
        let original =
            NumericTensor::<DynRank, SystemPool>::from_fn(vec![3], NumericDType::F32, &pool, |i| {
                crate::numeric_scalar::NumericScalar::from_f32(i as f32)
            })
            .unwrap();

        let cloned = original.clone();
        assert_eq!(cloned.shape(), original.shape());
        assert_eq!(cloned.dtype(), original.dtype());
        for i in 0..original.numel() {
            assert_eq!(cloned.read_element(i), original.read_element(i));
        }
    }
}
