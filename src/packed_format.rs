use serde::{Deserialize, Serialize};
use std::fmt;

/// Describes a block-packed tensor storage format.
///
/// These formats store groups of weights in fixed-size blocks with shared scale/offset
/// metadata. Individual elements cannot be independently addressed — the block is the
/// atomic unit of storage and dequantization.
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, PartialOrd, Serialize, Deserialize)]
#[allow(non_camel_case_types)]
pub enum PackedFormat {
    // -- Legacy types (block size = 32) --
    /// 4-bit symmetric: 18 bytes/block, 4.5 bpw
    Q4_0,
    /// 4-bit asymmetric (scale + min): 20 bytes/block, 5.0 bpw
    Q4_1,
    /// 5-bit symmetric: 22 bytes/block, 5.5 bpw
    Q5_0,
    /// 5-bit asymmetric (scale + min): 24 bytes/block, 6.0 bpw
    Q5_1,
    /// 8-bit symmetric: 34 bytes/block, 8.5 bpw
    Q8_0,
    /// 8-bit asymmetric (internal, for activations): 36 bytes/block, 9.0 bpw
    Q8_1,

    // -- K-quant types (super-block size = 256) --
    /// 2-bit with 4-bit sub-block scales: 84 bytes/block, 2.625 bpw
    Q2_K,
    /// 3-bit with 6-bit sub-block scales: 110 bytes/block, 3.4375 bpw
    Q3_K,
    /// 4-bit with 6-bit sub-block scales/mins: 144 bytes/block, 4.5 bpw
    Q4_K,
    /// 5-bit with 6-bit sub-block scales/mins: 176 bytes/block, 5.5 bpw
    Q5_K,
    /// 6-bit with 8-bit sub-block scales: 210 bytes/block, 6.5625 bpw
    Q6_K,
    /// 8-bit (internal, for K-quant dot products): 292 bytes/block, 9.125 bpw
    Q8_K,
}

impl PackedFormat {
    /// Number of logical elements per block.
    pub fn block_size(&self) -> usize {
        match self {
            PackedFormat::Q4_0
            | PackedFormat::Q4_1
            | PackedFormat::Q5_0
            | PackedFormat::Q5_1
            | PackedFormat::Q8_0
            | PackedFormat::Q8_1 => 32,

            PackedFormat::Q2_K
            | PackedFormat::Q3_K
            | PackedFormat::Q4_K
            | PackedFormat::Q5_K
            | PackedFormat::Q6_K
            | PackedFormat::Q8_K => 256,
        }
    }

    /// Number of bytes per block.
    pub fn block_bytes(&self) -> usize {
        match self {
            PackedFormat::Q4_0 => 18,
            PackedFormat::Q4_1 => 20,
            PackedFormat::Q5_0 => 22,
            PackedFormat::Q5_1 => 24,
            PackedFormat::Q8_0 => 34,
            PackedFormat::Q8_1 => 36,
            PackedFormat::Q2_K => 84,
            PackedFormat::Q3_K => 110,
            PackedFormat::Q4_K => 144,
            PackedFormat::Q5_K => 176,
            PackedFormat::Q6_K => 210,
            PackedFormat::Q8_K => 292,
        }
    }

    /// Compute the byte size required to store `num_elements` values in this format.
    ///
    /// `num_elements` must be a multiple of `block_size()`.
    pub fn storage_bytes(&self, num_elements: usize) -> usize {
        assert!(
            num_elements.is_multiple_of(self.block_size()),
            "num_elements ({}) must be a multiple of block_size ({})",
            num_elements,
            self.block_size()
        );
        (num_elements / self.block_size()) * self.block_bytes()
    }

    /// The ggml_type enum value for this format, matching the GGUF spec.
    pub fn ggml_type_id(&self) -> u32 {
        match self {
            PackedFormat::Q4_0 => 2,
            PackedFormat::Q4_1 => 3,
            PackedFormat::Q5_0 => 6,
            PackedFormat::Q5_1 => 7,
            PackedFormat::Q8_0 => 8,
            PackedFormat::Q8_1 => 9,
            PackedFormat::Q2_K => 10,
            PackedFormat::Q3_K => 11,
            PackedFormat::Q4_K => 12,
            PackedFormat::Q5_K => 13,
            PackedFormat::Q6_K => 14,
            PackedFormat::Q8_K => 15,
        }
    }

    /// Look up a PackedFormat from a ggml_type enum value.
    pub fn from_ggml_type_id(id: u32) -> Option<Self> {
        match id {
            2 => Some(PackedFormat::Q4_0),
            3 => Some(PackedFormat::Q4_1),
            6 => Some(PackedFormat::Q5_0),
            7 => Some(PackedFormat::Q5_1),
            8 => Some(PackedFormat::Q8_0),
            9 => Some(PackedFormat::Q8_1),
            10 => Some(PackedFormat::Q2_K),
            11 => Some(PackedFormat::Q3_K),
            12 => Some(PackedFormat::Q4_K),
            13 => Some(PackedFormat::Q5_K),
            14 => Some(PackedFormat::Q6_K),
            15 => Some(PackedFormat::Q8_K),
            _ => None,
        }
    }
}

impl fmt::Display for PackedFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PackedFormat::Q4_0 => write!(f, "Q4_0"),
            PackedFormat::Q4_1 => write!(f, "Q4_1"),
            PackedFormat::Q5_0 => write!(f, "Q5_0"),
            PackedFormat::Q5_1 => write!(f, "Q5_1"),
            PackedFormat::Q8_0 => write!(f, "Q8_0"),
            PackedFormat::Q8_1 => write!(f, "Q8_1"),
            PackedFormat::Q2_K => write!(f, "Q2_K"),
            PackedFormat::Q3_K => write!(f, "Q3_K"),
            PackedFormat::Q4_K => write!(f, "Q4_K"),
            PackedFormat::Q5_K => write!(f, "Q5_K"),
            PackedFormat::Q6_K => write!(f, "Q6_K"),
            PackedFormat::Q8_K => write!(f, "Q8_K"),
        }
    }
}
