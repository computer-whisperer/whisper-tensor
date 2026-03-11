use std::sync::Arc;

use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::dtype::DType;
use crate::packed_format::PackedFormat;
use crate::tensor_rank::{DimContainer, DynRank, Rank};

/// A block-packed tensor whose data is not element-addressable.
///
/// The data is stored as an opaque byte buffer. The `shape` is the logical element shape
/// (e.g. `[4096, 4096]` for a weight matrix), while the actual byte count is determined by
/// the packed format's block size and bytes-per-block.
///
/// Only the innermost dimension is packed — it must be a multiple of `format.block_size()`.
#[derive(Clone)]
pub struct PackedTensor<R: Rank> {
    data: Arc<[u8]>,
    shape: R::KnownDims,
    format: PackedFormat,
}

impl<R: Rank> std::fmt::Debug for PackedTensor<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PackedTensor")
            .field("format", &self.format)
            .field("shape", &self.shape)
            .field("data_bytes", &self.data.len())
            .finish()
    }
}

impl<R: Rank> PackedTensor<R> {
    /// Create a new packed tensor from raw block data.
    ///
    /// The caller must ensure that:
    /// - `data.len()` matches the expected storage size for `shape` in `format`
    /// - The innermost dimension of `shape` is a multiple of `format.block_size()`
    pub fn new(data: Arc<[u8]>, shape: R::KnownDims, format: PackedFormat) -> Self {
        let shape_slice = shape.as_slice();
        let num_elements: u64 = shape_slice.iter().product();
        let inner_dim = *shape_slice.last().expect("shape must be non-empty");

        assert!(
            (inner_dim as usize).is_multiple_of(format.block_size()),
            "innermost dimension ({inner_dim}) must be a multiple of block_size ({})",
            format.block_size()
        );
        assert_eq!(
            data.len(),
            format.storage_bytes(num_elements as usize),
            "data length ({}) doesn't match expected storage bytes ({}) for shape {:?} in format {}",
            data.len(),
            format.storage_bytes(num_elements as usize),
            shape_slice,
            format,
        );

        Self {
            data,
            shape,
            format,
        }
    }

    /// The logical element shape.
    pub fn shape(&self) -> &R::KnownDims {
        &self.shape
    }

    /// The packed storage format.
    pub fn format(&self) -> PackedFormat {
        self.format
    }

    /// The dtype, always `DType::Packed(self.format)`.
    pub fn dtype(&self) -> DType {
        DType::Packed(self.format)
    }

    /// The raw block data.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Number of logical elements (product of shape dimensions).
    pub fn num_elements(&self) -> u64 {
        self.shape.as_slice().iter().product()
    }

    /// Number of dimensions.
    pub fn rank(&self) -> usize {
        self.shape.as_slice().len()
    }

    /// Dequantize the entire tensor to f32, returning an NDArray tensor.
    pub fn dequantize(&self) -> NDArrayNumericTensor<R> {
        let num_elements = self.num_elements() as usize;
        let mut output = vec![0.0f32; num_elements];
        dequantize_blocks(self.format, &self.data, &mut output);
        NDArrayNumericTensor::try_from_vec_shape(
            output,
            &self
                .shape
                .as_slice()
                .iter()
                .map(|&d| d as usize)
                .collect::<Vec<_>>(),
        )
        .expect("shape should be valid after dequantization")
    }

    /// Convert to dynamic rank.
    pub fn to_dyn_rank(&self) -> PackedTensor<DynRank> {
        PackedTensor {
            data: self.data.clone(),
            shape: self.shape.as_slice().to_vec(),
            format: self.format,
        }
    }
}

// ---------------------------------------------------------------------------
// Dequantization implementations
// ---------------------------------------------------------------------------

/// Dequantize a contiguous buffer of packed blocks into f32 output.
///
/// `output` must have length equal to the number of logical elements.
fn dequantize_blocks(format: PackedFormat, data: &[u8], output: &mut [f32]) {
    match format {
        PackedFormat::Q4_0 => dequantize_q4_0(data, output),
        PackedFormat::Q4_1 => dequantize_q4_1(data, output),
        PackedFormat::Q5_0 => dequantize_q5_0(data, output),
        PackedFormat::Q5_1 => dequantize_q5_1(data, output),
        PackedFormat::Q8_0 => dequantize_q8_0(data, output),
        PackedFormat::Q8_1 => dequantize_q8_1(data, output),
        PackedFormat::Q2_K => dequantize_q2_k(data, output),
        PackedFormat::Q3_K => dequantize_q3_k(data, output),
        PackedFormat::Q4_K => dequantize_q4_k(data, output),
        PackedFormat::Q5_K => dequantize_q5_k(data, output),
        PackedFormat::Q6_K => dequantize_q6_k(data, output),
        PackedFormat::Q8_K => dequantize_q8_k(data, output),
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

fn read_f16(data: &[u8], offset: usize) -> f32 {
    let bits = u16::from_le_bytes([data[offset], data[offset + 1]]);
    f16_to_f32(bits)
}

// -- Legacy types (block size = 32) --

fn dequantize_q4_0(data: &[u8], output: &mut [f32]) {
    const QK: usize = 32;
    const BLOCK_BYTES: usize = 18; // 2 (d) + 16 (qs)
    let nb = output.len() / QK;

    for i in 0..nb {
        let block = &data[i * BLOCK_BYTES..];
        let d = read_f16(block, 0);
        let qs = &block[2..];

        for j in 0..QK / 2 {
            let x0 = (qs[j] & 0x0F) as i32 - 8;
            let x1 = (qs[j] >> 4) as i32 - 8;
            output[i * QK + j] = x0 as f32 * d;
            output[i * QK + j + QK / 2] = x1 as f32 * d;
        }
    }
}

fn dequantize_q4_1(data: &[u8], output: &mut [f32]) {
    const QK: usize = 32;
    const BLOCK_BYTES: usize = 20; // 2 (d) + 2 (m) + 16 (qs)
    let nb = output.len() / QK;

    for i in 0..nb {
        let block = &data[i * BLOCK_BYTES..];
        let d = read_f16(block, 0);
        let m = read_f16(block, 2);
        let qs = &block[4..];

        for j in 0..QK / 2 {
            let x0 = (qs[j] & 0x0F) as f32;
            let x1 = (qs[j] >> 4) as f32;
            output[i * QK + j] = x0 * d + m;
            output[i * QK + j + QK / 2] = x1 * d + m;
        }
    }
}

fn dequantize_q5_0(data: &[u8], output: &mut [f32]) {
    const QK: usize = 32;
    const BLOCK_BYTES: usize = 22; // 2 (d) + 4 (qh) + 16 (qs)
    let nb = output.len() / QK;

    for i in 0..nb {
        let block = &data[i * BLOCK_BYTES..];
        let d = read_f16(block, 0);
        let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
        let qs = &block[6..];

        for j in 0..QK / 2 {
            let xh_0 = ((qh >> (j as u32)) << 4) & 0x10;
            let xh_1 = (qh >> (j as u32 + 12)) & 0x10;
            let x0 = ((qs[j] & 0x0F) as u32 | xh_0) as i32 - 16;
            let x1 = ((qs[j] >> 4) as u32 | xh_1) as i32 - 16;
            output[i * QK + j] = x0 as f32 * d;
            output[i * QK + j + QK / 2] = x1 as f32 * d;
        }
    }
}

fn dequantize_q5_1(data: &[u8], output: &mut [f32]) {
    const QK: usize = 32;
    const BLOCK_BYTES: usize = 24; // 2 (d) + 2 (m) + 4 (qh) + 16 (qs)
    let nb = output.len() / QK;

    for i in 0..nb {
        let block = &data[i * BLOCK_BYTES..];
        let d = read_f16(block, 0);
        let m = read_f16(block, 2);
        let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
        let qs = &block[8..];

        for j in 0..QK / 2 {
            let xh_0 = ((qh >> (j as u32)) << 4) & 0x10;
            let xh_1 = (qh >> (j as u32 + 12)) & 0x10;
            let x0 = (qs[j] & 0x0F) as u32 | xh_0;
            let x1 = (qs[j] >> 4) as u32 | xh_1;
            output[i * QK + j] = x0 as f32 * d + m;
            output[i * QK + j + QK / 2] = x1 as f32 * d + m;
        }
    }
}

fn dequantize_q8_0(data: &[u8], output: &mut [f32]) {
    const QK: usize = 32;
    const BLOCK_BYTES: usize = 34; // 2 (d) + 32 (qs)
    let nb = output.len() / QK;

    for i in 0..nb {
        let block = &data[i * BLOCK_BYTES..];
        let d = read_f16(block, 0);
        let qs = &block[2..];

        for j in 0..QK {
            output[i * QK + j] = qs[j] as i8 as f32 * d;
        }
    }
}

fn dequantize_q8_1(data: &[u8], output: &mut [f32]) {
    const QK: usize = 32;
    const BLOCK_BYTES: usize = 36; // 2 (d) + 2 (s) + 32 (qs)
    let nb = output.len() / QK;

    for i in 0..nb {
        let block = &data[i * BLOCK_BYTES..];
        let d = read_f16(block, 0);
        // s (sum) at offset 2 is not needed for dequantization
        let qs = &block[4..];

        for j in 0..QK {
            output[i * QK + j] = qs[j] as i8 as f32 * d;
        }
    }
}

// -- K-quant types (super-block size = 256) --

fn dequantize_q2_k(data: &[u8], output: &mut [f32]) {
    const QK: usize = 256;
    const BLOCK_BYTES: usize = 84;
    let nb = output.len() / QK;

    for i in 0..nb {
        let block = &data[i * BLOCK_BYTES..];
        // Layout: scales[16], qs[64], d(f16), dmin(f16)
        let scales = &block[0..16];
        let qs = &block[16..80];
        let d = read_f16(block, 80);
        let dmin = read_f16(block, 82);

        let mut y = i * QK;
        let mut q_offset = 0;
        let mut is = 0;

        for _n in (0..QK).step_by(128) {
            for shift in (0..8).step_by(2) {
                let sc = scales[is];
                is += 1;
                let dl = d * (sc & 0xF) as f32;
                let ml = dmin * (sc >> 4) as f32;
                for l in 0..16 {
                    output[y] = dl * ((qs[q_offset + l] >> shift) & 3) as f32 - ml;
                    y += 1;
                }

                let sc = scales[is];
                is += 1;
                let dl = d * (sc & 0xF) as f32;
                let ml = dmin * (sc >> 4) as f32;
                for l in 0..16 {
                    output[y] = dl * ((qs[q_offset + 16 + l] >> shift) & 3) as f32 - ml;
                    y += 1;
                }
            }
            q_offset += 32;
        }
    }
}

fn dequantize_q3_k(data: &[u8], output: &mut [f32]) {
    const QK: usize = 256;
    const BLOCK_BYTES: usize = 110;
    let nb = output.len() / QK;

    const KMASK1: u32 = 0x03030303;
    const KMASK2: u32 = 0x0f0f0f0f;

    for i in 0..nb {
        let block = &data[i * BLOCK_BYTES..];
        // Layout: hmask[32], qs[64], scales[12], d(f16)
        let hmask = &block[0..32];
        let qs = &block[32..96];
        let scales_raw = &block[96..108];
        let d_all = read_f16(block, 108);

        // Unpack 16 x 6-bit scales from 12 bytes
        let mut aux = [0u32; 4];
        for j in 0..3 {
            aux[j] = u32::from_le_bytes([
                scales_raw[j * 4],
                scales_raw[j * 4 + 1],
                scales_raw[j * 4 + 2],
                scales_raw[j * 4 + 3],
            ]);
        }
        let tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
        aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
        aux[0] = (aux[0] & KMASK2) | (((tmp) & KMASK1) << 4);
        aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);

        let scales: &[i8; 16] = unsafe { &*(&aux as *const [u32; 4] as *const [i8; 16]) };

        let mut y = i * QK;
        let mut q_offset = 0;
        let mut is = 0;
        let mut m: u8 = 1;

        for _n in (0..QK).step_by(128) {
            for shift in (0..8).step_by(2) {
                let dl = d_all * (scales[is] as i32 - 32) as f32;
                is += 1;
                for l in 0..16 {
                    let q_val = ((qs[q_offset + l] >> shift) & 3) as i32;
                    let h_val = if hmask[l] & m != 0 { 0 } else { 4 };
                    output[y] = dl * (q_val - h_val) as f32;
                    y += 1;
                }

                let dl = d_all * (scales[is] as i32 - 32) as f32;
                is += 1;
                for l in 0..16 {
                    let q_val = ((qs[q_offset + 16 + l] >> shift) & 3) as i32;
                    let h_val = if hmask[16 + l] & m != 0 { 0 } else { 4 };
                    output[y] = dl * (q_val - h_val) as f32;
                    y += 1;
                }

                m = m.wrapping_shl(1);
            }
            q_offset += 32;
        }
    }
}

fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        (
            (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
        )
    }
}

fn dequantize_q4_k(data: &[u8], output: &mut [f32]) {
    const QK: usize = 256;
    const BLOCK_BYTES: usize = 144;
    let nb = output.len() / QK;

    for i in 0..nb {
        let block = &data[i * BLOCK_BYTES..];
        // Layout: d(f16), dmin(f16), scales[12], qs[128]
        let d = read_f16(block, 0);
        let dmin = read_f16(block, 2);
        let scales = &block[4..16];
        let qs = &block[16..];

        let mut y = i * QK;
        let mut q_offset = 0;
        let mut is = 0;

        for _j in (0..QK).step_by(64) {
            let (sc, m) = get_scale_min_k4(is, scales);
            let d1 = d * sc as f32;
            let m1 = dmin * m as f32;
            let (sc, m) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc as f32;
            let m2 = dmin * m as f32;

            for l in 0..32 {
                output[y] = d1 * (qs[q_offset + l] & 0xF) as f32 - m1;
                y += 1;
            }
            for l in 0..32 {
                output[y] = d2 * (qs[q_offset + l] >> 4) as f32 - m2;
                y += 1;
            }
            q_offset += 32;
            is += 2;
        }
    }
}

fn dequantize_q5_k(data: &[u8], output: &mut [f32]) {
    const QK: usize = 256;
    const BLOCK_BYTES: usize = 176;
    let nb = output.len() / QK;

    for i in 0..nb {
        let block = &data[i * BLOCK_BYTES..];
        // Layout: d(f16), dmin(f16), scales[12], qh[32], qs[128]
        let d = read_f16(block, 0);
        let dmin = read_f16(block, 2);
        let scales = &block[4..16];
        let qh = &block[16..48];
        let qs = &block[48..];

        let mut y = i * QK;
        let mut q_offset = 0;
        let mut is = 0;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;

        for _j in (0..QK).step_by(64) {
            let (sc, m) = get_scale_min_k4(is, scales);
            let d1 = d * sc as f32;
            let m1 = dmin * m as f32;
            let (sc, m) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc as f32;
            let m2 = dmin * m as f32;

            for l in 0..32 {
                let low = (qs[q_offset + l] & 0xF) as f32;
                let high = if qh[l] & u1 != 0 { 16.0 } else { 0.0 };
                output[y] = d1 * (low + high) - m1;
                y += 1;
            }
            for l in 0..32 {
                let low = (qs[q_offset + l] >> 4) as f32;
                let high = if qh[l] & u2 != 0 { 16.0 } else { 0.0 };
                output[y] = d2 * (low + high) - m2;
                y += 1;
            }
            q_offset += 32;
            is += 2;
            u1 = u1.wrapping_shl(2);
            u2 = u2.wrapping_shl(2);
        }
    }
}

fn dequantize_q6_k(data: &[u8], output: &mut [f32]) {
    const QK: usize = 256;
    const BLOCK_BYTES: usize = 210;
    let nb = output.len() / QK;

    for i in 0..nb {
        let block = &data[i * BLOCK_BYTES..];
        // Layout: ql[128], qh[64], scales[16], d(f16)
        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales = &block[192..208];
        let d = read_f16(block, 208);

        let mut y = i * QK;
        let mut ql_offset = 0;
        let mut qh_offset = 0;
        let mut sc_offset = 0;

        for _n in (0..QK).step_by(128) {
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[ql_offset + l] & 0xF) | (((qh[qh_offset + l]) & 3) << 4)) as i8 - 32;
                let q2 = ((ql[ql_offset + 32 + l] & 0xF) | (((qh[qh_offset + l] >> 2) & 3) << 4))
                    as i8
                    - 32;
                let q3 =
                    ((ql[ql_offset + l] >> 4) | (((qh[qh_offset + l] >> 4) & 3) << 4)) as i8 - 32;
                let q4 = ((ql[ql_offset + 32 + l] >> 4) | (((qh[qh_offset + l] >> 6) & 3) << 4))
                    as i8
                    - 32;

                output[y + l] = d * scales[sc_offset + is] as i8 as f32 * q1 as f32;
                output[y + 32 + l] = d * scales[sc_offset + 2 + is] as i8 as f32 * q2 as f32;
                output[y + 64 + l] = d * scales[sc_offset + 4 + is] as i8 as f32 * q3 as f32;
                output[y + 96 + l] = d * scales[sc_offset + 6 + is] as i8 as f32 * q4 as f32;
            }
            y += 128;
            ql_offset += 64;
            qh_offset += 32;
            sc_offset += 8;
        }
    }
}

fn dequantize_q8_k(data: &[u8], output: &mut [f32]) {
    const QK: usize = 256;
    const BLOCK_BYTES: usize = 292; // 4 (d as f32) + 256 (qs) + 32 (bsums)
    let nb = output.len() / QK;

    for i in 0..nb {
        let block = &data[i * BLOCK_BYTES..];
        // Layout: d(f32), qs[256], bsums[32]
        let d = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);
        let qs = &block[4..260];

        for j in 0..QK {
            output[i * QK + j] = qs[j] as i8 as f32 * d;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packed_format_storage_bytes() {
        // Q4_0: 32 elements → 18 bytes
        assert_eq!(PackedFormat::Q4_0.storage_bytes(32), 18);
        assert_eq!(PackedFormat::Q4_0.storage_bytes(64), 36);

        // Q4_K: 256 elements → 144 bytes
        assert_eq!(PackedFormat::Q4_K.storage_bytes(256), 144);
        assert_eq!(PackedFormat::Q4_K.storage_bytes(512), 288);
    }

    #[test]
    fn test_dequantize_q8_0_roundtrip() {
        // Q8_0 is nearly lossless: d * i8
        // Create a block manually: d=1.0 as f16, then 32 signed bytes
        let d_bits = half::f16::from_f32(0.1).to_bits();
        let mut block = vec![0u8; 34];
        block[0] = (d_bits & 0xFF) as u8;
        block[1] = (d_bits >> 8) as u8;
        for j in 0..32 {
            block[2 + j] = (j as i8 - 16) as u8;
        }

        let mut output = vec![0.0f32; 32];
        dequantize_q8_0(&block, &mut output);

        let d = half::f16::from_f32(0.1).to_f32();
        for (j, &val) in output.iter().enumerate() {
            let expected = (j as i32 - 16) as f32 * d;
            assert!(
                (val - expected).abs() < 1e-6,
                "mismatch at {j}: got {val}, expected {expected}",
            );
        }
    }

    #[test]
    fn test_dequantize_q4_0_basic() {
        // Q4_0: d * (nibble - 8)
        let d_bits = half::f16::from_f32(0.5).to_bits();
        let mut block = vec![0u8; 18];
        block[0] = (d_bits & 0xFF) as u8;
        block[1] = (d_bits >> 8) as u8;
        // Set all quant nibbles to 8 (which means value 0 after -8)
        for j in 0..16 {
            block[2 + j] = 0x88; // low=8, high=8
        }

        let mut output = vec![0.0f32; 32];
        dequantize_q4_0(&block, &mut output);

        for (j, &val) in output.iter().enumerate() {
            assert!(
                val.abs() < 1e-6,
                "expected ~0 at {j}, got {val}",
            );
        }
    }
}
