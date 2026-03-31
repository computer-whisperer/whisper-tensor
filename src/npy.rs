//! NumPy `.npy` file reader/writer for pool-backed [`NumericTensor`]s.
//!
//! Supports the subset of dtypes used in practice: float16/32/64, bfloat16,
//! int8/16/32/64, uint8/16/32/64, bool.

use crate::numeric_dtype::NumericDType;
use crate::numeric_tensor::{NumericTensor, NumericTensorView, TensorLayout};
use crate::pool::Pool;
use crate::tensor_rank::DynRank;
use std::path::Path;

/// Parse a `.npy` byte buffer into a pool-backed tensor.
pub fn read_npy<'p, P: Pool + 'p>(
    data: &[u8],
    pool: &'p P,
) -> Result<NumericTensor<'p, DynRank, P>, String> {
    // Validate magic: \x93NUMPY
    if data.len() < 10 || &data[..6] != b"\x93NUMPY" {
        return Err("Not a valid .npy file (bad magic)".into());
    }

    let major = data[6];
    let (header_start, header_len) = if major >= 2 {
        if data.len() < 12 {
            return Err("Truncated v2 npy header".into());
        }
        let len = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
        (12, len)
    } else {
        let len = u16::from_le_bytes([data[8], data[9]]) as usize;
        (10, len)
    };

    let header_end = header_start + header_len;
    if data.len() < header_end {
        return Err("Truncated npy header".into());
    }

    let header = std::str::from_utf8(&data[header_start..header_end])
        .map_err(|e| format!("npy header not UTF-8: {e}"))?;

    let dtype = parse_dtype(header)?;
    let shape = parse_shape(header)?;
    let raw = &data[header_end..];

    let layout = TensorLayout::<DynRank>::row_major(shape, dtype);
    let expected_bytes = layout.buffer_size_bytes();
    if raw.len() < expected_bytes {
        return Err(format!(
            "npy data too short: need {expected_bytes} bytes, got {}",
            raw.len()
        ));
    }

    let mut buf = pool
        .allocate(expected_bytes)
        .map_err(|e| format!("pool allocation failed: {e}"))?;
    buf[..expected_bytes].copy_from_slice(&raw[..expected_bytes]);
    Ok(NumericTensor::from_parts(buf, layout))
}

/// Read a `.npy` file from disk into a pool-backed tensor.
pub fn read_npy_file<'p, P: Pool + 'p>(
    path: &Path,
    pool: &'p P,
) -> Result<NumericTensor<'p, DynRank, P>, String> {
    let data =
        std::fs::read(path).map_err(|e| format!("Failed to read {}: {e}", path.display()))?;
    read_npy(&data, pool)
}

/// Serialize a tensor view to `.npy` v1 format bytes.
pub fn write_npy(view: &NumericTensorView<'_, DynRank>) -> Vec<u8> {
    let shape = view.shape();
    let dtype = view.dtype();
    let descr = dtype_to_npy_descr(dtype);

    let shape_str = if shape.is_empty() {
        "()".to_string()
    } else if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        format!("({})", dims.join(", "))
    };
    let header_content = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}",
        descr, shape_str
    );

    // Pad to 64-byte alignment
    let prefix_len = 10usize; // magic(6) + version(2) + header_len(2)
    let unpadded = prefix_len + header_content.len() + 1; // +1 for trailing \n
    let padded = (unpadded + 63) & !63;
    let padding = padded - unpadded;
    let header_len = (header_content.len() + padding + 1) as u16;

    let data_bytes = view.numel() * dtype.bytes_per_element();
    let mut buf = Vec::with_capacity(prefix_len + header_len as usize + data_bytes);
    buf.extend_from_slice(b"\x93NUMPY");
    buf.push(1); // major
    buf.push(0); // minor
    buf.extend_from_slice(&header_len.to_le_bytes());
    buf.extend_from_slice(header_content.as_bytes());
    buf.extend(std::iter::repeat(b' ').take(padding));
    buf.push(b'\n');

    // Write raw tensor data — element by element via scalar LE bytes.
    let bpe = dtype.bytes_per_element();
    for i in 0..view.numel() {
        let scalar = view.read_element(i);
        buf.extend_from_slice(&scalar.raw_bits()[..bpe]);
    }

    buf
}

/// Write a tensor view to a `.npy` file.
pub fn write_npy_file(path: &Path, view: &NumericTensorView<'_, DynRank>) -> Result<(), String> {
    let data = write_npy(view);
    std::fs::write(path, &data).map_err(|e| format!("Failed to write {}: {e}", path.display()))
}

fn dtype_to_npy_descr(dtype: NumericDType) -> &'static str {
    match dtype {
        NumericDType::F64 => "<f8",
        NumericDType::F32 => "<f4",
        NumericDType::F16 => "<f2",
        NumericDType::BF16 => "<f2", // BF16 has no standard npy descr; write as f16 bytes
        NumericDType::I64 => "<i8",
        NumericDType::I32 => "<i4",
        NumericDType::I16 => "<i2",
        NumericDType::I8 => "|i1",
        NumericDType::U64 => "<u8",
        NumericDType::U32 => "<u4",
        NumericDType::U16 => "<u2",
        NumericDType::U8 => "|u1",
        NumericDType::BOOL => "|b1",
        _ => "<f4", // Exotic sub-byte types: fall back to f4 descr
    }
}

fn parse_dtype(header: &str) -> Result<NumericDType, String> {
    // Find 'descr' value — pattern: 'descr': '<f4'
    let descr_start = header
        .find("'descr'")
        .or_else(|| header.find("\"descr\""))
        .ok_or("No 'descr' in npy header")?;
    let rest = &header[descr_start..];
    let colon = rest.find(':').ok_or("No colon after descr")?;
    let after_colon = &rest[colon + 1..];
    let quote_char = if after_colon.contains('\'') {
        '\''
    } else {
        '"'
    };
    let first_quote = after_colon
        .find(quote_char)
        .ok_or("No opening quote for descr")?;
    let inner = &after_colon[first_quote + 1..];
    let end_quote = inner.find(quote_char).ok_or("No closing quote for descr")?;
    let descr = &inner[..end_quote];

    // Strip endianness prefix
    let type_str = descr.trim_start_matches(['<', '>', '=', '|']);

    match type_str {
        "f8" | "float64" => Ok(NumericDType::F64),
        "f4" | "float32" => Ok(NumericDType::F32),
        "f2" | "float16" => Ok(NumericDType::F16),
        "i8" | "int64" => Ok(NumericDType::I64),
        "i4" | "int32" => Ok(NumericDType::I32),
        "i2" | "int16" => Ok(NumericDType::I16),
        "i1" | "int8" => Ok(NumericDType::I8),
        "u8" | "uint64" => Ok(NumericDType::U64),
        "u4" | "uint32" => Ok(NumericDType::U32),
        "u2" | "uint16" => Ok(NumericDType::U16),
        "u1" | "uint8" => Ok(NumericDType::U8),
        "b1" => Ok(NumericDType::BOOL),
        other => Err(format!("Unsupported npy dtype: {other}")),
    }
}

fn parse_shape(header: &str) -> Result<Vec<u64>, String> {
    let shape_start = header
        .find("'shape'")
        .or_else(|| header.find("\"shape\""))
        .ok_or("No 'shape' in npy header")?;
    let rest = &header[shape_start..];
    let open = rest.find('(').ok_or("No '(' in shape")?;
    let close = rest.find(')').ok_or("No ')' in shape")?;
    let inner = rest[open + 1..close].trim();

    if inner.is_empty() {
        return Ok(vec![]);
    }

    inner
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<u64>()
                .map_err(|e| format!("Bad shape dim '{s}': {e}"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pool::SystemPool;

    fn make_npy_v1(descr: &str, shape: &[u64], data: &[u8]) -> Vec<u8> {
        let fortran_order = "False";
        let shape_str = if shape.is_empty() {
            "()".to_string()
        } else if shape.len() == 1 {
            format!("({},)", shape[0])
        } else {
            let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
            format!("({})", dims.join(", "))
        };
        let header_content = format!(
            "{{'descr': '{}', 'fortran_order': {}, 'shape': {}, }}",
            descr, fortran_order, shape_str
        );
        // Pad to 64-byte alignment (header_len includes the newline)
        let prefix_len = 10; // magic(6) + version(2) + header_len(2)
        let unpadded = prefix_len + header_content.len() + 1; // +1 for trailing \n
        let padded = (unpadded + 63) & !63;
        let padding = padded - unpadded;
        let header_len = (header_content.len() + padding + 1) as u16;

        let mut buf = Vec::new();
        buf.extend_from_slice(b"\x93NUMPY");
        buf.push(1); // major
        buf.push(0); // minor
        buf.extend_from_slice(&header_len.to_le_bytes());
        buf.extend_from_slice(header_content.as_bytes());
        buf.extend(std::iter::repeat(b' ').take(padding));
        buf.push(b'\n');
        buf.extend_from_slice(data);
        buf
    }

    #[test]
    fn read_f32_tensor() {
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let npy = make_npy_v1("<f4", &[2, 3], &raw);

        let tensor = read_npy(&npy, &SystemPool).unwrap();
        assert_eq!(tensor.shape(), &vec![2, 3]);
        assert_eq!(tensor.dtype(), NumericDType::F32);
        assert_eq!(tensor.numel(), 6);
        for (i, &expected) in values.iter().enumerate() {
            assert_eq!(tensor.read_element(i).to_f32(), expected);
        }
    }

    #[test]
    fn read_i64_tensor() {
        let values: Vec<i64> = vec![10, -20, 30];
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let npy = make_npy_v1("<i8", &[3], &raw);

        let tensor = read_npy(&npy, &SystemPool).unwrap();
        assert_eq!(tensor.dtype(), NumericDType::I64);
        assert_eq!(tensor.numel(), 3);
        for (i, &expected) in values.iter().enumerate() {
            assert_eq!(tensor.read_element(i).to_f64(), expected as f64);
        }
    }

    #[test]
    fn read_scalar() {
        let raw = 42.0f64.to_le_bytes();
        let npy = make_npy_v1("<f8", &[], &raw);

        let tensor = read_npy(&npy, &SystemPool).unwrap();
        assert_eq!(tensor.shape(), &Vec::<u64>::new());
        assert_eq!(tensor.numel(), 1);
        assert_eq!(tensor.read_element(0).to_f64(), 42.0);
    }

    #[test]
    fn truncated_data_errors() {
        let npy = make_npy_v1("<f4", &[10], &[0u8; 4]); // needs 40 bytes, only 4
        assert!(read_npy(&npy, &SystemPool).is_err());
    }
}
