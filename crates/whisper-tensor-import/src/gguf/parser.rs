use std::collections::HashMap;
use std::io::{Read, Seek};
use std::path::{Path, PathBuf};
use whisper_tensor::dtype::DType;
use whisper_tensor::packed_format::PackedFormat;

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" as LE u32
const GGUF_VERSION_3: u32 = 3;
const DEFAULT_ALIGNMENT: usize = 32;

#[derive(Debug, thiserror::Error)]
pub enum GgufParseError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid GGUF magic: got 0x{0:08X}")]
    BadMagic(u32),
    #[error("Unsupported GGUF version: {0}")]
    UnsupportedVersion(u32),
    #[error("Unknown ggml type id: {0}")]
    UnknownGgmlType(u32),
    #[error("Unknown metadata value type: {0}")]
    UnknownMetadataType(u32),
}

/// GGUF metadata value types.
#[derive(Debug, Clone)]
pub enum GgufMetadataValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GgufMetadataValue>),
}

impl GgufMetadataValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufMetadataValue::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_u32(&self) -> Option<u32> {
        match self {
            GgufMetadataValue::U32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            GgufMetadataValue::U64(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_i32(&self) -> Option<i32> {
        match self {
            GgufMetadataValue::I32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            GgufMetadataValue::F32(v) => Some(*v),
            _ => None,
        }
    }

    /// Get as a u64, coercing from smaller integer types.
    pub fn as_u64_coerce(&self) -> Option<u64> {
        match self {
            GgufMetadataValue::U8(v) => Some(*v as u64),
            GgufMetadataValue::U16(v) => Some(*v as u64),
            GgufMetadataValue::U32(v) => Some(*v as u64),
            GgufMetadataValue::U64(v) => Some(*v),
            GgufMetadataValue::I8(v) => Some(*v as u64),
            GgufMetadataValue::I16(v) => Some(*v as u64),
            GgufMetadataValue::I32(v) => Some(*v as u64),
            GgufMetadataValue::I64(v) => Some(*v as u64),
            _ => None,
        }
    }

    /// Get as f32, coercing from f64.
    pub fn as_f32_coerce(&self) -> Option<f32> {
        match self {
            GgufMetadataValue::F32(v) => Some(*v),
            GgufMetadataValue::F64(v) => Some(*v as f32),
            _ => None,
        }
    }
}

/// Info for a single tensor in the GGUF file.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    /// Dimensions in standard row-major order (outermost first).
    /// Reversed from the GGUF on-disk order (which is innermost first).
    pub dimensions: Vec<u64>,
    /// ggml_type enum value.
    pub ggml_type: u32,
    /// Byte offset from the start of the tensor data section.
    pub offset_in_data: u64,
    /// Our DType (either a packed format or an element type like F32/F16).
    pub dtype: DType,
    /// Byte length of this tensor's data.
    pub byte_length: usize,
}

/// A parsed GGUF file.
#[derive(Debug)]
pub struct GgufFile {
    pub path: PathBuf,
    pub version: u32,
    pub metadata: HashMap<String, GgufMetadataValue>,
    pub tensors: Vec<GgufTensorInfo>,
    /// Byte offset of the tensor data section from the start of the file.
    pub data_offset: u64,
    pub alignment: usize,
}

impl GgufFile {
    pub fn open(path: &Path) -> Result<Self, GgufParseError> {
        let mut f = std::fs::File::open(path)?;
        Self::parse(&mut f, path.to_path_buf())
    }

    fn parse(f: &mut std::fs::File, path: PathBuf) -> Result<Self, GgufParseError> {
        // -- Header --
        let magic = read_u32(f)?;
        if magic != GGUF_MAGIC {
            return Err(GgufParseError::BadMagic(magic));
        }
        let version = read_u32(f)?;
        if version != GGUF_VERSION_3 {
            return Err(GgufParseError::UnsupportedVersion(version));
        }
        let tensor_count = read_u64(f)?;
        let metadata_kv_count = read_u64(f)?;

        // -- Metadata KVs --
        let mut metadata = HashMap::new();
        for _ in 0..metadata_kv_count {
            let key = read_string(f)?;
            let value = read_metadata_value(f)?;
            metadata.insert(key, value);
        }

        // Check for custom alignment in metadata
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u64_coerce())
            .map(|v| v as usize)
            .unwrap_or(DEFAULT_ALIGNMENT);

        // -- Tensor infos --
        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = read_string(f)?;
            let n_dimensions = read_u32(f)?;
            let mut dimensions = Vec::with_capacity(n_dimensions as usize);
            for _ in 0..n_dimensions {
                dimensions.push(read_u64(f)?);
            }
            // GGUF stores dimensions innermost-first; reverse to standard outermost-first
            dimensions.reverse();
            let ggml_type = read_u32(f)?;
            let offset_in_data = read_u64(f)?;

            let (dtype, byte_length) =
                resolve_ggml_type(ggml_type, &dimensions)?;

            tensors.push(GgufTensorInfo {
                name,
                dimensions,
                ggml_type,
                offset_in_data,
                dtype,
                byte_length,
            });
        }

        // The tensor data starts at the next alignment boundary after the current position
        let header_end = f.stream_position()?;
        let data_offset = align_up(header_end as usize, alignment) as u64;

        Ok(GgufFile {
            path,
            version,
            metadata,
            tensors,
            data_offset,
            alignment,
        })
    }

    /// Get a metadata value by key.
    pub fn get_metadata(&self, key: &str) -> Option<&GgufMetadataValue> {
        self.metadata.get(key)
    }

    /// Get the architecture name (e.g. "llama").
    pub fn architecture(&self) -> Option<&str> {
        self.get_metadata("general.architecture")?.as_str()
    }

    /// Absolute byte offset in the file for a given tensor.
    pub fn tensor_file_offset(&self, tensor: &GgufTensorInfo) -> u64 {
        self.data_offset + tensor.offset_in_data
    }

    /// Look up a tensor by name.
    pub fn get_tensor(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }
}

/// Map a ggml_type id + dimensions to our DType and compute the byte length.
fn resolve_ggml_type(
    ggml_type: u32,
    dimensions: &[u64],
) -> Result<(DType, usize), GgufParseError> {
    // Non-packed types first
    let num_elements: u64 = dimensions.iter().product();
    match ggml_type {
        0 => {
            // GGML_TYPE_F32
            Ok((DType::F32, num_elements as usize * 4))
        }
        1 => {
            // GGML_TYPE_F16
            Ok((DType::F16, num_elements as usize * 2))
        }
        _ => {
            // Try packed formats
            if let Some(fmt) = PackedFormat::from_ggml_type_id(ggml_type) {
                let byte_length = fmt.storage_bytes(num_elements as usize);
                Ok((DType::Packed(fmt), byte_length))
            } else {
                Err(GgufParseError::UnknownGgmlType(ggml_type))
            }
        }
    }
}

fn align_up(value: usize, alignment: usize) -> usize {
    (value + alignment - 1) & !(alignment - 1)
}

// -- Low-level readers (all little-endian) --

fn read_u8(f: &mut impl Read) -> Result<u8, std::io::Error> {
    let mut buf = [0u8; 1];
    f.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_i8(f: &mut impl Read) -> Result<i8, std::io::Error> {
    Ok(read_u8(f)? as i8)
}

fn read_u16(f: &mut impl Read) -> Result<u16, std::io::Error> {
    let mut buf = [0u8; 2];
    f.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_i16(f: &mut impl Read) -> Result<i16, std::io::Error> {
    let mut buf = [0u8; 2];
    f.read_exact(&mut buf)?;
    Ok(i16::from_le_bytes(buf))
}

fn read_u32(f: &mut impl Read) -> Result<u32, std::io::Error> {
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32(f: &mut impl Read) -> Result<i32, std::io::Error> {
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_u64(f: &mut impl Read) -> Result<u64, std::io::Error> {
    let mut buf = [0u8; 8];
    f.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i64(f: &mut impl Read) -> Result<i64, std::io::Error> {
    let mut buf = [0u8; 8];
    f.read_exact(&mut buf)?;
    Ok(i64::from_le_bytes(buf))
}

fn read_f32(f: &mut impl Read) -> Result<f32, std::io::Error> {
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_f64(f: &mut impl Read) -> Result<f64, std::io::Error> {
    let mut buf = [0u8; 8];
    f.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

fn read_string(f: &mut impl Read) -> Result<String, std::io::Error> {
    let len = read_u64(f)? as usize;
    let mut buf = vec![0u8; len];
    f.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

fn read_bool(f: &mut impl Read) -> Result<bool, std::io::Error> {
    Ok(read_u8(f)? != 0)
}

fn read_metadata_value(f: &mut impl Read) -> Result<GgufMetadataValue, GgufParseError> {
    let value_type = read_u32(f)?;
    read_metadata_value_of_type(f, value_type)
}

fn read_metadata_value_of_type(
    f: &mut impl Read,
    value_type: u32,
) -> Result<GgufMetadataValue, GgufParseError> {
    match value_type {
        0 => Ok(GgufMetadataValue::U8(read_u8(f)?)),
        1 => Ok(GgufMetadataValue::I8(read_i8(f)?)),
        2 => Ok(GgufMetadataValue::U16(read_u16(f)?)),
        3 => Ok(GgufMetadataValue::I16(read_i16(f)?)),
        4 => Ok(GgufMetadataValue::U32(read_u32(f)?)),
        5 => Ok(GgufMetadataValue::I32(read_i32(f)?)),
        6 => Ok(GgufMetadataValue::F32(read_f32(f)?)),
        7 => Ok(GgufMetadataValue::Bool(read_bool(f)?)),
        8 => Ok(GgufMetadataValue::String(read_string(f)?)),
        9 => {
            // Array: element_type (u32), count (u64), then elements
            let elem_type = read_u32(f)?;
            let count = read_u64(f)? as usize;
            let mut arr = Vec::with_capacity(count);
            for _ in 0..count {
                arr.push(read_metadata_value_of_type(f, elem_type)?);
            }
            Ok(GgufMetadataValue::Array(arr))
        }
        10 => Ok(GgufMetadataValue::U64(read_u64(f)?)),
        11 => Ok(GgufMetadataValue::I64(read_i64(f)?)),
        12 => Ok(GgufMetadataValue::F64(read_f64(f)?)),
        other => Err(GgufParseError::UnknownMetadataType(other)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_llama3_q4_0() {
        let path = Path::new("/ceph/public/neural_models/llms/Llama-3-8B.Q4_0.gguf");
        if !path.exists() {
            eprintln!("Skipping test: GGUF file not found at {}", path.display());
            return;
        }

        let gguf = GgufFile::open(path).expect("Failed to parse GGUF");
        assert_eq!(gguf.version, 3);
        assert_eq!(gguf.architecture(), Some("llama"));

        // Should have 291 tensors
        assert_eq!(gguf.tensors.len(), 291);

        // Check token_embd.weight — [vocab_size, hidden_dim] in standard order
        let embd = gguf.get_tensor("token_embd.weight").expect("missing token_embd");
        assert_eq!(embd.dtype, DType::Packed(PackedFormat::Q4_0));
        assert_eq!(embd.dimensions, vec![128256, 4096]);

        // Check that an F32 norm weight exists
        let norm = gguf.get_tensor("output_norm.weight").expect("missing output_norm");
        assert_eq!(norm.dtype, DType::F32);
        assert_eq!(norm.dimensions, vec![4096]);

        // Check metadata
        let block_count = gguf
            .get_metadata("llama.block_count")
            .expect("missing block_count");
        assert_eq!(block_count.as_u64_coerce(), Some(32));
    }

    #[test]
    fn parse_llama3_q4_k_m() {
        let path = Path::new("/ceph/public/neural_models/llms/Llama-3-8B.Q4_K_M.gguf");
        if !path.exists() {
            eprintln!("Skipping test: GGUF file not found at {}", path.display());
            return;
        }

        let gguf = GgufFile::open(path).expect("Failed to parse GGUF");
        assert_eq!(gguf.version, 3);
        assert_eq!(gguf.tensors.len(), 291);

        // K-quant files use a mix of Q4_K and Q6_K
        let q_proj = gguf
            .get_tensor("blk.0.attn_q.weight")
            .expect("missing blk.0.attn_q.weight");
        // Q4_K_M uses Q4_K for most attention weights
        assert!(
            matches!(q_proj.dtype, DType::Packed(PackedFormat::Q4_K) | DType::Packed(PackedFormat::Q6_K)),
            "Unexpected dtype for Q4_K_M attention: {:?}",
            q_proj.dtype
        );
    }
}
