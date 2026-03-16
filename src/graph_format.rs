//! Versioned, self-describing graph persistence format.
//!
//! The on-disk payload is CBOR with a small envelope:
//! - format name and encoding (self-description),
//! - container version,
//! - graph kind (SuperGraph vs MilliOpGraph),
//! - schema version for the payload,
//! - payload.
//!
//! This gives us a stable place for migrations when payload schemas change.

use crate::milli_graph::MilliOpGraph;
use crate::super_graph::SuperGraph;
use ciborium::value::Value;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::{Cursor, Read, Write};
use std::path::Path;

pub const GRAPH_FILE_FORMAT_NAME: &str = "whisper_tensor_graph";
pub const GRAPH_FILE_ENCODING: &str = "cbor";
pub const GRAPH_FILE_CONTAINER_VERSION: u32 = 1;
pub const SUPER_GRAPH_SCHEMA_VERSION: u32 = 1;
pub const MILLI_OP_GRAPH_SCHEMA_VERSION: u32 = 1;
pub const SUPER_GRAPH_FILE_EXTENSION: &str = "wtsg.cbor";
pub const MILLI_OP_GRAPH_FILE_EXTENSION: &str = "wtmg.cbor";

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphFileKind {
    SuperGraph,
    MilliOpGraph,
}

#[derive(Debug, thiserror::Error)]
pub enum GraphFormatError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("failed to encode CBOR: {0}")]
    CborEncode(String),
    #[error("failed to decode CBOR: {0}")]
    CborDecode(String),
    #[error("unsupported graph file format '{found}' (expected '{expected}')")]
    UnsupportedFormatName {
        expected: &'static str,
        found: String,
    },
    #[error("unsupported graph file encoding '{found}' (expected '{expected}')")]
    UnsupportedEncoding {
        expected: &'static str,
        found: String,
    },
    #[error("unsupported graph file container version {found} (max supported: {max_supported})")]
    UnsupportedContainerVersion { found: u32, max_supported: u32 },
    #[error("graph kind mismatch: expected {expected:?}, found {found:?}")]
    GraphKindMismatch {
        expected: GraphFileKind,
        found: GraphFileKind,
    },
    #[error("unsupported schema version {schema_version} for graph kind {kind:?}")]
    UnsupportedSchemaVersion {
        kind: GraphFileKind,
        schema_version: u32,
    },
    #[error(
        "failed to decode graph file envelope ({envelope_error}); \
         also failed legacy raw-CBOR decode ({legacy_error})"
    )]
    EnvelopeAndLegacyDecodeFailed {
        envelope_error: String,
        legacy_error: String,
    },
}

#[derive(Debug, Serialize)]
struct GraphEnvelopeEncode<'a, T: Serialize> {
    format_name: &'static str,
    encoding: &'static str,
    container_version: u32,
    kind: GraphFileKind,
    schema_version: u32,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    metadata: BTreeMap<String, String>,
    payload: &'a T,
}

#[derive(Debug, Deserialize)]
struct GraphEnvelopeDecode {
    #[serde(
        default = "default_format_name",
        alias = "format",
        alias = "file_format"
    )]
    format_name: String,
    #[serde(default = "default_encoding", alias = "codec")]
    encoding: String,
    #[serde(
        default = "default_container_version",
        alias = "format_version",
        alias = "version"
    )]
    container_version: u32,
    #[serde(default, alias = "graph_kind", alias = "type")]
    kind: Option<GraphFileKind>,
    #[serde(
        default = "default_schema_version",
        alias = "graph_version",
        alias = "schema"
    )]
    schema_version: u32,
    #[serde(default, rename = "metadata")]
    _metadata: BTreeMap<String, String>,
    #[serde(alias = "graph")]
    payload: Value,
}

fn default_format_name() -> String {
    GRAPH_FILE_FORMAT_NAME.to_string()
}

fn default_encoding() -> String {
    GRAPH_FILE_ENCODING.to_string()
}

fn default_container_version() -> u32 {
    GRAPH_FILE_CONTAINER_VERSION
}

fn default_schema_version() -> u32 {
    1
}

fn map_encode_error(err: ciborium::ser::Error<std::io::Error>) -> GraphFormatError {
    GraphFormatError::CborEncode(err.to_string())
}

fn map_decode_error(err: ciborium::de::Error<std::io::Error>) -> GraphFormatError {
    GraphFormatError::CborDecode(err.to_string())
}

fn encode_cbor<T: Serialize>(value: &T) -> Result<Vec<u8>, GraphFormatError> {
    let mut bytes = Vec::new();
    ciborium::ser::into_writer(value, &mut bytes).map_err(map_encode_error)?;
    Ok(bytes)
}

fn decode_cbor<T: for<'de> Deserialize<'de>>(bytes: &[u8]) -> Result<T, GraphFormatError> {
    let mut cursor = Cursor::new(bytes);
    ciborium::de::from_reader(&mut cursor).map_err(map_decode_error)
}

fn decode_payload<T: for<'de> Deserialize<'de>>(payload: Value) -> Result<T, GraphFormatError> {
    let payload_bytes = encode_cbor(&payload)?;
    decode_cbor(&payload_bytes)
}

fn validate_envelope_header(
    envelope: &GraphEnvelopeDecode,
    expected_kind: GraphFileKind,
) -> Result<(), GraphFormatError> {
    if envelope.format_name != GRAPH_FILE_FORMAT_NAME {
        return Err(GraphFormatError::UnsupportedFormatName {
            expected: GRAPH_FILE_FORMAT_NAME,
            found: envelope.format_name.clone(),
        });
    }
    if envelope.encoding != GRAPH_FILE_ENCODING {
        return Err(GraphFormatError::UnsupportedEncoding {
            expected: GRAPH_FILE_ENCODING,
            found: envelope.encoding.clone(),
        });
    }
    if envelope.container_version > GRAPH_FILE_CONTAINER_VERSION {
        return Err(GraphFormatError::UnsupportedContainerVersion {
            found: envelope.container_version,
            max_supported: GRAPH_FILE_CONTAINER_VERSION,
        });
    }
    if let Some(found_kind) = envelope.kind
        && found_kind != expected_kind
    {
        return Err(GraphFormatError::GraphKindMismatch {
            expected: expected_kind,
            found: found_kind,
        });
    }
    Ok(())
}

fn decode_super_graph_envelope(
    envelope: GraphEnvelopeDecode,
) -> Result<SuperGraph, GraphFormatError> {
    validate_envelope_header(&envelope, GraphFileKind::SuperGraph)?;
    match envelope.schema_version {
        0 | SUPER_GRAPH_SCHEMA_VERSION => decode_payload(envelope.payload),
        schema_version => Err(GraphFormatError::UnsupportedSchemaVersion {
            kind: GraphFileKind::SuperGraph,
            schema_version,
        }),
    }
}

fn decode_milli_op_graph_envelope(
    envelope: GraphEnvelopeDecode,
) -> Result<MilliOpGraph, GraphFormatError> {
    validate_envelope_header(&envelope, GraphFileKind::MilliOpGraph)?;
    match envelope.schema_version {
        0 | MILLI_OP_GRAPH_SCHEMA_VERSION => decode_payload(envelope.payload),
        schema_version => Err(GraphFormatError::UnsupportedSchemaVersion {
            kind: GraphFileKind::MilliOpGraph,
            schema_version,
        }),
    }
}

fn encode_super_graph_envelope(graph: &SuperGraph) -> Result<Vec<u8>, GraphFormatError> {
    let envelope = GraphEnvelopeEncode {
        format_name: GRAPH_FILE_FORMAT_NAME,
        encoding: GRAPH_FILE_ENCODING,
        container_version: GRAPH_FILE_CONTAINER_VERSION,
        kind: GraphFileKind::SuperGraph,
        schema_version: SUPER_GRAPH_SCHEMA_VERSION,
        metadata: BTreeMap::new(),
        payload: graph,
    };
    encode_cbor(&envelope)
}

fn encode_milli_op_graph_envelope(graph: &MilliOpGraph) -> Result<Vec<u8>, GraphFormatError> {
    let envelope = GraphEnvelopeEncode {
        format_name: GRAPH_FILE_FORMAT_NAME,
        encoding: GRAPH_FILE_ENCODING,
        container_version: GRAPH_FILE_CONTAINER_VERSION,
        kind: GraphFileKind::MilliOpGraph,
        schema_version: MILLI_OP_GRAPH_SCHEMA_VERSION,
        metadata: BTreeMap::new(),
        payload: graph,
    };
    encode_cbor(&envelope)
}

pub fn encode_super_graph_to_vec(graph: &SuperGraph) -> Result<Vec<u8>, GraphFormatError> {
    encode_super_graph_envelope(graph)
}

pub fn decode_super_graph_from_slice(bytes: &[u8]) -> Result<SuperGraph, GraphFormatError> {
    match decode_cbor::<GraphEnvelopeDecode>(bytes) {
        Ok(envelope) => decode_super_graph_envelope(envelope),
        Err(envelope_error) => match decode_cbor::<SuperGraph>(bytes) {
            Ok(graph) => Ok(graph),
            Err(legacy_error) => Err(GraphFormatError::EnvelopeAndLegacyDecodeFailed {
                envelope_error: envelope_error.to_string(),
                legacy_error: legacy_error.to_string(),
            }),
        },
    }
}

pub fn encode_milli_op_graph_to_vec(graph: &MilliOpGraph) -> Result<Vec<u8>, GraphFormatError> {
    encode_milli_op_graph_envelope(graph)
}

pub fn decode_milli_op_graph_from_slice(bytes: &[u8]) -> Result<MilliOpGraph, GraphFormatError> {
    match decode_cbor::<GraphEnvelopeDecode>(bytes) {
        Ok(envelope) => decode_milli_op_graph_envelope(envelope),
        Err(envelope_error) => match decode_cbor::<MilliOpGraph>(bytes) {
            Ok(graph) => Ok(graph),
            Err(legacy_error) => Err(GraphFormatError::EnvelopeAndLegacyDecodeFailed {
                envelope_error: envelope_error.to_string(),
                legacy_error: legacy_error.to_string(),
            }),
        },
    }
}

pub fn write_super_graph<W: Write>(
    mut writer: W,
    graph: &SuperGraph,
) -> Result<(), GraphFormatError> {
    let bytes = encode_super_graph_to_vec(graph)?;
    writer.write_all(&bytes)?;
    Ok(())
}

pub fn read_super_graph<R: Read>(mut reader: R) -> Result<SuperGraph, GraphFormatError> {
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;
    decode_super_graph_from_slice(&bytes)
}

pub fn save_super_graph_to_path(
    path: impl AsRef<Path>,
    graph: &SuperGraph,
) -> Result<(), GraphFormatError> {
    let bytes = encode_super_graph_to_vec(graph)?;
    std::fs::write(path, bytes)?;
    Ok(())
}

pub fn load_super_graph_from_path(path: impl AsRef<Path>) -> Result<SuperGraph, GraphFormatError> {
    let bytes = std::fs::read(path)?;
    decode_super_graph_from_slice(&bytes)
}

pub fn write_milli_op_graph<W: Write>(
    mut writer: W,
    graph: &MilliOpGraph,
) -> Result<(), GraphFormatError> {
    let bytes = encode_milli_op_graph_to_vec(graph)?;
    writer.write_all(&bytes)?;
    Ok(())
}

pub fn read_milli_op_graph<R: Read>(mut reader: R) -> Result<MilliOpGraph, GraphFormatError> {
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;
    decode_milli_op_graph_from_slice(&bytes)
}

pub fn save_milli_op_graph_to_path(
    path: impl AsRef<Path>,
    graph: &MilliOpGraph,
) -> Result<(), GraphFormatError> {
    let bytes = encode_milli_op_graph_to_vec(graph)?;
    std::fs::write(path, bytes)?;
    Ok(())
}

pub fn load_milli_op_graph_from_path(
    path: impl AsRef<Path>,
) -> Result<MilliOpGraph, GraphFormatError> {
    let bytes = std::fs::read(path)?;
    decode_milli_op_graph_from_slice(&bytes)
}

impl SuperGraph {
    pub fn to_cbor_bytes(&self) -> Result<Vec<u8>, GraphFormatError> {
        encode_super_graph_to_vec(self)
    }

    pub fn from_cbor_bytes(bytes: &[u8]) -> Result<Self, GraphFormatError> {
        decode_super_graph_from_slice(bytes)
    }

    pub fn save_cbor(&self, path: impl AsRef<Path>) -> Result<(), GraphFormatError> {
        save_super_graph_to_path(path, self)
    }

    pub fn load_cbor(path: impl AsRef<Path>) -> Result<Self, GraphFormatError> {
        load_super_graph_from_path(path)
    }
}

impl MilliOpGraph {
    pub fn to_cbor_bytes(&self) -> Result<Vec<u8>, GraphFormatError> {
        encode_milli_op_graph_to_vec(self)
    }

    pub fn from_cbor_bytes(bytes: &[u8]) -> Result<Self, GraphFormatError> {
        decode_milli_op_graph_from_slice(bytes)
    }

    pub fn save_cbor(&self, path: impl AsRef<Path>) -> Result<(), GraphFormatError> {
        save_milli_op_graph_to_path(path, self)
    }

    pub fn load_cbor(path: impl AsRef<Path>) -> Result<Self, GraphFormatError> {
        load_milli_op_graph_from_path(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{GlobalId, Graph};
    use crate::super_graph::SuperGraphBuilder;

    #[test]
    fn super_graph_roundtrip_envelope() {
        let mut rng = rand::rng();
        let mut builder = SuperGraphBuilder::new();
        let io_link = builder.new_tensor_link(&mut rng).to_any();
        let graph = builder.build(&mut rng, &[io_link], &[io_link]);

        let bytes = encode_super_graph_to_vec(&graph).unwrap();
        let decoded = decode_super_graph_from_slice(&bytes).unwrap();

        assert_eq!(Graph::global_id(&graph), Graph::global_id(&decoded));
        assert_eq!(graph.input_links, decoded.input_links);
        assert_eq!(graph.output_links, decoded.output_links);
        assert_eq!(graph.nodes.len(), decoded.nodes.len());
        assert_eq!(
            graph.links_by_global_id.len(),
            decoded.links_by_global_id.len()
        );
    }

    #[test]
    fn milli_op_graph_roundtrip_envelope() {
        let mut rng = rand::rng();
        let ext_input = GlobalId::new(&mut rng);
        let (mut graph, input_map) = MilliOpGraph::new([ext_input], &mut rng);
        graph.set_outputs(vec![input_map[&ext_input]]);

        let bytes = encode_milli_op_graph_to_vec(&graph).unwrap();
        let decoded = decode_milli_op_graph_from_slice(&bytes).unwrap();

        assert_eq!(Graph::global_id(&graph), Graph::global_id(&decoded));
        assert_eq!(graph.get_inputs(), decoded.get_inputs());
        assert_eq!(graph.get_outputs(), decoded.get_outputs());
        assert_eq!(graph.get_all_tensors(), decoded.get_all_tensors());
    }

    #[test]
    fn super_graph_legacy_raw_cbor_is_still_supported() {
        let mut rng = rand::rng();
        let mut builder = SuperGraphBuilder::new();
        let io_link = builder.new_tensor_link(&mut rng).to_any();
        let graph = builder.build(&mut rng, &[io_link], &[io_link]);

        let mut raw_bytes = Vec::new();
        ciborium::ser::into_writer(&graph, &mut raw_bytes).unwrap();

        let decoded = decode_super_graph_from_slice(&raw_bytes).unwrap();
        assert_eq!(graph.input_links, decoded.input_links);
        assert_eq!(graph.output_links, decoded.output_links);
    }

    #[test]
    fn kind_mismatch_is_rejected() {
        let mut rng = rand::rng();
        let mut builder = SuperGraphBuilder::new();
        let io_link = builder.new_tensor_link(&mut rng).to_any();
        let graph = builder.build(&mut rng, &[io_link], &[io_link]);
        let bytes = encode_super_graph_to_vec(&graph).unwrap();

        let err = decode_milli_op_graph_from_slice(&bytes).unwrap_err();
        assert!(matches!(err, GraphFormatError::GraphKindMismatch { .. }));
    }
}
