# Graph File Format (CBOR)

This project now stores `SuperGraph` and `MilliOpGraph` in a versioned CBOR envelope.

## Goals

- Self-describing payloads.
- Forward compatibility for added fields.
- Explicit handling for non-backwards-compatible schema changes.
- Clear migration points.

## Container

Both graph kinds use the same top-level envelope:

```text
{
  "format_name": "whisper_tensor_graph",
  "encoding": "cbor",
  "container_version": 1,
  "kind": "super_graph" | "milli_op_graph",
  "schema_version": <u32>,
  "metadata": { ... },         // optional, currently empty
  "payload": <graph payload>
}
```

Current schema versions:

- `SuperGraph`: `1`
- `MilliOpGraph`: `1`

## Compatibility behavior

- Unknown added fields in envelope/payload are ignored by serde default behavior.
- Legacy raw-CBOR payloads (without envelope) are still accepted on load.
- Graph kind mismatches are rejected.
- Unknown future `schema_version` values are rejected with an explicit error.

## Migration strategy

When a breaking payload change is needed:

1. Bump the relevant schema version constant.
2. Add a decode branch for the old version.
3. Deserialize old payload into a legacy Rust type.
4. Convert (migrate) into the current graph type.
5. Keep writing only the latest schema version.

This keeps the write path simple and the read path responsible for migrations.
