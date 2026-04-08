//! Emit a single `AtomGroup` as a loop or unrolled sequence.
//!
//! The base case for non-fused, non-reduce groups: load operands via
//! [`super::super::codec`], call into [`super::super::ops`], store the
//! result, and loop. Knows nothing about op semantics — just sequences
//! the codec/op calls per atom.
//!
//! Phase 0 stub. Implementation lands in phase 2.
