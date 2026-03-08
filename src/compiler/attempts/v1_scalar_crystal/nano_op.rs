//! Nano-op representation and expansion from milli ops.
//!
//! Nano ops are near-scalar operations — the maximally-verbose intermediate
//! representation between milli ops and compiled output. Each nano op is
//! either a single scalar operation, or (future) an operation over a single
//! collapsed unknown dimension.

use crate::graph::{GlobalId, Graph, Node};
use crate::milli_graph::MilliOpGraph;
use crate::milli_graph::ops::AnyMilliOp;
use std::collections::HashMap;

/// SSA-style value reference within the nano op stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NanoValue(pub u32);

/// Scalar binary operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarBinOp {
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,
}

/// Scalar unary operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarUnaryOp {
    Neg,
    Abs,
    Exp,
    Ln,
    Sqrt,
    Reciprocal,
    Tanh,
    Floor,
    Ceil,
}

/// A single nano op in the expanded stream.
#[derive(Debug, Clone)]
pub enum NanoOp {
    /// Load a scalar from a tensor buffer.
    Load {
        dst: NanoValue,
        tensor: GlobalId,
        flat_index: usize,
    },
    /// Store a scalar to a tensor buffer.
    Store {
        tensor: GlobalId,
        flat_index: usize,
        src: NanoValue,
    },
    /// Floating-point literal constant.
    Literal {
        dst: NanoValue,
        value: f64,
    },
    /// Binary scalar operation.
    BinOp {
        dst: NanoValue,
        op: ScalarBinOp,
        a: NanoValue,
        b: NanoValue,
    },
    /// Unary scalar operation.
    UnaryOp {
        dst: NanoValue,
        op: ScalarUnaryOp,
        input: NanoValue,
    },
}

impl NanoOp {
    pub fn dst(&self) -> Option<NanoValue> {
        match self {
            NanoOp::Load { dst, .. } => Some(*dst),
            NanoOp::Store { .. } => None,
            NanoOp::Literal { dst, .. } => Some(*dst),
            NanoOp::BinOp { dst, .. } => Some(*dst),
            NanoOp::UnaryOp { dst, .. } => Some(*dst),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum NanoExpandError {
    #[error("Missing shape for tensor {0}")]
    MissingShape(GlobalId),
    #[error("Unsupported op for nano expansion: {0}")]
    UnsupportedOp(String),
}

/// Expands a MilliOpGraph into a stream of NanoOps.
///
/// All tensor shapes must be fully concrete (no symbolic dimensions).
/// Constants are not expanded — the runtime must pre-fill their buffers.
pub struct NanoOpExpander {
    next_value: u32,
    tensor_shapes: HashMap<GlobalId, Vec<usize>>,
}

impl NanoOpExpander {
    pub fn new(tensor_shapes: HashMap<GlobalId, Vec<usize>>) -> Self {
        Self {
            next_value: 0,
            tensor_shapes,
        }
    }

    fn alloc_value(&mut self) -> NanoValue {
        let v = NanoValue(self.next_value);
        self.next_value += 1;
        v
    }

    fn get_shape(&self, tensor: &GlobalId) -> Result<&[usize], NanoExpandError> {
        self.tensor_shapes
            .get(tensor)
            .map(|s| s.as_slice())
            .ok_or_else(|| NanoExpandError::MissingShape(*tensor))
    }

    fn total_elements(&self, tensor: &GlobalId) -> Result<usize, NanoExpandError> {
        Ok(self.get_shape(tensor)?.iter().product())
    }

    /// Expand the entire graph into nano ops.
    pub fn expand(&mut self, graph: &MilliOpGraph) -> Result<Vec<NanoOp>, NanoExpandError> {
        let mut ops = Vec::new();
        for op_id in graph.op_ordering() {
            let op = graph.get_node_by_id(op_id).unwrap();
            self.expand_op(op, &mut ops)?;
        }
        Ok(ops)
    }

    fn expand_op(
        &mut self,
        op: &AnyMilliOp,
        out: &mut Vec<NanoOp>,
    ) -> Result<(), NanoExpandError> {
        let kind = op.op_kind();
        let inputs: Vec<GlobalId> = op.inputs().collect();
        let outputs: Vec<GlobalId> = op.outputs().collect();

        match kind.as_str() {
            // Constants: runtime pre-fills their buffers, no nano ops needed
            "Constant" | "ConstantOfShape" => Ok(()),

            // Elementwise binary ops
            "Add" | "Sub" | "Mul" | "Div" | "Max" | "Min" => {
                let scalar_op = match kind.as_str() {
                    "Add" => ScalarBinOp::Add,
                    "Sub" => ScalarBinOp::Sub,
                    "Mul" => ScalarBinOp::Mul,
                    "Div" => ScalarBinOp::Div,
                    "Max" => ScalarBinOp::Max,
                    "Min" => ScalarBinOp::Min,
                    _ => unreachable!(),
                };
                self.expand_elementwise_binary(inputs[0], inputs[1], outputs[0], scalar_op, out)
            }

            // Elementwise unary ops
            "Neg" | "Abs" | "Exp" | "Ln" | "Sqrt" | "Reciprocal" | "Floor" | "Ceil" => {
                let scalar_op = match kind.as_str() {
                    "Neg" => ScalarUnaryOp::Neg,
                    "Abs" => ScalarUnaryOp::Abs,
                    "Exp" => ScalarUnaryOp::Exp,
                    "Ln" => ScalarUnaryOp::Ln,
                    "Sqrt" => ScalarUnaryOp::Sqrt,
                    "Reciprocal" => ScalarUnaryOp::Reciprocal,
                    "Floor" => ScalarUnaryOp::Floor,
                    "Ceil" => ScalarUnaryOp::Ceil,
                    _ => unreachable!(),
                };
                self.expand_elementwise_unary(inputs[0], outputs[0], scalar_op, out)
            }

            // Trig ops (reported as their specific name by op_kind())
            "Tanh" => self.expand_elementwise_unary(
                inputs[0],
                outputs[0],
                ScalarUnaryOp::Tanh,
                out,
            ),

            _ => Err(NanoExpandError::UnsupportedOp(kind)),
        }
    }

    fn expand_elementwise_binary(
        &mut self,
        a_id: GlobalId,
        b_id: GlobalId,
        out_id: GlobalId,
        op: ScalarBinOp,
        out: &mut Vec<NanoOp>,
    ) -> Result<(), NanoExpandError> {
        let out_shape = self.get_shape(&out_id)?.to_vec();
        let a_shape = self.get_shape(&a_id)?.to_vec();
        let b_shape = self.get_shape(&b_id)?.to_vec();

        let a_strides = broadcast_strides(&a_shape, &out_shape);
        let b_strides = broadcast_strides(&b_shape, &out_shape);

        let total: usize = out_shape.iter().product();
        for flat_out in 0..total {
            let multi = flat_to_multi(flat_out, &out_shape);
            let flat_a = multi_to_flat(&multi, &a_strides);
            let flat_b = multi_to_flat(&multi, &b_strides);

            let va = self.alloc_value();
            out.push(NanoOp::Load {
                dst: va,
                tensor: a_id,
                flat_index: flat_a,
            });

            let vb = self.alloc_value();
            out.push(NanoOp::Load {
                dst: vb,
                tensor: b_id,
                flat_index: flat_b,
            });

            let vout = self.alloc_value();
            out.push(NanoOp::BinOp {
                dst: vout,
                op,
                a: va,
                b: vb,
            });

            out.push(NanoOp::Store {
                tensor: out_id,
                flat_index: flat_out,
                src: vout,
            });
        }
        Ok(())
    }

    fn expand_elementwise_unary(
        &mut self,
        input_id: GlobalId,
        out_id: GlobalId,
        op: ScalarUnaryOp,
        out: &mut Vec<NanoOp>,
    ) -> Result<(), NanoExpandError> {
        let total = self.total_elements(&out_id)?;
        for flat_idx in 0..total {
            let vin = self.alloc_value();
            out.push(NanoOp::Load {
                dst: vin,
                tensor: input_id,
                flat_index: flat_idx,
            });

            let vout = self.alloc_value();
            out.push(NanoOp::UnaryOp {
                dst: vout,
                op,
                input: vin,
            });

            out.push(NanoOp::Store {
                tensor: out_id,
                flat_index: flat_idx,
                src: vout,
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Index helpers
// ---------------------------------------------------------------------------

/// Convert a flat (row-major) index to a multi-dimensional index.
fn flat_to_multi(flat: usize, shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    if ndim == 0 {
        return vec![];
    }
    let mut indices = vec![0usize; ndim];
    let mut remaining = flat;
    for i in (0..ndim).rev() {
        indices[i] = remaining % shape[i];
        remaining /= shape[i];
    }
    indices
}

/// Compute row-major strides for `source` shape when broadcast to `target` shape.
/// Dimensions of size 1 (or missing leading dims) get stride 0.
fn broadcast_strides(source: &[usize], target: &[usize]) -> Vec<usize> {
    let t_ndim = target.len();
    let s_ndim = source.len();
    let offset = t_ndim.saturating_sub(s_ndim);
    let mut strides = vec![0usize; t_ndim];

    // Compute strides right-to-left
    let mut stride = 1usize;
    for i in (0..s_ndim).rev() {
        let ti = i + offset;
        if source[i] == target[ti] {
            strides[ti] = stride;
            stride *= source[i];
        } else if source[i] == 1 {
            strides[ti] = 0; // broadcast
        } else {
            panic!(
                "Incompatible shapes for broadcasting: {:?} vs {:?}",
                source, target
            );
        }
    }
    // Leading dimensions (before offset) stay 0 — implicit broadcast from shape [1]
    strides
}

/// Compute flat index from multi-dimensional index and strides.
fn multi_to_flat(indices: &[usize], strides: &[usize]) -> usize {
    indices.iter().zip(strides).map(|(&i, &s)| i * s).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GlobalId;
    use crate::milli_graph::MilliOpGraph;
    use crate::milli_graph::ops::SimpleBinary;

    fn make_id(rng: &mut impl rand::Rng) -> GlobalId {
        GlobalId::new(rng)
    }

    #[test]
    fn test_flat_to_multi() {
        assert_eq!(flat_to_multi(0, &[2, 3]), vec![0, 0]);
        assert_eq!(flat_to_multi(1, &[2, 3]), vec![0, 1]);
        assert_eq!(flat_to_multi(3, &[2, 3]), vec![1, 0]);
        assert_eq!(flat_to_multi(5, &[2, 3]), vec![1, 2]);
    }

    #[test]
    fn test_broadcast_strides() {
        // Same shape — normal strides
        assert_eq!(broadcast_strides(&[2, 3], &[2, 3]), vec![3, 1]);
        // Broadcast dim 0
        assert_eq!(broadcast_strides(&[1, 3], &[2, 3]), vec![0, 1]);
        // Broadcast dim 1
        assert_eq!(broadcast_strides(&[2, 1], &[2, 3]), vec![1, 0]);
        // Scalar broadcast
        assert_eq!(broadcast_strides(&[1], &[2, 3]), vec![0, 0]);
        // Missing leading dim
        assert_eq!(broadcast_strides(&[3], &[2, 3]), vec![0, 1]);
    }

    #[test]
    fn test_expand_add() {
        let mut rng = wyrand::WyRand::new(42);

        // Build graph: out = a + b, shapes [2, 2]
        let ext_a = make_id(&mut rng);
        let ext_b = make_id(&mut rng);
        let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
        let int_a = input_map[&ext_a];
        let int_b = input_map[&ext_b];
        let int_out = SimpleBinary::add(&mut graph, int_a, int_b, &mut rng);

        // Provide shapes for all tensors
        let mut shapes = HashMap::new();
        shapes.insert(int_a, vec![2, 2]);
        shapes.insert(int_b, vec![2, 2]);
        shapes.insert(int_out, vec![2, 2]);

        let mut expander = NanoOpExpander::new(shapes);
        let nano_ops = expander.expand(&graph).unwrap();

        // 4 elements × (Load, Load, BinOp, Store) = 16 nano ops
        assert_eq!(nano_ops.len(), 16);

        // Verify pattern: groups of 4
        for chunk in nano_ops.chunks(4) {
            assert!(matches!(chunk[0], NanoOp::Load { .. }));
            assert!(matches!(chunk[1], NanoOp::Load { .. }));
            assert!(matches!(chunk[2], NanoOp::BinOp { op: ScalarBinOp::Add, .. }));
            assert!(matches!(chunk[3], NanoOp::Store { .. }));
        }
    }

    #[test]
    fn test_expand_broadcast() {
        let mut rng = wyrand::WyRand::new(42);

        // Build graph: out = a + b, a=[2,3], b=[1,3] (broadcast dim 0)
        let ext_a = make_id(&mut rng);
        let ext_b = make_id(&mut rng);
        let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
        let int_a = input_map[&ext_a];
        let int_b = input_map[&ext_b];
        let int_out = SimpleBinary::add(&mut graph, int_a, int_b, &mut rng);

        let mut shapes = HashMap::new();
        shapes.insert(int_a, vec![2, 3]);
        shapes.insert(int_b, vec![1, 3]);
        shapes.insert(int_out, vec![2, 3]);

        let mut expander = NanoOpExpander::new(shapes);
        let nano_ops = expander.expand(&graph).unwrap();

        // 6 elements × 4 ops = 24
        assert_eq!(nano_ops.len(), 24);

        // Check that b's loads reuse flat indices 0,1,2 for both rows
        let b_loads: Vec<usize> = nano_ops
            .iter()
            .filter_map(|op| match op {
                NanoOp::Load { tensor, flat_index, .. } if *tensor == int_b => {
                    Some(*flat_index)
                }
                _ => None,
            })
            .collect();
        // Row 0: b[0], b[1], b[2]; Row 1: b[0], b[1], b[2]
        assert_eq!(b_loads, vec![0, 1, 2, 0, 1, 2]);
    }
}
