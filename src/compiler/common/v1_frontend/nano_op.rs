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
pub struct NanoValue(pub u64);

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
    Literal { dst: NanoValue, value: f64 },
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
    next_value: u64,
    tensor_shapes: HashMap<GlobalId, Vec<usize>>,
    /// Target number of output elements to sample per op.
    /// `0` means emit all elements (no sampling).
    max_output_samples: usize,
}

impl NanoOpExpander {
    pub fn new(tensor_shapes: HashMap<GlobalId, Vec<usize>>) -> Self {
        Self {
            next_value: 0,
            tensor_shapes,
            max_output_samples: 0,
        }
    }

    /// Create an expander that samples at most `n` output elements per op.
    ///
    /// Sampled elements are spread across the output space on a uniform grid
    /// so that every output axis has varied coordinates — enough for affine
    /// coefficient inference.  The full reduction depth (K) is always
    /// preserved; only the number of output positions is bounded.
    pub fn new_sampled(
        tensor_shapes: HashMap<GlobalId, Vec<usize>>,
        max_output_samples: usize,
    ) -> Self {
        Self {
            next_value: 0,
            tensor_shapes,
            max_output_samples,
        }
    }

    fn alloc_value(&mut self) -> NanoValue {
        let v = NanoValue(self.next_value);
        self.next_value = self
            .next_value
            .checked_add(1)
            .expect("NanoValue id overflow");
        v
    }

    fn get_shape(&self, tensor: &GlobalId) -> Result<&[usize], NanoExpandError> {
        self.tensor_shapes
            .get(tensor)
            .map(|s| s.as_slice())
            .ok_or(NanoExpandError::MissingShape(*tensor))
    }

    /// Expand the entire graph into nano ops.
    pub fn expand(&mut self, graph: &MilliOpGraph) -> Result<Vec<NanoOp>, NanoExpandError> {
        let mut out = Vec::new();
        self.expand_into(graph, |op| out.push(op))?;
        Ok(out)
    }

    /// Stream nano ops through a sink callback.
    pub fn expand_into<F>(
        &mut self,
        graph: &MilliOpGraph,
        mut sink: F,
    ) -> Result<(), NanoExpandError>
    where
        F: FnMut(NanoOp),
    {
        for op in self.expand_iter(graph) {
            sink(op?);
        }
        Ok(())
    }

    /// Returns an iterator that yields nano ops lazily.
    ///
    /// This API allows callers to process nano ops without materializing one
    /// large vector. Existing consumers can still call `collect()`.
    pub fn expand_iter<'a>(&'a mut self, graph: &'a MilliOpGraph) -> NanoOpIterator<'a> {
        NanoOpIterator::new(self, graph)
    }

    fn make_emitter(&self, op: &AnyMilliOp) -> Result<Option<OpEmitter>, NanoExpandError> {
        let kind = op.op_kind();
        let inputs: Vec<GlobalId> = op.inputs().collect();
        let outputs: Vec<GlobalId> = op.outputs().collect();

        match kind.as_str() {
            // Constants: runtime pre-fills their buffers, no nano ops needed.
            "Constant" | "ConstantOfShape" => Ok(None),

            // Elementwise binary ops.
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
                Ok(Some(OpEmitter::Binary(self.make_binary_emitter(
                    inputs[0], inputs[1], outputs[0], scalar_op,
                )?)))
            }

            // Elementwise unary ops.
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
                Ok(Some(OpEmitter::Unary(
                    self.make_unary_emitter(inputs[0], outputs[0], scalar_op)?,
                )))
            }

            // Trig ops (reported as their specific name by op_kind()).
            "Tanh" => Ok(Some(OpEmitter::Unary(self.make_unary_emitter(
                inputs[0],
                outputs[0],
                ScalarUnaryOp::Tanh,
            )?))),

            "MatMul" => Ok(Some(OpEmitter::MatMul(
                self.make_matmul_emitter(inputs[0], inputs[1], outputs[0])?,
            ))),

            _ => Err(NanoExpandError::UnsupportedOp(kind)),
        }
    }

    fn make_binary_emitter(
        &self,
        a_id: GlobalId,
        b_id: GlobalId,
        out_id: GlobalId,
        op: ScalarBinOp,
    ) -> Result<BinaryEmitter, NanoExpandError> {
        let out_shape = self.get_shape(&out_id)?.to_vec();
        let a_shape = self.get_shape(&a_id)?.to_vec();
        let b_shape = self.get_shape(&b_id)?.to_vec();
        let a_strides = broadcast_strides(&a_shape, &out_shape);
        let b_strides = broadcast_strides(&b_shape, &out_shape);

        let total: usize = out_shape.iter().product();
        let output_stride = if self.max_output_samples > 0 && total > self.max_output_samples {
            (total / self.max_output_samples).max(1)
        } else {
            1
        };

        Ok(BinaryEmitter {
            a_id,
            b_id,
            out_id,
            op,
            out_shape: out_shape.clone(),
            a_strides,
            b_strides,
            total,
            flat_out: 0,
            output_stride,
            stage: BinaryStage::LoadA,
            flat_a: 0,
            flat_b: 0,
            va: None,
            vb: None,
            vout: None,
        })
    }

    fn make_unary_emitter(
        &self,
        input_id: GlobalId,
        out_id: GlobalId,
        op: ScalarUnaryOp,
    ) -> Result<UnaryEmitter, NanoExpandError> {
        let out_shape = self.get_shape(&out_id)?.to_vec();
        let total: usize = out_shape.iter().product();
        let output_stride = if self.max_output_samples > 0 && total > self.max_output_samples {
            (total / self.max_output_samples).max(1)
        } else {
            1
        };
        Ok(UnaryEmitter {
            input_id,
            out_id,
            op,
            total,
            flat_idx: 0,
            output_stride,
            stage: UnaryStage::Load,
            vin: None,
            vout: None,
        })
    }

    fn make_matmul_emitter(
        &self,
        a_id: GlobalId,
        b_id: GlobalId,
        out_id: GlobalId,
    ) -> Result<MatMulEmitter, NanoExpandError> {
        let a_shape = self.get_shape(&a_id)?.to_vec();
        let b_shape = self.get_shape(&b_id)?.to_vec();
        let out_shape = self.get_shape(&out_id)?.to_vec();

        if a_shape.len() != 2 || b_shape.len() != 2 || out_shape.len() != 2 {
            return Err(NanoExpandError::UnsupportedOp(format!(
                "MatMul currently supports rank-2 only; got A{:?}, B{:?}, Out{:?}",
                a_shape, b_shape, out_shape
            )));
        }

        let (m, k) = (a_shape[0], a_shape[1]);
        let (k2, n) = (b_shape[0], b_shape[1]);
        if k != k2 || out_shape != vec![m, n] {
            return Err(NanoExpandError::UnsupportedOp(format!(
                "MatMul shape mismatch: A{:?}, B{:?}, Out{:?}",
                a_shape, b_shape, out_shape
            )));
        }

        let phase = if m == 0 || n == 0 {
            MatMulPhase::Done
        } else {
            MatMulPhase::OutputStart
        };

        // Compute sampling strides: pick ~sqrt(max_samples) points per axis.
        let (row_stride, col_stride) = if self.max_output_samples > 0 {
            let per_axis = (self.max_output_samples as f64).sqrt().ceil() as usize;
            let rs = if m > per_axis { m / per_axis } else { 1 };
            let cs = if n > per_axis { n / per_axis } else { 1 };
            (rs.max(1), cs.max(1))
        } else {
            (1, 1)
        };

        Ok(MatMulEmitter {
            a_id,
            b_id,
            out_id,
            m,
            n,
            k,
            row: 0,
            col: 0,
            kk: 0,
            row_stride,
            col_stride,
            phase,
            acc: None,
            va: None,
            vb: None,
            vm: None,
        })
    }
}

pub struct NanoOpIterator<'a> {
    expander: &'a mut NanoOpExpander,
    graph: &'a MilliOpGraph,
    op_ids: Vec<GlobalId>,
    next_op: usize,
    active: Option<OpEmitter>,
    finished: bool,
}

impl<'a> NanoOpIterator<'a> {
    fn new(expander: &'a mut NanoOpExpander, graph: &'a MilliOpGraph) -> Self {
        let op_ids: Vec<GlobalId> = graph.op_ordering().to_vec();
        Self {
            expander,
            graph,
            op_ids,
            next_op: 0,
            active: None,
            finished: false,
        }
    }
}

impl Iterator for NanoOpIterator<'_> {
    type Item = Result<NanoOp, NanoExpandError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        loop {
            if let Some(active) = &mut self.active {
                if let Some(op) = active.next(self.expander) {
                    return Some(Ok(op));
                }
                self.active = None;
            }

            if self.next_op >= self.op_ids.len() {
                self.finished = true;
                return None;
            }

            let op_id = self.op_ids[self.next_op];
            self.next_op += 1;
            let op = self.graph.get_node_by_id(&op_id).unwrap();
            match self.expander.make_emitter(op) {
                Ok(Some(emitter)) => {
                    self.active = Some(emitter);
                }
                Ok(None) => {}
                Err(err) => {
                    self.finished = true;
                    return Some(Err(err));
                }
            }
        }
    }
}

enum OpEmitter {
    Binary(BinaryEmitter),
    Unary(UnaryEmitter),
    MatMul(MatMulEmitter),
}

impl OpEmitter {
    fn next(&mut self, expander: &mut NanoOpExpander) -> Option<NanoOp> {
        match self {
            OpEmitter::Binary(inner) => inner.next(expander),
            OpEmitter::Unary(inner) => inner.next(expander),
            OpEmitter::MatMul(inner) => inner.next(expander),
        }
    }
}

#[derive(Clone, Copy)]
enum BinaryStage {
    LoadA,
    LoadB,
    Compute,
    Store,
}

struct BinaryEmitter {
    a_id: GlobalId,
    b_id: GlobalId,
    out_id: GlobalId,
    op: ScalarBinOp,
    out_shape: Vec<usize>,
    a_strides: Vec<usize>,
    b_strides: Vec<usize>,
    total: usize,
    flat_out: usize,
    /// Step between sampled output elements (1 = no sampling).
    output_stride: usize,
    stage: BinaryStage,
    flat_a: usize,
    flat_b: usize,
    va: Option<NanoValue>,
    vb: Option<NanoValue>,
    vout: Option<NanoValue>,
}

impl BinaryEmitter {
    fn next(&mut self, expander: &mut NanoOpExpander) -> Option<NanoOp> {
        if self.flat_out >= self.total {
            return None;
        }

        let op = match self.stage {
            BinaryStage::LoadA => {
                let multi = flat_to_multi(self.flat_out, &self.out_shape);
                self.flat_a = multi_to_flat(&multi, &self.a_strides);
                self.flat_b = multi_to_flat(&multi, &self.b_strides);

                let va = expander.alloc_value();
                self.va = Some(va);
                self.stage = BinaryStage::LoadB;
                NanoOp::Load {
                    dst: va,
                    tensor: self.a_id,
                    flat_index: self.flat_a,
                }
            }
            BinaryStage::LoadB => {
                let vb = expander.alloc_value();
                self.vb = Some(vb);
                self.stage = BinaryStage::Compute;
                NanoOp::Load {
                    dst: vb,
                    tensor: self.b_id,
                    flat_index: self.flat_b,
                }
            }
            BinaryStage::Compute => {
                let vout = expander.alloc_value();
                self.vout = Some(vout);
                self.stage = BinaryStage::Store;
                NanoOp::BinOp {
                    dst: vout,
                    op: self.op,
                    a: self.va.unwrap(),
                    b: self.vb.unwrap(),
                }
            }
            BinaryStage::Store => {
                let store_idx = self.flat_out;
                self.stage = BinaryStage::LoadA;
                self.flat_out += self.output_stride;
                NanoOp::Store {
                    tensor: self.out_id,
                    flat_index: store_idx,
                    src: self.vout.unwrap(),
                }
            }
        };

        Some(op)
    }
}

#[derive(Clone, Copy)]
enum UnaryStage {
    Load,
    Compute,
    Store,
}

struct UnaryEmitter {
    input_id: GlobalId,
    out_id: GlobalId,
    op: ScalarUnaryOp,
    total: usize,
    flat_idx: usize,
    /// Step between sampled output elements (1 = no sampling).
    output_stride: usize,
    stage: UnaryStage,
    vin: Option<NanoValue>,
    vout: Option<NanoValue>,
}

impl UnaryEmitter {
    fn next(&mut self, expander: &mut NanoOpExpander) -> Option<NanoOp> {
        if self.flat_idx >= self.total {
            return None;
        }

        let op = match self.stage {
            UnaryStage::Load => {
                let vin = expander.alloc_value();
                self.vin = Some(vin);
                self.stage = UnaryStage::Compute;
                NanoOp::Load {
                    dst: vin,
                    tensor: self.input_id,
                    flat_index: self.flat_idx,
                }
            }
            UnaryStage::Compute => {
                let vout = expander.alloc_value();
                self.vout = Some(vout);
                self.stage = UnaryStage::Store;
                NanoOp::UnaryOp {
                    dst: vout,
                    op: self.op,
                    input: self.vin.unwrap(),
                }
            }
            UnaryStage::Store => {
                let store_idx = self.flat_idx;
                self.stage = UnaryStage::Load;
                self.flat_idx += self.output_stride;
                NanoOp::Store {
                    tensor: self.out_id,
                    flat_index: store_idx,
                    src: self.vout.unwrap(),
                }
            }
        };

        Some(op)
    }
}

#[derive(Clone, Copy)]
enum MatMulPhase {
    OutputStart,
    LoadA,
    LoadB,
    Mul,
    Add,
    Store,
    Done,
}

struct MatMulEmitter {
    a_id: GlobalId,
    b_id: GlobalId,
    out_id: GlobalId,
    m: usize,
    n: usize,
    k: usize,
    row: usize,
    col: usize,
    kk: usize,
    /// Row step between sampled output elements (1 = no sampling).
    row_stride: usize,
    /// Column step between sampled output elements (1 = no sampling).
    col_stride: usize,
    phase: MatMulPhase,
    acc: Option<NanoValue>,
    va: Option<NanoValue>,
    vb: Option<NanoValue>,
    vm: Option<NanoValue>,
}

impl MatMulEmitter {
    fn next(&mut self, expander: &mut NanoOpExpander) -> Option<NanoOp> {
        match self.phase {
            MatMulPhase::Done => None,
            MatMulPhase::OutputStart => {
                if self.row >= self.m {
                    self.phase = MatMulPhase::Done;
                    return None;
                }

                self.kk = 0;
                let acc = expander.alloc_value();
                self.acc = Some(acc);
                self.phase = if self.k == 0 {
                    MatMulPhase::Store
                } else {
                    MatMulPhase::LoadA
                };

                Some(NanoOp::Literal {
                    dst: acc,
                    value: 0.0,
                })
            }
            MatMulPhase::LoadA => {
                let va = expander.alloc_value();
                self.va = Some(va);
                self.phase = MatMulPhase::LoadB;
                Some(NanoOp::Load {
                    dst: va,
                    tensor: self.a_id,
                    flat_index: self.row * self.k + self.kk,
                })
            }
            MatMulPhase::LoadB => {
                let vb = expander.alloc_value();
                self.vb = Some(vb);
                self.phase = MatMulPhase::Mul;
                Some(NanoOp::Load {
                    dst: vb,
                    tensor: self.b_id,
                    flat_index: self.kk * self.n + self.col,
                })
            }
            MatMulPhase::Mul => {
                let vm = expander.alloc_value();
                self.vm = Some(vm);
                self.phase = MatMulPhase::Add;
                Some(NanoOp::BinOp {
                    dst: vm,
                    op: ScalarBinOp::Mul,
                    a: self.va.unwrap(),
                    b: self.vb.unwrap(),
                })
            }
            MatMulPhase::Add => {
                let prev_acc = self.acc.unwrap();
                let vs = expander.alloc_value();
                self.acc = Some(vs);
                self.kk += 1;
                self.phase = if self.kk < self.k {
                    MatMulPhase::LoadA
                } else {
                    MatMulPhase::Store
                };
                Some(NanoOp::BinOp {
                    dst: vs,
                    op: ScalarBinOp::Add,
                    a: prev_acc,
                    b: self.vm.unwrap(),
                })
            }
            MatMulPhase::Store => {
                let store = NanoOp::Store {
                    tensor: self.out_id,
                    flat_index: self.row * self.n + self.col,
                    src: self.acc.unwrap(),
                };

                self.col += self.col_stride;
                if self.col >= self.n {
                    self.col = 0;
                    self.row += self.row_stride;
                }
                self.phase = MatMulPhase::OutputStart;
                Some(store)
            }
        }
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

    // Compute strides right-to-left.
    let mut stride = 1usize;
    for i in (0..s_ndim).rev() {
        let ti = i + offset;
        if source[i] == target[ti] {
            strides[ti] = stride;
            stride *= source[i];
        } else if source[i] == 1 {
            strides[ti] = 0;
        } else {
            panic!(
                "Incompatible shapes for broadcasting: {:?} vs {:?}",
                source, target
            );
        }
    }
    strides
}

/// Compute flat index from multi-dimensional index and strides.
fn multi_to_flat(indices: &[usize], strides: &[usize]) -> usize {
    indices.iter().zip(strides).map(|(&i, &s)| i * s).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::graph::GlobalId;
    use crate::milli_graph::MilliOpGraph;
    use crate::milli_graph::ops::{MatMul, SimpleBinary};

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
        assert_eq!(broadcast_strides(&[2, 3], &[2, 3]), vec![3, 1]);
        assert_eq!(broadcast_strides(&[1, 3], &[2, 3]), vec![0, 1]);
        assert_eq!(broadcast_strides(&[2, 1], &[2, 3]), vec![1, 0]);
        assert_eq!(broadcast_strides(&[1], &[2, 3]), vec![0, 0]);
        assert_eq!(broadcast_strides(&[3], &[2, 3]), vec![0, 1]);
    }

    #[test]
    fn test_expand_add() {
        let mut rng = wyrand::WyRand::new(42);

        let ext_a = make_id(&mut rng);
        let ext_b = make_id(&mut rng);
        let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
        let int_a = input_map[&ext_a];
        let int_b = input_map[&ext_b];
        let int_out = SimpleBinary::add(&mut graph, int_a, int_b, &mut rng);

        let mut shapes = HashMap::new();
        shapes.insert(int_a, vec![2, 2]);
        shapes.insert(int_b, vec![2, 2]);
        shapes.insert(int_out, vec![2, 2]);

        let mut expander = NanoOpExpander::new(shapes);
        let nano_ops = expander.expand(&graph).unwrap();

        assert_eq!(nano_ops.len(), 16);
        for chunk in nano_ops.chunks(4) {
            assert!(matches!(chunk[0], NanoOp::Load { .. }));
            assert!(matches!(chunk[1], NanoOp::Load { .. }));
            assert!(matches!(
                chunk[2],
                NanoOp::BinOp {
                    op: ScalarBinOp::Add,
                    ..
                }
            ));
            assert!(matches!(chunk[3], NanoOp::Store { .. }));
        }
    }

    #[test]
    fn test_expand_broadcast() {
        let mut rng = wyrand::WyRand::new(42);

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
        assert_eq!(nano_ops.len(), 24);

        let b_loads: Vec<usize> = nano_ops
            .iter()
            .filter_map(|op| match op {
                NanoOp::Load {
                    tensor, flat_index, ..
                } if *tensor == int_b => Some(*flat_index),
                _ => None,
            })
            .collect();
        assert_eq!(b_loads, vec![0, 1, 2, 0, 1, 2]);
    }

    #[test]
    fn test_expand_matmul_rank2() {
        let mut rng = wyrand::WyRand::new(123);
        let ext_a = make_id(&mut rng);
        let ext_b = make_id(&mut rng);
        let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
        let int_a = input_map[&ext_a];
        let int_b = input_map[&ext_b];
        let int_out =
            MatMul::push_new_default_precision(&mut graph, int_a, int_b, DType::F32, &mut rng);

        let mut shapes = HashMap::new();
        shapes.insert(int_a, vec![2, 3]);
        shapes.insert(int_b, vec![3, 4]);
        shapes.insert(int_out, vec![2, 4]);

        let mut expander = NanoOpExpander::new(shapes);
        let nano_ops = expander.expand(&graph).unwrap();

        let stores = nano_ops
            .iter()
            .filter(|op| matches!(op, NanoOp::Store { tensor, .. } if *tensor == int_out))
            .count();
        assert_eq!(stores, 8);
        assert_eq!(nano_ops.len(), 8 * (1 + 3 * 4 + 1));
    }

    #[test]
    fn test_expand_iter_collect_matches_expand() {
        let mut rng = wyrand::WyRand::new(7);
        let ext_a = make_id(&mut rng);
        let ext_b = make_id(&mut rng);
        let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
        let int_a = input_map[&ext_a];
        let int_b = input_map[&ext_b];
        let int_out = SimpleBinary::mul(&mut graph, int_a, int_b, &mut rng);

        let mut shapes = HashMap::new();
        shapes.insert(int_a, vec![3, 5]);
        shapes.insert(int_b, vec![3, 5]);
        shapes.insert(int_out, vec![3, 5]);

        let mut exp_a = NanoOpExpander::new(shapes.clone());
        let vec_ops = exp_a.expand(&graph).unwrap();

        let mut exp_b = NanoOpExpander::new(shapes);
        let iter_ops: Vec<NanoOp> = exp_b
            .expand_iter(&graph)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(vec_ops.len(), iter_ops.len());
        assert!(iter_ops.iter().all(|op| matches!(
            op,
            NanoOp::Load { .. }
                | NanoOp::Store { .. }
                | NanoOp::Literal { .. }
                | NanoOp::BinOp { .. }
                | NanoOp::UnaryOp { .. }
        )));
    }
}
