//! Planner: walks the milli graph and builds a kernel plan.
//!
//! The planner decides which ops to fuse, how to iterate, and what
//! loop structures to emit. Consecutive elementwise ops with compatible
//! shapes are fused into a single kernel so intermediates stay in registers.

use super::kernel::*;
use crate::graph::{GlobalId, Graph, Node};
use crate::milli_graph::MilliOpGraph;
use std::collections::{HashMap, HashSet};

#[derive(Debug, thiserror::Error)]
pub enum PlanError {
    #[error("Missing shape for tensor {0}")]
    MissingShape(GlobalId),
    #[error("Unsupported op: {0}")]
    UnsupportedOp(String),
    #[error("MatMul dimension mismatch: A inner {0} != B inner {1}")]
    MatMulDimMismatch(usize, usize),
    #[error("Incompatible broadcast shapes: {0:?} vs {1:?}")]
    IncompatibleBroadcast(Vec<usize>, Vec<usize>),
}

/// Plan compilation of a milli graph into kernel ops.
pub fn plan(
    graph: &MilliOpGraph,
    shapes: &HashMap<GlobalId, Vec<usize>>,
) -> Result<Vec<KernelOp>, PlanError> {
    // Precompute: for each tensor, how many ops consume it.
    let consumer_ops = build_consumer_map(graph);

    // The set of tensors that are graph outputs (must always be stored).
    let output_tensors: HashSet<GlobalId> = graph
        .output_link_ids()
        .map(|(_, internal)| internal)
        .collect();

    let mut kernels = Vec::new();
    let mut current_group: Option<FusionGroup> = None;

    for op_id in graph.op_ordering() {
        let op = graph.get_node_by_id(op_id).unwrap();
        let kind = op.op_kind();

        match classify_op(&kind) {
            OpClass::Constant => continue,
            OpClass::Elementwise => {
                let output_id = op.outputs().next().unwrap();
                let out_shape = shapes
                    .get(&output_id)
                    .ok_or(PlanError::MissingShape(output_id))?;

                if let Some(group) = &mut current_group {
                    if group.dims == *out_shape {
                        // Fuse into current group.
                        add_op_to_group(group, op, shapes)?;
                        continue;
                    } else {
                        // Shape mismatch — flush and start new group.
                        kernels.push(finalize_group(
                            current_group.take().unwrap(),
                            &consumer_ops,
                            &output_tensors,
                        ));
                    }
                }

                let mut group = FusionGroup::new(out_shape.clone());
                add_op_to_group(&mut group, op, shapes)?;
                current_group = Some(group);
            }
            OpClass::MatMul => {
                // Flush any pending elementwise group.
                if let Some(group) = current_group.take() {
                    kernels.push(finalize_group(group, &consumer_ops, &output_tensors));
                }

                let inputs: Vec<GlobalId> = op.inputs().collect();
                let outputs: Vec<GlobalId> = op.outputs().collect();
                kernels.push(build_gemm_kernel(
                    inputs[0], inputs[1], outputs[0], shapes,
                )?);
            }
            OpClass::Unsupported => {
                return Err(PlanError::UnsupportedOp(kind));
            }
        }
    }

    // Flush final group.
    if let Some(group) = current_group.take() {
        kernels.push(finalize_group(group, &consumer_ops, &output_tensors));
    }

    Ok(kernels)
}

// ---------------------------------------------------------------------------
// Op classification
// ---------------------------------------------------------------------------

enum OpClass {
    Constant,
    Elementwise,
    MatMul,
    Unsupported,
}

fn classify_op(kind: &str) -> OpClass {
    match kind {
        "Constant" | "ConstantOfShape" => OpClass::Constant,
        "Add" | "Sub" | "Mul" | "Div" | "Max" | "Min" | "Neg" | "Abs" | "Exp" | "Ln"
        | "Sqrt" | "Reciprocal" | "Tanh" | "Floor" | "Ceil" => OpClass::Elementwise,
        "MatMul" => OpClass::MatMul,
        _ => OpClass::Unsupported,
    }
}

fn scalar_bin_op(kind: &str) -> Option<ScalarBinOp> {
    match kind {
        "Add" => Some(ScalarBinOp::Add),
        "Sub" => Some(ScalarBinOp::Sub),
        "Mul" => Some(ScalarBinOp::Mul),
        "Div" => Some(ScalarBinOp::Div),
        "Max" => Some(ScalarBinOp::Max),
        "Min" => Some(ScalarBinOp::Min),
        _ => None,
    }
}

fn scalar_unary_op(kind: &str) -> Option<ScalarUnaryOp> {
    match kind {
        "Neg" => Some(ScalarUnaryOp::Neg),
        "Abs" => Some(ScalarUnaryOp::Abs),
        "Exp" => Some(ScalarUnaryOp::Exp),
        "Ln" => Some(ScalarUnaryOp::Ln),
        "Sqrt" => Some(ScalarUnaryOp::Sqrt),
        "Reciprocal" => Some(ScalarUnaryOp::Reciprocal),
        "Tanh" => Some(ScalarUnaryOp::Tanh),
        "Floor" => Some(ScalarUnaryOp::Floor),
        "Ceil" => Some(ScalarUnaryOp::Ceil),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Fusion group
// ---------------------------------------------------------------------------

struct FusionGroup {
    /// Iteration space = output shape.
    dims: Vec<usize>,
    /// Body ops built so far.
    body: Vec<BodyOp>,
    /// Tensor id → body op index for values available in-register.
    value_map: HashMap<GlobalId, usize>,
    /// Set of ops (by GlobalId) in this group.
    op_ids: HashSet<GlobalId>,
    /// Tensors produced by ops in this group.
    produced_tensors: HashSet<GlobalId>,
}

impl FusionGroup {
    fn new(dims: Vec<usize>) -> Self {
        FusionGroup {
            dims,
            body: Vec::new(),
            value_map: HashMap::new(),
            op_ids: HashSet::new(),
            produced_tensors: HashSet::new(),
        }
    }
}

fn add_op_to_group(
    group: &mut FusionGroup,
    op: &crate::milli_graph::ops::AnyMilliOp,
    shapes: &HashMap<GlobalId, Vec<usize>>,
) -> Result<(), PlanError> {
    let kind = op.op_kind();
    let inputs: Vec<GlobalId> = op.inputs().collect();
    let outputs: Vec<GlobalId> = op.outputs().collect();

    group.op_ids.insert(op.global_id());

    if let Some(scalar_op) = scalar_bin_op(&kind) {
        let a_ref = ensure_loaded(group, inputs[0], shapes)?;
        let b_ref = ensure_loaded(group, inputs[1], shapes)?;

        let result_ref = group.body.len();
        group.body.push(BodyOp::BinOp {
            op: scalar_op,
            a_ref,
            b_ref,
        });

        let out_id = outputs[0];
        group.value_map.insert(out_id, result_ref);
        group.produced_tensors.insert(out_id);
    } else if let Some(scalar_op) = scalar_unary_op(&kind) {
        let input_ref = ensure_loaded(group, inputs[0], shapes)?;

        let result_ref = group.body.len();
        group.body.push(BodyOp::UnaryOp {
            op: scalar_op,
            input_ref,
        });

        let out_id = outputs[0];
        group.value_map.insert(out_id, result_ref);
        group.produced_tensors.insert(out_id);
    } else {
        return Err(PlanError::UnsupportedOp(kind));
    }

    Ok(())
}

/// Ensure a tensor is available in the group, emitting a Load if needed.
fn ensure_loaded(
    group: &mut FusionGroup,
    tensor_id: GlobalId,
    shapes: &HashMap<GlobalId, Vec<usize>>,
) -> Result<usize, PlanError> {
    if let Some(&ref_idx) = group.value_map.get(&tensor_id) {
        return Ok(ref_idx);
    }

    let tensor_shape = shapes
        .get(&tensor_id)
        .ok_or(PlanError::MissingShape(tensor_id))?;
    let strides = broadcast_strides(tensor_shape, &group.dims);

    let ref_idx = group.body.len();
    group.body.push(BodyOp::Load {
        tensor: tensor_id,
        strides,
    });
    group.value_map.insert(tensor_id, ref_idx);
    Ok(ref_idx)
}

/// Finalize a fusion group into a KernelOp.
///
/// Decides which intermediate tensors need stores (used outside the group
/// or are graph outputs) and emits Store ops for them.
fn finalize_group(
    mut group: FusionGroup,
    consumer_ops: &HashMap<GlobalId, Vec<GlobalId>>,
    output_tensors: &HashSet<GlobalId>,
) -> KernelOp {
    // For each tensor produced by the group, check if it needs a Store.
    let produced: Vec<(GlobalId, usize)> = group
        .produced_tensors
        .iter()
        .filter_map(|tid| {
            let ref_idx = group.value_map[tid];
            Some((*tid, ref_idx))
        })
        .collect();

    for (tensor_id, value_ref) in produced {
        let needs_store = output_tensors.contains(&tensor_id)
            || consumer_ops
                .get(&tensor_id)
                .map_or(false, |consumers| {
                    consumers.iter().any(|c| !group.op_ids.contains(c))
                });

        if needs_store {
            // Store with same strides as if we loaded (it's the output shape).
            let strides = contiguous_strides(&group.dims);
            group.body.push(BodyOp::Store {
                tensor: tensor_id,
                strides,
                value_ref,
            });
        }
    }

    let kernel = ElementwiseKernel {
        dims: group.dims,
        body: group.body,
    };

    KernelOp::Elementwise(try_collapse(kernel))
}

// ---------------------------------------------------------------------------
// MatMul planning
// ---------------------------------------------------------------------------

fn build_gemm_kernel(
    a_id: GlobalId,
    b_id: GlobalId,
    c_id: GlobalId,
    shapes: &HashMap<GlobalId, Vec<usize>>,
) -> Result<KernelOp, PlanError> {
    let a_shape = shapes.get(&a_id).ok_or(PlanError::MissingShape(a_id))?;
    let b_shape = shapes.get(&b_id).ok_or(PlanError::MissingShape(b_id))?;

    // Promote 1-D to 2-D.
    let a_mat: Vec<usize> = if a_shape.len() == 1 {
        vec![1, a_shape[0]]
    } else {
        a_shape.clone()
    };
    let b_mat: Vec<usize> = if b_shape.len() == 1 {
        vec![b_shape[0], 1]
    } else {
        b_shape.clone()
    };

    let a_ndim = a_mat.len();
    let b_ndim = b_mat.len();

    let m = a_mat[a_ndim - 2];
    let k_a = a_mat[a_ndim - 1];
    let k_b = b_mat[b_ndim - 2];
    let n = b_mat[b_ndim - 1];

    if k_a != k_b {
        return Err(PlanError::MatMulDimMismatch(k_a, k_b));
    }
    let k = k_a;

    // Batch dimensions.
    let a_batch = &a_mat[..a_ndim - 2];
    let b_batch = &b_mat[..b_ndim - 2];

    let batch_size = if a_batch.is_empty() && b_batch.is_empty() {
        1
    } else if a_batch == b_batch {
        a_batch.iter().product::<usize>().max(1)
    } else if a_batch.is_empty() {
        b_batch.iter().product::<usize>().max(1)
    } else if b_batch.is_empty() {
        a_batch.iter().product::<usize>().max(1)
    } else {
        // For now, require matching batch dims.
        return Err(PlanError::IncompatibleBroadcast(
            a_batch.to_vec(),
            b_batch.to_vec(),
        ));
    };

    let a_mat_size = m * k;
    let b_mat_size = k * n;
    let c_mat_size = m * n;

    Ok(KernelOp::Gemm(GemmKernel {
        m,
        n,
        k,
        a: a_id,
        b: b_id,
        c: c_id,
        batch_size,
        a_batch_stride: if a_batch.is_empty() { 0 } else { a_mat_size },
        b_batch_stride: if b_batch.is_empty() { 0 } else { b_mat_size },
        c_batch_stride: c_mat_size,
    }))
}

// ---------------------------------------------------------------------------
// Shape helpers
// ---------------------------------------------------------------------------

/// Compute broadcast strides for a source shape within a target iteration space.
fn broadcast_strides(source: &[usize], target: &[usize]) -> Vec<isize> {
    let t_ndim = target.len();
    let s_ndim = source.len();
    let offset = t_ndim.saturating_sub(s_ndim);
    let mut strides = vec![0isize; t_ndim];

    let mut stride = 1isize;
    for i in (0..s_ndim).rev() {
        let ti = i + offset;
        if source[i] == target[ti] {
            strides[ti] = stride;
            stride *= source[i] as isize;
        } else if source[i] == 1 {
            strides[ti] = 0;
        }
        // Incompatible shapes would have been caught during graph construction.
    }
    strides
}

/// Standard contiguous (row-major) strides for a shape.
fn contiguous_strides(dims: &[usize]) -> Vec<isize> {
    let mut strides = vec![0isize; dims.len()];
    let mut stride = 1isize;
    for i in (0..dims.len()).rev() {
        strides[i] = stride;
        stride *= dims[i] as isize;
    }
    strides
}

/// Try to collapse an elementwise kernel to a flat 1D loop.
///
/// Possible when all loads/stores use contiguous or fully-broadcast strides.
fn try_collapse(kernel: ElementwiseKernel) -> ElementwiseKernel {
    if kernel.dims.len() <= 1 {
        return kernel;
    }

    let contiguous = contiguous_strides(&kernel.dims);

    let all_collapsible = kernel.body.iter().all(|op| {
        let strides = match op {
            BodyOp::Load { strides, .. } | BodyOp::Store { strides, .. } => strides,
            _ => return true,
        };
        // Either fully contiguous or fully broadcast (all zeros).
        *strides == contiguous || strides.iter().all(|&s| s == 0)
    });

    if !all_collapsible {
        return kernel;
    }

    let total: usize = kernel.dims.iter().product();
    let new_body = kernel
        .body
        .into_iter()
        .map(|op| match op {
            BodyOp::Load { tensor, strides } => {
                let flat_stride = if strides.iter().all(|&s| s == 0) {
                    0
                } else {
                    1
                };
                BodyOp::Load {
                    tensor,
                    strides: vec![flat_stride],
                }
            }
            BodyOp::Store {
                tensor,
                strides,
                value_ref,
            } => {
                let flat_stride = if strides.iter().all(|&s| s == 0) {
                    0
                } else {
                    1
                };
                BodyOp::Store {
                    tensor,
                    strides: vec![flat_stride],
                    value_ref,
                }
            }
            other => other,
        })
        .collect();

    ElementwiseKernel {
        dims: vec![total],
        body: new_body,
    }
}

/// Build a map: tensor_id → list of op_ids that consume it.
fn build_consumer_map(graph: &MilliOpGraph) -> HashMap<GlobalId, Vec<GlobalId>> {
    let mut consumers: HashMap<GlobalId, Vec<GlobalId>> = HashMap::new();
    for op_id in graph.op_ordering() {
        let op = graph.get_node_by_id(op_id).unwrap();
        for input_id in op.inputs() {
            consumers.entry(input_id).or_default().push(op.global_id());
        }
    }
    consumers
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::milli_graph::ops::{MatMul, SimpleBinary, SimpleUnaryOp};

    #[test]
    fn test_plan_fused_chain() {
        // neg(exp(a * b + c)) should fuse into 1 kernel with 4 compute ops.
        let mut rng = wyrand::WyRand::new(42);
        let ext_a = GlobalId::new(&mut rng);
        let ext_b = GlobalId::new(&mut rng);
        let ext_c = GlobalId::new(&mut rng);
        let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b, ext_c], &mut rng);
        let a = input_map[&ext_a];
        let b = input_map[&ext_b];
        let c = input_map[&ext_c];
        let mul_out = SimpleBinary::mul(&mut graph, a, b, &mut rng);
        let add_out = SimpleBinary::add(&mut graph, mul_out, c, &mut rng);
        let exp_out = SimpleUnaryOp::exp(&mut graph, add_out, &mut rng);
        let neg_out = SimpleUnaryOp::neg(&mut graph, exp_out, &mut rng);

        let ext_out = GlobalId::new(&mut rng);
        graph.set_output_map([(neg_out, ext_out)]);

        let shape = vec![4, 8];
        let mut shapes = HashMap::new();
        for &t in &[a, b, c, mul_out, add_out, exp_out, neg_out] {
            shapes.insert(t, shape.clone());
        }

        let kernels = plan(&graph, &shapes).unwrap();
        let s = super::super::kernel::stats(&kernels);

        // Should be 1 fused elementwise kernel.
        assert_eq!(s.num_kernels, 1);
        assert_eq!(s.num_elementwise, 1);
        assert_eq!(s.total_fused_ops, 4); // mul, add, exp, neg
    }

    #[test]
    fn test_plan_matmul() {
        let mut rng = wyrand::WyRand::new(42);
        let ext_a = GlobalId::new(&mut rng);
        let ext_b = GlobalId::new(&mut rng);
        let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
        let a = input_map[&ext_a];
        let b = input_map[&ext_b];
        let c = MatMul::push_new(&mut graph, a, b, &mut rng);

        let ext_out = GlobalId::new(&mut rng);
        graph.set_output_map([(c, ext_out)]);

        let mut shapes = HashMap::new();
        shapes.insert(a, vec![4, 8]);
        shapes.insert(b, vec![8, 3]);
        shapes.insert(c, vec![4, 3]);

        let kernels = plan(&graph, &shapes).unwrap();
        assert_eq!(kernels.len(), 1);
        if let KernelOp::Gemm(g) = &kernels[0] {
            assert_eq!(g.m, 4);
            assert_eq!(g.n, 3);
            assert_eq!(g.k, 8);
            assert_eq!(g.batch_size, 1);
        } else {
            panic!("Expected Gemm kernel");
        }
    }

    #[test]
    fn test_plan_matmul_then_elementwise() {
        // y = tanh(x @ w + b) — should produce Gemm then fused Elementwise.
        let mut rng = wyrand::WyRand::new(42);
        let ext_x = GlobalId::new(&mut rng);
        let ext_w = GlobalId::new(&mut rng);
        let ext_b = GlobalId::new(&mut rng);
        let (mut graph, input_map) = MilliOpGraph::new([ext_x, ext_w, ext_b], &mut rng);
        let x = input_map[&ext_x];
        let w = input_map[&ext_w];
        let b = input_map[&ext_b];

        let mm = MatMul::push_new(&mut graph, x, w, &mut rng);
        let add = SimpleBinary::add(&mut graph, mm, b, &mut rng);
        let tanh = SimpleUnaryOp::trig(
            &mut graph,
            add,
            crate::TrigOp::Tanh,
            &mut rng,
        );

        let ext_out = GlobalId::new(&mut rng);
        graph.set_output_map([(tanh, ext_out)]);

        let mut shapes = HashMap::new();
        shapes.insert(x, vec![4, 8]);
        shapes.insert(w, vec![8, 16]);
        shapes.insert(b, vec![1, 16]);
        shapes.insert(mm, vec![4, 16]);
        shapes.insert(add, vec![4, 16]);
        shapes.insert(tanh, vec![4, 16]);

        let kernels = plan(&graph, &shapes).unwrap();
        let s = super::super::kernel::stats(&kernels);

        assert_eq!(s.num_gemm, 1);
        assert_eq!(s.num_elementwise, 1);
        assert_eq!(s.total_fused_ops, 2); // add + tanh fused
    }

    #[test]
    fn test_broadcast_strides() {
        assert_eq!(broadcast_strides(&[2, 3], &[2, 3]), vec![3, 1]);
        assert_eq!(broadcast_strides(&[1, 3], &[2, 3]), vec![0, 1]);
        assert_eq!(broadcast_strides(&[2, 1], &[2, 3]), vec![1, 0]);
        assert_eq!(broadcast_strides(&[3], &[2, 3]), vec![0, 1]);
    }

    #[test]
    fn test_collapse_dims() {
        // Same shape, no broadcast → collapses to 1D.
        let kernel = ElementwiseKernel {
            dims: vec![4, 8],
            body: vec![
                BodyOp::Load {
                    tensor: GlobalId(1),
                    strides: vec![8, 1],
                },
                BodyOp::Load {
                    tensor: GlobalId(2),
                    strides: vec![8, 1],
                },
                BodyOp::BinOp {
                    op: ScalarBinOp::Add,
                    a_ref: 0,
                    b_ref: 1,
                },
                BodyOp::Store {
                    tensor: GlobalId(3),
                    strides: vec![8, 1],
                    value_ref: 2,
                },
            ],
        };

        let collapsed = try_collapse(kernel);
        assert_eq!(collapsed.dims, vec![32]);
        if let BodyOp::Load { strides, .. } = &collapsed.body[0] {
            assert_eq!(strides, &vec![1]);
        }
    }

    #[test]
    fn test_no_collapse_broadcast() {
        // With broadcast stride, should NOT collapse.
        let kernel = ElementwiseKernel {
            dims: vec![4, 8],
            body: vec![
                BodyOp::Load {
                    tensor: GlobalId(1),
                    strides: vec![8, 1],
                },
                BodyOp::Load {
                    tensor: GlobalId(2),
                    strides: vec![0, 1], // broadcast dim 0
                },
                BodyOp::BinOp {
                    op: ScalarBinOp::Add,
                    a_ref: 0,
                    b_ref: 1,
                },
                BodyOp::Store {
                    tensor: GlobalId(3),
                    strides: vec![8, 1],
                    value_ref: 2,
                },
            ],
        };

        let result = try_collapse(kernel);
        // Should keep 2D since broadcast strides aren't fully zero.
        assert_eq!(result.dims, vec![4, 8]);
    }
}
