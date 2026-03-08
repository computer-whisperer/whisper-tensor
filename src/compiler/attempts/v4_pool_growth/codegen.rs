//! Cranelift pipeline wrapper for v4 pool growth.
//!
//! This wrapper intentionally whitewashes nano-op order before recovery.

use super::growth::{GrowthError, PoolGrowthPlan, grow_from_pool};
use crate::compiler::attempts::v1_scalar_crystal::codegen as v1_codegen;
use crate::compiler::attempts::v1_scalar_crystal::nano_op::{
    NanoExpandError, NanoOp, NanoOpExpander,
};
use crate::graph::GlobalId;
use crate::milli_graph::MilliOpGraph;
use std::collections::HashMap;

pub use v1_codegen::{CodegenError, NativeCompiledGraph, TensorLayout};

#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error(transparent)]
    Expand(#[from] NanoExpandError),
    #[error(transparent)]
    Growth(#[from] GrowthError),
    #[error(transparent)]
    Codegen(#[from] CodegenError),
    #[error("v4 matmul codegen not wired yet; recovered {0} matmul crystal groups")]
    MatMulNotYetSupported(usize),
}

#[derive(Debug, Clone)]
pub struct PipelineArtifacts {
    pub ordered_nano_ops: Vec<NanoOp>,
    pub whitewashed_nano_ops: Vec<NanoOp>,
    pub plan: PoolGrowthPlan,
}

/// Deterministic, structure-destroying reorder for pool-style recovery tests.
pub fn whitewash_pool_order(mut ops: Vec<NanoOp>) -> Vec<NanoOp> {
    if ops.len() < 3 {
        return ops;
    }
    let len = ops.len();
    ops.rotate_left(len / 3);
    for chunk in ops.chunks_mut(7) {
        chunk.reverse();
    }
    ops
}

pub fn build_pipeline(
    graph: &MilliOpGraph,
    shapes: &HashMap<GlobalId, Vec<usize>>,
) -> Result<PipelineArtifacts, PipelineError> {
    let mut expander = NanoOpExpander::new(shapes.clone());
    let ordered_nano_ops = expander.expand(graph)?;
    let whitewashed_nano_ops = whitewash_pool_order(ordered_nano_ops.clone());
    let plan = grow_from_pool(&whitewashed_nano_ops, shapes)?;
    Ok(PipelineArtifacts {
        ordered_nano_ops,
        whitewashed_nano_ops,
        plan,
    })
}

pub fn compile_graph(
    graph: &MilliOpGraph,
    shapes: &HashMap<GlobalId, Vec<usize>>,
) -> Result<(NativeCompiledGraph, PipelineArtifacts), PipelineError> {
    let artifacts = build_pipeline(graph, shapes)?;
    if !artifacts.plan.matmul_crystals.is_empty() {
        return Err(PipelineError::MatMulNotYetSupported(
            artifacts.plan.matmul_crystals.len(),
        ));
    }
    let layout = TensorLayout::from_shapes(shapes);
    let compiled = v1_codegen::compile_crystallized(&artifacts.plan.fused_loops, &layout)?;
    Ok((compiled, artifacts))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GlobalId;
    use crate::milli_graph::MilliOpGraph;
    use crate::milli_graph::ops::{SimpleBinary, SimpleUnaryOp};

    #[test]
    fn test_compile_graph_elementwise_chain() {
        let mut rng = wyrand::WyRand::new(1234);
        let ext_a = GlobalId::new(&mut rng);
        let ext_b = GlobalId::new(&mut rng);
        let ext_c = GlobalId::new(&mut rng);
        let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b, ext_c], &mut rng);
        let a = input_map[&ext_a];
        let b = input_map[&ext_b];
        let c = input_map[&ext_c];

        let mul = SimpleBinary::mul(&mut graph, a, b, &mut rng);
        let add = SimpleBinary::add(&mut graph, mul, c, &mut rng);
        let neg = SimpleUnaryOp::neg(&mut graph, add, &mut rng);

        let mut shapes = HashMap::new();
        for &t in &[a, b, c, mul, add, neg] {
            shapes.insert(t, vec![8]);
        }

        let (compiled, artifacts) = compile_graph(&graph, &shapes).unwrap();
        assert_eq!(artifacts.plan.stats.fused_pairs, 2);
        assert!(artifacts.plan.matmul_crystals.is_empty());

        let layout = &compiled.layout;
        let mut a_data: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let mut b_data: Vec<f32> = (1..=8).map(|x| x as f32 * 0.5).collect();
        let mut c_data: Vec<f32> = vec![1.0; 8];
        let mut mul_data = vec![0.0f32; 8];
        let mut add_data = vec![0.0f32; 8];
        let mut neg_data = vec![0.0f32; 8];

        let mut buffers = vec![std::ptr::null_mut::<f32>(); layout.num_buffers];
        buffers[layout.tensor_index[&a]] = a_data.as_mut_ptr();
        buffers[layout.tensor_index[&b]] = b_data.as_mut_ptr();
        buffers[layout.tensor_index[&c]] = c_data.as_mut_ptr();
        buffers[layout.tensor_index[&mul]] = mul_data.as_mut_ptr();
        buffers[layout.tensor_index[&add]] = add_data.as_mut_ptr();
        buffers[layout.tensor_index[&neg]] = neg_data.as_mut_ptr();

        unsafe { compiled.execute(&mut buffers) };

        for i in 0..8 {
            let expected = -((i + 1) as f32 * ((i + 1) as f32 * 0.5) + 1.0);
            assert!((neg_data[i] - expected).abs() < 1e-5);
        }
    }
}
