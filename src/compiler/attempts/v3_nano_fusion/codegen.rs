//! Cranelift codegen entrypoints for v3 nano-fusion.
//!
//! v3 keeps the v1 nano-op frontend and v1 crystallized codegen backend,
//! but inserts the v3 loop-fusion pass between them.

use super::fusion::{FusionStats, fuse_crystal_ops, fuse_with_stats};
use crate::compiler::attempts::v1_scalar_crystal::codegen as v1_codegen;
use crate::compiler::attempts::v1_scalar_crystal::crystal::{self, CrystalOp};
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
    Codegen(#[from] CodegenError),
}

#[derive(Debug, Clone)]
pub struct PipelineArtifacts {
    pub nano_ops: Vec<NanoOp>,
    pub crystal_ops: Vec<CrystalOp>,
    pub fused_crystal_ops: Vec<CrystalOp>,
    pub fusion_stats: FusionStats,
}

pub fn build_pipeline(
    graph: &MilliOpGraph,
    shapes: &HashMap<GlobalId, Vec<usize>>,
) -> Result<PipelineArtifacts, PipelineError> {
    let mut expander = NanoOpExpander::new(shapes.clone());
    let nano_ops = expander.expand(graph)?;
    let crystal_ops = crystal::crystallize(&nano_ops);
    let (fused_crystal_ops, fusion_stats) = fuse_with_stats(&crystal_ops);

    Ok(PipelineArtifacts {
        nano_ops,
        crystal_ops,
        fused_crystal_ops,
        fusion_stats,
    })
}

pub fn compile_fused_crystallized(
    crystal_ops: &[CrystalOp],
    layout: &TensorLayout,
) -> Result<NativeCompiledGraph, CodegenError> {
    let fused = fuse_crystal_ops(crystal_ops);
    v1_codegen::compile_crystallized(&fused, layout)
}

pub fn compile_graph(
    graph: &MilliOpGraph,
    shapes: &HashMap<GlobalId, Vec<usize>>,
) -> Result<(NativeCompiledGraph, PipelineArtifacts), PipelineError> {
    let artifacts = build_pipeline(graph, shapes)?;
    let layout = TensorLayout::from_shapes(shapes);
    let compiled = v1_codegen::compile_crystallized(&artifacts.fused_crystal_ops, &layout)?;
    Ok((compiled, artifacts))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GlobalId;
    use crate::milli_graph::MilliOpGraph;
    use crate::milli_graph::ops::{SimpleBinary, SimpleUnaryOp};

    #[test]
    fn test_compile_graph_mul_then_neg() {
        let mut rng = wyrand::WyRand::new(123);
        let ext_a = GlobalId::new(&mut rng);
        let ext_b = GlobalId::new(&mut rng);
        let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
        let a = input_map[&ext_a];
        let b = input_map[&ext_b];
        let mul = SimpleBinary::mul(&mut graph, a, b, &mut rng);
        let neg = SimpleUnaryOp::neg(&mut graph, mul, &mut rng);

        let mut shapes = HashMap::new();
        for &tid in &[a, b, mul, neg] {
            shapes.insert(tid, vec![8]);
        }

        let (compiled, artifacts) = compile_graph(&graph, &shapes).unwrap();
        assert_eq!(artifacts.fusion_stats.fused_pairs, 1);

        let layout = &compiled.layout;
        let mut a_data: Vec<f32> = (1..=8).map(|v| v as f32).collect();
        let mut b_data: Vec<f32> = (1..=8).map(|v| v as f32 * 0.5).collect();
        let mut mul_data = vec![0.0f32; 8];
        let mut neg_data = vec![0.0f32; 8];

        let mut buffers = vec![std::ptr::null_mut::<f32>(); layout.num_buffers];
        buffers[layout.tensor_index[&a]] = a_data.as_mut_ptr();
        buffers[layout.tensor_index[&b]] = b_data.as_mut_ptr();
        buffers[layout.tensor_index[&mul]] = mul_data.as_mut_ptr();
        buffers[layout.tensor_index[&neg]] = neg_data.as_mut_ptr();

        unsafe { compiled.execute(&mut buffers) };

        for (i, got) in neg_data.iter().enumerate() {
            let expected = -((i + 1) as f32 * ((i + 1) as f32 * 0.5));
            assert!((got - expected).abs() < 1e-5);
        }
    }
}
