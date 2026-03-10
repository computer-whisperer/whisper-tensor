//! v9 fused expression compiler.
//!
//! First attempt consuming the v2 frontend's boundaryless expression trees.
//! Key differences from v8:
//! - No per-milli-op kernel boundaries — intermediate tensors can be fused away
//! - Pattern recovery operates on fully-inlined expression trees
//! - Single-threaded, no tiling — correctness first

pub mod codegen;
pub mod inline;
pub mod pattern;

#[cfg(feature = "cranelift")]
pub mod pipeline {
    use crate::compiler::attempts::v1_scalar_crystal::codegen::TensorLayout;
    use crate::compiler::attempts::v9_fused_expr::codegen::jit::{
        compile_patterns, CompiledGraph, V9Error,
    };
    use crate::compiler::attempts::v9_fused_expr::inline::inline_intermediates;
    use crate::compiler::attempts::v9_fused_expr::pattern::recover_patterns;
    use crate::compiler::common::v2_frontend::ExprExpander;
    use crate::graph::GlobalId;
    use crate::milli_graph::MilliOpGraph;
    use std::collections::{HashMap, HashSet};

    /// Compile a MilliOpGraph into a v9 CompiledGraph.
    ///
    /// `shapes` must contain concrete shapes for every tensor in the graph.
    /// `final_output_ids` are the internal tensor IDs that must be materialized
    /// (everything else is eligible for fusion/inlining).
    pub fn compile_graph(
        graph: &MilliOpGraph,
        shapes: &HashMap<GlobalId, Vec<usize>>,
        final_output_ids: &HashSet<GlobalId>,
        max_samples: usize,
    ) -> Result<CompiledGraph, V9Error> {
        // Phase 1: v2 frontend — produce expression trees
        let expander = if max_samples > 0 {
            ExprExpander::new_sampled(shapes.clone(), max_samples)
        } else {
            ExprExpander::new(shapes.clone())
        };
        let bindings = expander.expand(graph)?;

        // Phase 2: inline intermediates
        let inlined = inline_intermediates(&bindings, final_output_ids);

        // Phase 3: recover patterns from inlined expressions
        let patterns = recover_patterns(&inlined.bindings, shapes);

        // Phase 4: build tensor layout and compile
        let layout = TensorLayout::from_shapes(shapes);
        compile_patterns(&patterns, layout)
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::graph::GlobalId;
        use crate::milli_graph::MilliOpGraph;
        use crate::milli_graph::ops::{SimpleBinary, SimpleUnaryOp};

        #[test]
        fn test_fused_mul_neg() {
            // Graph: out = neg(a * b), where a*b is an intermediate (fused away)
            // Expected: -(a*b) = [-5, -12, -21, -32]
            let mut rng = wyrand::WyRand::new(42);
            let ext_a = GlobalId::new(&mut rng);
            let ext_b = GlobalId::new(&mut rng);
            let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
            let int_a = input_map[&ext_a];
            let int_b = input_map[&ext_b];
            let mul_out = SimpleBinary::mul(&mut graph, int_a, int_b, &mut rng);
            let neg_out = SimpleUnaryOp::neg(&mut graph, mul_out, &mut rng);

            let mut shapes = HashMap::new();
            shapes.insert(int_a, vec![4]);
            shapes.insert(int_b, vec![4]);
            shapes.insert(mul_out, vec![4]);
            shapes.insert(neg_out, vec![4]);

            let final_outputs: HashSet<GlobalId> = [neg_out].into_iter().collect();
            let compiled = compile_graph(&graph, &shapes, &final_outputs, 0).unwrap();

            // Should fuse into 1 kernel
            assert_eq!(compiled.kernels.len(), 1, "Expected 1 fused kernel");

            let mut buffers: Vec<Vec<f32>> = (0..compiled.layout.num_buffers)
                .map(|_| vec![0.0f32; 4])
                .collect();
            buffers[compiled.layout.tensor_index[&int_a]].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
            buffers[compiled.layout.tensor_index[&int_b]].copy_from_slice(&[5.0, 6.0, 7.0, 8.0]);

            let mut ptrs: Vec<*mut f32> = buffers.iter_mut().map(|b| b.as_mut_ptr()).collect();
            unsafe { compiled.execute(&mut ptrs) };

            let out_slot = compiled.layout.tensor_index[&neg_out];
            let expected = [-5.0, -12.0, -21.0, -32.0f32];
            assert_eq!(&buffers[out_slot], &expected);
        }

        #[test]
        fn test_pointwise_add() {
            // Simple: out = a + b (no fusion, just verifying basic codegen)
            let mut rng = wyrand::WyRand::new(42);
            let ext_a = GlobalId::new(&mut rng);
            let ext_b = GlobalId::new(&mut rng);
            let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
            let int_a = input_map[&ext_a];
            let int_b = input_map[&ext_b];
            let add_out = SimpleBinary::add(&mut graph, int_a, int_b, &mut rng);

            let mut shapes = HashMap::new();
            shapes.insert(int_a, vec![8]);
            shapes.insert(int_b, vec![8]);
            shapes.insert(add_out, vec![8]);

            let final_outputs: HashSet<GlobalId> = [add_out].into_iter().collect();
            let compiled = compile_graph(&graph, &shapes, &final_outputs, 0).unwrap();

            let a_data: Vec<f32> = (0..8).map(|i| i as f32).collect();
            let b_data: Vec<f32> = (0..8).map(|i| (i * 10) as f32).collect();

            let mut buffers: Vec<Vec<f32>> = (0..compiled.layout.num_buffers)
                .map(|_| vec![0.0f32; 8])
                .collect();
            let a_slot = compiled.layout.tensor_index[&int_a];
            let b_slot = compiled.layout.tensor_index[&int_b];
            buffers[a_slot].copy_from_slice(&a_data);
            buffers[b_slot].copy_from_slice(&b_data);

            let mut ptrs: Vec<*mut f32> = buffers.iter_mut().map(|b| b.as_mut_ptr()).collect();
            unsafe { compiled.execute(&mut ptrs) };

            let out_slot = compiled.layout.tensor_index[&add_out];
            let expected: Vec<f32> = (0..8).map(|i| i as f32 + (i * 10) as f32).collect();
            assert_eq!(&buffers[out_slot], &expected);
        }

        #[test]
        fn test_fused_add_mul_chain() {
            // out = (a + b) * c, where (a+b) is intermediate
            let mut rng = wyrand::WyRand::new(42);
            let ext_a = GlobalId::new(&mut rng);
            let ext_b = GlobalId::new(&mut rng);
            let ext_c = GlobalId::new(&mut rng);
            let (mut graph, input_map) =
                MilliOpGraph::new([ext_a, ext_b, ext_c], &mut rng);
            let int_a = input_map[&ext_a];
            let int_b = input_map[&ext_b];
            let int_c = input_map[&ext_c];
            let add_out = SimpleBinary::add(&mut graph, int_a, int_b, &mut rng);
            let mul_out = SimpleBinary::mul(&mut graph, add_out, int_c, &mut rng);

            let n = 6;
            let mut shapes = HashMap::new();
            shapes.insert(int_a, vec![n]);
            shapes.insert(int_b, vec![n]);
            shapes.insert(int_c, vec![n]);
            shapes.insert(add_out, vec![n]);
            shapes.insert(mul_out, vec![n]);

            let final_outputs: HashSet<GlobalId> = [mul_out].into_iter().collect();
            let compiled = compile_graph(&graph, &shapes, &final_outputs, 0).unwrap();

            // Verify: only 1 kernel (fusion happened)
            assert_eq!(
                compiled.kernels.len(),
                1,
                "Expected 1 fused kernel, got {}",
                compiled.kernels.len()
            );

            let a_data: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
            let b_data: Vec<f32> = (0..n).map(|i| (i * 2) as f32).collect();
            let c_data: Vec<f32> = (0..n).map(|i| (i + 3) as f32).collect();

            let mut buffers: Vec<Vec<f32>> = (0..compiled.layout.num_buffers)
                .map(|_| vec![0.0f32; n])
                .collect();
            buffers[compiled.layout.tensor_index[&int_a]].copy_from_slice(&a_data);
            buffers[compiled.layout.tensor_index[&int_b]].copy_from_slice(&b_data);
            buffers[compiled.layout.tensor_index[&int_c]].copy_from_slice(&c_data);

            let mut ptrs: Vec<*mut f32> = buffers.iter_mut().map(|b| b.as_mut_ptr()).collect();
            unsafe { compiled.execute(&mut ptrs) };

            let expected: Vec<f32> = (0..n)
                .map(|i| ((i + 1) as f32 + (i * 2) as f32) * (i + 3) as f32)
                .collect();
            let out_slot = compiled.layout.tensor_index[&mul_out];
            assert_eq!(&buffers[out_slot], &expected);
        }

        #[test]
        fn test_matmul_simple() {
            // out = A @ B, 2x3 * 3x2 = 2x2
            let mut rng = wyrand::WyRand::new(42);
            let ext_a = GlobalId::new(&mut rng);
            let ext_b = GlobalId::new(&mut rng);
            let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
            let int_a = input_map[&ext_a];
            let int_b = input_map[&ext_b];
            let mm_out =
                crate::milli_graph::ops::MatMul::push_new(&mut graph, int_a, int_b, &mut rng);

            let mut shapes = HashMap::new();
            shapes.insert(int_a, vec![2, 3]);
            shapes.insert(int_b, vec![3, 2]);
            shapes.insert(mm_out, vec![2, 2]);

            let final_outputs: HashSet<GlobalId> = [mm_out].into_iter().collect();
            let compiled = compile_graph(&graph, &shapes, &final_outputs, 0).unwrap();

            // A = [[1,2,3],[4,5,6]], B = [[1,0],[0,1],[1,1]]
            // A@B = [[1+0+3, 0+2+3],[4+0+6, 0+5+6]] = [[4,5],[10,11]]
            let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0f32];
            let b_data = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0f32];
            let expected = vec![4.0, 5.0, 10.0, 11.0f32];

            let mut buffers: Vec<Vec<f32>> = (0..compiled.layout.num_buffers)
                .map(|_| vec![0.0f32; 8])
                .collect();
            buffers[compiled.layout.tensor_index[&int_a]][..6].copy_from_slice(&a_data);
            buffers[compiled.layout.tensor_index[&int_b]][..6].copy_from_slice(&b_data);

            let mut ptrs: Vec<*mut f32> = buffers.iter_mut().map(|b| b.as_mut_ptr()).collect();
            unsafe { compiled.execute(&mut ptrs) };

            let out_slot = compiled.layout.tensor_index[&mm_out];
            assert_eq!(&buffers[out_slot][..4], &expected);
        }
    }
}
