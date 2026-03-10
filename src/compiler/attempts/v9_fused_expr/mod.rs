//! v9 fused expression compiler.
//!
//! First attempt consuming the v2 frontend's boundaryless expression trees.
//! Key differences from v8:
//! - No per-milli-op kernel boundaries — intermediate tensors can be fused away
//! - Pattern recovery operates on fully-inlined expression trees
//! - Single-threaded, no tiling — correctness first

pub mod codegen;
pub mod executor;
pub mod inline;
pub mod pattern;

#[cfg(feature = "cranelift")]
pub mod pipeline {
    use crate::compiler::attempts::v1_scalar_crystal::codegen::TensorLayout;
    pub use crate::compiler::attempts::v9_fused_expr::codegen::jit::{
        CompiledGraph, V9Error,
    };
    pub use crate::compiler::attempts::v9_fused_expr::executor::dag::execute_parallel;
    use crate::compiler::attempts::v9_fused_expr::codegen::jit::compile_patterns;
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

        #[test]
        fn test_fused_matmul_bias_tanh() {
            // out = tanh(A @ W + bias), 2x3 * 3x2 + [1,2] broadcast = 2x2
            // Tests the embedded-reduction architecture: reduction (matmul)
            // fused inside pointwise ops (add + tanh).
            let mut rng = wyrand::WyRand::new(42);
            let ext_a = GlobalId::new(&mut rng);
            let ext_w = GlobalId::new(&mut rng);
            let ext_bias = GlobalId::new(&mut rng);
            let (mut graph, input_map) =
                MilliOpGraph::new([ext_a, ext_w, ext_bias], &mut rng);
            let int_a = input_map[&ext_a];
            let int_w = input_map[&ext_w];
            let int_bias = input_map[&ext_bias];

            let mm_out =
                crate::milli_graph::ops::MatMul::push_new(&mut graph, int_a, int_w, &mut rng);
            let add_out = SimpleBinary::add(&mut graph, mm_out, int_bias, &mut rng);
            let tanh_out = SimpleUnaryOp::trig(
                &mut graph,
                add_out,
                crate::TrigOp::Tanh,
                &mut rng,
            );

            let mut shapes = HashMap::new();
            shapes.insert(int_a, vec![2, 3]);
            shapes.insert(int_w, vec![3, 2]);
            shapes.insert(int_bias, vec![1, 2]); // broadcast bias
            shapes.insert(mm_out, vec![2, 2]);
            shapes.insert(add_out, vec![2, 2]);
            shapes.insert(tanh_out, vec![2, 2]);

            // Only tanh_out is a final output — mm_out and add_out are intermediates
            let final_outputs: HashSet<GlobalId> = [tanh_out].into_iter().collect();
            let compiled = compile_graph(&graph, &shapes, &final_outputs, 0).unwrap();

            // Should fuse into 1 kernel (matmul + bias + tanh all fused)
            assert_eq!(compiled.kernels.len(), 1, "Expected 1 fused kernel");

            // A = [[1,2,3],[4,5,6]], W = [[1,0],[0,1],[1,1]], bias = [10, 20]
            // A@W = [[4,5],[10,11]]
            // A@W + bias = [[14,25],[20,31]]
            // tanh(A@W + bias) ≈ [[1.0, 1.0],[1.0, 1.0]] (saturated)
            let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0f32];
            let w_data = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0f32];
            let bias_data = vec![10.0, 20.0f32];

            let mut buffers: Vec<Vec<f32>> = (0..compiled.layout.num_buffers)
                .map(|_| vec![0.0f32; 8])
                .collect();
            buffers[compiled.layout.tensor_index[&int_a]][..6].copy_from_slice(&a_data);
            buffers[compiled.layout.tensor_index[&int_w]][..6].copy_from_slice(&w_data);
            buffers[compiled.layout.tensor_index[&int_bias]][..2].copy_from_slice(&bias_data);

            let mut ptrs: Vec<*mut f32> = buffers.iter_mut().map(|b| b.as_mut_ptr()).collect();
            unsafe { compiled.execute(&mut ptrs) };

            let out_slot = compiled.layout.tensor_index[&tanh_out];
            let result = &buffers[out_slot][..4];

            // Compute expected: tanh(mm + bias) for each element
            let mm = [4.0f32, 5.0, 10.0, 11.0];
            let biases = [10.0f32, 20.0, 10.0, 20.0]; // broadcast [10,20] to 2x2
            let expected: Vec<f32> = mm
                .iter()
                .zip(biases.iter())
                .map(|(m, b)| (m + b).tanh())
                .collect();

            for i in 0..4 {
                assert!(
                    (result[i] - expected[i]).abs() < 1e-6,
                    "element {}: got {}, expected {}",
                    i,
                    result[i],
                    expected[i]
                );
            }
        }

        #[test]
        fn test_two_layer_matmul_mlp() {
            // Two-layer MLP: out = tanh(tanh(x @ W1 + b1) @ W2 + b2)
            // x: 2x3, W1: 3x4, b1: 1x4, W2: 4x2, b2: 1x2 → out: 2x2
            // Tests materialization boundaries: act1 must be materialized
            // between the two matmul reductions.
            let mut rng = wyrand::WyRand::new(99);
            let ext_x = GlobalId::new(&mut rng);
            let ext_w1 = GlobalId::new(&mut rng);
            let ext_b1 = GlobalId::new(&mut rng);
            let ext_w2 = GlobalId::new(&mut rng);
            let ext_b2 = GlobalId::new(&mut rng);
            let (mut graph, input_map) =
                MilliOpGraph::new([ext_x, ext_w1, ext_b1, ext_w2, ext_b2], &mut rng);
            let x = input_map[&ext_x];
            let w1 = input_map[&ext_w1];
            let b1 = input_map[&ext_b1];
            let w2 = input_map[&ext_w2];
            let b2 = input_map[&ext_b2];

            // Layer 1: act1 = tanh(x @ W1 + b1)
            let mm1 = crate::milli_graph::ops::MatMul::push_new(&mut graph, x, w1, &mut rng);
            let add1 = SimpleBinary::add(&mut graph, mm1, b1, &mut rng);
            let act1 = SimpleUnaryOp::trig(&mut graph, add1, crate::TrigOp::Tanh, &mut rng);

            // Layer 2: out = tanh(act1 @ W2 + b2)
            let mm2 = crate::milli_graph::ops::MatMul::push_new(&mut graph, act1, w2, &mut rng);
            let add2 = SimpleBinary::add(&mut graph, mm2, b2, &mut rng);
            let out = SimpleUnaryOp::trig(&mut graph, add2, crate::TrigOp::Tanh, &mut rng);

            let mut shapes = HashMap::new();
            shapes.insert(x, vec![2, 3]);
            shapes.insert(w1, vec![3, 4]);
            shapes.insert(b1, vec![1, 4]);
            shapes.insert(mm1, vec![2, 4]);
            shapes.insert(add1, vec![2, 4]);
            shapes.insert(act1, vec![2, 4]);
            shapes.insert(w2, vec![4, 2]);
            shapes.insert(b2, vec![1, 2]);
            shapes.insert(mm2, vec![2, 2]);
            shapes.insert(add2, vec![2, 2]);
            shapes.insert(out, vec![2, 2]);

            let final_outputs: HashSet<GlobalId> = [out].into_iter().collect();
            let compiled = compile_graph(&graph, &shapes, &final_outputs, 0).unwrap();

            // Should produce multiple kernels (materialization boundary at act1)
            assert!(
                compiled.kernels.len() >= 2,
                "Expected >= 2 kernels (materialization boundary), got {}",
                compiled.kernels.len()
            );

            // Fill with simple data and verify against reference computation
            let x_data: Vec<f32> = (0..6).map(|i| (i + 1) as f32 * 0.1).collect();
            let w1_data: Vec<f32> = (0..12).map(|i| (i as f32 - 6.0) * 0.1).collect();
            let b1_data: Vec<f32> = vec![0.1, -0.1, 0.2, -0.2];
            let w2_data: Vec<f32> = (0..8).map(|i| (i as f32 - 4.0) * 0.1).collect();
            let b2_data: Vec<f32> = vec![0.05, -0.05];

            // Reference computation
            // mm1 = x @ W1 (2x3 * 3x4 = 2x4)
            let mut mm1_ref = vec![0.0f32; 8];
            for i in 0..2 {
                for j in 0..4 {
                    let mut sum = 0.0f32;
                    for k in 0..3 {
                        sum += x_data[i * 3 + k] * w1_data[k * 4 + j];
                    }
                    mm1_ref[i * 4 + j] = sum;
                }
            }
            // act1 = tanh(mm1 + b1)
            let mut act1_ref = vec![0.0f32; 8];
            for i in 0..2 {
                for j in 0..4 {
                    act1_ref[i * 4 + j] = (mm1_ref[i * 4 + j] + b1_data[j]).tanh();
                }
            }
            // mm2 = act1 @ W2 (2x4 * 4x2 = 2x2)
            let mut mm2_ref = vec![0.0f32; 4];
            for i in 0..2 {
                for j in 0..2 {
                    let mut sum = 0.0f32;
                    for k in 0..4 {
                        sum += act1_ref[i * 4 + k] * w2_data[k * 2 + j];
                    }
                    mm2_ref[i * 2 + j] = sum;
                }
            }
            // out = tanh(mm2 + b2)
            let expected: Vec<f32> = (0..4)
                .map(|idx| {
                    let i = idx / 2;
                    let j = idx % 2;
                    (mm2_ref[i * 2 + j] + b2_data[j]).tanh()
                })
                .collect();

            // Allocate buffers
            let max_size = 12; // largest tensor is 3x4=12
            let mut buffers: Vec<Vec<f32>> = (0..compiled.layout.num_buffers)
                .map(|_| vec![0.0f32; max_size])
                .collect();
            buffers[compiled.layout.tensor_index[&x]][..6].copy_from_slice(&x_data);
            buffers[compiled.layout.tensor_index[&w1]][..12].copy_from_slice(&w1_data);
            buffers[compiled.layout.tensor_index[&b1]][..4].copy_from_slice(&b1_data);
            buffers[compiled.layout.tensor_index[&w2]][..8].copy_from_slice(&w2_data);
            buffers[compiled.layout.tensor_index[&b2]][..2].copy_from_slice(&b2_data);

            let mut ptrs: Vec<*mut f32> = buffers.iter_mut().map(|b| b.as_mut_ptr()).collect();
            unsafe { compiled.execute(&mut ptrs) };

            let out_slot = compiled.layout.tensor_index[&out];
            let result = &buffers[out_slot][..4];

            for i in 0..4 {
                assert!(
                    (result[i] - expected[i]).abs() < 1e-5,
                    "element {}: got {}, expected {}",
                    i,
                    result[i],
                    expected[i]
                );
            }
        }

        fn make_random_f32(n: usize, seed: u64) -> Vec<f32> {
            let mut state = seed | 1;
            (0..n)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    (state as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
                })
                .collect()
        }

        fn max_diff(a: &[f32], b: &[f32]) -> f32 {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        }

        fn alloc_and_run(
            compiled: &CompiledGraph,
            inputs: &HashMap<GlobalId, Vec<f32>>,
            output_id: GlobalId,
        ) -> Vec<f32> {
            let layout = &compiled.layout;
            let mut bufs: Vec<Vec<f32>> = (0..layout.num_buffers)
                .map(|i| vec![0.0f32; layout.tensor_sizes.values()
                    .filter(|&&s| layout.tensor_index.values().any(|&idx| idx == i && s > 0))
                    .copied().next().unwrap_or(64)])
                .collect();
            // Properly size buffers
            for (id, &size) in &layout.tensor_sizes {
                let idx = layout.tensor_index[id];
                if bufs[idx].len() < size {
                    bufs[idx].resize(size, 0.0);
                }
            }
            for (id, data) in inputs {
                if let Some(&idx) = layout.tensor_index.get(id) {
                    bufs[idx][..data.len()].copy_from_slice(data);
                }
            }
            let mut ptrs: Vec<*mut f32> = bufs.iter_mut().map(|b| b.as_mut_ptr()).collect();
            unsafe { compiled.execute(&mut ptrs) };
            let out_idx = layout.tensor_index[&output_id];
            bufs[out_idx].clone()
        }

        fn alloc_and_run_parallel(
            compiled: &CompiledGraph,
            inputs: &HashMap<GlobalId, Vec<f32>>,
            output_id: GlobalId,
            num_threads: usize,
        ) -> Vec<f32> {
            let layout = &compiled.layout;
            let mut bufs: Vec<Vec<f32>> = (0..layout.num_buffers)
                .map(|_| vec![0.0f32; 64])
                .collect();
            for (id, &size) in &layout.tensor_sizes {
                let idx = layout.tensor_index[id];
                if bufs[idx].len() < size {
                    bufs[idx].resize(size, 0.0);
                }
            }
            for (id, data) in inputs {
                if let Some(&idx) = layout.tensor_index.get(id) {
                    bufs[idx][..data.len()].copy_from_slice(data);
                }
            }
            let ptrs: Vec<*mut f32> = bufs.iter_mut().map(|b| b.as_mut_ptr()).collect();
            unsafe { execute_parallel(compiled, &ptrs, num_threads) };
            let out_idx = layout.tensor_index[&output_id];
            bufs[out_idx].clone()
        }

        #[test]
        fn test_parallel_matmul() {
            let mut rng = wyrand::WyRand::new(7001);
            let (m, k, n) = (64, 96, 80);
            let ext_a = GlobalId::new(&mut rng);
            let ext_b = GlobalId::new(&mut rng);
            let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
            let a = input_map[&ext_a];
            let b = input_map[&ext_b];
            let c = crate::milli_graph::ops::MatMul::push_new(&mut graph, a, b, &mut rng);

            let mut shapes = HashMap::new();
            shapes.insert(a, vec![m, k]);
            shapes.insert(b, vec![k, n]);
            shapes.insert(c, vec![m, n]);

            let final_outputs: HashSet<GlobalId> = [c].into_iter().collect();
            let compiled = compile_graph(&graph, &shapes, &final_outputs, 0).unwrap();

            let a_data = make_random_f32(m * k, 7002);
            let b_data = make_random_f32(k * n, 7003);
            let mut inputs = HashMap::new();
            inputs.insert(a, a_data);
            inputs.insert(b, b_data);

            let serial = alloc_and_run(&compiled, &inputs, c);

            for threads in [2, 4, 8] {
                let parallel = alloc_and_run_parallel(&compiled, &inputs, c, threads);
                let diff = max_diff(&serial, &parallel);
                assert!(
                    diff < 1e-5,
                    "parallel ({threads}t) diverged from serial: max diff {diff}"
                );
            }
        }

        #[test]
        fn test_parallel_fused_mlp() {
            // Two-layer MLP: out = tanh(tanh(x @ W1 + b1) @ W2 + b2)
            // Tests inter-op parallelism (two kernels with dependency)
            let mut rng = wyrand::WyRand::new(8001);
            let (m, k1, k2, n) = (32, 128, 64, 32);
            let ext_x = GlobalId::new(&mut rng);
            let ext_w1 = GlobalId::new(&mut rng);
            let ext_b1 = GlobalId::new(&mut rng);
            let ext_w2 = GlobalId::new(&mut rng);
            let ext_b2 = GlobalId::new(&mut rng);
            let (mut graph, input_map) =
                MilliOpGraph::new([ext_x, ext_w1, ext_b1, ext_w2, ext_b2], &mut rng);
            let x = input_map[&ext_x];
            let w1 = input_map[&ext_w1];
            let b1 = input_map[&ext_b1];
            let w2 = input_map[&ext_w2];
            let b2 = input_map[&ext_b2];

            let mm1 = crate::milli_graph::ops::MatMul::push_new(&mut graph, x, w1, &mut rng);
            let add1 = SimpleBinary::add(&mut graph, mm1, b1, &mut rng);
            let act1 = SimpleUnaryOp::trig(&mut graph, add1, crate::TrigOp::Tanh, &mut rng);
            let mm2 = crate::milli_graph::ops::MatMul::push_new(&mut graph, act1, w2, &mut rng);
            let add2 = SimpleBinary::add(&mut graph, mm2, b2, &mut rng);
            let out = SimpleUnaryOp::trig(&mut graph, add2, crate::TrigOp::Tanh, &mut rng);

            let mut shapes = HashMap::new();
            shapes.insert(x, vec![m, k1]);
            shapes.insert(w1, vec![k1, k2]);
            shapes.insert(b1, vec![1, k2]);
            shapes.insert(mm1, vec![m, k2]);
            shapes.insert(add1, vec![m, k2]);
            shapes.insert(act1, vec![m, k2]);
            shapes.insert(w2, vec![k2, n]);
            shapes.insert(b2, vec![1, n]);
            shapes.insert(mm2, vec![m, n]);
            shapes.insert(add2, vec![m, n]);
            shapes.insert(out, vec![m, n]);

            let final_outputs: HashSet<GlobalId> = [out].into_iter().collect();
            let compiled = compile_graph(&graph, &shapes, &final_outputs, 0).unwrap();

            let mut inputs = HashMap::new();
            inputs.insert(x, make_random_f32(m * k1, 8002));
            inputs.insert(w1, make_random_f32(k1 * k2, 8003));
            inputs.insert(b1, make_random_f32(k2, 8004));
            inputs.insert(w2, make_random_f32(k2 * n, 8005));
            inputs.insert(b2, make_random_f32(n, 8006));

            let serial = alloc_and_run(&compiled, &inputs, out);

            for threads in [2, 4, 8] {
                let parallel = alloc_and_run_parallel(&compiled, &inputs, out, threads);
                let diff = max_diff(&serial, &parallel);
                assert!(
                    diff < 1e-4,
                    "fused MLP ({threads}t) diverged from serial: max diff {diff}"
                );
            }
        }
    }
}
