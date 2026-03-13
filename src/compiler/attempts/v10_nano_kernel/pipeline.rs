#![allow(clippy::all, dead_code, unreachable_patterns)]
//! v10 pipeline: NanoGraph → plan → compile → execute.

#[cfg(feature = "cranelift")]
pub mod run {
    use std::collections::HashMap;

    use crate::compiler::attempts::v10_nano_kernel::codegen::jit::{
        compile, CompiledGraph, CompiledKernel, V10Error,
    };
    use crate::compiler::attempts::v10_nano_kernel::plan::{
        build_plan, BufferId, CompilationPlan,
    };
    use crate::graph::GlobalId;
    use crate::nano_graph::lower::LowerResult;

    /// A ready-to-execute compiled model.
    pub struct V10Executable {
        pub compiled: CompiledGraph,
        pub plan: CompilationPlan,
        /// Pre-allocated buffer storage (one Vec<f32> per buffer slot).
        buffer_storage: Vec<Vec<f32>>,
        /// Maps GlobalId → buffer index for input tensors.
        pub input_map: HashMap<GlobalId, usize>,
        /// Maps GlobalId → buffer index for output tensors.
        pub output_map: HashMap<GlobalId, usize>,
    }

    impl V10Executable {
        /// Build from a LowerResult.
        pub fn build(
            lower: &LowerResult,
            input_tensor_ids: &[GlobalId],
            output_tensor_ids: &[GlobalId],
        ) -> Result<Self, V10Error> {
            let plan = build_plan(lower);

            // Report plan stats.
            eprintln!(
                "[v10] {} buffers, {} kernels, {} output buffers",
                plan.buffers.len(),
                plan.kernels.len(),
                plan.output_buffers.len(),
            );
            let total_fused = plan.kernels.iter()
                .filter(|k| k.reductions.len() > 0 || k.ops.len() > 2)
                .count();
            eprintln!("[v10] {} kernels with fusion/reduction", total_fused);

            let compiled = compile(&plan)?;

            // Allocate buffer storage.
            let mut buffer_storage: Vec<Vec<f32>> = plan
                .buffers
                .iter()
                .map(|b| vec![0.0f32; b.count as usize])
                .collect();

            // Pre-fill constant buffers.
            for (buf_id, pairs) in &plan.constants {
                let buf = &mut buffer_storage[buf_id.0 as usize];
                for &(offset, val) in pairs {
                    buf[offset as usize] = val;
                }
            }

            // Build input/output maps.
            // Input tensors: find which buffer corresponds to each input tensor's atoms.
            let mut input_map = HashMap::new();
            for &id in input_tensor_ids {
                if let Some(info) = lower.tensor_map.get(&id) {
                    if let Some(&buf_id) = plan.atom_to_buffer.get(&info.base_id.0) {
                        input_map.insert(id, buf_id.0 as usize);
                    }
                }
            }

            let mut output_map = HashMap::new();
            for &id in output_tensor_ids {
                if let Some(info) = lower.tensor_map.get(&id) {
                    if let Some(&buf_id) = plan.atom_to_buffer.get(&info.base_id.0) {
                        output_map.insert(id, buf_id.0 as usize);
                    }
                }
            }

            Ok(V10Executable {
                compiled,
                plan,
                buffer_storage,
                input_map,
                output_map,
            })
        }

        /// Set an input tensor's data.
        pub fn set_input(&mut self, id: &GlobalId, data: &[f32]) {
            if let Some(&buf_idx) = self.input_map.get(id) {
                let buf = &mut self.buffer_storage[buf_idx];
                let copy_len = data.len().min(buf.len());
                buf[..copy_len].copy_from_slice(&data[..copy_len]);
            }
        }

        /// Execute the compiled graph.
        pub fn execute(&mut self) {
            let mut ptrs: Vec<*mut f32> = self
                .buffer_storage
                .iter_mut()
                .map(|b| b.as_mut_ptr())
                .collect();

            let buf_ptr = ptrs.as_ptr();
            for kernel in &self.compiled.kernels {
                unsafe { kernel.execute(buf_ptr) };
            }
        }

        /// Execute with parallel tile-task scheduling.
        pub fn execute_parallel(&mut self, num_threads: usize) {
            let mut ptrs: Vec<*mut f32> = self
                .buffer_storage
                .iter_mut()
                .map(|b| b.as_mut_ptr())
                .collect();

            if num_threads <= 1 {
                self.execute();
                return;
            }

            // For now, just run kernels serially.
            // TODO: tile-task DAG executor.
            let buf_ptr = ptrs.as_ptr();
            for kernel in &self.compiled.kernels {
                unsafe { kernel.execute(buf_ptr) };
            }
        }

        /// Read an output tensor's data.
        pub fn get_output(&self, id: &GlobalId) -> Option<&[f32]> {
            self.output_map
                .get(id)
                .map(|&buf_idx| self.buffer_storage[buf_idx].as_slice())
        }

        /// Read a buffer by BufferId (for debugging).
        pub fn get_buffer(&self, buf_id: BufferId) -> &[f32] {
            &self.buffer_storage[buf_id.0 as usize]
        }
    }
}

#[cfg(all(test, feature = "cranelift"))]
mod tests {
    use std::collections::HashMap;

    use crate::backends::eval_backend::EvalBackend;
    use crate::graph::{GlobalId, Graph, Node};
    use crate::milli_graph::MilliOpGraph;
    use crate::nano_graph::eval::NanoEval;
    use crate::nano_graph::lower::{lower_with_info, LowerResult};
    use crate::numeric_scalar::NumericScalar;
    use crate::numeric_tensor::NumericTensor;
    use crate::tensor_info::TensorInfo;
    use crate::DynRank;

    use super::run::V10Executable;

    /// Build a milli graph, lower to nano, compile with v10, execute, compare to NanoEval.
    fn check_v10(
        build_graph: impl FnOnce(
            &mut MilliOpGraph,
            &mut rand::rngs::ThreadRng,
        ) -> (Vec<GlobalId>, Vec<GlobalId>),
        inputs: Vec<NumericTensor<DynRank>>,
    ) {
        let mut rng = rand::rng();
        let (mut milli, _ext_map) = MilliOpGraph::new(std::iter::empty(), &mut rng);

        let (input_ids, output_ids) = build_graph(&mut milli, &mut rng);
        assert_eq!(input_ids.len(), inputs.len());

        // Prepare info for lowering.
        let mut info_inputs: HashMap<GlobalId, TensorInfo> = HashMap::new();
        for (id, tensor) in input_ids.iter().zip(inputs.iter()) {
            info_inputs.insert(*id, TensorInfo::from(tensor.clone()));
        }

        // Lower to NanoGraph.
        let result = lower_with_info(&milli, &info_inputs).unwrap();
        assert!(
            result.unsupported.is_empty(),
            "Unsupported ops: {:?}",
            result.unsupported_details
        );

        // Build overrides including inputs.
        let mut overrides = result.numeric_overrides.clone();
        for (&id, tensor) in input_ids.iter().zip(inputs.iter()) {
            if let Some(tam) = result.tensor_map.get(&id) {
                let mut backend = EvalBackend::NDArray;
                let f32_t = tensor
                    .cast(crate::dtype::DType::F32, &mut backend)
                    .unwrap();
                let flat = f32_t.flatten().unwrap();
                let v: Vec<f32> = flat.to_ndarray().unwrap().try_into().unwrap();
                for (i, &val) in v.iter().enumerate() {
                    overrides.insert(
                        tam.base_id.0 + i as u32,
                        NumericScalar::F32(val),
                    );
                }
            }
        }

        // NanoEval reference.
        let nano_eval = NanoEval::eval(&result.graph, &overrides);

        // V10 compile & execute.
        let mut exe = V10Executable::build(&result, &input_ids, &output_ids)
            .expect("v10 build failed");

        // Set inputs.
        for (&id, tensor) in input_ids.iter().zip(inputs.iter()) {
            let mut backend = EvalBackend::NDArray;
            let f32_t = tensor
                .cast(crate::dtype::DType::F32, &mut backend)
                .unwrap();
            let flat = f32_t.flatten().unwrap();
            let v: Vec<f32> = flat.to_ndarray().unwrap().try_into().unwrap();
            exe.set_input(&id, &v);
        }

        exe.execute();
        // Compare outputs.
        for &out_id in &output_ids {
            let tam = result.tensor_map.get(&out_id).unwrap();
            let nano_vals: Vec<f64> = (0..tam.count)
                .map(|i| nano_eval.get(tam.base_id.offset(i)))
                .collect();

            let v10_vals = exe.get_output(&out_id).expect("output not found");
            assert_eq!(
                nano_vals.len(),
                v10_vals.len(),
                "Output {:?}: nano has {} elements, v10 has {}",
                out_id,
                nano_vals.len(),
                v10_vals.len()
            );

            for (i, (&nano, &v10)) in nano_vals.iter().zip(v10_vals.iter()).enumerate() {
                let diff = (nano - v10 as f64).abs();
                let tol = 1e-4 * nano.abs().max(1.0);
                assert!(
                    diff < tol,
                    "Output {:?} element {}: nano={} v10={} diff={}",
                    out_id,
                    i,
                    nano,
                    v10,
                    diff
                );
            }
        }
    }

    #[test]
    fn test_v10_add() {
        check_v10(
            |graph, rng| {
                let a = graph.add_input(rng);
                let b = graph.add_input(rng);
                let c = crate::milli_graph::ops::SimpleBinary::add(graph, a, b, rng);
                (vec![a, b], vec![c])
            },
            vec![
                NumericTensor::from_vec_shape(vec![1.0f32, 2.0, 3.0, 4.0], vec![4]).unwrap(),
                NumericTensor::from_vec_shape(vec![10.0f32, 20.0, 30.0, 40.0], vec![4]).unwrap(),
            ],
        );
    }

    #[test]
    fn test_v10_mul_add_chain() {
        check_v10(
            |graph, rng| {
                let a = graph.add_input(rng);
                let b = graph.add_input(rng);
                let c = graph.add_input(rng);
                let ab = crate::milli_graph::ops::SimpleBinary::mul(graph, a, b, rng);
                let abc = crate::milli_graph::ops::SimpleBinary::add(graph, ab, c, rng);
                (vec![a, b, c], vec![abc])
            },
            vec![
                NumericTensor::from_vec_shape(vec![1.0f32, 2.0, 3.0], vec![3]).unwrap(),
                NumericTensor::from_vec_shape(vec![4.0f32, 5.0, 6.0], vec![3]).unwrap(),
                NumericTensor::from_vec_shape(vec![0.1f32, 0.2, 0.3], vec![3]).unwrap(),
            ],
        );
    }

    #[test]
    fn test_v10_unary_exp() {
        check_v10(
            |graph, rng| {
                let a = graph.add_input(rng);
                let b = crate::milli_graph::ops::SimpleUnaryOp::exp(graph, a, rng);
                (vec![a], vec![b])
            },
            vec![
                NumericTensor::from_vec_shape(vec![0.0f32, 1.0, -1.0, 0.5], vec![4]).unwrap(),
            ],
        );
    }

    #[test]
    fn test_v10_matmul_2x3_3x2() {
        check_v10(
            |graph, rng| {
                let a = graph.add_input(rng);
                let b = graph.add_input(rng);
                let c = crate::milli_graph::ops::MatMul::push_new_default_precision(
                    graph,
                    a,
                    b,
                    crate::dtype::DType::F32,
                    rng,
                );
                (vec![a, b], vec![c])
            },
            vec![
                NumericTensor::from_vec_shape(
                    vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
                    vec![2, 3],
                )
                .unwrap(),
                NumericTensor::from_vec_shape(
                    vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0],
                    vec![3, 2],
                )
                .unwrap(),
            ],
        );
    }

    #[test]
    fn test_v10_broadcast_add() {
        // [2,3] + [3] → [2,3]
        check_v10(
            |graph, rng| {
                let a = graph.add_input(rng);
                let b = graph.add_input(rng);
                let c = crate::milli_graph::ops::SimpleBinary::add(graph, a, b, rng);
                (vec![a, b], vec![c])
            },
            vec![
                NumericTensor::from_vec_shape(
                    vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
                    vec![2, 3],
                )
                .unwrap(),
                NumericTensor::from_vec_shape(vec![10.0f32, 20.0, 30.0], vec![3]).unwrap(),
            ],
        );
    }

    #[test]
    fn test_v10_matmul_3x4_4x2() {
        // Larger matmul: [3,4] x [4,2] → [3,2]
        check_v10(
            |graph, rng| {
                let a = graph.add_input(rng);
                let b = graph.add_input(rng);
                let c = crate::milli_graph::ops::MatMul::push_new_default_precision(
                    graph,
                    a,
                    b,
                    crate::dtype::DType::F32,
                    rng,
                );
                (vec![a, b], vec![c])
            },
            vec![
                NumericTensor::from_vec_shape(
                    vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                    vec![3, 4],
                )
                .unwrap(),
                NumericTensor::from_vec_shape(
                    vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
                    vec![4, 2],
                )
                .unwrap(),
            ],
        );
    }

    #[test]
    fn test_v10_neg() {
        check_v10(
            |graph, rng| {
                let a = graph.add_input(rng);
                let b = crate::milli_graph::ops::SimpleUnaryOp::neg(graph, a, rng);
                (vec![a], vec![b])
            },
            vec![
                NumericTensor::from_vec_shape(vec![1.0f32, -2.0, 3.0, -4.0], vec![4]).unwrap(),
            ],
        );
    }

    #[test]
    fn test_v10_exp_add_chain() {
        // exp(a) + b — tests unary feeding into binary.
        check_v10(
            |graph, rng| {
                let a = graph.add_input(rng);
                let b = graph.add_input(rng);
                let ea = crate::milli_graph::ops::SimpleUnaryOp::exp(graph, a, rng);
                let c = crate::milli_graph::ops::SimpleBinary::add(graph, ea, b, rng);
                (vec![a, b], vec![c])
            },
            vec![
                NumericTensor::from_vec_shape(vec![0.0f32, 1.0, -1.0], vec![3]).unwrap(),
                NumericTensor::from_vec_shape(vec![10.0f32, 20.0, 30.0], vec![3]).unwrap(),
            ],
        );
    }

    #[test]
    fn test_v10_mul_then_matmul() {
        // Element-wise mul then matmul: (a * b) @ c
        // Tests pointwise kernel feeding into reduction kernel.
        check_v10(
            |graph, rng| {
                let a = graph.add_input(rng);
                let b = graph.add_input(rng);
                let c = graph.add_input(rng);
                let ab = crate::milli_graph::ops::SimpleBinary::mul(graph, a, b, rng);
                let d = crate::milli_graph::ops::MatMul::push_new_default_precision(
                    graph,
                    ab,
                    c,
                    crate::dtype::DType::F32,
                    rng,
                );
                (vec![a, b, c], vec![d])
            },
            vec![
                NumericTensor::from_vec_shape(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap(),
                NumericTensor::from_vec_shape(vec![2.0f32, 3.0, 4.0, 5.0], vec![2, 2]).unwrap(),
                NumericTensor::from_vec_shape(vec![1.0f32, 0.0, 0.0, 1.0], vec![2, 2]).unwrap(),
            ],
        );
    }

    #[test]
    fn test_v10_multiple_outputs() {
        // Two separate outputs from shared inputs: c = a+b, d = a*b
        check_v10(
            |graph, rng| {
                let a = graph.add_input(rng);
                let b = graph.add_input(rng);
                let c = crate::milli_graph::ops::SimpleBinary::add(graph, a, b, rng);
                let d = crate::milli_graph::ops::SimpleBinary::mul(graph, a, b, rng);
                (vec![a, b], vec![c, d])
            },
            vec![
                NumericTensor::from_vec_shape(vec![1.0f32, 2.0, 3.0], vec![3]).unwrap(),
                NumericTensor::from_vec_shape(vec![4.0f32, 5.0, 6.0], vec![3]).unwrap(),
            ],
        );
    }
}
