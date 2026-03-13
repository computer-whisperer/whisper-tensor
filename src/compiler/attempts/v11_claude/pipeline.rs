#![allow(clippy::all, dead_code, unreachable_patterns)]
//! v11 pipeline: NanoGraph → plan → compile → execute.

#[cfg(feature = "cranelift")]
pub mod run {
    use std::collections::HashMap;

    use crate::compiler::attempts::v11_claude::codegen::jit::{
        compile, CompiledGraph, V11Error,
    };
    use crate::compiler::attempts::v11_claude::plan::{
        build_plan, BufferId, CompilationPlan,
    };
    use crate::graph::GlobalId;
    use crate::nano_graph::lower::LowerResult;

    /// A ready-to-execute compiled model.
    pub struct V11Executable {
        pub compiled: CompiledGraph,
        pub plan: CompilationPlan,
        buffer_storage: Vec<Vec<f32>>,
        pub input_map: HashMap<GlobalId, usize>,
        pub output_map: HashMap<GlobalId, usize>,
    }

    impl V11Executable {
        pub fn build(
            lower: &LowerResult,
            input_tensor_ids: &[GlobalId],
            output_tensor_ids: &[GlobalId],
        ) -> Result<Self, V11Error> {
            let plan = build_plan(lower);

            eprintln!(
                "[v11] {} buffers, {} kernels, {} output buffers",
                plan.buffers.len(), plan.kernels.len(), plan.output_buffers.len(),
            );

            let compiled = compile(&plan)?;

            let mut buffer_storage: Vec<Vec<f32>> = plan.buffers.iter()
                .map(|b| vec![0.0f32; b.count as usize])
                .collect();

            // Allocate table buffers (IDs start after regular buffers).
            // Table buffers store u32 offsets reinterpreted as f32 bits.
            if !plan.table_data.is_empty() {
                let max_table_id = plan.table_data.keys().map(|b| b.0).max().unwrap_or(0);
                while buffer_storage.len() <= max_table_id as usize {
                    buffer_storage.push(Vec::new());
                }
                for (buf_id, offsets) in &plan.table_data {
                    let data: Vec<f32> = offsets.iter()
                        .map(|&o| f32::from_bits(o))
                        .collect();
                    buffer_storage[buf_id.0 as usize] = data;
                }
            }

            for (buf_id, pairs) in &plan.constants {
                let buf = &mut buffer_storage[buf_id.0 as usize];
                for &(offset, val) in pairs {
                    if (offset as usize) < buf.len() {
                        buf[offset as usize] = val;
                    }
                }
            }

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

            Ok(V11Executable { compiled, plan, buffer_storage, input_map, output_map })
        }

        pub fn set_input(&mut self, id: &GlobalId, data: &[f32]) {
            if let Some(&buf_idx) = self.input_map.get(id) {
                let buf = &mut self.buffer_storage[buf_idx];
                let copy_len = data.len().min(buf.len());
                buf[..copy_len].copy_from_slice(&data[..copy_len]);
            }
        }

        pub fn execute(&mut self) {
            self.execute_range(0, self.compiled.kernels.len());
        }

        pub fn execute_range(&mut self, start: usize, end: usize) {
            let mut ptrs: Vec<*mut f32> = self.buffer_storage.iter_mut()
                .map(|b| b.as_mut_ptr())
                .collect();
            let buf_ptr = ptrs.as_ptr();
            for ki in start..end.min(self.compiled.kernels.len()) {
                unsafe { self.compiled.kernels[ki].execute(buf_ptr) };
            }
        }

        pub fn get_output(&self, id: &GlobalId) -> Option<&[f32]> {
            self.output_map.get(id)
                .map(|&buf_idx| self.buffer_storage[buf_idx].as_slice())
        }

        pub fn get_buffer(&self, buf_id: BufferId) -> &[f32] {
            &self.buffer_storage[buf_id.0 as usize]
        }

        pub fn num_buffers(&self) -> usize {
            self.buffer_storage.len()
        }
    }
}

#[cfg(all(test, feature = "cranelift"))]
mod tests {
    use std::collections::HashMap;

    use crate::backends::eval_backend::EvalBackend;
    use crate::backends::ndarray_backend::NDArrayNumericTensor;
    use crate::graph::{GlobalId, Graph, Node};
    use crate::milli_graph::MilliOpGraph;
    use crate::nano_graph::eval::NanoEval;
    use crate::nano_graph::lower::{lower_with_info, LowerResult};
    use crate::numeric_scalar::NumericScalar;
    use crate::numeric_tensor::NumericTensor;
    use crate::tensor_info::TensorInfo;
    use crate::DynRank;

    use super::run::V11Executable;

    /// Build milli graph, lower to nano, compile with v11, execute, compare to NanoEval.
    fn check_v11(
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

        let mut info_inputs: HashMap<GlobalId, TensorInfo> = HashMap::new();
        for (id, tensor) in input_ids.iter().zip(inputs.iter()) {
            info_inputs.insert(*id, TensorInfo::from(tensor.clone()));
        }

        let result = lower_with_info(&milli, &info_inputs).unwrap();
        assert!(
            result.unsupported.is_empty(),
            "Unsupported ops: {:?}", result.unsupported_details
        );

        let mut overrides = result.numeric_overrides.clone();
        for (&id, tensor) in input_ids.iter().zip(inputs.iter()) {
            if let Some(tam) = result.tensor_map.get(&id) {
                let mut backend = EvalBackend::NDArray;
                let f32_t = tensor.cast(crate::dtype::DType::F32, &mut backend).unwrap();
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

        let nano_eval = NanoEval::eval(&result.graph, &overrides);

        let mut exe = V11Executable::build(&result, &input_ids, &output_ids)
            .expect("v11 build failed");

        for (&id, tensor) in input_ids.iter().zip(inputs.iter()) {
            let mut backend = EvalBackend::NDArray;
            let f32_t = tensor.cast(crate::dtype::DType::F32, &mut backend).unwrap();
            let flat = f32_t.flatten().unwrap();
            let v: Vec<f32> = flat.to_ndarray().unwrap().try_into().unwrap();
            exe.set_input(&id, &v);
        }

        exe.execute();

        for &out_id in &output_ids {
            let tam = result.tensor_map.get(&out_id).unwrap();
            let nano_vals: Vec<f64> = (0..tam.count)
                .map(|i| nano_eval.get(tam.base_id.offset(i)))
                .collect();

            let v11_vals = exe.get_output(&out_id).expect("output not found");
            assert_eq!(
                nano_vals.len(), v11_vals.len(),
                "Output {:?}: nano has {} elements, v11 has {}",
                out_id, nano_vals.len(), v11_vals.len()
            );

            for (i, (&nano, &v11)) in nano_vals.iter().zip(v11_vals.iter()).enumerate() {
                let diff = (nano - v11 as f64).abs();
                let tol = 1e-4 * nano.abs().max(1.0);
                assert!(
                    diff < tol,
                    "Output {:?} element {}: nano={} v11={} diff={}",
                    out_id, i, nano, v11, diff
                );
            }
        }
    }

    #[test]
    fn test_v11_add() {
        check_v11(
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
    fn test_v11_mul_add_chain() {
        check_v11(
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
    fn test_v11_unary_exp() {
        check_v11(
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
    fn test_v11_neg() {
        check_v11(
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
    fn test_v11_broadcast_add() {
        // [2,3] + [3] → [2,3]
        check_v11(
            |graph, rng| {
                let a = graph.add_input(rng);
                let b = graph.add_input(rng);
                let c = crate::milli_graph::ops::SimpleBinary::add(graph, a, b, rng);
                (vec![a, b], vec![c])
            },
            vec![
                NumericTensor::from_vec_shape(
                    vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3],
                ).unwrap(),
                NumericTensor::from_vec_shape(vec![10.0f32, 20.0, 30.0], vec![3]).unwrap(),
            ],
        );
    }

    #[test]
    fn test_v11_exp_add_chain() {
        check_v11(
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
    fn test_v11_matmul_2x3_3x2() {
        check_v11(
            |graph, rng| {
                let a = graph.add_input(rng);
                let b = graph.add_input(rng);
                let c = crate::milli_graph::ops::MatMul::push_new_default_precision(
                    graph, a, b, crate::dtype::DType::F32, rng,
                );
                (vec![a, b], vec![c])
            },
            vec![
                NumericTensor::from_vec_shape(
                    vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3],
                ).unwrap(),
                NumericTensor::from_vec_shape(
                    vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0], vec![3, 2],
                ).unwrap(),
            ],
        );
    }

    #[test]
    fn test_v11_matmul_3x4_4x2() {
        check_v11(
            |graph, rng| {
                let a = graph.add_input(rng);
                let b = graph.add_input(rng);
                let c = crate::milli_graph::ops::MatMul::push_new_default_precision(
                    graph, a, b, crate::dtype::DType::F32, rng,
                );
                (vec![a, b], vec![c])
            },
            vec![
                NumericTensor::from_vec_shape(
                    vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                    vec![3, 4],
                ).unwrap(),
                NumericTensor::from_vec_shape(
                    vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
                    vec![4, 2],
                ).unwrap(),
            ],
        );
    }

    #[test]
    fn test_v11_mul_then_matmul() {
        // (a * b) @ c
        check_v11(
            |graph, rng| {
                let a = graph.add_input(rng);
                let b = graph.add_input(rng);
                let c = graph.add_input(rng);
                let ab = crate::milli_graph::ops::SimpleBinary::mul(graph, a, b, rng);
                let d = crate::milli_graph::ops::MatMul::push_new_default_precision(
                    graph, ab, c, crate::dtype::DType::F32, rng,
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
    fn test_v11_multiple_outputs() {
        check_v11(
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

    // --- All binary ops ---

    #[test]
    fn test_v11_sub() {
        check_v11(
            |graph, rng| {
                let a = graph.add_input(rng);
                let b = graph.add_input(rng);
                let c = crate::milli_graph::ops::SimpleBinary::sub(graph, a, b, rng);
                (vec![a, b], vec![c])
            },
            vec![
                NumericTensor::from_vec_shape(vec![10.0f32, 20.0, 30.0, 40.0], vec![4]).unwrap(),
                NumericTensor::from_vec_shape(vec![1.0f32, 2.0, 3.0, 4.0], vec![4]).unwrap(),
            ],
        );
    }

    #[test]
    fn test_v11_div() {
        check_v11(
            |graph, rng| {
                let a = graph.add_input(rng);
                let b = graph.add_input(rng);
                let c = crate::milli_graph::ops::SimpleBinary::div(graph, a, b, rng);
                (vec![a, b], vec![c])
            },
            vec![
                NumericTensor::from_vec_shape(vec![10.0f32, 20.0, 30.0, 40.0], vec![4]).unwrap(),
                NumericTensor::from_vec_shape(vec![2.0f32, 4.0, 5.0, 8.0], vec![4]).unwrap(),
            ],
        );
    }

    #[test]
    fn test_v11_max_min() {
        check_v11(
            |graph, rng| {
                let a = graph.add_input(rng);
                let b = graph.add_input(rng);
                let c = crate::milli_graph::ops::SimpleBinary::max(graph, a, b, rng);
                let d = crate::milli_graph::ops::SimpleBinary::min(graph, a, b, rng);
                (vec![a, b], vec![c, d])
            },
            vec![
                NumericTensor::from_vec_shape(vec![1.0f32, 5.0, 3.0, 8.0], vec![4]).unwrap(),
                NumericTensor::from_vec_shape(vec![4.0f32, 2.0, 6.0, 1.0], vec![4]).unwrap(),
            ],
        );
    }

    // --- All unary ops ---

    #[test]
    fn test_v11_abs() {
        check_v11(
            |graph, rng| {
                let a = graph.add_input(rng);
                let b = crate::milli_graph::ops::SimpleUnaryOp::abs(graph, a, rng);
                (vec![a], vec![b])
            },
            vec![
                NumericTensor::from_vec_shape(vec![-1.0f32, 2.0, -3.0, 4.0], vec![4]).unwrap(),
            ],
        );
    }

    #[test]
    fn test_v11_ln() {
        check_v11(
            |graph, rng| {
                let a = graph.add_input(rng);
                let b = crate::milli_graph::ops::SimpleUnaryOp::ln(graph, a, rng);
                (vec![a], vec![b])
            },
            vec![
                NumericTensor::from_vec_shape(vec![1.0f32, 2.0, 0.5, 10.0], vec![4]).unwrap(),
            ],
        );
    }

    #[test]
    fn test_v11_sqrt() {
        check_v11(
            |graph, rng| {
                let a = graph.add_input(rng);
                let b = crate::milli_graph::ops::SimpleUnaryOp::sqrt(graph, a, rng);
                (vec![a], vec![b])
            },
            vec![
                NumericTensor::from_vec_shape(vec![1.0f32, 4.0, 9.0, 16.0], vec![4]).unwrap(),
            ],
        );
    }

    #[test]
    fn test_v11_reciprocal() {
        check_v11(
            |graph, rng| {
                let a = graph.add_input(rng);
                let b = crate::milli_graph::ops::SimpleUnaryOp::reciprocal(graph, a, rng);
                (vec![a], vec![b])
            },
            vec![
                NumericTensor::from_vec_shape(vec![1.0f32, 2.0, 4.0, 0.5], vec![4]).unwrap(),
            ],
        );
    }

    #[test]
    fn test_v11_tanh() {
        check_v11(
            |graph, rng| {
                let a = graph.add_input(rng);
                let b = crate::milli_graph::ops::SimpleUnaryOp::trig(
                    graph, a, crate::TrigOp::Tanh, rng,
                );
                (vec![a], vec![b])
            },
            vec![
                NumericTensor::from_vec_shape(vec![0.0f32, 1.0, -1.0, 0.5], vec![4]).unwrap(),
            ],
        );
    }

    #[test]
    fn test_v11_floor_ceil() {
        check_v11(
            |graph, rng| {
                let a = graph.add_input(rng);
                let b = crate::milli_graph::ops::SimpleUnaryOp::floor(graph, a, rng);
                let c = crate::milli_graph::ops::SimpleUnaryOp::ceil(graph, a, rng);
                (vec![a], vec![b, c])
            },
            vec![
                NumericTensor::from_vec_shape(vec![1.3f32, -2.7, 3.0, 0.5], vec![4]).unwrap(),
            ],
        );
    }

    // --- Larger matmuls ---

    #[test]
    fn test_v11_matmul_8x16_16x4() {
        let m = 8; let k = 16; let n = 4;
        let a_data: Vec<f32> = (0..m*k).map(|i| 0.01 * (i as f32 + 1.0)).collect();
        let b_data: Vec<f32> = (0..k*n).map(|i| 0.01 * ((i as f32) - 0.5)).collect();
        check_v11(
            |graph, rng| {
                let a = graph.add_input(rng);
                let b = graph.add_input(rng);
                let c = crate::milli_graph::ops::MatMul::push_new_default_precision(
                    graph, a, b, crate::dtype::DType::F32, rng,
                );
                (vec![a, b], vec![c])
            },
            vec![
                NumericTensor::from_vec_shape(a_data, vec![m, k]).unwrap(),
                NumericTensor::from_vec_shape(b_data, vec![k, n]).unwrap(),
            ],
        );
    }

    #[test]
    fn test_v11_matmul_32x32_32x32() {
        let n = 32;
        let a_data: Vec<f32> = (0..n*n).map(|i| 0.001 * (i as f32 + 1.0)).collect();
        let b_data: Vec<f32> = (0..n*n).map(|i| 0.001 * ((i as f32) - 0.5)).collect();
        check_v11(
            |graph, rng| {
                let a = graph.add_input(rng);
                let b = graph.add_input(rng);
                let c = crate::milli_graph::ops::MatMul::push_new_default_precision(
                    graph, a, b, crate::dtype::DType::F32, rng,
                );
                (vec![a, b], vec![c])
            },
            vec![
                NumericTensor::from_vec_shape(a_data, vec![n, n]).unwrap(),
                NumericTensor::from_vec_shape(b_data, vec![n, n]).unwrap(),
            ],
        );
    }

    // --- Composite patterns (softmax-like, layer-norm-like) ---

    /// Helper to create an i64 scalar constant (for axes parameters).
    fn make_axes_constant(graph: &mut MilliOpGraph, axis: i64, rng: &mut impl rand::Rng) -> GlobalId {
        let data = NDArrayNumericTensor::<DynRank>::from_vec_shape(vec![axis], &vec![1]).unwrap();
        crate::milli_graph::ops::Constant::push_new(graph, data, rng)
    }

    #[test]
    fn test_v11_reduce_sum() {
        // reduce_sum on a [3,4] tensor along axis 1 → [3]
        check_v11(
            |graph, rng| {
                let a = graph.add_input(rng);
                let axes = make_axes_constant(graph, 1, rng);
                let b = crate::milli_graph::ops::ReduceSum::push_new(
                    graph, a, Some(axes), false, false, rng,
                );
                (vec![a], vec![b])
            },
            vec![
                NumericTensor::from_vec_shape(
                    (1..=12).map(|i| i as f32).collect(),
                    vec![3, 4],
                ).unwrap(),
            ],
        );
    }

    #[test]
    fn test_v11_reduce_max() {
        check_v11(
            |graph, rng| {
                let a = graph.add_input(rng);
                let axes = make_axes_constant(graph, 1, rng);
                let b = crate::milli_graph::ops::ReduceMax::push_new(
                    graph, a, Some(axes), false, false, rng,
                );
                (vec![a], vec![b])
            },
            vec![
                NumericTensor::from_vec_shape(
                    vec![3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0],
                    vec![3, 3],
                ).unwrap(),
            ],
        );
    }

    #[test]
    fn test_v11_reduce_mean() {
        check_v11(
            |graph, rng| {
                let a = graph.add_input(rng);
                let axes = make_axes_constant(graph, 1, rng);
                let b = crate::milli_graph::ops::ReduceMean::push_new(
                    graph, a, Some(axes), false, false, rng,
                );
                (vec![a], vec![b])
            },
            vec![
                NumericTensor::from_vec_shape(
                    (1..=12).map(|i| i as f32).collect(),
                    vec![3, 4],
                ).unwrap(),
            ],
        );
    }

    #[test]
    fn test_v11_softmax_pattern() {
        // softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
        check_v11(
            |graph, rng| {
                let x = graph.add_input(rng);
                let axes = make_axes_constant(graph, 1, rng);
                let max_x = crate::milli_graph::ops::ReduceMax::push_new(
                    graph, x, Some(axes), true, false, rng,
                );
                let shifted = crate::milli_graph::ops::SimpleBinary::sub(graph, x, max_x, rng);
                let exp_x = crate::milli_graph::ops::SimpleUnaryOp::exp(graph, shifted, rng);
                let axes2 = make_axes_constant(graph, 1, rng);
                let sum_exp = crate::milli_graph::ops::ReduceSum::push_new(
                    graph, exp_x, Some(axes2), true, false, rng,
                );
                let result = crate::milli_graph::ops::SimpleBinary::div(graph, exp_x, sum_exp, rng);
                (vec![x], vec![result])
            },
            vec![
                NumericTensor::from_vec_shape(
                    vec![1.0f32, 2.0, 3.0, 4.0, 1.0, 5.0, 2.0, 3.0],
                    vec![2, 4],
                ).unwrap(),
            ],
        );
    }

    #[test]
    fn test_v11_chain_unary() {
        // exp(tanh(x)) - tests fusing multiple unary ops
        check_v11(
            |graph, rng| {
                let a = graph.add_input(rng);
                let b = crate::milli_graph::ops::SimpleUnaryOp::trig(
                    graph, a, crate::TrigOp::Tanh, rng,
                );
                let c = crate::milli_graph::ops::SimpleUnaryOp::exp(graph, b, rng);
                (vec![a], vec![c])
            },
            vec![
                NumericTensor::from_vec_shape(vec![0.0f32, 0.5, -0.5, 1.0], vec![4]).unwrap(),
            ],
        );
    }

    #[test]
    fn test_v11_matmul_add_bias() {
        // y = x @ w + b  (common linear layer pattern)
        check_v11(
            |graph, rng| {
                let x = graph.add_input(rng);
                let w = graph.add_input(rng);
                let b = graph.add_input(rng);
                let xw = crate::milli_graph::ops::MatMul::push_new_default_precision(
                    graph, x, w, crate::dtype::DType::F32, rng,
                );
                let result = crate::milli_graph::ops::SimpleBinary::add(graph, xw, b, rng);
                (vec![x, w, b], vec![result])
            },
            vec![
                NumericTensor::from_vec_shape(
                    vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3],
                ).unwrap(),
                NumericTensor::from_vec_shape(
                    vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6], vec![3, 2],
                ).unwrap(),
                NumericTensor::from_vec_shape(vec![1.0f32, 2.0], vec![2]).unwrap(),
            ],
        );
    }
}
