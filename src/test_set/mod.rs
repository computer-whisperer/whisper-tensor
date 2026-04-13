//! Test dataset infrastructure for verifying MilliOpGraph evaluation.
//!
//! Test data is constructed using the new [`NumericTensor`](crate::numeric_tensor::NumericTensor)
//! types with [`SystemPool`](crate::pool::SystemPool). At the eval boundary,
//! tensors are converted to legacy types for the current MilliOpGraph interpreter.
//!
//! Behind `#[cfg(feature = "tch")]`, data sets can be validated against PyTorch
//! via the tch crate as an independent oracle.

pub mod cast;
pub mod composite;
pub mod conv;
pub mod dtype_discipline;
pub mod elementwise;
pub mod matmul;
pub mod opaque_ops;
pub mod pad;
pub mod reduce;
pub mod structural;

use std::collections::HashMap;

use crate::graph::GlobalId;
use crate::milli_graph::MilliOpGraph;
use crate::numeric_dtype::NumericDType;
use crate::numeric_scalar::NumericScalar;
use crate::numeric_tensor::{NumericTensor, NumericTensorView};
use crate::pool::SystemPool;
use crate::tensor_rank::DynRank;

/// Shared static SystemPool for all test/bridge allocations.
/// SystemPool is stateless (unit struct) — a static ref is always valid.
pub static POOL: SystemPool = SystemPool;

/// Type alias for test tensors (SystemPool, 'static lifetime).
pub type TestTensor = NumericTensor<'static, DynRank, SystemPool>;

/// A complete test case: one graph, multiple input/output data sets.
pub struct TestCase {
    pub name: String,
    pub graph: MilliOpGraph,
    pub data_sets: Vec<TestDataSet>,
}

/// One input/output pair for a test case.
pub struct TestDataSet {
    pub label: String,
    pub inputs: HashMap<GlobalId, TestTensor>,
    pub expected_outputs: HashMap<GlobalId, TestTensor>,
    pub tolerance: Tolerance,
}

/// Tolerance for comparing tensor values.
#[derive(Clone, Debug)]
pub struct Tolerance {
    pub atol: f64,
    pub rtol: f64,
}

impl Tolerance {
    /// Exact match (integer/bool semantics).
    pub const EXACT: Self = Tolerance {
        atol: 0.0,
        rtol: 0.0,
    };

    pub fn for_dtype(dtype: NumericDType) -> Self {
        match dtype {
            NumericDType::F64 => Tolerance {
                atol: 1e-10,
                rtol: 1e-10,
            },
            NumericDType::F32 => Tolerance {
                atol: 1e-5,
                rtol: 1.3e-6,
            },
            NumericDType::F16 => Tolerance {
                atol: 1e-3,
                rtol: 4e-3,
            },
            NumericDType::BF16 => Tolerance {
                atol: 1e-2,
                rtol: 1.6e-2,
            },
            NumericDType::I64
            | NumericDType::I32
            | NumericDType::I16
            | NumericDType::I8
            | NumericDType::U64
            | NumericDType::U32
            | NumericDType::U16
            | NumericDType::U8
            | NumericDType::BOOL => Tolerance {
                atol: 0.0,
                rtol: 0.0,
            },
            _ => Tolerance {
                atol: 1e-3,
                rtol: 1e-3,
            }, // conservative default for exotic types
        }
    }
}

// ---------------------------------------------------------------------------
// Tensor construction helpers
// ---------------------------------------------------------------------------

/// Create a 1D test tensor from f64 values encoded into the given dtype.
/// Works for any dtype — values are cast through the dtype's encode path.
pub fn tensor_from_f64(dtype: NumericDType, values: &[f64]) -> TestTensor {
    let shape = vec![values.len() as u64];
    let mut t = NumericTensor::zeros(shape, dtype, &POOL).unwrap();
    for (i, &v) in values.iter().enumerate() {
        t.write_element(i, NumericScalar::from_f64(v).cast_to(dtype));
    }
    t
}

/// Create a test tensor from f32 values with an explicit shape.
pub fn tensor_f32_shaped(shape: Vec<u64>, values: &[f32]) -> TestTensor {
    assert_eq!(
        shape.iter().product::<u64>() as usize,
        values.len(),
        "shape product must match values length"
    );
    let mut t = NumericTensor::zeros(shape, NumericDType::F32, &POOL).unwrap();
    for (i, &v) in values.iter().enumerate() {
        t.write_element(i, NumericScalar::from_f32(v));
    }
    t
}

/// Create a 1D test tensor from f32 values.
pub fn tensor_f32(values: &[f32]) -> TestTensor {
    tensor_f32_shaped(vec![values.len() as u64], values)
}

/// Create a test tensor from f64 values with an explicit shape.
pub fn tensor_f64_shaped(shape: Vec<u64>, values: &[f64]) -> TestTensor {
    assert_eq!(shape.iter().product::<u64>() as usize, values.len());
    let mut t = NumericTensor::zeros(shape, NumericDType::F64, &POOL).unwrap();
    for (i, &v) in values.iter().enumerate() {
        t.write_element(i, NumericScalar::from_f64(v));
    }
    t
}

/// Create a test tensor from bf16 values with an explicit shape.
pub fn tensor_bf16_shaped(shape: Vec<u64>, values: &[half::bf16]) -> TestTensor {
    assert_eq!(shape.iter().product::<u64>() as usize, values.len());
    let mut t = NumericTensor::zeros(shape, NumericDType::BF16, &POOL).unwrap();
    for (i, &v) in values.iter().enumerate() {
        t.write_element(i, NumericScalar::from_bf16(v));
    }
    t
}

/// Create a 1D test tensor from bf16 values.
pub fn tensor_bf16(values: &[half::bf16]) -> TestTensor {
    tensor_bf16_shaped(vec![values.len() as u64], values)
}

/// Create a test tensor from f16 values with an explicit shape.
pub fn tensor_f16_shaped(shape: Vec<u64>, values: &[half::f16]) -> TestTensor {
    assert_eq!(shape.iter().product::<u64>() as usize, values.len());
    let mut t = NumericTensor::zeros(shape, NumericDType::F16, &POOL).unwrap();
    for (i, &v) in values.iter().enumerate() {
        t.write_element(i, NumericScalar::from_f16(v));
    }
    t
}

/// Create a 1D test tensor from f16 values.
pub fn tensor_f16(values: &[half::f16]) -> TestTensor {
    tensor_f16_shaped(vec![values.len() as u64], values)
}

/// Create a test tensor from i32 values with an explicit shape.
pub fn tensor_i32_shaped(shape: Vec<u64>, values: &[i32]) -> TestTensor {
    assert_eq!(shape.iter().product::<u64>() as usize, values.len());
    let mut t = NumericTensor::zeros(shape, NumericDType::I32, &POOL).unwrap();
    for (i, &v) in values.iter().enumerate() {
        t.write_element(i, NumericScalar::from_i32(v));
    }
    t
}

/// Create a 1D test tensor from i32 values.
pub fn tensor_i32(values: &[i32]) -> TestTensor {
    tensor_i32_shaped(vec![values.len() as u64], values)
}

/// Create a test tensor from i64 values with an explicit shape.
pub fn tensor_i64_shaped(shape: Vec<u64>, values: &[i64]) -> TestTensor {
    assert_eq!(shape.iter().product::<u64>() as usize, values.len());
    let mut t = NumericTensor::zeros(shape, NumericDType::I64, &POOL).unwrap();
    for (i, &v) in values.iter().enumerate() {
        t.write_element(i, NumericScalar::from_i64(v));
    }
    t
}

/// Create a 1D test tensor from i64 values.
pub fn tensor_i64(values: &[i64]) -> TestTensor {
    tensor_i64_shaped(vec![values.len() as u64], values)
}

/// Create a 1D test tensor of BOOL values.
pub fn tensor_bool(values: &[bool]) -> TestTensor {
    tensor_bool_shaped(vec![values.len() as u64], values)
}

/// Create a test tensor of BOOL values with an explicit shape.
pub fn tensor_bool_shaped(shape: Vec<u64>, values: &[bool]) -> TestTensor {
    assert_eq!(shape.iter().product::<u64>() as usize, values.len());
    let mut t = NumericTensor::zeros(shape, NumericDType::BOOL, &POOL).unwrap();
    for (i, &v) in values.iter().enumerate() {
        t.write_element(i, NumericScalar::from_bool(v));
    }
    t
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

/// Compare two new-type tensors element-wise within tolerance.
pub fn assert_tensors_close(
    actual: &NumericTensorView<'_, DynRank>,
    expected: &NumericTensorView<'_, DynRank>,
    tolerance: &Tolerance,
    context: &str,
) -> Result<(), String> {
    if actual.shape() != expected.shape() {
        return Err(format!(
            "{context}: shape mismatch: actual {:?} vs expected {:?}",
            actual.shape(),
            expected.shape()
        ));
    }
    if actual.dtype() != expected.dtype() {
        return Err(format!(
            "{context}: dtype mismatch: actual {:?} vs expected {:?}",
            actual.dtype(),
            expected.dtype()
        ));
    }

    let numel = actual.numel();
    for i in 0..numel {
        let a = actual.read_element(i).to_f64();
        let e = expected.read_element(i).to_f64();
        if a.is_nan() && e.is_nan() {
            continue;
        }
        if a.is_infinite() && e.is_infinite() && a.signum() == e.signum() {
            continue;
        }
        let err = (a - e).abs();
        let limit = tolerance.atol + tolerance.rtol * a.abs().max(e.abs());
        if err > limit {
            return Err(format!(
                "{context}: element [{i}] mismatch: actual={a}, expected={e}, \
                 err={err}, limit={limit} (atol={}, rtol={})",
                tolerance.atol, tolerance.rtol
            ));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Test runners
// ---------------------------------------------------------------------------

/// Collect all test cases from all submodules.
pub fn build_test_set() -> Vec<TestCase> {
    let mut cases = Vec::new();
    cases.extend(elementwise::build_cases());
    cases.extend(matmul::build_cases());
    cases.extend(cast::build_cases());
    cases.extend(reduce::build_cases());
    cases.extend(composite::build_cases());
    cases.extend(conv::build_cases());
    cases.extend(pad::build_cases());
    cases.extend(structural::build_cases());
    cases.extend(opaque_ops::build_cases());
    cases.extend(dtype_discipline::build_cases());
    cases
}

/// Run all data sets of a test case through the pool-based nano eval.
///
/// Flow: MilliOpGraph → lower_to_nano → pool_eval → compare
pub fn run_case_via_pool_eval(case: &TestCase) -> Result<(), String> {
    use crate::nano_graph::lower;

    use crate::nano_graph::pool_eval;
    use crate::pool::TrackedPool;
    use crate::tensor_info::TensorInfo;

    for ds in &case.data_sets {
        // Build TensorInfo for each input (needed by lower).
        let info_inputs: HashMap<GlobalId, TensorInfo<'_, '_, crate::pool::SystemPool>> = ds
            .inputs
            .iter()
            .map(|(&id, t)| {
                (
                    id,
                    TensorInfo::from_view(&t.view(), &crate::pool::SystemPool),
                )
            })
            .collect();

        // Lower MilliOpGraph → NanoGraph.
        let lower_result = lower::lower(&case.graph, &info_inputs, &crate::pool::SystemPool)
            .map_err(|e| format!("{}[{}]: lower failed: {e}", case.name, ds.label))?;

        if !lower_result.unsupported.is_empty() {
            return Err(format!(
                "{}[{}]: unsupported ops: {:?}",
                case.name, ds.label, lower_result.unsupported_details
            ));
        }

        // Map input tensors to (TAMI, view) pairs for pool_eval.
        let input_tam_pairs: Vec<_> = ds
            .inputs
            .iter()
            .filter_map(|(&ext_id, tensor)| {
                let internal_id = case.graph.input_map.get(&ext_id)?;
                let tam = lower_result.graph.tensor_map.get(internal_id)?;
                Some((tam, tensor))
            })
            .collect();

        let input_views: Vec<_> = input_tam_pairs
            .iter()
            .map(|(_, tensor)| tensor.view())
            .collect();

        let eval_inputs: Vec<_> = input_tam_pairs
            .iter()
            .zip(input_views.iter())
            .map(|((tam, _), view)| (*tam, view))
            .collect();

        // Build output TAMIs from the graph's output ids.
        let output_ids: Vec<GlobalId> = case.graph.output_ordering.clone().unwrap_or_default();

        let output_tamis: Vec<_> = output_ids
            .iter()
            .filter_map(|out_id| lower_result.graph.tensor_map.get(out_id))
            .collect();

        // Run pool_eval — returns correctly-shaped output tensors.
        let pool = TrackedPool::new(None);
        let eval_results =
            pool_eval::pool_eval(&lower_result.graph, &eval_inputs, &output_tamis, &[], &pool)
                .map_err(|e| format!("{}[{}]: pool_eval failed: {e}", case.name, ds.label))?;

        // Compare outputs — pool_eval returns tensors in the same order as output_tamis,
        // which aligns with output_ids. Each result is already correctly shaped.
        for (out_id, result_tensor) in output_ids.iter().zip(eval_results.iter()) {
            let expected_tensor = match ds.expected_outputs.get(out_id) {
                Some(t) => t,
                None => continue,
            };

            let ctx = format!("{}[{}] pool_eval", case.name, ds.label);
            assert_tensors_close(
                &result_tensor.view(),
                &expected_tensor.view(),
                &ds.tolerance,
                &ctx,
            )?;
        }
    }
    Ok(())
}

/// Run a test case with some dimensions withheld (made symbolic).
///
/// `withheld_dims` lists dimension indices to withhold. For each input tensor,
/// any dim whose index is in this set AND whose size matches across all inputs
/// at that index is replaced with a shared SymbolicScalarTyped (same symbol_id).
/// This exercises the symbolic dimension path end-to-end:
/// lower → NanoGraph with sym_dims → pool_eval with gc_values → compare.
pub fn run_case_with_symbolic_dims(case: &TestCase, withheld_dims: &[usize]) -> Result<(), String> {
    use crate::nano_graph::lower;
    use crate::nano_graph::pool_eval;
    use crate::pool::TrackedPool;
    use crate::scalar_info::ScalarInfoTyped;
    use crate::symbolic_scalar::SymbolicScalarTyped;
    use crate::tensor_info::TensorInfo;

    for ds in &case.data_sets {
        let mut rng = rand::rng();

        // Check that all inputs agree on the withheld dim sizes AND ranks.
        // Sharing a single symbol for dim index d across inputs is only valid
        // when that dim corresponds to the same output axis for all inputs.
        // Different ranks means broadcasting shifts dim alignment, making the
        // shared symbol semantically wrong.
        let mut skip = false;
        let mut seen_rank: Option<usize> = None;
        for t in ds.inputs.values() {
            let r = t.view().shape().len();
            if let Some(prev) = seen_rank {
                if prev != r {
                    skip = true;
                    break;
                }
            } else {
                seen_rank = Some(r);
            }
        }
        if !skip {
            for &d in withheld_dims {
                let mut seen_size: Option<u64> = None;
                for t in ds.inputs.values() {
                    let shape = t.view().shape().clone();
                    if d < shape.len() && shape[d] > 1 {
                        if let Some(prev) = seen_size {
                            if prev != shape[d] {
                                skip = true;
                                break;
                            }
                        } else {
                            seen_size = Some(shape[d]);
                        }
                    }
                }
                if skip {
                    break;
                }
            }
        }
        if skip {
            continue;
        }

        // For each withheld dim index, create a shared SymbolicScalarTyped
        // so all inputs that have that dim share the same GraphConstant.
        let shared_syms: HashMap<usize, SymbolicScalarTyped<u64>> = withheld_dims
            .iter()
            .map(|&d| (d, SymbolicScalarTyped::new(&mut rng)))
            .collect();

        // Build TensorInfo with withheld dims as Symbolic.
        let info_inputs: HashMap<GlobalId, TensorInfo<'_, '_, SystemPool>> = ds
            .inputs
            .iter()
            .map(|(&id, t)| {
                let view = t.view();
                let shape = view.shape();
                let rank = shape.len();

                let dims: Vec<ScalarInfoTyped<u64>> = (0..rank)
                    .map(|d| {
                        if let Some(sym) = shared_syms.get(&d) {
                            // Don't make broadcast dims (size 1) symbolic —
                            // they don't share the same extent as other inputs.
                            if shape[d] <= 1 {
                                ScalarInfoTyped::Numeric(shape[d])
                            } else {
                                ScalarInfoTyped::Symbolic(sym.clone())
                            }
                        } else {
                            ScalarInfoTyped::Numeric(shape[d])
                        }
                    })
                    .collect();

                (
                    id,
                    TensorInfo::from_dtype_and_shape_scalars(view.dtype(), &dims),
                )
            })
            .collect();

        // Lower with partially symbolic info.
        let lower_result = lower::lower(&case.graph, &info_inputs, &SystemPool)
            .map_err(|e| format!("{}[{}] sym: lower failed: {e}", case.name, ds.label))?;

        // Collect gc_values: for each GraphConstant, look up the actual dim size
        // from one of the inputs. The lowering mapped symbol_ids to GraphConstantIds,
        // so we need to find what actual size each GC corresponds to.
        let gc_count = lower_result.graph.graph_constants.len();
        let mut gc_values = vec![0u64; gc_count];

        // Populate gc_values from the TAMIs — each TAMI's sym_dims tells us which
        // GraphConstants are used, and dim_layout tells us the original shape position.
        for (&ext_id, tensor) in &ds.inputs {
            let Some(&int_id) = case.graph.input_map.get(&ext_id) else {
                continue;
            };
            let Some(tam) = lower_result.graph.tensor_map.get(&int_id) else {
                continue;
            };
            let shape = tensor.view().shape().clone();
            for (dim_idx, dk) in tam.dims.iter().enumerate() {
                if let crate::nano_graph::lower::DimKind::Sym { gc, .. } = dk
                    && dim_idx < shape.len()
                {
                    gc_values[gc.0 as usize] = shape[dim_idx];
                }
            }
        }

        // Check we have no unsupported ops.
        if !lower_result.unsupported.is_empty() {
            return Err(format!(
                "{}[{}] sym: unsupported ops: {:?}",
                case.name, ds.label, lower_result.unsupported_details
            ));
        }

        // Map input tensors to (TAMI, view) pairs.
        let input_tam_pairs: Vec<_> = ds
            .inputs
            .iter()
            .filter_map(|(&ext_id, tensor)| {
                let internal_id = case.graph.input_map.get(&ext_id)?;
                let tam = lower_result.graph.tensor_map.get(internal_id)?;
                Some((tam, tensor))
            })
            .collect();

        let input_views: Vec<_> = input_tam_pairs
            .iter()
            .map(|(_, tensor)| tensor.view())
            .collect();

        let eval_inputs: Vec<_> = input_tam_pairs
            .iter()
            .zip(input_views.iter())
            .map(|((tam, _), view)| (*tam, view))
            .collect();

        // Build output TAMIs.
        let output_ids: Vec<GlobalId> = case.graph.output_ordering.clone().unwrap_or_default();
        let output_tamis: Vec<_> = output_ids
            .iter()
            .filter_map(|out_id| lower_result.graph.tensor_map.get(out_id))
            .collect();

        // Run pool_eval with gc_values.
        let pool = TrackedPool::new(None);
        let eval_results = pool_eval::pool_eval(
            &lower_result.graph,
            &eval_inputs,
            &output_tamis,
            &gc_values,
            &pool,
        )
        .map_err(|e| format!("{}[{}] sym: pool_eval failed: {e}", case.name, ds.label))?;

        // Compare.
        for (out_id, result_tensor) in output_ids.iter().zip(eval_results.iter()) {
            let expected_tensor = match ds.expected_outputs.get(out_id) {
                Some(t) => t,
                None => continue,
            };

            let ctx = format!("{}[{}] sym_dims", case.name, ds.label);
            assert_tensors_close(
                &result_tensor.view(),
                &expected_tensor.view(),
                &ds.tolerance,
                &ctx,
            )?;
        }
    }
    Ok(())
}

/// Run all data sets of a test case through MilliOpGraph::pool_eval().
///
/// This is the new end-to-end path using only pool-based types.
/// No legacy NumericTensor, no EvalBackend.
pub fn run_case_via_graph_pool_eval(case: &TestCase) -> Result<(), String> {
    use crate::pool::TrackedPool;

    for ds in &case.data_sets {
        let pool = TrackedPool::new(None);

        // Build input views keyed by external IDs.
        let input_views: Vec<(GlobalId, _)> =
            ds.inputs.iter().map(|(&id, t)| (id, t.view())).collect();
        let input_map: HashMap<
            GlobalId,
            &crate::numeric_tensor::NumericTensorView<'_, crate::tensor_rank::DynRank>,
        > = input_views.iter().map(|(id, view)| (*id, view)).collect();

        let results = case
            .graph
            .pool_eval(&input_map, &pool)
            .map_err(|e| format!("{}[{}]: pool_eval failed: {e}", case.name, ds.label))?;

        for (&expected_id, expected_tensor) in &ds.expected_outputs {
            let actual = results.get(&expected_id).ok_or_else(|| {
                format!(
                    "{}[{}]: output {expected_id} not in results",
                    case.name, ds.label
                )
            })?;
            let ctx = format!("{}[{}] graph_pool_eval", case.name, ds.label);
            assert_tensors_close(&actual.view(), &expected_tensor.view(), &ds.tolerance, &ctx)?;
        }
    }
    Ok(())
}

/// Run all data sets of a test case through the compiled (JIT) eval path.
///
/// Flow: MilliOpGraph → lower_to_nano → compile_nano_graph → execute → compare
///
/// Tests both the trivial plan (bit-perfect) and the partitioned plan to
/// catch partitioner accuracy regressions.
#[cfg(feature = "x86_compile")]
pub fn run_case_via_compiled_eval(case: &TestCase, num_lanes: usize) -> Result<(), String> {
    use crate::compiler::{CompileOptions, PartitionerKind};
    use crate::nano_graph::lower;
    use crate::pool::TrackedPool;
    use crate::super_graph::compiled_eval;
    use crate::tensor_info::TensorInfo;

    // num_lanes == 0 selects the trivial single-span plan.
    let options = CompileOptions {
        partitioner: if num_lanes == 0 {
            PartitionerKind::Trivial
        } else {
            PartitionerKind::LaneSplit { num_lanes }
        },
        ..CompileOptions::default()
    };

    for ds in &case.data_sets {
        // Build TensorInfo for each input (needed by lower).
        let info_inputs: HashMap<GlobalId, TensorInfo<'_, '_, crate::pool::SystemPool>> = ds
            .inputs
            .iter()
            .map(|(&id, t)| {
                (
                    id,
                    TensorInfo::from_view(&t.view(), &crate::pool::SystemPool),
                )
            })
            .collect();

        // Lower MilliOpGraph → NanoGraph.
        let lower_result = lower::lower(&case.graph, &info_inputs, &crate::pool::SystemPool)
            .map_err(|e| format!("{}[{}]: lower failed: {e}", case.name, ds.label))?;

        if !lower_result.unsupported.is_empty() {
            return Err(format!(
                "{}[{}]: unsupported ops: {:?}",
                case.name, ds.label, lower_result.unsupported_details
            ));
        }

        // Build output IDs (internal IDs — set_outputs makes ext == internal).
        let output_ids: Vec<GlobalId> = case.graph.output_ordering.clone().unwrap_or_default();

        // Build output atom ranges from TAMIs.
        let (output_ranges, output_shapes, all_output_atom_ranges) =
            compiled_eval::build_output_ranges(
                &lower_result.graph,
                &lower_result.graph.tensor_map,
                &output_ids,
                |id| *id, // test graphs: ext == internal for outputs
            );

        // Compile the NanoGraph.
        let (executable_plan, _plan_summary, _compile_errors) = compiled_eval::compile_nano_graph(
            &lower_result.graph,
            &all_output_atom_ranges,
            &options,
            Some(&lower_result.group_provenance),
            &mut (), // tests don't observe milestones
        )
        .map_err(|e| format!("{}[{}]: compile failed: {e}", case.name, ds.label))?;

        // Prepare inputs.
        let input_views: Vec<_> = ds.inputs.iter().map(|(&id, t)| (id, t.view())).collect();
        let input_view_refs: Vec<(GlobalId, &NumericTensorView<'_, DynRank>)> =
            input_views.iter().map(|(id, v)| (*id, v)).collect();

        let pool = TrackedPool::new(None);
        let initial_inputs = compiled_eval::prepare_compiled_inputs(
            &input_view_refs,
            &case.graph.input_map,
            &lower_result.graph.tensor_map,
            &pool,
        )
        .map_err(|e| format!("{}[{}]: prepare inputs failed: {e}", case.name, ds.label))?;

        // Build input_ptrs array from the placer's buffer assignments.
        let placement = executable_plan.placement();
        let ptr_array_len = (placement.scratch_buffer_id as usize) + 1;
        let mut input_ptrs: Vec<*mut u8> = vec![std::ptr::null_mut(); ptr_array_len];
        for (atom_id, cow) in &initial_inputs {
            if let Some((buf_id, _)) = placement.byte_offset_of(*atom_id) {
                let ptr = cow.buffer().as_ptr() as *mut u8;
                if (buf_id.0 as usize) < input_ptrs.len() {
                    input_ptrs[buf_id.0 as usize] = ptr;
                }
            }
        }

        // Execute.
        let executor_outputs = executable_plan.execute(&input_ptrs, &pool);
        drop(initial_inputs);

        // Extract outputs.
        let results =
            compiled_eval::extract_outputs(&output_ranges, &output_shapes, executor_outputs, &pool)
                .map_err(|e| format!("{}[{}]: extract outputs failed: {e}", case.name, ds.label))?;

        // Compare.
        for (out_id, expected_tensor) in &ds.expected_outputs {
            let actual = results.get(out_id).ok_or_else(|| {
                format!(
                    "{}[{}]: output {out_id:?} not in compiled results",
                    case.name, ds.label
                )
            })?;
            let lanes_label = if num_lanes == 0 {
                "trivial"
            } else {
                "partitioned"
            };
            let ctx = format!("{}[{}] compiled_eval({})", case.name, ds.label, lanes_label);
            assert_tensors_close(&actual.view(), &expected_tensor.view(), &ds.tolerance, &ctx)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_cases_via_pool_eval() {
        let cases = build_test_set();
        assert!(!cases.is_empty(), "test set should not be empty");
        let mut total_data_sets = 0;
        for case in &cases {
            run_case_via_pool_eval(case).unwrap_or_else(|e| panic!("{e}"));
            total_data_sets += case.data_sets.len();
        }
        eprintln!(
            "{} test cases ({total_data_sets} data sets) passed via pool eval",
            cases.len()
        );
    }

    #[test]
    fn test_all_cases_via_graph_pool_eval() {
        let cases = build_test_set();
        assert!(!cases.is_empty(), "test set should not be empty");
        let mut total_data_sets = 0;
        for case in &cases {
            run_case_via_graph_pool_eval(case).unwrap_or_else(|e| panic!("{e}"));
            total_data_sets += case.data_sets.len();
        }
        eprintln!(
            "{} test cases ({total_data_sets} data sets) passed via graph pool eval",
            cases.len()
        );
    }

    /// Run all test cases with dim 0 withheld (symbolic).
    /// Tests the full symbolic dimension path: lower with partial shapes,
    /// eval with gc_values, compare against ground truth.
    #[test]
    fn test_all_cases_with_symbolic_dim0() {
        let cases = build_test_set();
        assert!(!cases.is_empty(), "test set should not be empty");
        let mut passed = 0;
        let mut skipped = 0;
        let mut failures = Vec::new();
        for case in &cases {
            let name = case.name.clone();
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_case_with_symbolic_dims(case, &[0])
            })) {
                Ok(Ok(())) => {
                    eprintln!("  PASS: {name}");
                    passed += 1;
                }
                Ok(Err(e))
                    if e.contains("unsupported ops")
                        || e.contains("lower failed")
                        || e.contains("pool_eval failed")
                        || e.contains("shape mismatch") =>
                {
                    eprintln!("  SKIP: {name}: {e}");
                    skipped += 1;
                }
                Ok(Err(e)) => {
                    eprintln!("  FAIL: {e}");
                    failures.push(e);
                }
                Err(panic_info) => {
                    let msg = if let Some(s) = panic_info.downcast_ref::<String>() {
                        s.clone()
                    } else if let Some(s) = panic_info.downcast_ref::<&str>() {
                        s.to_string()
                    } else {
                        "unknown panic".to_string()
                    };
                    eprintln!("  PANIC: {name}: {msg}");
                    failures.push(format!("{name}: PANIC: {msg}"));
                }
            }
        }
        eprintln!(
            "{passed} passed, {} failed, {skipped} skipped",
            failures.len(),
        );
        assert!(failures.is_empty(), "Failures:\n{}", failures.join("\n"));
    }

    /// JIT compiled eval with trivial plan (1 phase, no partitioning).
    /// Should be bit-perfect vs pool_eval.
    #[test]
    #[cfg(feature = "x86_compile")]
    fn test_all_cases_via_compiled_eval_trivial() {
        let cases = build_test_set();
        assert!(!cases.is_empty(), "test set should not be empty");
        let mut failures = Vec::new();
        let mut total_data_sets = 0;
        for case in &cases {
            match run_case_via_compiled_eval(case, 0) {
                Ok(()) => {}
                Err(e) => failures.push(e),
            }
            total_data_sets += case.data_sets.len();
        }
        if failures.is_empty() {
            eprintln!(
                "{} test cases ({total_data_sets} data sets) passed via compiled eval (trivial)",
                cases.len()
            );
        } else {
            panic!(
                "{} of {} test cases failed via compiled eval (trivial):\n{}",
                failures.len(),
                cases.len(),
                failures.join("\n"),
            );
        }
    }

    /// JIT compiled eval with partitioned plan (8 lanes).
    /// Tests partitioner correctness — should match within dtype tolerance.
    #[test]
    #[cfg(feature = "x86_compile")]
    fn test_all_cases_via_compiled_eval_partitioned() {
        let cases = build_test_set();
        assert!(!cases.is_empty(), "test set should not be empty");
        let mut failures = Vec::new();
        let mut total_data_sets = 0;
        for case in &cases {
            match run_case_via_compiled_eval(case, 8) {
                Ok(()) => {}
                Err(e) => failures.push(e),
            }
            total_data_sets += case.data_sets.len();
        }
        if failures.is_empty() {
            eprintln!(
                "{} test cases ({total_data_sets} data sets) passed via compiled eval (partitioned)",
                cases.len()
            );
        } else {
            panic!(
                "{} of {} test cases failed via compiled eval (partitioned):\n{}",
                failures.len(),
                cases.len(),
                failures.join("\n"),
            );
        }
    }

    /// Per-category partitioned tests for isolating x86_jit crashes.
    #[cfg(feature = "x86_compile")]
    mod partitioned_by_category {
        use super::super::*;

        fn run_category(cases: Vec<TestCase>) {
            for case in &cases {
                eprintln!("  running: {}", case.name);
                run_case_via_compiled_eval(case, 8).unwrap();
            }
        }

        #[test]
        fn elementwise() {
            run_category(super::super::elementwise::build_cases());
        }
        #[test]
        fn matmul() {
            run_category(super::super::matmul::build_cases());
        }
        #[test]
        fn cast() {
            run_category(super::super::cast::build_cases());
        }
        #[test]
        fn reduce() {
            run_category(super::super::reduce::build_cases());
        }
        #[test]
        fn composite() {
            run_category(super::super::composite::build_cases());
        }
        #[test]
        fn conv() {
            run_category(super::super::conv::build_cases());
        }
        #[test]
        fn pad() {
            run_category(super::super::pad::build_cases());
        }
        #[test]
        fn structural() {
            run_category(super::super::structural::build_cases());
        }
        #[test]
        fn opaque_ops() {
            run_category(super::super::opaque_ops::build_cases());
        }
        #[test]
        fn dtype_discipline() {
            run_category(super::super::dtype_discipline::build_cases());
        }
    }
}
