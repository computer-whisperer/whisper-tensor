//! Elementwise operation test cases (add, sub, mul, neg, exp, etc.)
//!
//! Each test case builds one MilliOpGraph and provides multiple data sets
//! testing different value ranges and dtypes.

use std::collections::HashMap;

use half::{bf16, f16};
use rand::rngs::SmallRng;
use rand::SeedableRng;

use crate::dtype::DType;
use crate::graph::GlobalId;
use crate::migration::numeric_tensor::NumericTensor;
use crate::milli_graph::MilliOpGraph;
use crate::milli_graph::ops::binary::SimpleBinary;
use crate::milli_graph::ops::unary::SimpleUnaryOp;
use crate::DynRank;

use super::{TestCase, TestDataSet, Tolerance};

pub fn build_cases() -> Vec<TestCase> {
    let mut cases = Vec::new();
    cases.push(add_case());
    cases.push(mul_case());
    cases.push(neg_case());
    cases.push(exp_case());
    cases
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn rng() -> SmallRng {
    SmallRng::seed_from_u64(42)
}

pub(crate) struct BinaryGraphIds {
    pub ext_a: GlobalId,
    pub ext_b: GlobalId,
    pub out: GlobalId,
}

fn build_binary_graph(
    build_op: fn(&mut MilliOpGraph, GlobalId, GlobalId, &mut SmallRng) -> GlobalId,
) -> (MilliOpGraph, BinaryGraphIds) {
    let mut rng = rng();
    let ext_a = GlobalId::new(&mut rng);
    let ext_b = GlobalId::new(&mut rng);
    let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
    let int_a = input_map[&ext_a];
    let int_b = input_map[&ext_b];
    let out = build_op(&mut graph, int_a, int_b, &mut rng);
    graph.set_outputs(vec![out]);
    (graph, BinaryGraphIds { ext_a, ext_b, out })
}

pub(crate) struct UnaryGraphIds {
    pub ext_in: GlobalId,
    pub out: GlobalId,
}

fn build_unary_graph(
    build_op: fn(&mut MilliOpGraph, GlobalId, &mut SmallRng) -> GlobalId,
) -> (MilliOpGraph, UnaryGraphIds) {
    let mut rng = rng();
    let ext_in = GlobalId::new(&mut rng);
    let (mut graph, input_map) = MilliOpGraph::new([ext_in], &mut rng);
    let int_in = input_map[&ext_in];
    let out = build_op(&mut graph, int_in, &mut rng);
    graph.set_outputs(vec![out]);
    (graph, UnaryGraphIds { ext_in, out })
}

fn binary_data_set(
    label: &str,
    ids: &BinaryGraphIds,
    a: NumericTensor<DynRank>,
    b: NumericTensor<DynRank>,
    expected: NumericTensor<DynRank>,
    tolerance: Tolerance,
) -> TestDataSet {
    TestDataSet {
        label: label.to_string(),
        inputs: HashMap::from([(ids.ext_a, a), (ids.ext_b, b)]),
        expected_outputs: HashMap::from([(ids.out, expected)]),
        tolerance,
    }
}

fn unary_data_set(
    label: &str,
    ids: &UnaryGraphIds,
    input: NumericTensor<DynRank>,
    expected: NumericTensor<DynRank>,
    tolerance: Tolerance,
) -> TestDataSet {
    TestDataSet {
        label: label.to_string(),
        inputs: HashMap::from([(ids.ext_in, input)]),
        expected_outputs: HashMap::from([(ids.out, expected)]),
        tolerance,
    }
}

// ---------------------------------------------------------------------------
// Add
// ---------------------------------------------------------------------------

fn add_case() -> TestCase {
    let (graph, ids) = build_binary_graph(SimpleBinary::add);

    TestCase {
        name: "add".to_string(),
        graph,
        data_sets: vec![
            binary_data_set(
                "f32_normal",
                &ids,
                NumericTensor::from_vec(vec![0.15163845f32, 0.31361532, 5.393808]).to_dyn_rank(),
                NumericTensor::from_vec(vec![1.3424649f32, 0.004955234, 6.920299]).to_dyn_rank(),
                NumericTensor::from_vec(vec![1.4941034f32, 0.31857055, 12.314107]).to_dyn_rank(),
                Tolerance::for_dtype(DType::F32),
            ),
            binary_data_set(
                "f32_zeros_negatives",
                &ids,
                NumericTensor::from_vec(vec![0.0f32, -1.0, 1e38]).to_dyn_rank(),
                NumericTensor::from_vec(vec![0.0f32, 1.0, -1e38]).to_dyn_rank(),
                NumericTensor::from_vec(vec![0.0f32, 0.0, 0.0]).to_dyn_rank(),
                Tolerance::for_dtype(DType::F32),
            ),
            binary_data_set(
                "bf16",
                &ids,
                NumericTensor::from_vec(vec![
                    bf16::from_f32(0.75390625),
                    bf16::from_f32(0.93359375),
                    bf16::from_f32(0.13671875),
                ]).to_dyn_rank(),
                NumericTensor::from_vec(vec![
                    bf16::from_f32(4.40625),
                    bf16::from_f32(5.65625),
                    bf16::from_f32(38.25),
                ]).to_dyn_rank(),
                NumericTensor::from_vec(vec![
                    bf16::from_f32(5.15625),
                    bf16::from_f32(6.59375),
                    bf16::from_f32(38.25),
                ]).to_dyn_rank(),
                Tolerance::for_dtype(DType::BF16),
            ),
            binary_data_set(
                "f16",
                &ids,
                NumericTensor::from_vec(vec![
                    f16::from_f32(0.75390625),
                    f16::from_f32(0.93359375),
                    f16::from_f32(0.13671875),
                ]).to_dyn_rank(),
                NumericTensor::from_vec(vec![
                    f16::from_f32(4.40625),
                    f16::from_f32(5.65625),
                    f16::from_f32(38.25),
                ]).to_dyn_rank(),
                NumericTensor::from_vec(vec![
                    f16::from_f32(5.15625),
                    f16::from_f32(6.59375),
                    f16::from_f32(38.375),
                ]).to_dyn_rank(),
                Tolerance::for_dtype(DType::F16),
            ),
        ],
    }
}

// ---------------------------------------------------------------------------
// Mul
// ---------------------------------------------------------------------------

fn mul_case() -> TestCase {
    let (graph, ids) = build_binary_graph(SimpleBinary::mul);

    TestCase {
        name: "mul".to_string(),
        graph,
        data_sets: vec![
            binary_data_set(
                "f32_normal",
                &ids,
                NumericTensor::from_vec(vec![2.0f32, 3.0, 0.5]).to_dyn_rank(),
                NumericTensor::from_vec(vec![4.0f32, 5.0, 6.0]).to_dyn_rank(),
                NumericTensor::from_vec(vec![8.0f32, 15.0, 3.0]).to_dyn_rank(),
                Tolerance::for_dtype(DType::F32),
            ),
            binary_data_set(
                "f32_special",
                &ids,
                NumericTensor::from_vec(vec![0.0f32, f32::INFINITY, -1.0]).to_dyn_rank(),
                NumericTensor::from_vec(vec![100.0f32, 2.0, -1.0]).to_dyn_rank(),
                NumericTensor::from_vec(vec![0.0f32, f32::INFINITY, 1.0]).to_dyn_rank(),
                Tolerance::for_dtype(DType::F32),
            ),
            binary_data_set(
                "bf16",
                &ids,
                NumericTensor::from_vec(vec![
                    bf16::from_f32(2.0),
                    bf16::from_f32(3.0),
                    bf16::from_f32(0.5),
                ]).to_dyn_rank(),
                NumericTensor::from_vec(vec![
                    bf16::from_f32(4.0),
                    bf16::from_f32(5.0),
                    bf16::from_f32(6.0),
                ]).to_dyn_rank(),
                NumericTensor::from_vec(vec![
                    bf16::from_f32(8.0),
                    bf16::from_f32(15.0),
                    bf16::from_f32(3.0),
                ]).to_dyn_rank(),
                Tolerance::for_dtype(DType::BF16),
            ),
        ],
    }
}

// ---------------------------------------------------------------------------
// Neg
// ---------------------------------------------------------------------------

fn neg_case() -> TestCase {
    let (graph, ids) = build_unary_graph(SimpleUnaryOp::neg);

    TestCase {
        name: "neg".to_string(),
        graph,
        data_sets: vec![
            unary_data_set(
                "f32",
                &ids,
                NumericTensor::from_vec(vec![1.0f32, -2.5, 0.0, 3.14]).to_dyn_rank(),
                NumericTensor::from_vec(vec![-1.0f32, 2.5, -0.0, -3.14]).to_dyn_rank(),
                Tolerance::for_dtype(DType::F32),
            ),
            unary_data_set(
                "bf16",
                &ids,
                NumericTensor::from_vec(vec![
                    bf16::from_f32(1.0),
                    bf16::from_f32(-2.5),
                ]).to_dyn_rank(),
                NumericTensor::from_vec(vec![
                    bf16::from_f32(-1.0),
                    bf16::from_f32(2.5),
                ]).to_dyn_rank(),
                Tolerance::for_dtype(DType::BF16),
            ),
        ],
    }
}

// ---------------------------------------------------------------------------
// Exp
// ---------------------------------------------------------------------------

fn exp_case() -> TestCase {
    let (graph, ids) = build_unary_graph(SimpleUnaryOp::exp);

    TestCase {
        name: "exp".to_string(),
        graph,
        data_sets: vec![
            unary_data_set(
                "f32",
                &ids,
                NumericTensor::from_vec(vec![0.0f32, 1.0, -1.0, 2.0]).to_dyn_rank(),
                NumericTensor::from_vec(vec![
                    1.0f32,
                    std::f32::consts::E,
                    1.0 / std::f32::consts::E,
                    std::f32::consts::E * std::f32::consts::E,
                ]).to_dyn_rank(),
                Tolerance::for_dtype(DType::F32),
            ),
        ],
    }
}

// ---------------------------------------------------------------------------
// tch validation: verify expected outputs match PyTorch
// ---------------------------------------------------------------------------

#[cfg(feature = "tch")]
pub mod tch_validation {
    use super::*;

    /// Convert a NumericTensor to a tch::Tensor.
    fn to_tch_tensor(t: &NumericTensor<DynRank>) -> tch::Tensor {
        let ndarray = t.to_ndarray().unwrap();
        let shape: Vec<i64> = t.shape().iter().map(|&d| d as i64).collect();
        match t.dtype() {
            DType::F32 => {
                let data: Vec<f32> = ndarray.cast(DType::F32).unwrap()
                    .flatten().try_to_vec().unwrap();
                tch::Tensor::from_slice(&data).reshape(&shape)
            }
            DType::F64 => {
                let data: Vec<f64> = ndarray.cast(DType::F64).unwrap()
                    .flatten().try_to_vec().unwrap();
                tch::Tensor::from_slice(&data).reshape(&shape)
            }
            DType::BF16 => {
                // tch doesn't have a direct bf16 path; go through f32
                let data: Vec<f32> = ndarray.cast(DType::F32).unwrap()
                    .flatten().try_to_vec().unwrap();
                tch::Tensor::from_slice(&data).reshape(&shape).to_kind(tch::Kind::BFloat16)
            }
            DType::F16 => {
                let data: Vec<f32> = ndarray.cast(DType::F32).unwrap()
                    .flatten().try_to_vec().unwrap();
                tch::Tensor::from_slice(&data).reshape(&shape).to_kind(tch::Kind::Half)
            }
            other => panic!("tch_validation: unsupported dtype {other}"),
        }
    }

    /// Convert a tch::Tensor back to a NumericTensor.
    fn from_tch_tensor(t: &tch::Tensor, target_dtype: DType) -> NumericTensor<DynRank> {
        // Always go through f64 for comparison
        let t_f64 = t.to_kind(tch::Kind::Float).flatten(0, -1);
        let numel = t_f64.numel();
        let mut data = vec![0.0f32; numel];
        t_f64.copy_data(&mut data, numel);
        let shape: Vec<usize> = t.size().iter().map(|&d| d as usize).collect();

        match target_dtype {
            DType::F32 => NumericTensor::<DynRank>::from_vec_shape(data, shape).unwrap(),
            DType::BF16 => {
                let bf_data: Vec<bf16> = data.iter().map(|&v| bf16::from_f32(v)).collect();
                NumericTensor::<DynRank>::from_vec_shape(bf_data, shape).unwrap()
            }
            DType::F16 => {
                let hf_data: Vec<f16> = data.iter().map(|&v| f16::from_f32(v)).collect();
                NumericTensor::<DynRank>::from_vec_shape(hf_data, shape).unwrap()
            }
            DType::F64 => {
                let f64_data: Vec<f64> = data.iter().map(|&v| v as f64).collect();
                NumericTensor::<DynRank>::from_vec_shape(f64_data, shape).unwrap()
            }
            other => panic!("from_tch_tensor: unsupported dtype {other}"),
        }
    }

    /// Validate an add test case against PyTorch.
    pub fn validate_add(ds: &TestDataSet, ids: &super::BinaryGraphIds) -> Result<(), String> {
        let a = to_tch_tensor(ds.inputs.get(&ids.ext_a).unwrap());
        let b = to_tch_tensor(ds.inputs.get(&ids.ext_b).unwrap());
        let pytorch_result = &a + &b;
        let expected = ds.expected_outputs.get(&ids.out).unwrap();
        let pytorch_tensor = from_tch_tensor(&pytorch_result, expected.dtype());
        super::super::assert_tensors_close(
            &pytorch_tensor, expected, &ds.tolerance,
            &format!("tch_validate_add[{}]", ds.label),
        )
    }

    /// Validate a neg test case against PyTorch.
    pub fn validate_neg(ds: &TestDataSet, ids: &super::UnaryGraphIds) -> Result<(), String> {
        let input = to_tch_tensor(ds.inputs.get(&ids.ext_in).unwrap());
        let pytorch_result = input.neg();
        let expected = ds.expected_outputs.get(&ids.out).unwrap();
        let pytorch_tensor = from_tch_tensor(&pytorch_result, expected.dtype());
        super::super::assert_tensors_close(
            &pytorch_tensor, expected, &ds.tolerance,
            &format!("tch_validate_neg[{}]", ds.label),
        )
    }

    /// Validate an exp test case against PyTorch.
    pub fn validate_exp(ds: &TestDataSet, ids: &super::UnaryGraphIds) -> Result<(), String> {
        let input = to_tch_tensor(ds.inputs.get(&ids.ext_in).unwrap());
        let pytorch_result = input.exp();
        let expected = ds.expected_outputs.get(&ids.out).unwrap();
        let pytorch_tensor = from_tch_tensor(&pytorch_result, expected.dtype());
        super::super::assert_tensors_close(
            &pytorch_tensor, expected, &ds.tolerance,
            &format!("tch_validate_exp[{}]", ds.label),
        )
    }
}

#[cfg(all(test, feature = "tch"))]
mod tch_tests {
    use super::*;

    #[test]
    fn validate_add_against_pytorch() {
        let (_, ids) = build_binary_graph(SimpleBinary::add);
        let case = add_case();
        for ds in &case.data_sets {
            tch_validation::validate_add(ds, &ids).unwrap_or_else(|e| panic!("{e}"));
        }
    }

    #[test]
    fn validate_neg_against_pytorch() {
        let (_, ids) = build_unary_graph(SimpleUnaryOp::neg);
        let case = neg_case();
        for ds in &case.data_sets {
            tch_validation::validate_neg(ds, &ids).unwrap_or_else(|e| panic!("{e}"));
        }
    }

    #[test]
    fn validate_exp_against_pytorch() {
        let (_, ids) = build_unary_graph(SimpleUnaryOp::exp);
        let case = exp_case();
        for ds in &case.data_sets {
            tch_validation::validate_exp(ds, &ids).unwrap_or_else(|e| panic!("{e}"));
        }
    }
}
