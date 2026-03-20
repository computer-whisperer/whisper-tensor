use crate::backends::eval_backend::EvalBackend;
use crate::dtype::DType;
use crate::graph::{GlobalId, Node, Property, PropertyValue};
use crate::milli_graph::{MilliLoweringContext, MilliOpGraph};
use crate::numeric_tensor::NumericTensor;
use crate::onnx::AttributeProto;
use crate::symbolic_graph::ops::{EvalError, Operation};
use crate::symbolic_graph::{ONNXDecodingError, query_attribute_string};
use crate::tensor_rank::DynRank;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// ONNX Einsum operator.
///
/// Evaluates Einstein summation convention on the inputs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EinsumOperation {
    global_id: GlobalId,
    inputs: Vec<GlobalId>,
    output: GlobalId,
    equation: String,
}

impl EinsumOperation {
    pub fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.is_empty() {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Einsum"));
        }
        if outputs.is_empty() {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Einsum"));
        }

        let equation = query_attribute_string(attributes, "equation")
            .ok_or(ONNXDecodingError::MissingField("equation"))?;

        let input_ids: Vec<GlobalId> = inputs.iter().filter_map(|x| *x).collect();

        Ok(Self {
            global_id: GlobalId::new(rng),
            inputs: input_ids,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Einsum"))?,
            equation,
        })
    }
}

impl Node for EinsumOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Einsum".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId> + '_> {
        Box::new(self.inputs.iter().copied())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId> + '_> {
        Box::new([self.output].into_iter())
    }
}

/// Parse an einsum equation like "bij, bjk -> bik" or "ij->i".
/// Returns (input_subscripts, output_subscripts).
/// Each subscript is a list of chars representing axes.
fn parse_equation(eq: &str) -> Result<(Vec<Vec<char>>, Vec<char>), EvalError> {
    let eq = eq.replace(' ', "");
    let (lhs, rhs) = if let Some((l, r)) = eq.split_once("->") {
        (l, Some(r))
    } else {
        (eq.as_str(), None)
    };

    let input_subs: Vec<Vec<char>> = lhs.split(',').map(parse_subscript).collect();

    let output_sub = if let Some(r) = rhs {
        parse_subscript(r)
    } else {
        // Implicit output: sorted unique labels that appear exactly once
        let mut counts: HashMap<char, usize> = HashMap::new();
        for sub in &input_subs {
            for &c in sub {
                *counts.entry(c).or_default() += 1;
            }
        }
        let mut out: Vec<char> = counts
            .into_iter()
            .filter(|&(_, count)| count == 1)
            .map(|(c, _)| c)
            .collect();
        out.sort();
        out
    };

    Ok((input_subs, output_sub))
}

/// Parse a single subscript, handling ellipsis (...) by expanding to uppercase placeholders.
fn parse_subscript(s: &str) -> Vec<char> {
    let mut result = Vec::new();
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if i + 2 < chars.len() && chars[i] == '.' && chars[i + 1] == '.' && chars[i + 2] == '.' {
            // Ellipsis placeholder — we'll expand later
            result.push('\u{2026}'); // Unicode ellipsis as placeholder
            i += 3;
        } else {
            result.push(chars[i]);
            i += 1;
        }
    }
    result
}

impl Operation for EinsumOperation {
    fn parameters(&self) -> Vec<Property> {
        vec![Property::new(
            "equation",
            PropertyValue::String(self.equation.clone()),
        )]
    }

    fn eval(
        &self,
        backend: &mut EvalBackend,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, EvalError> {
        let tensors: Vec<&NumericTensor<DynRank>> =
            self.inputs.iter().map(|id| &inputs[id]).collect();

        let orig_dtype = tensors[0].dtype();

        let (mut input_subs, mut output_sub) = parse_equation(&self.equation)?;

        // Expand ellipsis: determine how many dimensions the ellipsis covers
        let ellipsis_char = '\u{2026}';
        let has_ellipsis = input_subs.iter().any(|s| s.contains(&ellipsis_char))
            || output_sub.contains(&ellipsis_char);

        if has_ellipsis {
            // Find the number of ellipsis dimensions from the first input that has one
            let mut ellipsis_ndim = 0;
            for (i, sub) in input_subs.iter().enumerate() {
                if sub.contains(&ellipsis_char) {
                    let explicit_dims = sub.len() - 1; // subtract the ellipsis placeholder
                    ellipsis_ndim = tensors[i].rank() - explicit_dims;
                    break;
                }
            }

            // Generate unique labels for ellipsis dims (use Unicode private use area
            // to avoid collision with any valid einsum label)
            let ellipsis_labels: Vec<char> = (0..ellipsis_ndim)
                .map(|i| char::from_u32(0xE000 + i as u32).unwrap())
                .collect();

            // Replace ellipsis in all subscripts
            for sub in input_subs.iter_mut() {
                if let Some(pos) = sub.iter().position(|&c| c == ellipsis_char) {
                    sub.splice(pos..pos + 1, ellipsis_labels.iter().copied());
                }
            }
            if let Some(pos) = output_sub.iter().position(|&c| c == ellipsis_char) {
                output_sub.splice(pos..pos + 1, ellipsis_labels.iter().copied());
            }
        }

        // Collect all unique labels and assign dimension sizes
        let mut label_to_dim: HashMap<char, usize> = HashMap::new();
        for (i, sub) in input_subs.iter().enumerate() {
            let shape = tensors[i].shape();
            assert_eq!(
                sub.len(),
                shape.len(),
                "Einsum subscript rank mismatch for input {}: {} vs {}",
                i,
                sub.len(),
                shape.len()
            );
            for (j, &label) in sub.iter().enumerate() {
                label_to_dim.entry(label).or_insert(shape[j] as usize);
            }
        }

        // Build ordered list of all labels: output labels first, then contracted labels
        let mut all_labels: Vec<char> = output_sub.clone();
        for sub in &input_subs {
            for &c in sub {
                if !all_labels.contains(&c) {
                    all_labels.push(c);
                }
            }
        }

        let n_labels = all_labels.len();
        let label_sizes: Vec<usize> = all_labels
            .iter()
            .map(|c| *label_to_dim.get(c).unwrap())
            .collect();

        // Compute total iterations and strides for the multi-index
        let total: usize = label_sizes.iter().product();
        let mut strides = vec![1usize; n_labels];
        for i in (0..n_labels.saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * label_sizes[i + 1];
        }

        // Pre-compute input flat strides: for each input, for each label in all_labels,
        // what stride does it contribute (0 if the label isn't in this input's subscript)
        let input_data: Vec<Vec<f64>> = tensors
            .iter()
            .map(|t| {
                let t_f64 = t.cast(DType::F64, backend).unwrap();
                let flat: Vec<f64> = t_f64.to_ndarray().unwrap().flatten().try_into().unwrap();
                flat
            })
            .collect();

        let input_strides: Vec<Vec<usize>> = input_subs
            .iter()
            .enumerate()
            .map(|(inp_idx, sub)| {
                let shape = tensors[inp_idx].shape();
                // Input's own strides
                let mut inp_strides = vec![1usize; sub.len()];
                for i in (0..sub.len().saturating_sub(1)).rev() {
                    inp_strides[i] = inp_strides[i + 1] * shape[i + 1] as usize;
                }
                // Map to all_labels — sum strides for repeated labels (diagonal)
                all_labels
                    .iter()
                    .map(|label| {
                        let mut total_stride = 0usize;
                        for (pos, c) in sub.iter().enumerate() {
                            if c == label {
                                total_stride += inp_strides[pos];
                            }
                        }
                        total_stride
                    })
                    .collect()
            })
            .collect();

        // Compute output
        let out_rank = output_sub.len();
        let out_shape: Vec<usize> = output_sub
            .iter()
            .map(|c| *label_to_dim.get(c).unwrap())
            .collect();
        let out_total: usize = out_shape.iter().product::<usize>().max(1);
        let mut out_flat = vec![0.0f64; out_total];

        // Output strides in the all_labels space (output labels are the first ones)
        let mut out_strides = vec![1usize; out_rank];
        for i in (0..out_rank.saturating_sub(1)).rev() {
            out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
        }

        // Main loop over all label combinations
        for flat_idx in 0..total {
            // Decompose into multi-index
            let mut remaining = flat_idx;
            let mut multi_idx = vec![0usize; n_labels];
            for d in 0..n_labels {
                multi_idx[d] = remaining / strides[d];
                remaining %= strides[d];
            }

            // Compute product of all input values at this multi-index
            let mut product = 1.0f64;
            for (inp_idx, data) in input_data.iter().enumerate() {
                let mut inp_flat = 0;
                for d in 0..n_labels {
                    inp_flat += multi_idx[d] * input_strides[inp_idx][d];
                }
                product *= data[inp_flat];
            }

            // Compute output flat index (first out_rank labels)
            let mut out_idx = 0;
            for d in 0..out_rank {
                out_idx += multi_idx[d] * out_strides[d];
            }

            out_flat[out_idx] += product;
        }

        let output = NumericTensor::<DynRank>::from_vec_shape(out_flat, out_shape)
            .map_err(|e| EvalError::InvalidInput(format!("Einsum: {e}")))?;
        let output = output.cast(orig_dtype, backend)?;

        Ok(Box::new([(self.output, output)].into_iter()))
    }

    fn is_differentiable(&self) -> bool {
        false
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, _rng: &mut impl Rng) -> MilliOpGraph {
        panic!("Einsum uses custom eval")
    }
}
