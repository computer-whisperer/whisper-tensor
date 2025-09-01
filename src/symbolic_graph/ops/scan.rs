use crate::backends::eval_backend::EvalBackend;
use crate::milli_graph::MilliOpGraph;
use crate::numeric_tensor::NumericTensor;
use crate::symbolic_graph::ops::{EvalError, Operation};
use crate::symbolic_graph::{
    ONNXDecodingError, SymbolicGraphInner, SymbolicGraphMutator, SymbolicGraphTensorId,
    query_attribute_graph, query_attribute_int, query_attribute_ints,
};
use crate::{DynRank, onnx};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScanOperation {
    scan_inputs: Vec<Option<SymbolicGraphTensorId>>,
    state_inputs: Vec<Option<SymbolicGraphTensorId>>,
    scan_outputs: Vec<Option<SymbolicGraphTensorId>>,
    state_outputs: Vec<Option<SymbolicGraphTensorId>>,
    scan_input_axes: Option<Vec<i64>>,
    scan_input_directions: Option<Vec<i64>>,
    scan_output_axes: Option<Vec<i64>>,
    scan_output_directions: Option<Vec<i64>>,
    body: SymbolicGraphInner,
}

impl ScanOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<SymbolicGraphTensorId>],
        outputs: &[Option<SymbolicGraphTensorId>],
        attributes: &[onnx::AttributeProto],
        symbolic_graph_mutator: &mut SymbolicGraphMutator,
        core_opset_version: usize,
    ) -> Result<Self, ONNXDecodingError> {
        let body = query_attribute_graph(attributes, "body")
            .ok_or(ONNXDecodingError::MissingField("body"))?;
        let body = {
            let mut inner_graph = SymbolicGraphInner::new();
            inner_graph.populate(symbolic_graph_mutator, body, core_opset_version)?;
            inner_graph
        };

        let scan_inputs_start = if core_opset_version < 9 {
            //1
            panic!("We don't support this version of scan!")
        } else {
            0
        };

        let num_scan_inputs = query_attribute_int(attributes, "num_scan_inputs")
            .ok_or(ONNXDecodingError::MissingField("num_scan_inputs"))?;
        assert!(num_scan_inputs <= inputs.len() as i64);
        assert!(num_scan_inputs >= 1);
        let num_state_tensors = (inputs.len() - scan_inputs_start) - num_scan_inputs as usize;

        let scan_input_axes = query_attribute_ints(attributes, "scan_input_axes");
        let scan_input_directions = query_attribute_ints(attributes, "scan_input_directions");
        let scan_output_axes = query_attribute_ints(attributes, "scan_output_axes");
        let scan_output_directions = query_attribute_ints(attributes, "scan_output_directions");

        let state_inputs = inputs[scan_inputs_start..num_state_tensors].to_vec();
        let scan_inputs = inputs[scan_inputs_start + num_state_tensors..].to_vec();

        let state_outputs = outputs[..num_state_tensors].to_vec();
        let scan_outputs = outputs[num_state_tensors..].to_vec();

        Ok(Self {
            state_inputs,
            scan_inputs,
            state_outputs,
            scan_outputs,
            body,
            scan_input_axes,
            scan_input_directions,
            scan_output_axes,
            scan_output_directions,
        })
    }
}

impl Operation for ScanOperation {
    fn get_op_type_name(&self) -> String {
        "Scan".to_string()
    }

    fn get_inputs(&self) -> Vec<SymbolicGraphTensorId> {
        let mut inputs_set = HashSet::new();
        inputs_set.extend(self.state_inputs.iter().filter_map(|x| *x));
        inputs_set.extend(self.scan_inputs.iter().filter_map(|x| *x));
        inputs_set.extend(self.body.get_foreign_tensor_ids());
        let mut inputs_vec: Vec<_> = inputs_set.into_iter().collect();
        inputs_vec.sort(); // Deterministic ordering
        inputs_vec
    }

    fn get_outputs(&self) -> Vec<SymbolicGraphTensorId> {
        let mut outputs = Vec::new();
        outputs.extend(self.state_outputs.iter().filter_map(|x| *x));
        outputs.extend(self.scan_outputs.iter().filter_map(|x| *x));
        outputs
    }

    fn eval(
        &self,
        backend: &mut EvalBackend,
        inputs: &HashMap<SymbolicGraphTensorId, NumericTensor<DynRank>>,
    ) -> Result<HashMap<SymbolicGraphTensorId, NumericTensor<DynRank>>, EvalError> {
        let state_inputs: Vec<_> = self
            .state_inputs
            .iter()
            .map(|x| x.map(|x| inputs[&x].clone()))
            .collect();
        let scan_inputs: Vec<_> = self
            .scan_inputs
            .iter()
            .map(|x| inputs[&x.unwrap()].clone())
            .collect();

        let scan_input_axes = if let Some(scan_input_axes) = &self.scan_input_axes {
            let mut output = Vec::new();
            for (i, axis) in scan_input_axes.iter().enumerate() {
                if *axis >= 0 {
                    output.push(*axis as usize);
                } else {
                    output.push((scan_inputs[i].rank() as i64 + *axis) as usize);
                }
                assert!(output[i] < scan_inputs[i].rank());
            }
            output
        } else {
            vec![0; scan_inputs.len()]
        };

        let iter_count = scan_inputs[0].shape()[scan_input_axes[0]];

        assert!(self.scan_input_directions.is_none());
        assert!(self.scan_output_directions.is_none());

        let mut accumulated_scan_outputs = Vec::new();
        for _ in 0..self.scan_outputs.len() {
            accumulated_scan_outputs.push(Vec::new());
        }

        let mut state_tensors = state_inputs;

        for i in 0..iter_count {
            let iter_scan_inputs = {
                let mut iter_scan_inputs = Vec::new();
                for j in 0..scan_inputs.len() {
                    let mut slice_indexes = Vec::new();
                    for k in 0..scan_inputs[j].rank() {
                        if k == scan_input_axes[j] {
                            slice_indexes.push(i..i + 1);
                        } else {
                            slice_indexes.push(0..scan_inputs[j].shape()[k]);
                        }
                    }
                    let sliced = scan_inputs[j].slice(slice_indexes.as_slice(), backend)?;
                    let squeezed = sliced.squeeze(scan_input_axes[j])?;
                    iter_scan_inputs.push(squeezed);
                }
                iter_scan_inputs
            };

            let input_map = {
                let mut input_map = HashMap::new();
                let mut i = 0;
                for input in state_tensors.iter().flatten() {
                    input_map.insert(self.body.ordered_inputs[i], input.clone());
                    i += 1;
                }
                for input in &iter_scan_inputs {
                    input_map.insert(self.body.ordered_inputs[i], input.clone());
                    i += 1;
                }
                for input in self.body.get_foreign_tensor_ids() {
                    input_map.insert(input, inputs[&input].clone());
                }
                input_map
            };

            let eval_outputs = self.body.eval(&input_map, backend)?;

            let temp_outputs = self.body.ordered_outputs[0..state_tensors.len()]
                .iter()
                .map(|x| eval_outputs[x].clone())
                .collect::<Vec<_>>();
            let scan_outputs = self.body.ordered_outputs[state_tensors.len()..]
                .iter()
                .map(|x| eval_outputs[x].clone())
                .collect::<Vec<_>>();

            for (i, output) in scan_outputs.iter().enumerate() {
                accumulated_scan_outputs[i].push(output.clone());
            }
            state_tensors = temp_outputs
                .iter()
                .map(|x| Some(x.clone()))
                .collect::<Vec<_>>();
        }

        // Concatenate the accumulated outputs
        let scan_outputs = {
            let mut scan_outputs: Vec<NumericTensor<DynRank>> = Vec::new();
            for (i, outputs) in accumulated_scan_outputs.iter().enumerate() {
                let concat_dim = if let Some(x) = &self.scan_output_axes {
                    let v = x[i];
                    if v < 0 {
                        (v + (scan_outputs[0].rank() + 1) as i64) as usize
                    } else {
                        v as usize
                    }
                } else {
                    0
                };
                let mut unsqueezed_outputs = Vec::new();
                for output in outputs {
                    unsqueezed_outputs.push(output.unsqueeze(concat_dim)?);
                }
                let mut unsqueezed_outputs_refs = Vec::new();
                for output in &unsqueezed_outputs {
                    unsqueezed_outputs_refs.push(output);
                }
                scan_outputs.push(NumericTensor::concat(
                    unsqueezed_outputs_refs.as_slice(),
                    concat_dim,
                    backend,
                )?);
            }
            scan_outputs
        };

        let mut outputs = HashMap::new();
        for (i, state_tensor) in state_tensors.iter().enumerate() {
            if let Some(x) = self.state_outputs[i] {
                outputs.insert(x, state_tensor.clone().unwrap());
            }
        }
        for (i, output) in scan_outputs.iter().enumerate() {
            if let Some(x) = self.scan_outputs[i] {
                outputs.insert(x, output.clone());
            }
        }

        Ok(outputs)
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        todo!()
    }
}
