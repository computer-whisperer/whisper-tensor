use crate::graph::{GlobalId, Node, Property, PropertyValue};
use crate::milli_graph::{MilliLoweringContext, MilliOpGraph};
use crate::symbolic_graph::ops::{EvalError, Operation};
use crate::symbolic_graph::{
    ONNXDecodingError, SymbolicGraph, SymbolicGraphMutator, query_attribute_graph,
    query_attribute_int, query_attribute_ints,
};
use crate::{DynRank, onnx};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScanOperation {
    global_id: GlobalId,
    scan_inputs: Vec<Option<GlobalId>>,
    state_inputs: Vec<Option<GlobalId>>,
    scan_outputs: Vec<Option<GlobalId>>,
    state_outputs: Vec<Option<GlobalId>>,
    scan_input_axes: Option<Vec<i64>>,
    scan_input_directions: Option<Vec<i64>>,
    scan_output_axes: Option<Vec<i64>>,
    scan_output_directions: Option<Vec<i64>>,
    body: SymbolicGraph,
}

impl ScanOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        symbolic_graph_mutator: &mut SymbolicGraphMutator,
        core_opset_version: usize,
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        let body = query_attribute_graph(attributes, "body")
            .ok_or(ONNXDecodingError::MissingField("body"))?;
        let body = {
            let mut inner_graph = SymbolicGraph::new(rng);
            inner_graph.populate(symbolic_graph_mutator, body, core_opset_version, rng, None)?;
            inner_graph
        };

        // In opset < 9, input[0] is an optional sequence_lens tensor (skip it).
        let scan_inputs_start = if core_opset_version < 9 { 1 } else { 0 };

        let num_scan_inputs = query_attribute_int(attributes, "num_scan_inputs")
            .ok_or(ONNXDecodingError::MissingField("num_scan_inputs"))?;
        assert!(num_scan_inputs <= inputs.len() as i64);
        assert!(num_scan_inputs >= 1);
        let num_state_tensors = (inputs.len() - scan_inputs_start) - num_scan_inputs as usize;

        let scan_input_axes = query_attribute_ints(attributes, "scan_input_axes");
        let scan_input_directions = query_attribute_ints(attributes, "scan_input_directions");
        let scan_output_axes = query_attribute_ints(attributes, "scan_output_axes");
        let scan_output_directions = query_attribute_ints(attributes, "scan_output_directions");

        let state_inputs =
            inputs[scan_inputs_start..scan_inputs_start + num_state_tensors].to_vec();
        let scan_inputs = inputs[scan_inputs_start + num_state_tensors..].to_vec();

        let state_outputs = outputs[..num_state_tensors].to_vec();
        let scan_outputs = outputs[num_state_tensors..].to_vec();

        Ok(Self {
            global_id: GlobalId::new(rng),
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

impl Node for ScanOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Scan".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        let mut v = vec![];
        v.extend(self.state_inputs.iter().filter_map(|x| *x));
        v.extend(self.scan_inputs.iter().filter_map(|x| *x));
        v.extend(self.body.get_foreign_tensor_ids());
        Box::new(v.into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        let mut v = vec![];
        v.extend(self.state_outputs.iter().filter_map(|x| *x));
        v.extend(self.scan_outputs.iter().filter_map(|x| *x));
        Box::new(v.into_iter())
    }
}

impl Operation for ScanOperation {
    fn parameters(&self) -> Vec<Property> {
        let mut params = Vec::new();
        params.push(Property::new(
            "num_scan_inputs",
            PropertyValue::Int(self.scan_inputs.len() as i64),
        ));
        params.push(Property::new(
            "num_state_inputs",
            PropertyValue::Int(self.state_inputs.len() as i64),
        ));
        if let Some(axes) = &self.scan_input_axes {
            params.push(Property::new(
                "scan_input_axes",
                PropertyValue::IntList(axes.clone()),
            ));
        }
        if let Some(dirs) = &self.scan_input_directions {
            params.push(Property::new(
                "scan_input_directions",
                PropertyValue::IntList(dirs.clone()),
            ));
        }
        if let Some(axes) = &self.scan_output_axes {
            params.push(Property::new(
                "scan_output_axes",
                PropertyValue::IntList(axes.clone()),
            ));
        }
        if let Some(dirs) = &self.scan_output_directions {
            params.push(Property::new(
                "scan_output_directions",
                PropertyValue::IntList(dirs.clone()),
            ));
        }
        params
    }

    fn get_sub_graphs(&self) -> Vec<&SymbolicGraph> {
        vec![&self.body]
    }

    fn eval_pool<'p, P: crate::pool::Pool + 'p>(
        &self,
        inputs: &HashMap<GlobalId, &crate::numeric_tensor::NumericTensorView<'_, DynRank>>,
        pool: &'p P,
    ) -> Result<HashMap<GlobalId, crate::numeric_tensor::NumericTensor<'p, DynRank, P>>, EvalError>
    {
        use crate::numeric_dtype::NumericDType;
        use crate::numeric_scalar::NumericScalar;
        use crate::numeric_tensor::{NumericTensor as PoolTensor, NumericTensorView, TensorLayout};

        // Helper: slice a pool tensor along an axis at a single index, squeeze that axis.
        fn slice_and_squeeze<'p2, P2: crate::pool::Pool + 'p2>(
            src: &NumericTensorView<'_, DynRank>,
            axis: usize,
            index: u64,
            pool: &'p2 P2,
        ) -> Result<PoolTensor<'p2, DynRank, P2>, EvalError> {
            let shape = src.shape();
            let rank = shape.len();
            let mut out_shape = Vec::with_capacity(rank - 1);
            for (i, &d) in shape.iter().enumerate() {
                if i != axis {
                    out_shape.push(d);
                }
            }
            let dtype = src.dtype();
            let out_numel: usize = out_shape.iter().product::<u64>() as usize;
            let layout = TensorLayout::<DynRank>::row_major(out_shape, dtype);
            let buf = pool
                .allocate(layout.buffer_size_bytes())
                .map_err(|e| EvalError::InvalidInput(format!("pool allocation: {e}")))?;
            let mut out = PoolTensor::from_parts(buf, layout);

            // Compute strides for the source tensor.
            let mut strides = vec![1usize; rank];
            for i in (0..rank - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1] as usize;
            }

            // Copy elements: iterate over the squeezed output shape, mapping back to src.
            for flat_out in 0..out_numel {
                // Decompose flat_out into indices (skipping the axis dim).
                let mut remaining = flat_out;
                let mut src_flat = index as usize * strides[axis];
                let mut dim_idx = 0;
                for i in 0..rank {
                    if i == axis {
                        continue;
                    }
                    let out_dim_size = if dim_idx + 1 < out_numel.max(1) {
                        // compute stride in output space
                        let mut s = 1usize;
                        for j in (dim_idx + 1)..out.view().shape().len() {
                            s *= out.view().shape()[j] as usize;
                        }
                        s
                    } else {
                        1
                    };
                    let _ = out_dim_size; // unused in this approach

                    // Simpler: just use the source strides directly.
                    // Decompose flat_out into per-dim indices using output row-major strides.
                    break;
                }
                // Actually, simplest approach: iterate all src elements and filter.
                // But that's O(n * axis_size). Let me do it properly.

                // Recompute: flat_out → output indices → source flat index.
                let out_view = out.view();
                let out_shape_ref = out_view.shape();
                let mut out_indices = vec![0u64; out_shape_ref.len()];
                let mut rem = flat_out;
                for i in (0..out_shape_ref.len()).rev() {
                    out_indices[i] = (rem % out_shape_ref[i] as usize) as u64;
                    rem /= out_shape_ref[i] as usize;
                }
                // Map output indices → source indices (insert `index` at axis).
                let mut src_idx = 0usize;
                let mut oi = 0;
                for i in 0..rank {
                    let idx = if i == axis {
                        index
                    } else {
                        let v = out_indices[oi];
                        oi += 1;
                        v
                    };
                    src_idx += idx as usize * strides[i];
                }
                out.write_element(flat_out, src.read_element(src_idx));
            }
            Ok(out)
        }

        // Helper: unsqueeze a tensor (insert size-1 dim at axis).
        fn unsqueeze<'p2, P2: crate::pool::Pool + 'p2>(
            src: &PoolTensor<'p2, DynRank, P2>,
            axis: usize,
            pool: &'p2 P2,
        ) -> Result<PoolTensor<'p2, DynRank, P2>, EvalError> {
            let src_view = src.view();
            let mut new_shape: Vec<u64> = src_view.shape().to_vec();
            new_shape.insert(axis, 1);
            let layout = TensorLayout::<DynRank>::row_major(new_shape, src_view.dtype());
            let buf = pool
                .allocate(layout.buffer_size_bytes())
                .map_err(|e| EvalError::InvalidInput(format!("pool allocation: {e}")))?;
            let mut out = PoolTensor::from_parts(buf, layout);
            for i in 0..src_view.numel() {
                out.write_element(i, src_view.read_element(i));
            }
            Ok(out)
        }

        // Helper: concat tensors along an axis.
        fn concat_along<'p2, P2: crate::pool::Pool + 'p2>(
            tensors: &[PoolTensor<'p2, DynRank, P2>],
            axis: usize,
            pool: &'p2 P2,
        ) -> Result<PoolTensor<'p2, DynRank, P2>, EvalError> {
            if tensors.is_empty() {
                return Err(EvalError::InvalidInput("concat: empty input".into()));
            }
            let first = tensors[0].view();
            let rank = first.shape().len();
            let dtype = first.dtype();

            let mut out_shape: Vec<u64> = first.shape().to_vec();
            for t in &tensors[1..] {
                out_shape[axis] += t.view().shape()[axis];
            }
            let out_numel: usize = out_shape.iter().product::<u64>() as usize;
            let layout = TensorLayout::<DynRank>::row_major(out_shape.clone(), dtype);
            let buf = pool
                .allocate(layout.buffer_size_bytes())
                .map_err(|e| EvalError::InvalidInput(format!("pool allocation: {e}")))?;
            let mut out = PoolTensor::from_parts(buf, layout);

            // Compute output strides.
            let mut out_strides = vec![1usize; rank];
            for i in (0..rank - 1).rev() {
                out_strides[i] = out_strides[i + 1] * out_shape[i + 1] as usize;
            }

            let mut axis_offset = 0u64;
            for t in tensors {
                let tv = t.view();
                let t_shape = tv.shape();
                let t_numel = tv.numel();
                let mut t_strides = vec![1usize; rank];
                for i in (0..rank - 1).rev() {
                    t_strides[i] = t_strides[i + 1] * t_shape[i + 1] as usize;
                }
                for flat in 0..t_numel {
                    // Decompose into per-dim indices.
                    let mut rem = flat;
                    let mut out_flat = 0usize;
                    for i in 0..rank {
                        let idx = rem / t_strides[i];
                        rem %= t_strides[i];
                        let out_idx = if i == axis {
                            idx + axis_offset as usize
                        } else {
                            idx
                        };
                        out_flat += out_idx * out_strides[i];
                    }
                    out.write_element(out_flat, tv.read_element(flat));
                }
                axis_offset += t_shape[axis];
            }
            Ok(out)
        }

        // --- Main Scan loop ---

        // Collect scan input views.
        let scan_input_views: Vec<_> = self
            .scan_inputs
            .iter()
            .map(|x| *inputs.get(&x.unwrap()).unwrap())
            .collect();

        let scan_input_axes = if let Some(axes) = &self.scan_input_axes {
            axes.iter()
                .enumerate()
                .map(|(i, &a)| {
                    if a >= 0 {
                        a as usize
                    } else {
                        (scan_input_views[i].shape().len() as i64 + a) as usize
                    }
                })
                .collect::<Vec<_>>()
        } else {
            vec![0; scan_input_views.len()]
        };

        let iter_count = scan_input_views[0].shape()[scan_input_axes[0]];

        assert!(self.scan_input_directions.is_none());
        assert!(self.scan_output_directions.is_none());

        // Initialize state tensors from inputs.
        let mut state_tensors: Vec<Option<PoolTensor<'p, DynRank, P>>> = self
            .state_inputs
            .iter()
            .map(|x| {
                x.map(|id| {
                    let view = inputs[&id];
                    let layout =
                        TensorLayout::<DynRank>::row_major(view.shape().to_vec(), view.dtype());
                    let buf = pool.allocate(layout.buffer_size_bytes()).unwrap();
                    let mut t = PoolTensor::from_parts(buf, layout);
                    for i in 0..view.numel() {
                        t.write_element(i, view.read_element(i));
                    }
                    t
                })
            })
            .collect();

        let mut accumulated_scan_outputs: Vec<Vec<PoolTensor<'p, DynRank, P>>> =
            (0..self.scan_outputs.len()).map(|_| Vec::new()).collect();

        for iter_idx in 0..iter_count {
            // Slice scan inputs for this iteration.
            let iter_scan_inputs: Vec<PoolTensor<'p, DynRank, P>> = scan_input_views
                .iter()
                .enumerate()
                .map(|(j, view)| slice_and_squeeze(view, scan_input_axes[j], iter_idx, pool))
                .collect::<Result<Vec<_>, _>>()?;

            // Build input map for body: state tensors + scan inputs + foreign tensors.
            let mut body_input_views: Vec<(GlobalId, PoolTensor<'p, DynRank, P>)> = Vec::new();
            let mut input_idx = 0;
            for st in &state_tensors {
                if let Some(t) = st {
                    let view = t.view();
                    let layout =
                        TensorLayout::<DynRank>::row_major(view.shape().to_vec(), view.dtype());
                    let buf = pool
                        .allocate(layout.buffer_size_bytes())
                        .map_err(|e| EvalError::InvalidInput(format!("pool allocation: {e}")))?;
                    let mut copy = PoolTensor::from_parts(buf, layout);
                    for i in 0..view.numel() {
                        copy.write_element(i, view.read_element(i));
                    }
                    body_input_views.push((self.body.ordered_inputs[input_idx], copy));
                    input_idx += 1;
                }
            }
            for t in &iter_scan_inputs {
                body_input_views.push((self.body.ordered_inputs[input_idx], {
                    // Already a pool tensor from slice_and_squeeze.
                    let view = t.view();
                    let layout =
                        TensorLayout::<DynRank>::row_major(view.shape().to_vec(), view.dtype());
                    let buf = pool
                        .allocate(layout.buffer_size_bytes())
                        .map_err(|e| EvalError::InvalidInput(format!("pool allocation: {e}")))?;
                    let mut copy = PoolTensor::from_parts(buf, layout);
                    for i in 0..view.numel() {
                        copy.write_element(i, view.read_element(i));
                    }
                    copy
                }));
                input_idx += 1;
            }
            // Foreign tensors (outer-scope tensors referenced by the body).
            for foreign_id in self.body.get_foreign_tensor_ids() {
                if let Some(&view) = inputs.get(&foreign_id) {
                    let layout =
                        TensorLayout::<DynRank>::row_major(view.shape().to_vec(), view.dtype());
                    let buf = pool
                        .allocate(layout.buffer_size_bytes())
                        .map_err(|e| EvalError::InvalidInput(format!("pool allocation: {e}")))?;
                    let mut copy = PoolTensor::from_parts(buf, layout);
                    for i in 0..view.numel() {
                        copy.write_element(i, view.read_element(i));
                    }
                    body_input_views.push((foreign_id, copy));
                }
            }

            // Build view refs and call body.eval_pool.
            let body_views: HashMap<
                GlobalId,
                crate::numeric_tensor::NumericTensorView<'_, DynRank>,
            > = body_input_views
                .iter()
                .map(|(id, t)| (*id, t.view()))
                .collect();
            let body_view_refs: HashMap<
                GlobalId,
                &crate::numeric_tensor::NumericTensorView<'_, DynRank>,
            > = body_views.iter().map(|(&id, v)| (id, v)).collect();
            let eval_outputs = self.body.eval_pool(&body_view_refs, pool)?;

            // Extract state outputs (first N body outputs).
            let new_state: Vec<Option<PoolTensor<'p, DynRank, P>>> = self.body.ordered_outputs
                [..state_tensors.len()]
                .iter()
                .map(|id| {
                    eval_outputs.get(id).map(|t| {
                        let view = t.view();
                        let layout =
                            TensorLayout::<DynRank>::row_major(view.shape().to_vec(), view.dtype());
                        let buf = pool.allocate(layout.buffer_size_bytes()).unwrap();
                        let mut copy = PoolTensor::from_parts(buf, layout);
                        for i in 0..view.numel() {
                            copy.write_element(i, view.read_element(i));
                        }
                        copy
                    })
                })
                .collect();

            // Extract scan outputs (remaining body outputs).
            let scan_out_ids = &self.body.ordered_outputs[state_tensors.len()..];
            for (i, id) in scan_out_ids.iter().enumerate() {
                if let Some(t) = eval_outputs.get(id) {
                    let view = t.view();
                    let layout =
                        TensorLayout::<DynRank>::row_major(view.shape().to_vec(), view.dtype());
                    let buf = pool
                        .allocate(layout.buffer_size_bytes())
                        .map_err(|e| EvalError::InvalidInput(format!("pool allocation: {e}")))?;
                    let mut copy = PoolTensor::from_parts(buf, layout);
                    for j in 0..view.numel() {
                        copy.write_element(j, view.read_element(j));
                    }
                    accumulated_scan_outputs[i].push(copy);
                }
            }

            state_tensors = new_state;
        }

        // Concatenate accumulated scan outputs.
        let mut outputs = HashMap::new();
        for (i, st) in state_tensors.iter().enumerate() {
            if let Some(out_id) = self.state_outputs[i] {
                if let Some(t) = st {
                    let view = t.view();
                    let layout =
                        TensorLayout::<DynRank>::row_major(view.shape().to_vec(), view.dtype());
                    let buf = pool
                        .allocate(layout.buffer_size_bytes())
                        .map_err(|e| EvalError::InvalidInput(format!("pool allocation: {e}")))?;
                    let mut copy = PoolTensor::from_parts(buf, layout);
                    for j in 0..view.numel() {
                        copy.write_element(j, view.read_element(j));
                    }
                    outputs.insert(out_id, copy);
                }
            }
        }

        for (i, acc) in accumulated_scan_outputs.iter().enumerate() {
            if let Some(out_id) = self.scan_outputs[i] {
                let concat_axis = if let Some(axes) = &self.scan_output_axes {
                    let v = axes[i];
                    if v < 0 {
                        (v + acc[0].view().shape().len() as i64 + 1) as usize
                    } else {
                        v as usize
                    }
                } else {
                    0
                };
                let unsqueezed: Vec<PoolTensor<'p, DynRank, P>> = acc
                    .iter()
                    .map(|t| unsqueeze(t, concat_axis, pool))
                    .collect::<Result<Vec<_>, _>>()?;
                let concatenated = concat_along(&unsqueezed, concat_axis, pool)?;
                outputs.insert(out_id, concatenated);
            }
        }

        Ok(outputs)
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, _rng: &mut impl Rng) -> MilliOpGraph {
        todo!()
    }
}
