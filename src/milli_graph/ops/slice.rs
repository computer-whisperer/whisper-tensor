use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::dtype::DType;
use crate::graph::GlobalId;
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::numeric_tensor::NumericTensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use typenum::P1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Slice {
    global_id: crate::graph::GlobalId,
    output: GlobalId,
    data: GlobalId,
    starts: GlobalId,
    ends: GlobalId,
    steps: Option<GlobalId>,
    axes: Option<GlobalId>,
}

impl Slice {
    pub fn push_new(
        graph: &mut MilliOpGraph,
        data: GlobalId,
        starts: GlobalId,
        ends: GlobalId,
        steps: Option<GlobalId>,
        axes: Option<GlobalId>,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            output,
            data,
            starts,
            ends,
            steps,
            axes,
            global_id: GlobalId::new(rng),
        };
        graph.push_op(AnyMilliOp::Slice(node));
        output
    }
}

impl Slice {
    pub(crate) fn data_id(&self) -> GlobalId {
        self.data
    }
    pub(crate) fn starts_id(&self) -> GlobalId {
        self.starts
    }
    pub(crate) fn ends_id(&self) -> GlobalId {
        self.ends
    }
    pub(crate) fn steps_id(&self) -> Option<GlobalId> {
        self.steps
    }
    pub(crate) fn axes_id(&self) -> Option<GlobalId> {
        self.axes
    }

    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl rand::Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.data, map);
        super::remap(&mut self.starts, map);
        super::remap(&mut self.ends, map);
        super::remap_opt(&mut self.steps, map);
        super::remap_opt(&mut self.axes, map);
    }
}

impl crate::graph::Node for Slice {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Slice".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        let mut res = vec![self.data, self.starts, self.ends];
        if let Some(steps) = &self.steps {
            res.push(*steps);
        }
        if let Some(axes) = &self.axes {
            res.push(*axes);
        }
        Box::new(res.into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new([self.output].into_iter())
    }
}

impl MilliOp for Slice {
    fn infer(
        &self,
        known_inputs: &HashMap<GlobalId, crate::tensor_info::TensorInfo>,
        _symbolic_resolver: &mut crate::symbolic_scalar::SymbolicResolver,
        backend: &mut EvalBackend,
    ) -> Result<
        Box<dyn Iterator<Item = (GlobalId, crate::tensor_info::TensorInfo)>>,
        MilliOpGraphError,
    > {
        use crate::tensor_info::TensorInfo;
        use crate::scalar_info::ScalarInfoTyped;

        let data_info = known_inputs.get(&self.data).ok_or(MilliOpGraphError::UnableToInfer)?;

        // If all inputs are concrete, fall back to eval.
        let all_numeric = data_info.as_numeric().is_some()
            && known_inputs.get(&self.starts).and_then(|i| i.as_numeric()).is_some()
            && known_inputs.get(&self.ends).and_then(|i| i.as_numeric()).is_some()
            && self.steps.map_or(true, |id| known_inputs.get(&id).and_then(|i| i.as_numeric()).is_some())
            && self.axes.map_or(true, |id| known_inputs.get(&id).and_then(|i| i.as_numeric()).is_some());
        if all_numeric {
            use crate::graph::Node;
            let mut resolved = HashMap::new();
            for id in Node::inputs(self) {
                let info = known_inputs.get(&id).ok_or(MilliOpGraphError::UnableToInfer)?;
                resolved.insert(id, info.as_numeric().unwrap().clone());
            }
            let collected: Vec<(GlobalId, TensorInfo)> = self
                .eval(&resolved, backend)?
                .map(|(a, b)| (a, TensorInfo::from(b)))
                .collect();
            return Ok(Box::new(collected.into_iter()));
        }

        // Shape-only inference: compute output shape from data shape + slice params.
        let data_ranked = data_info.as_ranked().ok_or(MilliOpGraphError::UnableToInfer)?;
        let data_shape = data_ranked.shape();
        let data_rank = data_shape.len();

        // Try to extract concrete i64 values from a tensor.
        let extract_i64 = |id: &GlobalId| -> Option<Vec<i64>> {
            let info = known_inputs.get(id)?;
            let tensor = info.as_numeric()?;
            let as_i64 = tensor.cast(DType::I64, &mut EvalBackend::NDArray).ok()?;
            let rank1 = as_i64.try_to_rank::<P1>().ok()?;
            Vec::<i64>::try_from(rank1.to_ndarray().ok()?).ok()
        };

        let starts = extract_i64(&self.starts);
        let ends = extract_i64(&self.ends);
        let steps: Option<Vec<i64>> = if let Some(steps_id) = &self.steps {
            extract_i64(steps_id)
        } else {
            starts.as_ref().map(|s| s.iter().map(|_| 1i64).collect())
        };
        let axes: Option<Vec<usize>> = if let Some(axes_id) = &self.axes {
            extract_i64(axes_id).map(|a| {
                a.iter().map(|&v| if v < 0 { (v + data_rank as i64) as usize } else { v as usize }).collect()
            })
        } else {
            starts.as_ref().map(|s| (0..s.len()).collect())
        };

        let mut out_dims = data_shape.clone();

        // If we have concrete slice params, compute exact output dims.
        // Otherwise, make sliced axes symbolic (we know the rank but not the dim sizes).
        if let (Some(starts), Some(ends), Some(steps), Some(axes)) = (&starts, &ends, &steps, &axes) {
            for (i, &axis) in axes.iter().enumerate() {
                if let ScalarInfoTyped::Numeric(dim_val) = &data_shape[axis] {
                    let dim = *dim_val as i64;
                    let step = steps[i];
                    let (start, end) = if step > 0 {
                        let s = starts[i].clamp(-dim, dim);
                        let s = if s < 0 { s + dim } else { s };
                        let e = ends[i].clamp(-dim, dim);
                        let e = if e < 0 { e + dim } else { e };
                        (s, e)
                    } else {
                        let s = starts[i].clamp(-dim, dim - 1);
                        let s = if s < 0 { s + dim } else { s };
                        let e = ends[i].clamp(-dim - 1, dim);
                        let e = if e < 0 { e + dim } else { e };
                        (s, e)
                    };
                    let sliced = ((end - start + (step - step.signum())) / step).max(0) as u64;
                    out_dims[axis] = ScalarInfoTyped::Numeric(sliced);
                }
                // If dim is symbolic, leave it symbolic.
            }
        } else if let Some(axes) = &axes {
            // We know which axes are sliced but not the exact values —
            // make those dims symbolic.
            for &axis in axes {
                out_dims[axis] = ScalarInfoTyped::Symbolic(
                    crate::symbolic_scalar::SymbolicScalarTyped::new(_symbolic_resolver),
                );
            }
        } else {
            // We don't know the axes — any dim could be sliced.
            // Make all dims symbolic to avoid claiming incorrect concrete sizes.
            for dim in out_dims.iter_mut() {
                *dim = ScalarInfoTyped::Symbolic(
                    crate::symbolic_scalar::SymbolicScalarTyped::new(_symbolic_resolver),
                );
            }
        }

        let out_dtype = data_info.dtype();
        let out_info = TensorInfo::from_dtype_and_shape_scalars(out_dtype, &out_dims);
        Ok(Box::new([(self.output, out_info)].into_iter()))
    }

    fn eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, MilliOpGraphError>
    {
        let data_input = &inputs[&self.data];
        let input_shape = data_input.shape();
        let input_rank = data_input.rank();
        let axes: Vec<i64> = if let Some(axes) = &self.axes {
            inputs[axes]
                .cast(DType::I64, backend)?
                .try_to_rank::<P1>()?
                .try_into()?
        } else {
            (0i64..(input_rank as i64)).collect()
        };
        let steps: Vec<i64> = if let Some(steps) = &self.steps {
            inputs[steps]
                .cast(DType::I64, backend)?
                .try_to_rank::<P1>()?
                .try_into()?
        } else {
            axes.iter().map(|_| 1).collect()
        };
        let starts: Vec<i64> = inputs[&self.starts]
            .cast(DType::I64, backend)?
            .try_to_rank::<P1>()?
            .try_into()?;
        let ends: Vec<i64> = inputs[&self.ends]
            .cast(DType::I64, backend)?
            .try_to_rank::<P1>()?
            .try_into()?;

        // Build per-axis (start, end, step) in ndarray isize convention.
        // Unmentioned axes get full extent with step 1.
        let mut slices: Vec<(isize, isize, isize)> = input_shape
            .iter()
            .map(|&dim| (0, dim as isize, 1))
            .collect();

        for (i, axis) in axes.into_iter().enumerate() {
            let axis = if axis < 0 {
                (input_rank as i64 + axis) as usize
            } else {
                axis as usize
            };
            let dim = input_shape[axis] as i64;
            let step = steps[i];
            if step == 0 {
                return Err(MilliOpGraphError::InvalidInput(
                    "Step must not be 0".to_string(),
                ));
            }

            // Clamp start/end per ONNX spec
            let (start, end) = if step > 0 {
                let s = starts[i].clamp(-dim, dim);
                let s = if s < 0 { s + dim } else { s };
                let e = ends[i].clamp(-dim, dim);
                let e = if e < 0 { e + dim } else { e };
                (s as isize, e as isize)
            } else {
                // Negative step: start clamped to [-1, dim-1], end to [-dim-1, dim-1]
                let s = starts[i].clamp(-dim, dim - 1);
                let s = if s < 0 { s + dim } else { s };
                let e = ends[i].clamp(-dim - 1, dim);
                let e = if e < 0 { e + dim } else { e };
                (s as isize, e as isize)
            };

            slices[axis] = (start, end, step as isize);
        }

        let output = data_input.slice_with_steps(&slices, backend)?;
        Ok(Box::new([(self.output, output)].into_iter()))
    }
}
