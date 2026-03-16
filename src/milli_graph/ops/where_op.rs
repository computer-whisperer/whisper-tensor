use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::graph::{GlobalId, Node};
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::numeric_tensor::NumericTensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Where {
    global_id: GlobalId,
    pub(crate) label: Option<String>,
    output: GlobalId,
    condition: GlobalId,
    x: GlobalId,
    y: GlobalId,
}

impl Where {
    pub fn push_new(
        graph: &mut MilliOpGraph,
        condition: GlobalId,
        x: GlobalId,
        y: GlobalId,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        Self::push_new_with_label(graph, condition, x, y, None, rng)
    }

    pub fn push_new_with_label(
        graph: &mut MilliOpGraph,
        condition: GlobalId,
        x: GlobalId,
        y: GlobalId,
        label: Option<String>,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            global_id: GlobalId::new(rng),
            label,
            output,
            condition,
            x,
            y,
        };
        graph.push_op(AnyMilliOp::Where(node));
        output
    }
}

impl Where {
    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl rand::Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.condition, map);
        super::remap(&mut self.x, map);
        super::remap(&mut self.y, map);
    }
}

impl Node for Where {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> String {
        "Where".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new([self.condition, self.x, self.y].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new([self.output].into_iter())
    }
}

impl MilliOp for Where {
    fn infer(
        &self,
        known_inputs: &HashMap<GlobalId, crate::tensor_info::TensorInfo>,
        symbolic_resolver: &mut crate::symbolic_scalar::SymbolicResolver,
        backend: &mut EvalBackend,
    ) -> Result<
        Box<dyn Iterator<Item = (GlobalId, crate::tensor_info::TensorInfo)>>,
        MilliOpGraphError,
    > {
        use crate::tensor_info::TensorInfo;

        let cond_info = known_inputs
            .get(&self.condition)
            .ok_or(MilliOpGraphError::UnableToInfer)?;
        let x_info = known_inputs
            .get(&self.x)
            .ok_or(MilliOpGraphError::UnableToInfer)?;
        let y_info = known_inputs
            .get(&self.y)
            .ok_or(MilliOpGraphError::UnableToInfer)?;

        // If all concrete, fall back to eval.
        if cond_info.as_numeric().is_some()
            && x_info.as_numeric().is_some()
            && y_info.as_numeric().is_some()
        {
            let mut resolved = HashMap::new();
            resolved.insert(self.condition, cond_info.as_numeric().unwrap().clone());
            resolved.insert(self.x, x_info.as_numeric().unwrap().clone());
            resolved.insert(self.y, y_info.as_numeric().unwrap().clone());
            let collected: Vec<(GlobalId, TensorInfo)> = self
                .eval(&resolved, backend)?
                .map(|(a, b)| (a, TensorInfo::from(b)))
                .collect();
            return Ok(Box::new(collected.into_iter()));
        }

        let out_dtype = x_info.dtype();

        // Try per-dim broadcast shape inference.
        if let (Some(c_ranked), Some(x_ranked), Some(y_ranked)) = (
            cond_info.as_ranked(),
            x_info.as_ranked(),
            y_info.as_ranked(),
        ) && let Ok(out_dims) = super::infer_multidirectional_broadcasting_shape(
            &[c_ranked.shape(), x_ranked.shape(), y_ranked.shape()],
            symbolic_resolver,
        ) {
            let out_info = TensorInfo::from_dtype_and_shape_scalars(out_dtype, &out_dims);
            return Ok(Box::new([(self.output, out_info)].into_iter()));
        }

        // Fallback: rank-only inference.
        let c_shape = cond_info.shape(symbolic_resolver);
        let x_shape = x_info.shape(symbolic_resolver);
        let y_shape = y_info.shape(symbolic_resolver);
        let out_rank = super::infer_multidirectional_broadcasting_rank(
            &[c_shape, x_shape, y_shape],
            symbolic_resolver,
        )?;
        let first_elem = crate::scalar_info::ScalarInfo::Symbolic(
            crate::symbolic_scalar::SymbolicScalar::new(out_dtype, symbolic_resolver),
        );
        let out_info =
            TensorInfo::new_from_first_element_and_rank(first_elem, out_rank, symbolic_resolver);
        Ok(Box::new([(self.output, out_info)].into_iter()))
    }

    fn eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, MilliOpGraphError>
    {
        let out = inputs[&self.condition].where_op(&inputs[&self.x], &inputs[&self.y], backend)?;
        Ok(Box::new([(self.output, out)].into_iter()))
    }
}
