use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::graph::GlobalId;
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::migration::numeric_tensor::NumericTensor;
use crate::pool::Pool;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use typenum::P1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Range {
    global_id: GlobalId,
    pub(crate) label: Option<String>,
    output: GlobalId,
    start: GlobalId,
    end: GlobalId,
    delta: GlobalId,
}

impl Range {
    pub fn push_new(
        graph: &mut MilliOpGraph,
        start: GlobalId,
        end: GlobalId,
        delta: GlobalId,
        rng: &mut impl Rng,
    ) -> GlobalId {
        Self::push_new_with_label(graph, start, end, delta, None, rng)
    }

    pub fn push_new_with_label(
        graph: &mut MilliOpGraph,
        start: GlobalId,
        end: GlobalId,
        delta: GlobalId,
        label: Option<String>,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            global_id: GlobalId::new(rng),
            label,
            output,
            start,
            end,
            delta,
        };
        graph.push_op(AnyMilliOp::Range(node));
        output
    }
}

impl Range {
    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl rand::Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.start, map);
        super::remap(&mut self.end, map);
        super::remap(&mut self.delta, map);
    }
}

impl crate::graph::Node for Range {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Range".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.start, self.end, self.delta].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.output].into_iter())
    }
}

impl MilliOp for Range {
    fn infer<'p, P: Pool + 'p>(
        &self,
        known_inputs: &HashMap<GlobalId, crate::tensor_info::TensorInfo<'p, P>>,
        symbolic_resolver: &mut crate::symbolic_scalar::SymbolicResolver,
        _pool: &'p P,
    ) -> Result<Vec<(GlobalId, crate::tensor_info::TensorInfo<'p, P>)>, MilliOpGraphError> {
        use crate::scalar_info::{ScalarInfo, ScalarInfoTyped};
        use crate::symbolic_scalar::SymbolicScalar;
        use crate::tensor_info::TensorInfo;

        let start_info = known_inputs
            .get(&self.start)
            .ok_or(MilliOpGraphError::UnableToInfer)?;
        let end_info = known_inputs
            .get(&self.end)
            .ok_or(MilliOpGraphError::UnableToInfer)?;
        let delta_info = known_inputs
            .get(&self.delta)
            .ok_or(MilliOpGraphError::UnableToInfer)?;
        let out_dtype = start_info.dtype();

        // If all inputs are concrete, compute the output length.
        if let (Some(start_v), Some(end_v), Some(delta_v)) = (
            start_info
                .to_f64_vec()
                .and_then(|v| v.first().copied()),
            end_info
                .to_f64_vec()
                .and_then(|v| v.first().copied()),
            delta_info
                .to_f64_vec()
                .and_then(|v| v.first().copied()),
        ) {
            let n = ((end_v - start_v) / delta_v).ceil().max(0.0) as u64;
            return Ok(vec![(
                self.output,
                TensorInfo::from_dtype_and_shape(out_dtype, &[n]),
            )]);
        }

        // Can't determine output length without concrete values.
        let first = ScalarInfo::Symbolic(SymbolicScalar::new(out_dtype, symbolic_resolver));
        let rank = ScalarInfoTyped::Numeric(1u32);
        Ok(vec![(
            self.output,
            TensorInfo::new_from_first_element_and_rank(first, rank, symbolic_resolver),
        )])
    }

    fn eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        _config: &super::MilliEvalConfig,
        backend: &mut EvalBackend,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, MilliOpGraphError>
    {
        let out: NumericTensor<DynRank> = NumericTensor::<P1>::range(
            inputs[&self.start].first_element(),
            inputs[&self.end].first_element(),
            inputs[&self.delta].first_element(),
            backend,
        )?
        .to_dyn_rank();
        Ok(Box::new([(self.output, out)].into_iter()))
    }

    fn eval_new<'p, P2: crate::pool::Pool + 'p>(
        &self,
        inputs: &[crate::numeric_tensor::NumericTensorView<'_, crate::tensor_rank::DynRank>],
        pool: &'p P2,
    ) -> Result<Vec<crate::numeric_tensor::NumericTensor<'p, crate::tensor_rank::DynRank, P2>>, crate::nano_graph::pool_eval::PoolEvalError> {
        use crate::numeric_scalar::NumericScalar;
        use crate::numeric_tensor::{NumericTensor, TensorLayout};
        use crate::tensor_rank::DynRank;

        let start = inputs[0].read_element(0);
        let end = inputs[1].read_element(0);
        let delta = inputs[2].read_element(0);
        let dtype = start.dtype();

        // Compute length: ceil((end - start) / delta).
        let start_f = start.to_f64();
        let end_f = end.to_f64();
        let delta_f = delta.to_f64();
        let n = ((end_f - start_f) / delta_f).ceil().max(0.0) as usize;

        let layout = TensorLayout::<DynRank>::row_major(vec![n as u64], dtype);
        let buf = pool.allocate(layout.buffer_size_bytes())
            .map_err(crate::nano_graph::pool_eval::PoolEvalError::Allocation)?;
        let mut out = NumericTensor::from_parts(buf, layout);

        let mut cur = start;
        for i in 0..n {
            out.write_element(i, cur);
            cur = cur.add(delta);
        }

        Ok(vec![out])
    }
}
