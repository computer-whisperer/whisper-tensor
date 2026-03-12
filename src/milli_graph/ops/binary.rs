use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::dtype::DType;
use crate::milli_graph::MilliOpGraphError;
use crate::milli_graph::ops::MilliOp;
use crate::numeric_tensor::NumericTensor;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) enum WhichSimpleBinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Modulo(Option<bool>),
    And,
    Or,
    Xor,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    Equal,
    Greater,
    GreaterOrEqual,
    Less,
    LessOrEqual,
    Max,
    Min,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleBinary {
    global_id: GlobalId,
    output: GlobalId,
    which_op: WhichSimpleBinaryOp,
    a: GlobalId,
    b: GlobalId,
}

use crate::graph::{GlobalId, Node};
use crate::milli_graph::MilliOpGraph;
use crate::milli_graph::ops::AnyMilliOp;

impl SimpleBinary {
    pub(crate) fn which_op(&self) -> &WhichSimpleBinaryOp {
        &self.which_op
    }

    fn push_new(
        graph: &mut MilliOpGraph,
        a: GlobalId,
        b: GlobalId,
        which_op: WhichSimpleBinaryOp,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            global_id: GlobalId::new(rng),
            output,
            which_op,
            a,
            b,
        };
        graph.push_op(AnyMilliOp::SimpleBinary(node));
        output
    }
    pub fn add(graph: &mut MilliOpGraph, a: GlobalId, b: GlobalId, rng: &mut impl Rng) -> GlobalId {
        Self::push_new(graph, a, b, WhichSimpleBinaryOp::Add, rng)
    }

    pub fn sub(graph: &mut MilliOpGraph, a: GlobalId, b: GlobalId, rng: &mut impl Rng) -> GlobalId {
        Self::push_new(graph, a, b, WhichSimpleBinaryOp::Sub, rng)
    }

    pub fn mul(graph: &mut MilliOpGraph, a: GlobalId, b: GlobalId, rng: &mut impl Rng) -> GlobalId {
        Self::push_new(graph, a, b, WhichSimpleBinaryOp::Mul, rng)
    }

    pub fn div(graph: &mut MilliOpGraph, a: GlobalId, b: GlobalId, rng: &mut impl Rng) -> GlobalId {
        Self::push_new(graph, a, b, WhichSimpleBinaryOp::Div, rng)
    }

    pub fn modulo(
        graph: &mut MilliOpGraph,
        a: GlobalId,
        b: GlobalId,
        fmod: Option<bool>,
        rng: &mut impl Rng,
    ) -> GlobalId {
        Self::push_new(graph, a, b, WhichSimpleBinaryOp::Modulo(fmod), rng)
    }
    pub fn and(graph: &mut MilliOpGraph, a: GlobalId, b: GlobalId, rng: &mut impl Rng) -> GlobalId {
        Self::push_new(graph, a, b, WhichSimpleBinaryOp::And, rng)
    }

    pub fn or(graph: &mut MilliOpGraph, a: GlobalId, b: GlobalId, rng: &mut impl Rng) -> GlobalId {
        Self::push_new(graph, a, b, WhichSimpleBinaryOp::Or, rng)
    }

    pub fn xor(graph: &mut MilliOpGraph, a: GlobalId, b: GlobalId, rng: &mut impl Rng) -> GlobalId {
        Self::push_new(graph, a, b, WhichSimpleBinaryOp::Xor, rng)
    }

    pub fn bitwise_and(
        graph: &mut MilliOpGraph,
        a: GlobalId,
        b: GlobalId,
        rng: &mut impl Rng,
    ) -> GlobalId {
        Self::push_new(graph, a, b, WhichSimpleBinaryOp::BitwiseAnd, rng)
    }

    pub fn bitwise_or(
        graph: &mut MilliOpGraph,
        a: GlobalId,
        b: GlobalId,
        rng: &mut impl Rng,
    ) -> GlobalId {
        Self::push_new(graph, a, b, WhichSimpleBinaryOp::BitwiseOr, rng)
    }

    pub fn bitwise_xor(
        graph: &mut MilliOpGraph,
        a: GlobalId,
        b: GlobalId,
        rng: &mut impl Rng,
    ) -> GlobalId {
        Self::push_new(graph, a, b, WhichSimpleBinaryOp::BitwiseXor, rng)
    }

    pub fn equal(
        graph: &mut MilliOpGraph,
        a: GlobalId,
        b: GlobalId,
        rng: &mut impl Rng,
    ) -> GlobalId {
        Self::push_new(graph, a, b, WhichSimpleBinaryOp::Equal, rng)
    }

    pub fn greater(
        graph: &mut MilliOpGraph,
        a: GlobalId,
        b: GlobalId,
        rng: &mut impl Rng,
    ) -> GlobalId {
        Self::push_new(graph, a, b, WhichSimpleBinaryOp::Greater, rng)
    }

    pub fn greater_or_equal(
        graph: &mut MilliOpGraph,
        a: GlobalId,
        b: GlobalId,
        rng: &mut impl Rng,
    ) -> GlobalId {
        Self::push_new(graph, a, b, WhichSimpleBinaryOp::GreaterOrEqual, rng)
    }

    pub fn less(
        graph: &mut MilliOpGraph,
        a: GlobalId,
        b: GlobalId,
        rng: &mut impl Rng,
    ) -> GlobalId {
        Self::push_new(graph, a, b, WhichSimpleBinaryOp::Less, rng)
    }

    pub fn less_or_equal(
        graph: &mut MilliOpGraph,
        a: GlobalId,
        b: GlobalId,
        rng: &mut impl Rng,
    ) -> GlobalId {
        Self::push_new(graph, a, b, WhichSimpleBinaryOp::LessOrEqual, rng)
    }

    pub fn max(graph: &mut MilliOpGraph, a: GlobalId, b: GlobalId, rng: &mut impl Rng) -> GlobalId {
        Self::push_new(graph, a, b, WhichSimpleBinaryOp::Max, rng)
    }

    pub fn min(graph: &mut MilliOpGraph, a: GlobalId, b: GlobalId, rng: &mut impl Rng) -> GlobalId {
        Self::push_new(graph, a, b, WhichSimpleBinaryOp::Min, rng)
    }
}

impl SimpleBinary {
    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.a, map);
        super::remap(&mut self.b, map);
    }
}

impl MilliOp for SimpleBinary {
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

        let a_info = known_inputs
            .get(&self.a)
            .ok_or(MilliOpGraphError::UnableToInfer)?;
        let b_info = known_inputs
            .get(&self.b)
            .ok_or(MilliOpGraphError::UnableToInfer)?;

        // If both inputs are concrete, fall back to eval.
        if a_info.as_numeric().is_some() && b_info.as_numeric().is_some() {
            let mut resolved = HashMap::new();
            resolved.insert(self.a, a_info.as_numeric().unwrap().clone());
            resolved.insert(self.b, b_info.as_numeric().unwrap().clone());
            let collected: Vec<(GlobalId, TensorInfo)> = self
                .eval(&resolved, backend)?
                .map(|(a, b)| (a, TensorInfo::from(b)))
                .collect();
            return Ok(Box::new(collected.into_iter()));
        }

        // Determine output dtype: comparison ops produce Bool, others match input.
        let out_dtype = match self.which_op {
            WhichSimpleBinaryOp::Equal
            | WhichSimpleBinaryOp::Greater
            | WhichSimpleBinaryOp::GreaterOrEqual
            | WhichSimpleBinaryOp::Less
            | WhichSimpleBinaryOp::LessOrEqual
            | WhichSimpleBinaryOp::And
            | WhichSimpleBinaryOp::Or
            | WhichSimpleBinaryOp::Xor => DType::BOOL,
            _ => a_info.dtype(),
        };

        // Try per-dim broadcast shape inference for better precision.
        let a_ranked = a_info.as_ranked();
        let b_ranked = b_info.as_ranked();
        if let (Some(a_ranked), Some(b_ranked)) = (a_ranked, b_ranked) {
            let a_dims = a_ranked.shape();
            let b_dims = b_ranked.shape();
            if let Ok(out_dims) = super::infer_multidirectional_broadcasting_shape(
                &[a_dims.clone(), b_dims.clone()], symbolic_resolver,
            ) {
                let out_info = TensorInfo::from_dtype_and_shape_scalars(out_dtype, &out_dims);
                return Ok(Box::new([(self.output, out_info)].into_iter()));
            }
        }

        // Fallback: rank-only inference.
        let a_shape = a_info.shape(symbolic_resolver);
        let b_shape = b_info.shape(symbolic_resolver);
        let out_rank =
            super::infer_multidirectional_broadcasting_rank(&[a_shape, b_shape], symbolic_resolver)?;

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
        let a = &inputs[&self.a];
        let b = &inputs[&self.b];
        let out = match self.which_op {
            WhichSimpleBinaryOp::Add => NumericTensor::<DynRank>::add(a, b, backend)?,
            WhichSimpleBinaryOp::Sub => NumericTensor::<DynRank>::sub(a, b, backend)?,
            WhichSimpleBinaryOp::Mul => NumericTensor::<DynRank>::mul(a, b, backend)?,
            WhichSimpleBinaryOp::Div => NumericTensor::<DynRank>::div(a, b, backend)?,
            WhichSimpleBinaryOp::Modulo(fmod) => {
                let is_float =
                    [DType::F64, DType::F32, DType::BF16, DType::F16].contains(&a.dtype());
                let fmod = if is_float {
                    true
                } else {
                    fmod.unwrap_or(false)
                };
                if fmod {
                    NumericTensor::<DynRank>::fmod(a, b, backend)?
                } else {
                    NumericTensor::<DynRank>::imod(a, b, backend)?
                }
            }
            WhichSimpleBinaryOp::And => NumericTensor::<DynRank>::and(a, b, backend)?,
            WhichSimpleBinaryOp::Or => NumericTensor::<DynRank>::or(a, b, backend)?,
            WhichSimpleBinaryOp::Xor => NumericTensor::<DynRank>::xor(a, b, backend)?,
            WhichSimpleBinaryOp::BitwiseAnd => {
                NumericTensor::<DynRank>::bitwise_and(a, b, backend)?
            }
            WhichSimpleBinaryOp::BitwiseOr => NumericTensor::<DynRank>::bitwise_or(a, b, backend)?,
            WhichSimpleBinaryOp::BitwiseXor => {
                NumericTensor::<DynRank>::bitwise_xor(a, b, backend)?
            }
            WhichSimpleBinaryOp::Equal => NumericTensor::<DynRank>::equal(a, b, backend)?,
            WhichSimpleBinaryOp::Greater => NumericTensor::<DynRank>::greater(a, b, backend)?,
            WhichSimpleBinaryOp::GreaterOrEqual => {
                NumericTensor::<DynRank>::greater_or_equal(a, b, backend)?
            }
            WhichSimpleBinaryOp::Less => NumericTensor::<DynRank>::less(a, b, backend)?,
            WhichSimpleBinaryOp::LessOrEqual => {
                NumericTensor::<DynRank>::less_or_equal(a, b, backend)?
            }
            WhichSimpleBinaryOp::Max => NumericTensor::<DynRank>::max(a, b, backend)?,
            WhichSimpleBinaryOp::Min => NumericTensor::<DynRank>::min(a, b, backend)?,
        };
        Ok(Box::new([(self.output, out)].into_iter()))
    }

    fn backward(
        &self,
        output_grads: &HashMap<GlobalId, GlobalId>,
        graph: &mut MilliOpGraph,
        rng: &mut impl Rng,
    ) -> Option<HashMap<GlobalId, GlobalId>> {
        let grad_output = *output_grads.get(&self.output)?;

        // Compute per-input gradients as (input_id, grad_tensor_id) pairs.
        // Using a Vec instead of HashMap so that when self.a == self.b,
        // both contributions are preserved and summed below.
        let pairs: Vec<(GlobalId, GlobalId)> = match self.which_op {
            // d/da(a+b) = 1, d/db(a+b) = 1
            WhichSimpleBinaryOp::Add => {
                vec![(self.a, grad_output), (self.b, grad_output)]
            }
            // d/da(a-b) = 1, d/db(a-b) = -1
            WhichSimpleBinaryOp::Sub => {
                let neg_grad = super::SimpleUnaryOp::neg(graph, grad_output, rng);
                vec![(self.a, grad_output), (self.b, neg_grad)]
            }
            // d/da(a*b) = b, d/db(a*b) = a
            WhichSimpleBinaryOp::Mul => {
                let grad_a = SimpleBinary::mul(graph, grad_output, self.b, rng);
                let grad_b = SimpleBinary::mul(graph, grad_output, self.a, rng);
                vec![(self.a, grad_a), (self.b, grad_b)]
            }
            // d/da(a/b) = 1/b, d/db(a/b) = -a/b^2
            WhichSimpleBinaryOp::Div => {
                let grad_a = SimpleBinary::div(graph, grad_output, self.b, rng);
                let b_sq = SimpleBinary::mul(graph, self.b, self.b, rng);
                let a_over_b_sq = SimpleBinary::div(graph, self.a, b_sq, rng);
                let neg = super::SimpleUnaryOp::neg(graph, a_over_b_sq, rng);
                let grad_b = SimpleBinary::mul(graph, grad_output, neg, rng);
                vec![(self.a, grad_a), (self.b, grad_b)]
            }
            _ => return None,
        };

        // Reduce gradients to match input shapes (un-broadcast)
        // and accumulate: if self.a == self.b, sum both gradient contributions
        let mut result = HashMap::new();
        for (input_id, grad_id) in pairs {
            let shape = super::Shape::push_new(graph, input_id, rng);
            let reduced = super::SumTo::push_new(graph, grad_id, shape, rng);
            result
                .entry(input_id)
                .and_modify(|existing: &mut GlobalId| {
                    *existing = SimpleBinary::add(graph, *existing, reduced, rng);
                })
                .or_insert(reduced);
        }
        Some(result)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pow {
    global_id: GlobalId,
    output: GlobalId,
    a: GlobalId,
    b: GlobalId,
}

impl Pow {
    pub fn push_new(
        graph: &mut MilliOpGraph,
        a: GlobalId,
        b: GlobalId,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            output,
            a,
            b,
            global_id: GlobalId::new(rng),
        };
        graph.push_op(AnyMilliOp::Pow(node));
        output
    }
}

impl Pow {
    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.a, map);
        super::remap(&mut self.b, map);
    }
}

impl MilliOp for Pow {
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

        let a_info = known_inputs
            .get(&self.a)
            .ok_or(MilliOpGraphError::UnableToInfer)?;
        let b_info = known_inputs
            .get(&self.b)
            .ok_or(MilliOpGraphError::UnableToInfer)?;

        // If both inputs are concrete, fall back to eval.
        if a_info.as_numeric().is_some() && b_info.as_numeric().is_some() {
            let mut resolved = HashMap::new();
            resolved.insert(self.a, a_info.as_numeric().unwrap().clone());
            resolved.insert(self.b, b_info.as_numeric().unwrap().clone());
            let collected: Vec<(GlobalId, TensorInfo)> = self
                .eval(&resolved, backend)?
                .map(|(a, b)| (a, TensorInfo::from(b)))
                .collect();
            return Ok(Box::new(collected.into_iter()));
        }

        // Output dtype = input dtype.
        let out_dtype = a_info.dtype();

        // Try per-dim broadcast shape inference.
        let a_ranked = a_info.as_ranked();
        let b_ranked = b_info.as_ranked();
        if let (Some(a_ranked), Some(b_ranked)) = (a_ranked, b_ranked) {
            let a_dims = a_ranked.shape();
            let b_dims = b_ranked.shape();
            if let Ok(out_dims) = super::infer_multidirectional_broadcasting_shape(
                &[a_dims.clone(), b_dims.clone()], symbolic_resolver,
            ) {
                let out_info = TensorInfo::from_dtype_and_shape_scalars(out_dtype, &out_dims);
                return Ok(Box::new([(self.output, out_info)].into_iter()));
            }
        }

        // Fallback: rank-only inference.
        let a_shape = a_info.shape(symbolic_resolver);
        let b_shape = b_info.shape(symbolic_resolver);
        let out_rank =
            super::infer_multidirectional_broadcasting_rank(&[a_shape, b_shape], symbolic_resolver)?;

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
        let out = NumericTensor::<DynRank>::pow(&inputs[&self.a], &inputs[&self.b], backend)?;
        Ok(Box::new([(self.output, out)].into_iter()))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatMul {
    global_id: GlobalId,
    output: GlobalId,
    a: GlobalId,
    b: GlobalId,
}

impl MatMul {
    pub fn push_new(
        graph: &mut MilliOpGraph,
        a: GlobalId,
        b: GlobalId,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            output,
            a,
            b,
            global_id: GlobalId::new(rng),
        };
        graph.push_op(AnyMilliOp::MatMul(node));
        output
    }
}

impl MatMul {
    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.a, map);
        super::remap(&mut self.b, map);
    }
}

impl MilliOp for MatMul {
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

        let a_info = known_inputs
            .get(&self.a)
            .ok_or(MilliOpGraphError::UnableToInfer)?;
        let b_info = known_inputs
            .get(&self.b)
            .ok_or(MilliOpGraphError::UnableToInfer)?;

        // If both inputs are concrete, fall back to eval.
        if a_info.as_numeric().is_some() && b_info.as_numeric().is_some() {
            let mut resolved = HashMap::new();
            resolved.insert(self.a, a_info.as_numeric().unwrap().clone());
            resolved.insert(self.b, b_info.as_numeric().unwrap().clone());
            let collected: Vec<(GlobalId, TensorInfo)> = self
                .eval(&resolved, backend)?
                .map(|(a, b)| (a, TensorInfo::from(b)))
                .collect();
            return Ok(Box::new(collected.into_iter()));
        }

        // MatMul output dtype matches input dtype.
        let out_dtype = a_info.dtype();

        // Compute output rank from input ranks.
        // numpy matmul: for rank >= 2, output rank = max(a_rank, b_rank).
        // Batch dims are broadcast, last two dims follow [M,K]@[K,N]->[M,N].
        let out_rank = match (a_info.rank(), b_info.rank()) {
            (
                crate::scalar_info::ScalarInfoTyped::Numeric(a_rank),
                crate::scalar_info::ScalarInfoTyped::Numeric(b_rank),
            ) => {
                let out_r = if a_rank >= 2 && b_rank >= 2 {
                    a_rank.max(b_rank)
                } else if a_rank == 1 && b_rank >= 2 {
                    // vector @ matrix: result drops the prepended dim
                    b_rank - 1
                } else if a_rank >= 2 && b_rank == 1 {
                    // matrix @ vector: result drops the appended dim
                    a_rank - 1
                } else {
                    // both rank 1: dot product -> scalar
                    0
                };
                crate::scalar_info::ScalarInfoTyped::Numeric(out_r)
            }
            _ => crate::scalar_info::ScalarInfoTyped::Symbolic(
                crate::symbolic_scalar::SymbolicScalarTyped::new(symbolic_resolver),
            ),
        };

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
        let a_input = &inputs[&self.a];
        let b_input = &inputs[&self.b];
        let accumulate_dtype = match a_input.dtype() {
            DType::BF16 | DType::F16 => Some(DType::F32),
            _ => None,
        };
        let out = NumericTensor::<DynRank>::matmul(a_input, b_input, accumulate_dtype, backend)?;
        Ok(Box::new([(self.output, out)].into_iter()))
    }

    fn backward(
        &self,
        output_grads: &HashMap<GlobalId, GlobalId>,
        graph: &mut crate::milli_graph::MilliOpGraph,
        rng: &mut impl Rng,
    ) -> Option<HashMap<GlobalId, GlobalId>> {
        let grad_output = *output_grads.get(&self.output)?;
        // d/dA (A @ B) = grad @ B^T
        let b_t = super::Transpose::push_new(graph, self.b, Some(vec![-1, -2]), rng);
        let grad_a = MatMul::push_new(graph, grad_output, b_t, rng);
        // d/dB (A @ B) = A^T @ grad
        let a_t = super::Transpose::push_new(graph, self.a, Some(vec![-1, -2]), rng);
        let grad_b = MatMul::push_new(graph, a_t, grad_output, rng);

        // Reduce gradients to match input shapes (un-broadcast batch dims)
        // and accumulate if self.a == self.b (self-matmul)
        let mut result = HashMap::new();
        for (input_id, grad_id) in [(self.a, grad_a), (self.b, grad_b)] {
            let shape = super::Shape::push_new(graph, input_id, rng);
            let reduced = super::SumTo::push_new(graph, grad_id, shape, rng);
            result
                .entry(input_id)
                .and_modify(|existing: &mut GlobalId| {
                    *existing = SimpleBinary::add(graph, *existing, reduced, rng);
                })
                .or_insert(reduced);
        }
        Some(result)
    }
}

impl Node for SimpleBinary {
    type OpKind = String;
    fn op_kind(&self) -> Self::OpKind {
        match self.which_op {
            WhichSimpleBinaryOp::Add => "Add",
            WhichSimpleBinaryOp::Sub => "Sub",
            WhichSimpleBinaryOp::Mul => "Mul",
            WhichSimpleBinaryOp::Div => "Div",
            WhichSimpleBinaryOp::Modulo(_) => "Modulo",
            WhichSimpleBinaryOp::And => "And",
            WhichSimpleBinaryOp::Or => "Or",
            WhichSimpleBinaryOp::Xor => "Xor",
            WhichSimpleBinaryOp::BitwiseAnd => "Bitwise And",
            WhichSimpleBinaryOp::BitwiseOr => "Bitwise Or",
            WhichSimpleBinaryOp::BitwiseXor => "Bitwise Xor",
            WhichSimpleBinaryOp::Equal => "Equal",
            WhichSimpleBinaryOp::Greater => "Greater",
            WhichSimpleBinaryOp::GreaterOrEqual => "Greater or Equal",
            WhichSimpleBinaryOp::Less => "Less",
            WhichSimpleBinaryOp::LessOrEqual => "Less or Equal",
            WhichSimpleBinaryOp::Max => "Max",
            WhichSimpleBinaryOp::Min => "Min",
        }
        .to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.a, self.b].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.output].into_iter())
    }
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

impl Node for Pow {
    type OpKind = String;
    fn op_kind(&self) -> Self::OpKind {
        "Pow".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.a, self.b].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.output].into_iter())
    }
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

impl Node for MatMul {
    type OpKind = String;
    fn op_kind(&self) -> Self::OpKind {
        "MatMul".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.a, self.b].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.output].into_iter())
    }
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}
