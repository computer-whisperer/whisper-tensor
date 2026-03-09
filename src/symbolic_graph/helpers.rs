//! Ergonomic helpers for programmatic SymbolicGraph construction.
//!
//! These methods add high-level `push_*` operations to [`SymbolicGraphMutator`],
//! each creating an output tensor, wiring the operation, and returning the
//! output's [`GlobalId`]. The `name` parameter becomes both the output tensor
//! name and the operation name.
//!
//! For operations that need to target a pre-existing output tensor (e.g. for
//! explicit naming of graph outputs), use the `push_*_into` variants.

use super::ops::*;
use super::{SymbolicGraphMutator, TensorType};
use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::dtype::DType;
use crate::graph::GlobalId;
use rand::Rng;

impl SymbolicGraphMutator {
    // ── internal helpers ─────────────────────────────────────────────

    fn push_intermediate(&mut self, name: &str, rng: &mut impl Rng) -> GlobalId {
        self.push_unknown_tensor(name, TensorType::Intermediate, rng)
    }

    fn push_op(&mut self, name: &str, op: AnyOperation, rng: &mut impl Rng) {
        self.push_operation(Some(name.to_string()), op, rng);
    }

    // ── binary operations ────────────────────────────────────────────

    fn push_binary(
        &mut self,
        name: &str,
        a: GlobalId,
        b: GlobalId,
        which: WhichBinaryOperation,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let out = self.push_intermediate(name, rng);
        self.push_binary_into(name, a, b, which, out, rng);
        out
    }

    fn push_binary_into(
        &mut self,
        name: &str,
        a: GlobalId,
        b: GlobalId,
        which: WhichBinaryOperation,
        out: GlobalId,
        rng: &mut impl Rng,
    ) {
        let op = BinaryOperation::new(a, b, out, which, rng);
        self.push_op(name, AnyOperation::Binary(op), rng);
    }

    pub fn push_matmul(
        &mut self,
        name: &str,
        a: GlobalId,
        b: GlobalId,
        rng: &mut impl Rng,
    ) -> GlobalId {
        self.push_binary(name, a, b, WhichBinaryOperation::MatMul, rng)
    }

    pub fn push_matmul_into(
        &mut self,
        name: &str,
        a: GlobalId,
        b: GlobalId,
        out: GlobalId,
        rng: &mut impl Rng,
    ) {
        self.push_binary_into(name, a, b, WhichBinaryOperation::MatMul, out, rng);
    }

    pub fn push_add(
        &mut self,
        name: &str,
        a: GlobalId,
        b: GlobalId,
        rng: &mut impl Rng,
    ) -> GlobalId {
        self.push_binary(name, a, b, WhichBinaryOperation::Add, rng)
    }

    pub fn push_mul(
        &mut self,
        name: &str,
        a: GlobalId,
        b: GlobalId,
        rng: &mut impl Rng,
    ) -> GlobalId {
        self.push_binary(name, a, b, WhichBinaryOperation::Mul, rng)
    }

    pub fn push_div(
        &mut self,
        name: &str,
        a: GlobalId,
        b: GlobalId,
        rng: &mut impl Rng,
    ) -> GlobalId {
        self.push_binary(name, a, b, WhichBinaryOperation::Div, rng)
    }

    // ── unary operations ─────────────────────────────────────────────

    fn push_unary(
        &mut self,
        name: &str,
        input: GlobalId,
        which: WhichUnaryOperation,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let out = self.push_intermediate(name, rng);
        let op = UnaryOperation::new(input, out, which, rng);
        self.push_op(name, AnyOperation::Unary(op), rng);
        out
    }

    pub fn push_sigmoid(
        &mut self,
        name: &str,
        input: GlobalId,
        rng: &mut impl Rng,
    ) -> GlobalId {
        self.push_unary(name, input, WhichUnaryOperation::Sigmoid, rng)
    }

    pub fn push_softmax(
        &mut self,
        name: &str,
        input: GlobalId,
        axis: i64,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let out = self.push_intermediate(name, rng);
        let op = SoftmaxOperation::new(input, out, Some(axis), rng);
        self.push_op(name, AnyOperation::Softmax(op), rng);
        out
    }

    // ── shape operations ─────────────────────────────────────────────

    /// Reshape with an inline shape constant (ONNX semantics: 0 = copy dim, -1 = infer).
    pub fn push_reshape(
        &mut self,
        name: &str,
        input: GlobalId,
        shape: &[i64],
        rng: &mut impl Rng,
    ) -> GlobalId {
        let shape_tensor = self.push_constant_tensor(
            NDArrayNumericTensor::from_vec_shape(shape.to_vec(), &vec![shape.len() as u64])
                .unwrap(),
            None,
            rng,
        );
        let out = self.push_intermediate(name, rng);
        let op = ReshapeOperation::new(input, shape_tensor, out, rng);
        self.push_op(name, AnyOperation::Reshape(op), rng);
        out
    }

    pub fn push_transpose(
        &mut self,
        name: &str,
        input: GlobalId,
        perm: &[i64],
        rng: &mut impl Rng,
    ) -> GlobalId {
        let out = self.push_intermediate(name, rng);
        let op = TransposeOperation::new(input, out, Some(perm.to_vec()), rng);
        self.push_op(name, AnyOperation::Transpose(op), rng);
        out
    }

    pub fn push_gather(
        &mut self,
        name: &str,
        data: GlobalId,
        indices: GlobalId,
        axis: i64,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let out = self.push_intermediate(name, rng);
        let op = GatherOperation::new(data, indices, out, axis, rng);
        self.push_op(name, AnyOperation::Gather(op), rng);
        out
    }

    pub fn push_concat(
        &mut self,
        name: &str,
        inputs: Vec<GlobalId>,
        axis: i64,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let out = self.push_intermediate(name, rng);
        self.push_concat_into(name, inputs, axis, out, rng);
        out
    }

    pub fn push_concat_into(
        &mut self,
        name: &str,
        inputs: Vec<GlobalId>,
        axis: i64,
        out: GlobalId,
        rng: &mut impl Rng,
    ) {
        let op = ConcatOperation::new(inputs, out, axis, rng);
        self.push_op(name, AnyOperation::Concat(op), rng);
    }

    /// Extract shape (or a slice of it) as a 1D i64 tensor.
    pub fn push_shape(
        &mut self,
        name: &str,
        input: GlobalId,
        start: Option<i64>,
        end: Option<i64>,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let out = self.push_intermediate(name, rng);
        let op = ShapeOperation::new(input, out, start, end, rng);
        self.push_op(name, AnyOperation::Shape(op), rng);
        out
    }

    pub fn push_cast(
        &mut self,
        name: &str,
        input: GlobalId,
        to: DType,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let out = self.push_intermediate(name, rng);
        let op = CastOperation::new(input, out, to, rng);
        self.push_op(name, AnyOperation::Cast(op), rng);
        out
    }

    // ── normalization ────────────────────────────────────────────────

    pub fn push_rms_norm(
        &mut self,
        name: &str,
        input: GlobalId,
        scale: GlobalId,
        epsilon: f32,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let out = self.push_intermediate(name, rng);
        let op = RMSNormalizationOperation::new(input, scale, None, out, epsilon, rng);
        self.push_op(name, AnyOperation::RMSNormalization(op), rng);
        out
    }

    pub fn push_rotary_embedding(
        &mut self,
        name: &str,
        data: GlobalId,
        cos: GlobalId,
        sin: GlobalId,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let out = self.push_intermediate(name, rng);
        let op = RotaryEmbeddingOperation::new(
            data, cos, sin, None, // no position_ids
            out, false, // not interleaved
            None, // num_heads
            0,    // full head rotation
            rng,
        );
        self.push_op(name, AnyOperation::RotaryEmbedding(op), rng);
        out
    }

    // ── composites ───────────────────────────────────────────────────

    /// SiLU activation: x * sigmoid(x). Synthesized since there is no SiLU primitive.
    pub fn push_silu(
        &mut self,
        name: &str,
        input: GlobalId,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let sig = self.push_sigmoid(&format!("{name}/sigmoid"), input, rng);
        self.push_mul(name, input, sig, rng)
    }

    /// Linear layer: input @ weight^T. Weight is [out_features, in_features].
    pub fn push_linear(
        &mut self,
        name: &str,
        input: GlobalId,
        weight: GlobalId,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let wt = self.push_transpose(&format!("{name}/wt"), weight, &[1, 0], rng);
        self.push_matmul(name, input, wt, rng)
    }

    /// Linear layer targeting a pre-existing output tensor.
    pub fn push_linear_into(
        &mut self,
        name: &str,
        input: GlobalId,
        weight: GlobalId,
        out: GlobalId,
        rng: &mut impl Rng,
    ) {
        let wt = self.push_transpose(&format!("{name}/wt"), weight, &[1, 0], rng);
        self.push_matmul_into(name, input, wt, out, rng);
    }

    /// Quantized linear layer: input @ dequant(weight)^T.
    ///
    /// Weight is a packed (quantized) tensor with shape [out_features, in_features].
    /// Dequantization and transpose are fused into a single QuantMatMul operation.
    pub fn push_quant_linear(
        &mut self,
        name: &str,
        input: GlobalId,
        weight: GlobalId,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let out = self.push_intermediate(name, rng);
        self.push_quant_linear_into(name, input, weight, out, rng);
        out
    }

    /// Quantized linear layer targeting a pre-existing output tensor.
    pub fn push_quant_linear_into(
        &mut self,
        name: &str,
        input: GlobalId,
        weight: GlobalId,
        out: GlobalId,
        rng: &mut impl Rng,
    ) {
        let op = QuantMatMulOperation::new(input, weight, out, rng);
        self.push_op(name, AnyOperation::QuantMatMul(op), rng);
    }
}
