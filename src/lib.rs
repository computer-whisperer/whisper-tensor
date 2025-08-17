use num_traits::Float;
use serde::{Deserialize, Serialize};
pub mod backends;
pub mod dtype;
pub mod interfaces;
pub mod milli_graph;
pub mod model;
pub mod numeric_scalar;
pub mod numeric_tensor;
pub mod numeric_tensor_typed;
pub mod scalar_info;
pub mod super_graph;
pub mod symbolic_graph;
pub mod symbolic_scalar;
pub mod tensor_info;
pub mod tensor_rank;
pub mod tokenizer;

pub use tensor_rank::DynRank;

pub mod onnx {
    #![allow(clippy::all)]
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize, strum_macros::Display)]
pub enum TrigOp {
    Asin,
    Asinh,
    Acos,
    Acosh,
    Atan,
    Atanh,
    Sin,
    Sinh,
    Cos,
    Cosh,
    Tan,
    Tanh,
}

impl TrigOp {
    fn apply<F: Float>(&self, x: F) -> F {
        match self {
            TrigOp::Asin => x.asin(),
            TrigOp::Asinh => x.asinh(),
            TrigOp::Acos => x.acos(),
            TrigOp::Acosh => x.acosh(),
            TrigOp::Atan => x.atan(),
            TrigOp::Atanh => x.atanh(),
            TrigOp::Sin => x.sin(),
            TrigOp::Sinh => x.sinh(),
            TrigOp::Cos => x.cos(),
            TrigOp::Cosh => x.cosh(),
            TrigOp::Tan => x.tan(),
            TrigOp::Tanh => x.tanh(),
        }
    }

    fn get_name(&self) -> &'static str {
        match self {
            TrigOp::Asin => "Asin",
            TrigOp::Asinh => "Asinh",
            TrigOp::Acos => "Acos",
            TrigOp::Acosh => "Acosh",
            TrigOp::Atan => "Atan",
            TrigOp::Atanh => "Atanh",
            TrigOp::Sin => "Sin",
            TrigOp::Sinh => "Sinh",
            TrigOp::Cos => "Cos",
            TrigOp::Cosh => "Cosh",
            TrigOp::Tan => "Tan",
            TrigOp::Tanh => "Tanh",
        }
    }
}
