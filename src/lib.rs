use num_traits::Float;
use serde::{Deserialize, Serialize};
pub mod symbolic_graph;
pub mod numeric_tensor;
pub mod dtype;
pub mod sampler;
pub mod language_model;
pub mod tokenizer;
pub mod backends;
pub mod numeric_scalar;
pub mod tensor_rank;
pub mod numeric_tensor_typed;
pub mod model;
pub mod super_graph;
pub mod milli_graph;
pub mod scalar_info;
pub mod tensor_info;
pub mod symbolic_scalar;
pub mod interfaces;

pub use tensor_rank::DynRank;


pub mod onnx {
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
    Tanh
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
            TrigOp::Tanh => x.tanh()
        }
    }
}


