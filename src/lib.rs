use std::collections::HashMap;
use num_traits::Float;

use prost::{Message};
use serde::{Deserialize, Serialize};
use crate::numeric_tensor::{NumericTensor};

pub mod symbolic_graph;
pub mod numeric_tensor;
pub mod dtype;
pub mod sampler;
pub mod language_model;
pub mod tokenizer;
pub mod eval_backend;
pub mod ndarray_backend;
mod onnx_testing;
pub mod numeric_scalar;
pub mod tensor_rank;
pub mod numeric_tensor_typed;

pub mod model;

pub use ndarray_backend::NDArrayNumericTensor;
pub use tensor_rank::DynRank;

#[cfg(feature = "ort")]
pub mod ort_backend;
#[cfg(feature = "onnx-reference")]
mod onnx_reference_backend;
#[cfg(feature = "candle")]
mod candle_backend;
#[cfg(feature = "vulkan")]
pub mod vulkan_backend;

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


