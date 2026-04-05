use serde::{Deserialize, Serialize};
pub mod compiler;
pub mod graph;
pub mod graph_format;
pub mod interfaces;
pub mod loader;
pub mod metadata;
pub mod milli_graph;
pub mod model;
pub mod nano_graph;
pub mod npy;
pub mod numeric_dtype;
pub mod numeric_scalar;
pub mod numeric_tensor;
pub mod phonemization;
pub mod pool;
pub mod pth;
pub mod range_map;
pub mod scalar_info;
pub mod scalar_ops;
pub mod super_graph;
pub mod symbolic_graph;
pub mod symbolic_scalar;
pub mod tensor_info;
pub mod tensor_rank;
pub mod test_set;
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
