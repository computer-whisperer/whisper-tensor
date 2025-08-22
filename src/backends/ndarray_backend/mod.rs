pub mod conversions;
mod full_generic_matmul;
pub mod numeric_tensor;
pub mod ops;
pub mod specialized_matmul;

pub use numeric_tensor::{NDArrayNumericTensor, NDArrayNumericTensorError};
