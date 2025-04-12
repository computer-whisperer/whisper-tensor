use crate::symbolic_graph::SymbolicGraph;

mod vulkan_context;
mod symbolic_graph;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("ONNX decoding error: {0}")]
    ONNXDecodingError(#[from] symbolic_graph::ONNXDecodingError)
}

pub fn build(onnx_data: &[u8]) -> Result<(), Error> {
    println!("onnx data is {} bytes long", onnx_data.len());
    
    let symbolic_graph = SymbolicGraph::from_onnx_bytes(onnx_data).map_err(|x| Error::ONNXDecodingError(x))?;
    
    Ok(())
}