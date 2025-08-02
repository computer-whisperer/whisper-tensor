use std::io::Result;
use std::path::Path;

fn main() -> Result<()> {
    let onnx_dir = Path::new("libs/onnx/onnx");
    prost_build::compile_protos(&[onnx_dir.join("onnx.proto3")], &[onnx_dir])?;
    Ok(())
}
