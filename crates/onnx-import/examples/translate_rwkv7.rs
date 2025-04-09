use std::fs::File;
use std::io::Write;
use std::path::Path;

fn main() {
    let file_in = Path::new("/mnt/secondary/temp-latest-training-models/RWKV7-G1-1.5B-50%trained-20250330-ctx4k.pth");
    let onnx_out = Path::new("out.onnx");
    let bin_out = Path::new("out.bin");
    let onnx_data = onnx_import::rwkv7::translate(file_in, Some(bin_out)).unwrap();
    File::create(onnx_out).unwrap().write_all(&onnx_data).unwrap();
}