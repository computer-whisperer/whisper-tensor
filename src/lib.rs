use prost::Message;
use vulkano::{VulkanError, VulkanLibrary};
mod vulkan_context;
mod symbolic_graph;

pub mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

#[derive(Debug)]
pub enum Error {
    GeneralError,
    NoSuitableVulkanDeviceError,
    NoSuitableVulkanQueueError,
    ProtobufError(prost::DecodeError),
    VulkanError(VulkanError),
    ValidatedVulkanError(vulkano::Validated<VulkanError>),
    VulkanLoadingError(vulkano::LoadingError),
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl core::error::Error for Error{
    fn cause(&self) -> Option<&dyn core::error::Error> {
        match self {
            Error::ProtobufError(x) => Some(x),
            Error::VulkanError(x) => Some(x),
            Error::ValidatedVulkanError(x) => Some(x),
            Error::VulkanLoadingError(x) => Some(x),
            _ => None,
        }
    }
}


pub fn build(onnx_data: &[u8]) -> Result<(), Error> {
    println!("onnx data is {} bytes long", onnx_data.len());
    
    let model = onnx::ModelProto::decode(onnx_data).map_err(|x| Error::ProtobufError(x))?;
    //println!("{:?}", model);
    
    Ok(())
}