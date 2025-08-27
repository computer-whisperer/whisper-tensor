use whisper_tensor::backends::eval_backend::EvalBackend;
use whisper_tensor::backends::ndarray_backend::NDArrayNumericTensor;
use whisper_tensor::backends::vulkan_backend::{VulkanContext, VulkanImmediateExecutor};
use whisper_tensor::dtype::DType;
use whisper_tensor::numeric_tensor::NumericTensor;

fn main() {
    tracing_subscriber::fmt::init();

    let vulkan_context = VulkanContext::new().unwrap();
    let mut vulkan_runtime = VulkanImmediateExecutor::new(vulkan_context).unwrap();

    let mut eval_backend = EvalBackend::Vulkan(&mut vulkan_runtime);

    let start_data = vec![1.0f64, -2.0, 3.0, -4.0];

    let start_tensor = NumericTensor::NDArray(NDArrayNumericTensor::from(start_data).to_dyn());

    //let vulkan_tensor = VulkanTensor::from_ndarray(start_tensor, &mut vulkan_runtime).unwrap();
    //let returned_tensor = vulkan_tensor.to_ndarray();

    let _v = start_tensor.cast(DType::BF16, &mut eval_backend).unwrap();

    let v1 = start_tensor.neg(&mut eval_backend).unwrap();

    let end_tensor = v1.to_ndarray().unwrap();

    println!("{end_tensor:?}");
}
