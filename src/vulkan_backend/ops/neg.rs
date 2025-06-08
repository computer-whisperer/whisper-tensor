use std::collections::BTreeMap;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::descriptor_set::layout::{DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType};
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::{PipelineLayoutCreateInfo, PushConstantRange};
use vulkano::shader::ShaderStages;
use vulkano::sync::GpuFuture;
use zerocopy::IntoBytes;
use crate::dtype::DType;
use crate::tensor_rank::{DimContainer, DimProduct, Rank};
use crate::vulkan_backend::tensor::VulkanTensor;
use crate::vulkan_backend::{VulkanError, VulkanImmediateExecutor};

#[repr(C)]
#[derive(zerocopy_derive::IntoBytes, zerocopy_derive::Immutable, bytemuck::Pod, bytemuck::Zeroable, Copy, Clone, Debug)]
struct OpMetadata {
    input_offset: u32,
    input_stride: u32,
    output_offset: u32,
    size: u32,
}

mod cs_f64 {
    vulkano_shaders::shader!{
        ty: "compute",
        src: r"
            #version 460

            layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

            layout(set = 0, binding = 0) buffer Input {double data[]; } input_0;
            layout(set = 0, binding = 1) buffer Output {double data[]; } output_0;
            layout(push_constant) uniform PushConstants { uint input_offset; uint input_stride; uint output_offset; uint size; } metadata;

            void main() {
                uint idx = gl_GlobalInvocationID.x;
                if (idx < metadata.size) {
                   output_0.data[metadata.output_offset/8 + idx] = -input_0.data[metadata.input_offset/8 + idx*metadata.input_stride];
                }
            }
        ",
    }
}

mod cs_f32 {
    vulkano_shaders::shader!{
        ty: "compute",
        src: r"
            #version 460

            layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

            layout(set = 0, binding = 0) buffer Input {float data[]; } input_0;
            layout(set = 0, binding = 1) buffer Output {float data[]; } output_0;
            layout(push_constant) uniform PushConstants { uint input_offset; uint input_stride; uint output_offset; uint size; } metadata;

            void main() {
                uint idx = gl_GlobalInvocationID.x;
                if (idx < metadata.size) {
                   output_0.data[metadata.output_offset/4 + idx] = -input_0.data[metadata.input_offset/4 + idx*metadata.input_stride];
                }
            }
        ",
    }
}

impl<R: Rank> VulkanTensor<R> {
    pub fn neg(&self, vulkan_immediate_executor: &mut VulkanImmediateExecutor) -> Result<VulkanTensor<R>, VulkanError> {

        let output_tensor = unsafe{VulkanTensor::new_uninitialized(self.shape().clone(), self.dtype(), vulkan_immediate_executor)}?;

        let shader = match self.dtype() {
            DType::F64 => {
                cs_f64::load(vulkan_immediate_executor.context.device.clone())?
            }
            DType::F32 => {
                cs_f32::load(vulkan_immediate_executor.context.device.clone())?
            }
            _ => panic!("Unsupported dtype")
        };

        let cs = shader.entry_point("main").unwrap();
        let stage = PipelineShaderStageCreateInfo::new(cs);

        let (descriptor_set, descriptor_set_layout) = vulkan_immediate_executor.get_descriptor_set_and_layout(&BTreeMap::from([
            (0, self.buffer.clone()),
            (1, output_tensor.buffer.clone())
        ]))?;

        let layout = PipelineLayout::new(
            vulkan_immediate_executor.context.device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![descriptor_set_layout],
                push_constant_ranges: vec![PushConstantRange{
                    stages: ShaderStages::COMPUTE,
                    offset: 0,
                    size: std::mem::size_of::<OpMetadata>() as u32,
                }],
                .. Default::default()
            }
        ).unwrap();

        let compute_pipeline = ComputePipeline::new(
            vulkan_immediate_executor.context.device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout.clone())
        ).unwrap();
        
        let min_input_stride = self.stride.as_slice().iter().min().unwrap();

        let op_metadata = OpMetadata {
            input_offset: self.offset as u32 + self.suballocation.offset as u32,
            input_stride: 1,
            output_offset: output_tensor.offset as u32 + output_tensor.suballocation.offset as u32,
            size: self.shape().dim_product() as u32
        };

        let mut builder = AutoCommandBufferBuilder::primary(
            vulkan_immediate_executor.context.command_buffer_allocator.clone(),
            vulkan_immediate_executor.context.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit
        ).unwrap();
        
        builder.push_constants(layout, 0, op_metadata).unwrap();

        // Note that we clone the pipeline and the set. Since they are both wrapped in an `Arc`,
        // this only clones the `Arc` and not the whole pipeline or set (which aren't cloneable
        // anyway). In this example we would avoid cloning them since this is the last time we use
        // them, but in real code you would probably need to clone them.
        builder
            .bind_pipeline_compute(compute_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                compute_pipeline.layout().clone(),
                0,
                descriptor_set,
            )
            .unwrap();

        // The command buffer only does one thing: execute the compute pipeline. This is called a
        // *dispatch* operation.
        unsafe { builder.dispatch([1024, 1, 1]) }.unwrap();

        // Finish building the command buffer by calling `build`.
        let command_buffer = builder.build().unwrap();

        // Let's execute this command buffer now.
        let future = vulkano::sync::now(vulkan_immediate_executor.context.device.clone())
            .then_execute(vulkan_immediate_executor.context.queue.clone(), command_buffer)
            .unwrap()
            // This line instructs the GPU to signal a *fence* once the command buffer has finished
            // execution. A fence is a Vulkan object that allows the CPU to know when the GPU has
            // reached a certain point. We need to signal a fence here because below we want to block
            // the CPU until the GPU has reached that point in the execution.
            .then_signal_fence_and_flush()
            .unwrap();

        // Blocks execution until the GPU has finished the operation. This method only exists on the
        // future that corresponds to a signalled fence. In other words, this method wouldn't be
        // available if we didn't call `.then_signal_fence_and_flush()` earlier. The `None` parameter
        // is an optional timeout.
        //
        // Note however that dropping the `future` variable (with `drop(future)` for example) would
        // block execution as well, and this would be the case even if we didn't call
        // `.then_signal_fence_and_flush()`. Therefore the actual point of calling
        // `.then_signal_fence_and_flush()` and `.wait()` is to make things more explicit. In the
        // future, if the Rust language gets linear types vulkano may get modified so that only
        // fence-signalled futures can get destroyed like this.
        future.wait(None).unwrap();
        Ok(output_tensor)
    }
}