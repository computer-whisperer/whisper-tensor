use std::collections::BTreeMap;
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::descriptor_set::layout::{DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType};
use vulkano::memory::allocator::{AllocationCreateInfo, FreeListAllocator, MemoryTypeFilter, Suballocator};
use vulkano::pipeline::layout::{PipelineDescriptorSetLayoutCreateInfo, PipelineLayoutCreateInfo};
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::shader::ShaderStages;
use vulkano::sync::GpuFuture;
use whisper_tensor::vulkan_backend::VulkanContext;


mod cs {
    vulkano_shaders::shader!{
        ty: "compute",
        src: r"
            #version 460

            layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

            layout(set = 0, binding = 0) buffer Data {
                uint data[];
            } buf;

            void main() {
                uint idx = gl_GlobalInvocationID.x;
                buf.data[idx] *= 12;
            }
        ",
    }
}

fn main() {
    let vk_context = VulkanContext::new().unwrap();



    let shader = cs::load(vk_context.device.clone()).unwrap();
    let cs = shader.entry_point("main").unwrap();
    let stage = PipelineShaderStageCreateInfo::new(cs);

    let descriptor_set_bindings = {
        let mut ret = BTreeMap::new();
        ret.insert(
            0,
            DescriptorSetLayoutBinding {
                stages: ShaderStages::COMPUTE,
                ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer)
            }
        );
        ret
    };
    let descriptor_set_layout = DescriptorSetLayout::new(vk_context.device.clone(), DescriptorSetLayoutCreateInfo {
        bindings: descriptor_set_bindings,
        ..Default::default()
    }).unwrap();

    let layout = PipelineLayout::new(
        vk_context.device.clone(),
        PipelineLayoutCreateInfo {
            set_layouts: vec![descriptor_set_layout.clone()],
            .. Default::default()
        }
    ).unwrap();
    
    let compute_pipeline = ComputePipeline::new(
        vk_context.device.clone(),
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout)
    ).unwrap();
    
    let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(vk_context.device.clone(), Default::default()));
    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(vk_context.device.clone(), Default::default()));

    // We start by creating the buffer that will store the data.
    let data_buffer = Buffer::from_iter(
        vk_context.memory_allocator,
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        // Iterator that produces the data.
        0..65536u32,
    ).unwrap();
    


    let descriptor_set = DescriptorSet::new(
        descriptor_set_allocator.clone(),
        descriptor_set_layout.clone(),
        [WriteDescriptorSet::buffer(0, data_buffer.clone())], // 0 is the binding
        [],
    ).unwrap();
    

    let mut builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator,
        vk_context.queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit
    ).unwrap();
    
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
    let future = vulkano::sync::now(vk_context.device)
        .then_execute(vk_context.queue, command_buffer)
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

    // Now that the GPU is done, the content of the buffer should have been modified. Let's check
    // it out. The call to `read()` would return an error if the buffer was still in use by the
    // GPU.
    let data_buffer_content = data_buffer.read().unwrap();
    for n in 0..65536u32 {
        assert_eq!(data_buffer_content[n as usize], n * 12);
    }

    println!("Success");
}