use std::collections::BTreeMap;
use rspirv::binary::Assemble;
use rspirv::dr::Operand;
use rspirv::dr::Operand::LiteralBit32;
use rspirv::spirv;
use rspirv::spirv::{BuiltIn, Decoration, ExecutionMode, ExecutionModel, SelectionControl, StorageClass};
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
    pub fn unary(&self, vulkan_immediate_executor: &mut VulkanImmediateExecutor) -> Result<VulkanTensor<R>, VulkanError> {
        
        // Generate spirv
        let mut b = rspirv::dr::Builder::new();
        b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);
        let void = b.type_void();
        let data_type = match self.dtype() {
            DType::F64 => b.type_float(64),
            DType::F32 => b.type_float(32),
            DType::BF16 => b.type_float(16),
            DType::F16 => b.type_float(16),
            DType::I64 => b.type_int(64, 1),
            DType::I32 => b.type_int(32, 1),
            DType::U64 => b.type_int(64, 0),
            DType::U32 => b.type_int(32, 0),
            DType::U8 => b.type_int(8, 0),
            DType::I8 => b.type_int(8, 0),
            DType::BOOL => b.type_bool(),
            _ => panic!("Unsupported dtype"),
        };
        let data_type_array = b.type_runtime_array(data_type);
        let data_type_array_ptr = b.type_pointer(None, StorageClass::StorageBuffer, data_type_array);
        let input_0_var = b.variable(data_type_array_ptr, None, StorageClass::StorageBuffer, None);
        b.decorate(input_0_var, Decoration::DescriptorSet, [LiteralBit32(0)]);
        b.decorate(input_0_var, Decoration::Binding, [LiteralBit32(0)]);

        let output_0_var = b.variable(data_type_array_ptr, None, StorageClass::StorageBuffer, None);
        b.decorate(output_0_var, Decoration::DescriptorSet, [LiteralBit32(0)]);
        b.decorate(output_0_var, Decoration::Binding, [LiteralBit32(1)]);

        let u32_t = b.type_int(32, 0);
        let pc_struct = b.type_struct([u32_t, u32_t, u32_t, u32_t]);

        b.decorate(pc_struct, Decoration::Block, []);
        for (i, off) in [0,4,8,12].iter().enumerate() {
            b.member_decorate(pc_struct, i as u32, Decoration::Offset,
                                    [Operand::LiteralBit32(*off)]);
        }

        let pc_ptr = b.type_pointer(None, StorageClass::PushConstant, pc_struct);
        let metadata_var = b.variable(pc_ptr, None, StorageClass::PushConstant, None);

        let vec3u32_t  = b.type_vector(u32_t, 3);
        let in_ptr = b.type_pointer(None, StorageClass::Input, vec3u32_t);
        let gid    = b.variable(in_ptr, None, StorageClass::Input, None);
        b.decorate(gid, Decoration::BuiltIn,[Operand::BuiltIn(BuiltIn::GlobalInvocationId)]);

        let voidf = b.type_function(void, vec![void]);
        let main_fn = b.begin_function(void, None, (spirv::FunctionControl::DONT_INLINE), voidf).unwrap();
        b.entry_point(ExecutionModel::GLCompute, main_fn, "main", []);
        b.execution_mode(main_fn, ExecutionMode::LocalSize, [1, 1, 1]);
        b.begin_block(None).unwrap();

        /* Constants */
        let c0 = b.constant_bit32(u32_t, 0);
        let c1 = b.constant_bit32(u32_t, 1);
        let c2 = b.constant_bit32(u32_t, 2);
        let c3 = b.constant_bit32(u32_t, 3);
        let c4 = b.constant_bit32(u32_t, 4);

        /* idx = gl_GlobalInvocationID.x */
        let gid  = b.load(vec3u32_t, None, gid.clone(), None, []).unwrap();
        let idx  = b.composite_extract(u32_t, None, gid, [0u32]).unwrap();

        /* size  = metadata.size */
        let u32_pc_ptr_t = b.type_pointer(None, StorageClass::PushConstant, u32_t);
        let size_ptr = b.access_chain(u32_pc_ptr_t, None, metadata_var,
                                      [c0 /* struct */, c3]).unwrap();
        let size_val = b.load(u32_t, None, size_ptr, None, []).unwrap();

        /* cmp & branch */
        
        let bool_type = b.type_bool();
        let cmp = b.u_less_than(bool_type, None, idx, size_val).unwrap();
        let merge_blk = b.id();
        let then_blk  = b.id();

        b.selection_merge(merge_blk, SelectionControl::NONE).unwrap();
        b.branch_conditional(cmp, then_blk, merge_blk, None).unwrap();

        /* ---- THEN block ---- */
        b.begin_block(None).unwrap();

        /* input_index = input_offset/4 + idx*stride */
        let inoff_ptr = b.access_chain(u32_pc_ptr_t, None, metadata_var,
                                       [c0, c0]).unwrap();
        let inoff_val = b.load(u32_t, None, inoff_ptr, None, []).unwrap();
        let inoff_div4= b.u_div(u32_t, None, inoff_val, c4).unwrap();

        let stride_ptr= b.access_chain(u32_pc_ptr_t, None, metadata_var,
                                       [c0, c1]).unwrap();
        let stride_val= b.load(u32_t, None, stride_ptr, None, []).unwrap();
        let mul       = b.i_mul(u32_t, None, idx, stride_val).unwrap();
        let in_index  = b.i_add(u32_t, None, inoff_div4, mul).unwrap();

        /* value = -input[in_index] */
        let data_input_ptr_t = b.type_pointer(None, StorageClass::Input, data_type);
        let in_ptr = b.access_chain(data_input_ptr_t, None, input_0_var, [c0, in_index]).unwrap();
        let in_val = b.load(data_type, None, in_ptr, None, []).unwrap();
        let neg_val= b.f_negate(data_type, None, in_val).unwrap();

        /* out_index = output_offset/4 + idx */
        let outoff_ptr= b.access_chain(u32_pc_ptr_t, None, metadata_var,
                                       [c0, c2]).unwrap();
        let outoff_val= b.load(u32_t, None, outoff_ptr, None, []).unwrap();
        let outoff_div4= b.u_div(u32_t, None, outoff_val, c4).unwrap();
        let out_index = b.i_add(u32_t, None, outoff_div4, idx).unwrap();

        /* store */
        let out_ptr = b.access_chain(data_type, None, output_0_var, [c0, out_index]).unwrap();
        b.store(out_ptr, neg_val, None, []).unwrap();

        /* branch to merge */
        b.branch(merge_blk).unwrap();

        /* ---- MERGE block ---- */
        b.begin_block(None).unwrap();
        b.ret().unwrap();                       // OpReturn
        b.end_function().unwrap();              // OpFunction
        
        let module = b.module();
        let code = module.assemble();

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

    pub fn neg(&self, vulkan_immediate_executor: &mut VulkanImmediateExecutor) -> Result<VulkanTensor<R>, VulkanError> {
        self.unary(vulkan_immediate_executor)
    }
    
    pub fn abs(&self, vulkan_immediate_executor: &mut VulkanImmediateExecutor) -> Result<VulkanTensor<R>, VulkanError> {
        unimplemented!()
    }
}