use std::any::Any;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::hash::Hash;
use std::sync::{Arc, Mutex};
use rspirv::binary::Assemble;
use rspirv::dr::{InsertPoint, Operand};
use rspirv::dr::Operand::LiteralBit32;
use rspirv::spirv;
use rspirv::spirv::{BuiltIn, Capability, Decoration, ExecutionMode, ExecutionModel, GLOp, Op, SelectionControl, StorageClass};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::{PipelineLayoutCreateInfo, PushConstantRange};
use vulkano::shader::{ShaderModule, ShaderModuleCreateInfo, ShaderStages};
use vulkano::sync::GpuFuture;
use zerocopy::IntoBytes;
use crate::dtype::DType;
use crate::tensor_rank::{DimContainer, DimProduct, Rank};
use crate::TrigOp;
use crate::vulkan_backend::tensor::VulkanTensor;
use crate::vulkan_backend::{VulkanError, VulkanImmediateExecutor};
use crate::vulkan_backend::spirv_helpers::{cast_bf16_to_f32, cast_f32_to_bf16, get_spirv_datatype};

impl<R: Rank> VulkanTensor<R> {
    
    fn build_unary_pipeline(
        vulkan_immediate_executor: &mut VulkanImmediateExecutor,
        input_dtype: DType,
        output_dtype: DType,
        rank: usize,
        op: fn(&mut rspirv::dr::Builder, rspirv::spirv::Word, DType) -> Result<rspirv::spirv::Word, VulkanError>
    ) -> Result<(Arc<PipelineLayout>, Arc<ComputePipeline>), VulkanError>
    {
        let mut b = rspirv::dr::Builder::new();
        b.capability(Capability::Shader);
        b.capability(Capability::Float64);
        b.capability(Capability::Float16);
        b.capability(Capability::Int16);
        b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);
        let void = b.type_void();

        let input_data_type = match input_dtype {
            DType::BF16 => b.type_int(16, 0),
            _ => get_spirv_datatype(&mut b, input_dtype)?
        };

        let output_data_type = match output_dtype {
            DType::BF16 => b.type_int(16, 0),
            _ => get_spirv_datatype(&mut b, input_dtype)?
        };

        let input_data_type_array = b.type_runtime_array(input_data_type);
        b.decorate(input_data_type_array, Decoration::ArrayStride, [Operand::LiteralBit32(input_dtype.size() as u32)]);
        let input_data_type_array_struct = b.type_struct([input_data_type_array]);
        b.decorate(input_data_type_array_struct, Decoration::Block, []);
        b.member_decorate(input_data_type_array_struct, 0, Decoration::Offset, [Operand::LiteralBit32(0)]);
        let input_sb_ptr  = b.type_pointer(None, StorageClass::StorageBuffer, input_data_type_array_struct);
        let input_0_var = b.variable(input_sb_ptr, None, StorageClass::StorageBuffer, None);
        b.decorate(input_0_var, Decoration::DescriptorSet, [LiteralBit32(0)]);
        b.decorate(input_0_var, Decoration::Binding, [LiteralBit32(0)]);

        let output_data_type_array = b.type_runtime_array(output_data_type);
        b.decorate(output_data_type_array, Decoration::ArrayStride, [Operand::LiteralBit32(input_dtype.size() as u32)]);
        let output_data_type_array_struct = b.type_struct([output_data_type_array]);
        b.decorate(output_data_type_array_struct, Decoration::Block, []);
        b.member_decorate(output_data_type_array_struct, 0, Decoration::Offset, [Operand::LiteralBit32(0)]);
        let output_sb_ptr  = b.type_pointer(None, StorageClass::StorageBuffer, input_data_type_array_struct);
        let output_0_var = b.variable(output_sb_ptr, None, StorageClass::StorageBuffer, None);
        b.decorate(output_0_var, Decoration::DescriptorSet, [LiteralBit32(0)]);
        b.decorate(output_0_var, Decoration::Binding, [LiteralBit32(1)]);

        let u32_t = b.type_int(32, 0);
        // Data:
        // input_offset,
        // output_offset,
        // num_elements,
        // input_output_shape (RANK u32s)
        // input_stride (RANK u32s)
        // output_stride (RANK u32s)
        let metadata_value_count = rank*3 + 3;
        let pc_struct = {
            let mut v = Vec::new();
            v.resize(metadata_value_count, u32_t);
            b.type_struct(v)
        };

        b.decorate(pc_struct, Decoration::Block, []);
        for i in 0..metadata_value_count {
            b.member_decorate(pc_struct, i as u32, Decoration::Offset,
                              [Operand::LiteralBit32((i*4) as u32)]);
        }

        let pc_ptr = b.type_pointer(None, StorageClass::PushConstant, pc_struct);
        let metadata_var = b.variable(pc_ptr, None, StorageClass::PushConstant, None);

        let vec3u32_t  = b.type_vector(u32_t, 3);
        let in_ptr = b.type_pointer(None, StorageClass::Input, vec3u32_t);
        let gid    = b.variable(in_ptr, None, StorageClass::Input, None);
        b.decorate(gid, Decoration::BuiltIn,[Operand::BuiltIn(BuiltIn::GlobalInvocationId)]);

        let voidf = b.type_function(void, []);
        let main_fn = b.begin_function(void, None, (spirv::FunctionControl::DONT_INLINE), voidf).unwrap();
        b.entry_point(ExecutionModel::GLCompute, main_fn, "main", [gid, metadata_var, output_0_var, input_0_var]);
        b.execution_mode(main_fn, ExecutionMode::LocalSize, [64, 1, 1]);
        b.begin_block(None).unwrap();

        /* Constants */
        let c0 = b.constant_bit32(u32_t, 0);

        /* idx = gl_GlobalInvocationID.x */
        let gid  = b.load(vec3u32_t, None, gid.clone(), None, []).unwrap();
        let idx  = b.composite_extract(u32_t, None, gid, [0u32]).unwrap();

        /* size  = metadata.size */
        let push_constant_vals = {
            let mut vals = Vec::new();
            let u32_pc_ptr_t = b.type_pointer(None, StorageClass::PushConstant, u32_t);
            for i in 0..metadata_value_count {
                let c_i = b.constant_bit32(u32_t, i as u32);
                let size_ptr = b.access_chain(u32_pc_ptr_t, None, metadata_var, [c_i]).unwrap();
                vals.push(b.load(u32_t, None, size_ptr, None, []).unwrap());
            }
            vals
        };

        let input_offset = push_constant_vals[0];
        let output_offset = push_constant_vals[1];
        let num_elements = push_constant_vals[2];
        let io_shape = &push_constant_vals[3..3+rank];
        let input_strides = &push_constant_vals[3+rank..3+rank+rank];
        let output_strides = &push_constant_vals[3+rank+rank..3+rank+rank+rank];

        /* cmp & branch */

        let bool_type = b.type_bool();
        let cmp = b.u_less_than(bool_type, None, idx, num_elements).unwrap();
        let merge_blk = b.id();
        let then_blk  = b.id();

        b.selection_merge(merge_blk, SelectionControl::NONE).unwrap();
        b.branch_conditional(cmp, then_blk, merge_blk, None).unwrap();

        /* ---- THEN block ---- */
        b.begin_block(Some(then_blk)).unwrap();

        // Calculate index
        let mut remaining_idx = idx;
        let index = {
            let mut v = vec![];
            for i in 0..rank {
                let shape_val = io_shape[i];
                let rem = b.u_mod(u32_t, None, remaining_idx, shape_val).unwrap();
                let div = b.u_div(u32_t, None, remaining_idx, shape_val).unwrap();
                remaining_idx = div;
                v.push(rem);
            }
            v
        };

        let input_index = {
            let mut v = input_offset;
            for i in 0..rank {
                let mul = b.i_mul(u32_t, None, index[i], input_strides[i]).unwrap();
                v = b.i_add(u32_t, None, v, mul).unwrap();
            }
            v
        };

        let output_index = {
            let mut v = output_offset;
            for i in 0..rank {
                let mul = b.i_mul(u32_t, None, index[i], output_strides[i]).unwrap();
                v = b.i_add(u32_t, None, v, mul).unwrap();
            }
            v
        };

        /* value = -input[in_index] */
        let data_i_ptr_t = b.type_pointer(None, StorageClass::StorageBuffer, input_data_type);
        let in_ptr = b.access_chain(data_i_ptr_t, None, input_0_var, [c0, input_index]).unwrap();
        let in_val = b.load(input_data_type, None, in_ptr, None, []).unwrap();

        let (in_val, in_type) = if let DType::BF16 = input_dtype {
            (cast_bf16_to_f32(&mut b, in_val), DType::F32)
        } else {
            (in_val, input_dtype)
        };

        let out_val = op(&mut b, in_val, in_type)?;

        let out_val = if let DType::BF16 = output_dtype {
            cast_f32_to_bf16(&mut b, out_val)
        } else {
            out_val
        };

        /* store */
        let data_o_ptr_t = b.type_pointer(None, StorageClass::StorageBuffer, output_data_type);
        let out_ptr = b.access_chain(data_o_ptr_t, None, output_0_var, [c0, output_index]).unwrap();
        b.store(out_ptr, out_val, None, []).unwrap();

        /* branch to merge */
        b.branch(merge_blk).unwrap();

        /* ---- MERGE block ---- */
        b.begin_block(Some(merge_blk)).unwrap();
        b.ret().unwrap();                       // OpReturn
        b.end_function().unwrap();              // OpFunction

        let module = b.module();
        let code = module.assemble();

        VulkanImmediateExecutor::debug_dump_spirv(&code);

        let shader = unsafe{ ShaderModule::new(
            vulkan_immediate_executor.context.device.clone(),
            ShaderModuleCreateInfo::new(&code)
        )}?;

        let cs = shader.entry_point("main").unwrap();

        let stage = PipelineShaderStageCreateInfo::new(cs);

        let descriptor_set_layout = vulkan_immediate_executor.get_descriptor_set_layout(BTreeSet::from([0, 1]))?;

        let layout = PipelineLayout::new(
            vulkan_immediate_executor.context.device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![descriptor_set_layout],
                push_constant_ranges: vec![PushConstantRange{
                    stages: ShaderStages::COMPUTE,
                    offset: 0,
                    size: (rank*12 + 12) as u32,
                }],
                .. Default::default()
            }
        )?;

        let compute_pipeline = ComputePipeline::new(
            vulkan_immediate_executor.context.device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout.clone())
        )?;

        Ok((layout, compute_pipeline))
    }

    fn unary(&self, vulkan_immediate_executor: &mut VulkanImmediateExecutor, cache_id: u32, op: fn(&mut rspirv::dr::Builder, rspirv::spirv::Word, DType) -> Result<rspirv::spirv::Word, VulkanError>) -> Result<VulkanTensor<R>, VulkanError> {

        let (pipeline_layout, compute_pipeline) = {
            let key = (cache_id, self.dtype(), self.dtype(), self.rank() as u32);
            let res = vulkan_immediate_executor.pipeline_cache.unary_op.get(&key);
            if let Some(res) = res {
                res.clone()
            } else {
                let res = Self::build_unary_pipeline(vulkan_immediate_executor, self.dtype(), self.dtype(), self.rank(), op)?;
                vulkan_immediate_executor.pipeline_cache.unary_op.insert(key, res.clone());
                res
            }
        };

        let output_tensor = unsafe{VulkanTensor::new_uninitialized(self.shape().clone(), self.dtype(), vulkan_immediate_executor)}?;

        let (descriptor_set, _) = vulkan_immediate_executor.get_descriptor_set_and_layout(&BTreeMap::from([
            (0, self.buffer.clone()),
            (1, output_tensor.buffer.clone())
        ]))?;

        let op_metadata_vec = {
            let mut v = vec![];
            v.push((self.offset as u32 + self.suballocation.offset as u32)/self.dtype().size() as u32);
            v.push((output_tensor.offset as u32 + output_tensor.suballocation.offset as u32)/self.dtype().size() as u32);
            v.push((self.shape().dim_product() as u32));
            for dim in self.shape().as_slice() {
                v.push(*dim as u32);
            }
            for dim in self.stride.as_slice() {
                v.push(*dim as u32);
            }
            for dim in R::KnownDims::as_slice(&output_tensor.stride) {
                v.push(*dim as u32);
            }
            v
        };


        let mut builder = AutoCommandBufferBuilder::primary(
            vulkan_immediate_executor.context.command_buffer_allocator.clone(),
            vulkan_immediate_executor.context.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit
        )?;

        for (a, b) in op_metadata_vec.iter().enumerate() {
            builder.push_constants(pipeline_layout.clone(), (a*4) as u32, *b).unwrap();
        }

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
        unsafe { builder.dispatch([((self.shape().dim_product()/64) + 1) as u32, 1, 1]) }.unwrap();

        // Finish building the command buffer by calling `build`.
        let command_buffer = builder.build()?;

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
        self.unary(vulkan_immediate_executor, 0, |b, i, dtype| {
            let data_type = get_spirv_datatype(b, dtype)?;
            match dtype {
                DType::F16 | DType::F32 | DType::F64 => Ok(b.f_negate(data_type, None, i).unwrap()),
                DType::I64 | DType::I32 | DType::I16 | DType::I8 => Ok(b.s_negate(data_type, None, i).unwrap()),
                _ => Err(VulkanError::UnsupportedByBackendError),
            }
        })
    }

    pub fn abs(&self, vulkan_immediate_executor: &mut VulkanImmediateExecutor) -> Result<VulkanTensor<R>, VulkanError> {
        self.unary(vulkan_immediate_executor, 1, |b, i, dtype| {
            let data_type = get_spirv_datatype(b, dtype)?;
            let glsl   = b.ext_inst_import("GLSL.std.450");
            match dtype {
                DType::F16 | DType::F32 | DType::F64 =>
                    Ok(b.ext_inst(data_type, None, glsl, GLOp::FAbs as u32, [rspirv::dr::Operand::IdRef(i)]).unwrap()),
                DType::I64 | DType::I32 | DType::I16 | DType::I8 | DType::U64 | DType::U32 | DType::U16 | DType::U8  =>
                    Ok(b.ext_inst(data_type, None, glsl, GLOp::SAbs as u32, [rspirv::dr::Operand::IdRef(i)]).unwrap()),
                _ =>
                    Err(VulkanError::UnsupportedByBackendError)
            }
        })
    }

    pub fn exp(&self, vulkan_immediate_executor: &mut VulkanImmediateExecutor) -> Result<VulkanTensor<R>, VulkanError> {
        self.unary(vulkan_immediate_executor, 2, |b, i, dtype| {
            let data_type = get_spirv_datatype(b, dtype)?;
            let glsl   = b.ext_inst_import("GLSL.std.450");
            match dtype {
                // F64 fails for some reason, must investigate later
                DType::F16 | DType::F32 => Ok(b.ext_inst(data_type, None, glsl, GLOp::Exp as u32, [rspirv::dr::Operand::IdRef(i)]).unwrap()),
                _ => Err(VulkanError::UnsupportedByBackendError),
            }
        })
    }

    pub fn ln(&self, vulkan_immediate_executor: &mut VulkanImmediateExecutor) -> Result<VulkanTensor<R>, VulkanError> {
        self.unary(vulkan_immediate_executor, 3, |b, i, dtype| {
            let data_type = get_spirv_datatype(b, dtype)?;
            let glsl   = b.ext_inst_import("GLSL.std.450");
            match dtype {
                DType::F16 | DType::F32 | DType::F64 => Ok(b.ext_inst(data_type, None, glsl, GLOp::Log as u32, [rspirv::dr::Operand::IdRef(i)]).unwrap()),
                _ => Err(VulkanError::UnsupportedByBackendError),
            }
        })
    }

    pub fn sqrt(&self, vulkan_immediate_executor: &mut VulkanImmediateExecutor) -> Result<VulkanTensor<R>, VulkanError> {
        self.unary(vulkan_immediate_executor, 4, |b, i, dtype| {
            let data_type = get_spirv_datatype(b, dtype)?;
            let glsl   = b.ext_inst_import("GLSL.std.450");
            match dtype {
                DType::F16 | DType::F32 | DType::F64 => Ok(b.ext_inst(data_type, None, glsl, GLOp::Sqrt as u32, [rspirv::dr::Operand::IdRef(i)]).unwrap()),
                _ => Err(VulkanError::UnsupportedByBackendError),
            }
        })
    }

    pub fn not(&self, vulkan_immediate_executor: &mut VulkanImmediateExecutor) -> Result<VulkanTensor<R>, VulkanError> {
        self.unary(vulkan_immediate_executor, 5, |b, i, dtype| {
            let data_type = get_spirv_datatype(b, dtype)?;
            match dtype {
                DType::BOOL => Ok(b.not(data_type, None, i).unwrap()),
                _ => Err(VulkanError::UnsupportedByBackendError),
            }
        })
    }

    pub fn sign(&self, vulkan_immediate_executor: &mut VulkanImmediateExecutor) -> Result<VulkanTensor<R>, VulkanError> {
        self.unary(vulkan_immediate_executor, 6, |b, i, dtype| {
            let data_type = get_spirv_datatype(b, dtype)?;
            let glsl   = b.ext_inst_import("GLSL.std.450");
            match dtype {
                DType::F16 | DType::F32 | DType::F64 => 
                    Ok(b.ext_inst(data_type, None, glsl, GLOp::FSign as u32, [rspirv::dr::Operand::IdRef(i)]).unwrap()),
                DType::I64 | DType::I32 | DType::I16 | DType::I8 | DType::U64 | DType::U32 | DType::U16 | DType::U8  =>
                    Ok(b.ext_inst(data_type, None, glsl, GLOp::SSign as u32, [rspirv::dr::Operand::IdRef(i)]).unwrap()),
                _ => Err(VulkanError::UnsupportedByBackendError),
            }
        })
    }

    pub fn bitwise_not(&self, vulkan_immediate_executor: &mut VulkanImmediateExecutor) -> Result<VulkanTensor<R>, VulkanError> {
        self.unary(vulkan_immediate_executor, 7, |b, i, dtype| {
            let data_type = get_spirv_datatype(b, dtype)?;
            match dtype {
                DType::I64 | DType::I32 | DType::I16 | DType::I8 | DType::U64 | DType::U32 | DType::U16 | DType::U8  =>
                    Ok(b.not(data_type, None, i).unwrap()),
                _ => Err(VulkanError::UnsupportedByBackendError),
            }
        })
    }

    pub fn reciprocal(&self, vulkan_immediate_executor: &mut VulkanImmediateExecutor) -> Result<VulkanTensor<R>, VulkanError> {
        self.unary(vulkan_immediate_executor, 8, |b, i, dtype| {
            let data_type = get_spirv_datatype(b, dtype)?;
            let f32_type = b.type_float(32);
            match dtype {
                DType::F16 | DType::F32 | DType::F64 => {
                    let const_one = b.constant_bit32(f32_type, 1);
                    let const_one = b.f_convert(data_type, None, const_one).unwrap();
                    Ok(b.f_div(data_type, None, const_one, i).unwrap())
                }
                _ => Err(VulkanError::UnsupportedByBackendError),
            }
        })
    }

    pub fn trig(&self, which_trig_op: TrigOp, vulkan_immediate_executor: &mut VulkanImmediateExecutor) -> Result<VulkanTensor<R>, VulkanError> {
        match which_trig_op {
            TrigOp::Asin => {
                self.unary(vulkan_immediate_executor, 9, |b, i, dtype| {
                    let data_type = get_spirv_datatype(b, dtype)?;
                    let glsl   = b.ext_inst_import("GLSL.std.450");
                    match dtype {
                        DType::F16 | DType::F32 | DType::F64 =>
                            Ok(b.ext_inst(data_type, None, glsl, GLOp::Asin as u32, [rspirv::dr::Operand::IdRef(i)]).unwrap()),
                        _ => Err(VulkanError::UnsupportedByBackendError),
                    }
                })
            }
            TrigOp::Asinh => {
                self.unary(vulkan_immediate_executor, 10, |b, i, dtype| {
                    let data_type = get_spirv_datatype(b, dtype)?;
                    let glsl   = b.ext_inst_import("GLSL.std.450");
                    match dtype {
                        DType::F16 | DType::F32 | DType::F64 => 
                            Ok(b.ext_inst(data_type, None, glsl, GLOp::Asinh as u32, [rspirv::dr::Operand::IdRef(i)]).unwrap()),
                        _ => Err(VulkanError::UnsupportedByBackendError),
                    }
                })
            }
            TrigOp::Acos => {
                self.unary(vulkan_immediate_executor, 11, |b, i, dtype| {
                    let data_type = get_spirv_datatype(b, dtype)?;
                    let glsl   = b.ext_inst_import("GLSL.std.450");
                    match dtype {
                        DType::F16 | DType::F32 | DType::F64 =>
                            Ok(b.ext_inst(data_type, None, glsl, GLOp::Acos as u32, [rspirv::dr::Operand::IdRef(i)]).unwrap()),
                        _ => Err(VulkanError::UnsupportedByBackendError),
                    }
                })
            }
            TrigOp::Acosh => {
                self.unary(vulkan_immediate_executor, 12, |b, i, dtype| {
                    let data_type = get_spirv_datatype(b, dtype)?;
                    let glsl   = b.ext_inst_import("GLSL.std.450");
                    match dtype {
                        DType::F16 | DType::F32 | DType::F64 =>
                            Ok(b.ext_inst(data_type, None, glsl, GLOp::Acosh as u32, [rspirv::dr::Operand::IdRef(i)]).unwrap()),
                        _ => Err(VulkanError::UnsupportedByBackendError),
                    }
                })
            }
            TrigOp::Atan => {
                self.unary(vulkan_immediate_executor, 13, |b, i, dtype| {
                    let data_type = get_spirv_datatype(b, dtype)?;
                    let glsl   = b.ext_inst_import("GLSL.std.450");
                    match dtype {
                        DType::F16 | DType::F32 | DType::F64 =>
                            Ok(b.ext_inst(data_type, None, glsl, GLOp::Atan as u32, [rspirv::dr::Operand::IdRef(i)]).unwrap()),
                        _ => Err(VulkanError::UnsupportedByBackendError),
                    }
                })
            }
            TrigOp::Atanh => {
                self.unary(vulkan_immediate_executor, 14, |b, i, dtype| {
                    let data_type = get_spirv_datatype(b, dtype)?;
                    let glsl   = b.ext_inst_import("GLSL.std.450");
                    match dtype {
                        DType::F16 | DType::F32 | DType::F64 =>
                            Ok(b.ext_inst(data_type, None, glsl, GLOp::Atanh as u32, [rspirv::dr::Operand::IdRef(i)]).unwrap()),
                        _ => Err(VulkanError::UnsupportedByBackendError),
                    }
                })
            }
            TrigOp::Sin => {
                self.unary(vulkan_immediate_executor, 15, |b, i, dtype| {
                    let data_type = get_spirv_datatype(b, dtype)?;
                    let glsl   = b.ext_inst_import("GLSL.std.450");
                    match dtype {
                        DType::F16 | DType::F32 | DType::F64 =>
                            Ok(b.ext_inst(data_type, None, glsl, GLOp::Sin as u32, [rspirv::dr::Operand::IdRef(i)]).unwrap()),
                        _ => Err(VulkanError::UnsupportedByBackendError),
                    }
                })
            }
            TrigOp::Sinh => {
                self.unary(vulkan_immediate_executor, 16, |b, i, dtype| {
                    let data_type = get_spirv_datatype(b, dtype)?;
                    let glsl   = b.ext_inst_import("GLSL.std.450");
                    match dtype {
                        DType::F16 | DType::F32 | DType::F64 =>
                            Ok(b.ext_inst(data_type, None, glsl, GLOp::Sinh as u32, [rspirv::dr::Operand::IdRef(i)]).unwrap()),
                        _ => Err(VulkanError::UnsupportedByBackendError),
                    }
                })
            }
            TrigOp::Cos => {
                self.unary(vulkan_immediate_executor, 17, |b, i, dtype| {
                    let data_type = get_spirv_datatype(b, dtype)?;
                    let glsl   = b.ext_inst_import("GLSL.std.450");
                    match dtype {
                        DType::F16 | DType::F32 | DType::F64 =>
                            Ok(b.ext_inst(data_type, None, glsl, GLOp::Cos as u32, [rspirv::dr::Operand::IdRef(i)]).unwrap()),
                        _ => Err(VulkanError::UnsupportedByBackendError),
                    }
                })
            }
            TrigOp::Cosh => {
                self.unary(vulkan_immediate_executor, 18, |b, i, dtype| {
                    let data_type = get_spirv_datatype(b, dtype)?;
                    let glsl   = b.ext_inst_import("GLSL.std.450");
                    match dtype {
                        DType::F16 | DType::F32 | DType::F64 =>
                            Ok(b.ext_inst(data_type, None, glsl, GLOp::Cos as u32, [rspirv::dr::Operand::IdRef(i)]).unwrap()),
                        _ => Err(VulkanError::UnsupportedByBackendError),
                    }
                })
            }
            TrigOp::Tan => {
                self.unary(vulkan_immediate_executor, 19, |b, i, dtype| {
                    let data_type = get_spirv_datatype(b, dtype)?;
                    let glsl   = b.ext_inst_import("GLSL.std.450");
                    match dtype {
                        DType::F16 | DType::F32 | DType::F64 =>
                            Ok(b.ext_inst(data_type, None, glsl, GLOp::Tan as u32, [rspirv::dr::Operand::IdRef(i)]).unwrap()),
                        _ => Err(VulkanError::UnsupportedByBackendError),
                    }
                })
            }
            TrigOp::Tanh => {
                self.unary(vulkan_immediate_executor, 20, |b, i, dtype| {
                    let data_type = get_spirv_datatype(b, dtype)?;
                    let glsl   = b.ext_inst_import("GLSL.std.450");
                    match dtype {
                        DType::F16 | DType::F32 | DType::F64 =>
                            Ok(b.ext_inst(data_type, None, glsl, GLOp::Tanh as u32, [rspirv::dr::Operand::IdRef(i)]).unwrap()),
                        _ => Err(VulkanError::UnsupportedByBackendError),
                    }
                })
            }
        }
    }

    pub fn floor(&self, vulkan_immediate_executor: &mut VulkanImmediateExecutor) -> Result<VulkanTensor<R>, VulkanError> {
        self.unary(vulkan_immediate_executor, 21, |b, i, dtype| {
            let data_type = get_spirv_datatype(b, dtype)?;
            let glsl   = b.ext_inst_import("GLSL.std.450");
            match dtype {
                DType::F16 | DType::F32 | DType::F64 =>
                    Ok(b.ext_inst(data_type, None, glsl, GLOp::Floor as u32, [rspirv::dr::Operand::IdRef(i)]).unwrap()),
                _ => Err(VulkanError::UnsupportedByBackendError),
            }
        })
    }

    pub fn ceil(&self, vulkan_immediate_executor: &mut VulkanImmediateExecutor) -> Result<VulkanTensor<R>, VulkanError> {
        self.unary(vulkan_immediate_executor, 22, |b, i, dtype| {
            let data_type = get_spirv_datatype(b, dtype)?;
            let glsl   = b.ext_inst_import("GLSL.std.450");
            match dtype {
                DType::F16 | DType::F32 | DType::F64 =>
                    Ok(b.ext_inst(data_type, None, glsl, GLOp::Ceil as u32, [rspirv::dr::Operand::IdRef(i)]).unwrap()),
                _ => Err(VulkanError::UnsupportedByBackendError),
            }
        })
    }

    pub fn round(&self, vulkan_immediate_executor: &mut VulkanImmediateExecutor) -> Result<VulkanTensor<R>, VulkanError> {
        self.unary(vulkan_immediate_executor, 23, |b, i, dtype| {
            let data_type = get_spirv_datatype(b, dtype)?;
            let glsl   = b.ext_inst_import("GLSL.std.450");
            match dtype {
                DType::F16 | DType::F32 | DType::F64 =>
                    Ok(b.ext_inst(data_type, None, glsl, GLOp::RoundEven as u32, [rspirv::dr::Operand::IdRef(i)]).unwrap()),
                _ => Err(VulkanError::UnsupportedByBackendError),
            }
        })
    }

    pub fn is_nan(&self, vulkan_immediate_executor: &mut VulkanImmediateExecutor) -> Result<VulkanTensor<R>, VulkanError> {
        self.unary(vulkan_immediate_executor, 24, |b, i, dtype| {
            let data_type = get_spirv_datatype(b, dtype)?;
            match dtype {
                DType::F16 | DType::F32 | DType::F64 =>
                    Ok(b.is_nan(data_type, None, i).unwrap()),
                _ => Err(VulkanError::UnsupportedByBackendError),
            }
        })
    }

    pub fn to_contiguous(&self, vulkan_immediate_executor: &mut VulkanImmediateExecutor) -> Result<VulkanTensor<R>, VulkanError> {
        if self.is_contiguous() {
            Ok(self.clone())
        } else {
            self.unary(vulkan_immediate_executor, 24, |b, i, dtype| {
                Ok(i)
            })
        }
    }
}

mod test {
    use typenum::P1;
    use crate::dtype::DType;
    use crate::eval_backend::EvalBackend;
    use crate::NDArrayNumericTensor;
    use crate::numeric_tensor::NumericTensor;
    use crate::vulkan_backend::{VulkanContext, VulkanImmediateExecutor};
    use crate::vulkan_backend::tensor::VulkanTensor;

    #[test]
    fn test_neg() {
        let vulkan_context = VulkanContext::new().unwrap();
        let mut vulkan_runtime = VulkanImmediateExecutor::new(vulkan_context).unwrap();

        let start_data = vec![1.0, -2.0, 3.0, -4.0];
        let expected_data = vec![-1.0, 2.0, -3.0, 4.0];

        let start_tensor = NDArrayNumericTensor::from(start_data).to_dyn();

        let dtypes_to_test = [
            DType::F64,
            DType::F32,
            DType::F16,
            DType::BF16,
            DType::I64,
            DType::I32,
            DType::I16,
            DType::I8,
        ];

        for dtype in dtypes_to_test {
            let start_tensor_cast = start_tensor.cast(dtype).unwrap();
            let start_tensor_vk = VulkanTensor::from_ndarray(start_tensor_cast, &mut vulkan_runtime).unwrap();
            let end_tensor_vk = start_tensor_vk.neg(&mut vulkan_runtime).unwrap();
            let end_tensor = end_tensor_vk.to_ndarray();
            let end_tensor_ranked = end_tensor.cast(DType::F64).unwrap().try_to_rank::<P1>().unwrap();
            let end_data: Vec<f64> = end_tensor_ranked.try_to_vec().unwrap();
            assert_eq!(end_data, expected_data);
        }
    }

    #[test]
    fn test_abs() {
        let vulkan_context = VulkanContext::new().unwrap();
        let mut vulkan_runtime = VulkanImmediateExecutor::new(vulkan_context).unwrap();

        let start_data = vec![1.0, -2.0, 3.0, -4.0];
        let expected_data = vec![1.0, 2.0, 3.0, 4.0];

        let start_tensor = NDArrayNumericTensor::from(start_data).to_dyn();

        let dtypes_to_test = [
            DType::F64,
            DType::F32,
            DType::F16,
            DType::BF16,
            DType::I64,
            DType::I32,
            DType::I16,
            DType::I8,
        ];

        for dtype in dtypes_to_test {
            let start_tensor_cast = start_tensor.cast(dtype).unwrap();
            let start_tensor_vk = VulkanTensor::from_ndarray(start_tensor_cast, &mut vulkan_runtime).unwrap();
            let end_tensor_vk = start_tensor_vk.abs(&mut vulkan_runtime).unwrap();
            let end_tensor = end_tensor_vk.to_ndarray();
            let end_tensor_ranked = end_tensor.cast(DType::F64).unwrap().try_to_rank::<P1>().unwrap();
            let end_data: Vec<f64> = end_tensor_ranked.try_to_vec().unwrap();
            assert_eq!(end_data, expected_data);
        }
    }

    #[test]
    fn test_exp() {
        let vulkan_context = VulkanContext::new().unwrap();
        let mut vulkan_runtime = VulkanImmediateExecutor::new(vulkan_context).unwrap();

        let start_data = vec![1.0, -2.0, 3.0, -4.0];
        let expected_data = vec![2.718, 0.135, 20.0855, 0.01832];

        let start_tensor = NDArrayNumericTensor::from(start_data).to_dyn();

        let dtypes_to_test = [
           // DType::F64,
            DType::F32,
            DType::F16,
            DType::BF16,
        ];

        for dtype in dtypes_to_test {
            let start_tensor_cast = start_tensor.cast(dtype).unwrap();
            let start_tensor_vk = VulkanTensor::from_ndarray(start_tensor_cast, &mut vulkan_runtime).unwrap();
            let end_tensor_vk = start_tensor_vk.exp(&mut vulkan_runtime).unwrap();
            let end_tensor = end_tensor_vk.to_ndarray();
            let end_tensor_ranked = end_tensor.cast(DType::F64).unwrap().try_to_rank::<P1>().unwrap();
            let end_data: Vec<f64> = end_tensor_ranked.try_to_vec().unwrap();
            for (a, b) in end_data.iter().zip(expected_data.iter()) {
                assert!((a - b).abs() < 1.0);
            }
        }
    }
}