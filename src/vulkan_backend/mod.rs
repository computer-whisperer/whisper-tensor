pub mod tensor;
pub mod ops;
mod spirv_helpers;

use std::any::TypeId;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::{Arc, Mutex};
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo, QueueFlags};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, AllocationType, DeviceLayout, FreeListAllocator, MemoryTypeFilter, StandardMemoryAllocator, Suballocation, Suballocator};
use vulkano::{DeviceSize, VulkanLibrary};
use vulkano::buffer::{AllocateBufferError, Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::descriptor_set::layout::{DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType};
use vulkano::memory::allocator::suballocator::Region;
use vulkano::memory::DeviceAlignment;
use vulkano::pipeline::{ComputePipeline, PipelineLayout};
use vulkano::shader::ShaderStages;
use crate::dtype::DType;
use crate::NDArrayNumericTensor;
use crate::tensor_rank::{DimContainer, Rank};

#[derive(Debug, thiserror::Error)]
pub enum VulkanError {
    #[error("No suitable Vulkan device found")]
    NoSuitableVulkanDeviceError,
    #[error("No suitable Vulkan queue found")]
    NoSuitableVulkanQueueError,
    #[error(transparent)]
    VulkanError(#[from] vulkano::VulkanError),
    #[error(transparent)]
    ValidatedVulkanError(#[from] vulkano::Validated<vulkano::VulkanError>),
    #[error(transparent)]
    ValidatedVulkanAllocateBufferError(#[from] vulkano::Validated<AllocateBufferError>),
    #[error(transparent)]
    VulkanLoadingError(#[from] vulkano::LoadingError),
    #[error("Unsupported in backend")]
    UnsupportedByBackendError,
    #[error("Invalid Shape")]
    InvalidShape
}

#[derive(Debug)]
pub struct VulkanContext {
    pub instance: Arc<Instance>,
    pub queue: Arc<Queue>,
    pub device: Arc<Device>,
    pub memory_allocator: Arc<StandardMemoryAllocator>,
    pub descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    pub command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
}

impl VulkanContext {
    pub fn new() -> Result<Self, VulkanError> {
        let library = VulkanLibrary::new().map_err(|x| VulkanError::VulkanLoadingError(x))?;
        let instance = Instance::new(library, InstanceCreateInfo{
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            .. Default::default()
        }).map_err(|x| VulkanError::ValidatedVulkanError(x))?;

        let device_extensions = DeviceExtensions {
            ..DeviceExtensions::empty()
        };
        
        let device_features = DeviceFeatures {
            shader_float64: true,
            shader_float16: true,
            shader_int16: true,
            shader_int8: true,
            .. Default::default()
        };
        
        let physical_device = instance
            .enumerate_physical_devices()
            .map_err(|x| VulkanError::VulkanError(x))?
            .filter(|p| {
                // Some devices may not support the extensions or features that your application,
                // or report properties and limits that are not sufficient for your application.
                // These should be filtered out here.
                //p.properties().device_name != "AMD Radeon RX 6900 XT (RADV NAVI21)" &&
                p.supported_extensions().contains(&device_extensions) &&
                    p.supported_features().contains(&device_features)
            })
            .next()
            .ok_or(VulkanError::NoSuitableVulkanDeviceError)?;

        let queue_family_index = physical_device
            .queue_family_properties()
            .iter()
            .enumerate()
            .position(|(_queue_family_index, queue_family_properties)| {
                queue_family_properties.queue_flags.contains(QueueFlags::COMPUTE)
            })
            .ok_or(VulkanError::NoSuitableVulkanQueueError)?;

        
        
        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                // here we pass the desired queue family to use by index
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index: queue_family_index as u32,
                    ..Default::default()
                }],
                enabled_features: device_features,
                enabled_extensions: device_extensions,
                ..Default::default()
            },
        ).map_err(|x| VulkanError::ValidatedVulkanError(x))?;

        let queue = queues.next().ok_or(VulkanError::NoSuitableVulkanQueueError)?;

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(device.clone(), Default::default()));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(device.clone(), Default::default()));

        Ok(Self {
            instance,
            device,
            queue,
            memory_allocator,
            descriptor_set_allocator,
            command_buffer_allocator
        })
    }
}

#[derive(Debug)]
pub struct VulkanImmediateExecutorBuffer {
    buffer: Subbuffer<[u8]>,
    allocator: FreeListAllocator
}

#[derive(Debug)]
pub struct PipelineCache {
    pub unary_op: HashMap<(u32, DType, DType, u32), (Arc<PipelineLayout>, Arc<ComputePipeline>)>,
    pub binary_op: HashMap<(u32, DType, DType, u32), (Arc<PipelineLayout>, Arc<ComputePipeline>)>
}

impl PipelineCache {
    pub fn new() -> Self {
        Self {
            unary_op: HashMap::new(),
            binary_op: HashMap::new()
        }
    }
}

#[derive(Debug)]
pub struct VulkanImmediateExecutor {
    context: VulkanContext,
    buffers: Vec<VulkanImmediateExecutorBuffer>,
    descriptor_set_layouts: HashMap<BTreeSet<u32>, Arc<DescriptorSetLayout>>,
    descriptor_set_cache: HashMap<(BTreeMap<u32, Subbuffer<[u8]>>), Arc<DescriptorSet>>,
    pipeline_cache: PipelineCache
}

impl VulkanImmediateExecutor {
    pub fn new(context: VulkanContext) -> Result<Self, VulkanError> {
        Ok(Self {
            context,
            descriptor_set_layouts: HashMap::new(),
            descriptor_set_cache: HashMap::new(),
            buffers: Vec::new(),
            pipeline_cache: PipelineCache::new()
        })
    }


    pub fn get_descriptor_set_layout(&mut self, binding_ids: BTreeSet<u32>) -> Result<Arc<DescriptorSetLayout>, VulkanError> {
        if !self.descriptor_set_layouts.contains_key(&binding_ids) {
            // Build new one
            let descriptor_set_bindings = {
                let mut ret = BTreeMap::new();
                // inputs
                for id in &binding_ids {
                    ret.insert(
                        *id,
                        DescriptorSetLayoutBinding {
                            stages: ShaderStages::COMPUTE,
                            ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer)
                        }
                    );
                }
                ret
            };
            let res = DescriptorSetLayout::new(self.context.device.clone(), DescriptorSetLayoutCreateInfo {
                bindings: descriptor_set_bindings,
                ..Default::default()
            })?;
            self.descriptor_set_layouts.insert(binding_ids.clone(), res.clone());
        }

        // Fetch from cache
        Ok(self.descriptor_set_layouts[&binding_ids].clone())
    }

    pub fn get_descriptor_set_and_layout(&mut self, input_buffer_ids: &BTreeMap<u32, Subbuffer<[u8]>>) -> Result<(Arc<DescriptorSet>, Arc<DescriptorSetLayout>), VulkanError> {
        let layout = self.get_descriptor_set_layout(input_buffer_ids.keys().cloned().collect())?;

        if !self.descriptor_set_cache.contains_key(&input_buffer_ids) {
            // Build new one
            let writes = input_buffer_ids.iter().map(|(a, b)|
                WriteDescriptorSet::buffer(*a, b.clone())
            ).collect::<Vec<_>>();
            let descriptor_set = DescriptorSet::new(
                self.context.descriptor_set_allocator.clone(),
                layout.clone(),
                writes, // 0 is the binding
                [],
            ).unwrap();
            self.descriptor_set_cache.insert(input_buffer_ids.clone(), descriptor_set.clone());
        }
        Ok((self.descriptor_set_cache[&input_buffer_ids].clone(), layout))
    }

    pub fn allocate_buffer(&mut self, size: usize) -> Result<usize, VulkanError> {
        let buffer = Buffer::new_slice(
            self.context.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                   | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
                },
                size as u64
            ).map_err(|x| VulkanError::ValidatedVulkanAllocateBufferError(x))?;
        let allocator = FreeListAllocator::new(Region::new(0, buffer.size()).unwrap());
        let buffer = VulkanImmediateExecutorBuffer {
            buffer,
            allocator
        };
        self.buffers.push(buffer);
        Ok(self.buffers.len() - 1)
    }

    pub fn alloc_space(&mut self, size: usize) -> (Subbuffer<[u8]>, Suballocation) {
        if self.buffers.len() == 0 {
            self.allocate_buffer(100000000).unwrap();
        }
        let layout = DeviceLayout::from_size_alignment(size as u64, 8).unwrap();
        let v = self.buffers[0].allocator.allocate(
            layout,
            AllocationType::Linear,
            DeviceAlignment::MIN
        ).unwrap();
        (self.buffers[0].buffer.clone(), v)
    }
    
    pub fn debug_dump_spirv(spirv: &[u32]) {
        let output_file = File::create("shader.spv").unwrap();
        let mut writer = BufWriter::new(output_file);
        for word in spirv {
            writer.write(&word.to_le_bytes()).unwrap();
        }
        writer.flush().unwrap();
    }
}

