use std::sync::{Arc, Mutex};
use vulkano::device::{Device, DeviceCreateInfo, Queue, QueueCreateInfo, QueueFlags};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, AllocationType, DeviceLayout, FreeListAllocator, MemoryTypeFilter, StandardMemoryAllocator, Suballocation, Suballocator};
use vulkano::{DeviceSize, VulkanError, VulkanLibrary};
use vulkano::buffer::{AllocateBufferError, Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::memory::allocator::suballocator::Region;
use vulkano::memory::DeviceAlignment;
use crate::dtype::DType;
use crate::NDArrayNumericTensor;
use crate::tensor_rank::{DimContainer, Rank};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("No suitable Vulkan device found")]
    NoSuitableVulkanDeviceError,
    #[error("No suitable Vulkan queue found")]
    NoSuitableVulkanQueueError,
    #[error(transparent)]
    VulkanError(#[from] VulkanError),
    #[error(transparent)]
    ValidatedVulkanError(#[from] vulkano::Validated<VulkanError>),
    #[error(transparent)]
    ValidatedVulkanAllocateBufferError(#[from] vulkano::Validated<AllocateBufferError>),
    #[error(transparent)]
    VulkanLoadingError(#[from] vulkano::LoadingError)
}

#[derive(Debug)]
pub struct VulkanContext {
    pub instance: Arc<Instance>,
    pub queue: Arc<Queue>,
    pub device: Arc<Device>,
    pub memory_allocator: Arc<StandardMemoryAllocator>,
}

impl VulkanContext {
    pub fn new() -> Result<Self, Error> {
        let library = VulkanLibrary::new().map_err(|x| Error::VulkanLoadingError(x))?;
        let instance = Instance::new(library, InstanceCreateInfo{
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            .. Default::default()
        }).map_err(|x| Error::ValidatedVulkanError(x))?;

        let physical_device = instance
            .enumerate_physical_devices()
            .map_err(|x| Error::VulkanError(x))?
            .next()
            .ok_or(Error::NoSuitableVulkanDeviceError)?;

        let queue_family_index = physical_device
            .queue_family_properties()
            .iter()
            .enumerate()
            .position(|(_queue_family_index, queue_family_properties)| {
                queue_family_properties.queue_flags.contains(QueueFlags::COMPUTE)
            })
            .ok_or(Error::NoSuitableVulkanQueueError)?;

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                // here we pass the desired queue family to use by index
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index: queue_family_index as u32,
                    ..Default::default()
                }],
                ..Default::default()
            },
        ).map_err(|x| Error::ValidatedVulkanError(x))?;

        let queue = queues.next().ok_or(Error::NoSuitableVulkanQueueError)?;

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        Ok(Self {
            instance,
            device,
            queue,
            memory_allocator
        })
    }
}

#[derive(Debug)]
pub struct VulkanImmediateExecutorBuffer {
    buffer: Arc<Subbuffer<[u8]>>,
    allocator: FreeListAllocator
}

#[derive(Debug)]
pub struct VulkanImmediateExecutor {
    context: VulkanContext,
    buffers: Vec<VulkanImmediateExecutorBuffer>,
}

impl VulkanImmediateExecutor {
    pub fn new(context: VulkanContext) -> Result<Self, Error> {
        Ok(Self {
            context,
            buffers: Vec::new()
        })
    }

    pub fn allocate_buffer(&mut self, size: usize) -> Result<usize, Error> {
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
            ).map_err(|x| Error::ValidatedVulkanAllocateBufferError(x))?;
        let allocator = FreeListAllocator::new(Region::new(0, buffer.size()).unwrap());
        let buffer = VulkanImmediateExecutorBuffer {
            buffer: Arc::new(buffer),
            allocator
        };
        self.buffers.push(buffer);
        Ok(self.buffers.len() - 1)
    }

    pub fn alloc_space(&mut self, size: usize) -> (Arc<Subbuffer<[u8]>>, Suballocation) {
        if self.buffers.len() == 0 {
            self.allocate_buffer(100000).unwrap();
        }
        let layout = DeviceLayout::from_size_alignment(size as u64, 8).unwrap();
        let v = self.buffers[0].allocator.allocate(
            layout,
            AllocationType::Linear,
            DeviceAlignment::MIN
        ).unwrap();
        (self.buffers[0].buffer.clone(), v)
    }
}

#[derive(Debug, Clone)]
pub struct VulkanTensor<R: Rank> {
    dtype: DType,
    shape: R::KnownDims,
    stride: R::KnownDims,
    suballocation: Arc<Suballocation>,
    offset: usize,
    buffer: Arc<Subbuffer<[u8]>>
}

impl<R: Rank> VulkanTensor<R> {

    pub fn from_ndarray(source: NDArrayNumericTensor<R>, executor: &mut VulkanImmediateExecutor) -> Result<Self, Error> {

        let shape = source.shape();
        let dtype = source.dtype();
        let needed_space = shape.as_slice().iter().product::<u64>() as usize*dtype.size();
        let (buffer, suballocation) = executor.alloc_space(needed_space);

        let mut stride = vec![];
        let mut v = 1;
        for &i in shape.as_slice() {
            stride.push(v);
            v = v * i;
        }
        let stride = R::KnownDims::try_from_slice(stride.as_slice()).unwrap();
        {
            let mut writer = buffer.write().unwrap();
            for i in 0..shape.as_slice().iter().product() {
                let mut index = vec![];
                let mut v = i;
                for &j in shape.as_slice() {
                    index.push(v % j);
                    v = v / j;
                }
                let index = R::KnownDims::try_from_slice(index.as_slice()).unwrap();
                let value = source.get(&index).unwrap().to_bytes();
                // Calculate destination
                let outer_offset = suballocation.offset as usize;
                let inner_offset = index.as_slice().iter().zip(stride.as_slice().iter()).map(|(a, b)| a*b).sum::<u64>() as usize;
                writer[outer_offset+inner_offset..outer_offset+inner_offset+value.len()].copy_from_slice(&value);
            }
        }

        Ok(VulkanTensor {
            dtype,
            shape,
            suballocation: Arc::new(suballocation),
            offset: 0,
            buffer,
            stride
        })
    }

    pub fn to_ndarray(&self) -> NDArrayNumericTensor<R> {
        {
            let reader = self.buffer.read().unwrap();
            let bytes_to_read = self.shape.as_slice().iter().zip(self.stride.as_slice().iter()).map(|(a, b)| a*b).sum::<u64>() as usize*self.dtype.size();
            let start_offset = self.suballocation.offset as usize + self.offset;
            NDArrayNumericTensor::from_bytes(
                &reader[start_offset ..start_offset + bytes_to_read], 
                self.dtype, &self.shape, &self.stride).unwrap()
        }
    }
    
    pub fn shape(&self) -> &R::KnownDims {
        &self.shape
    }
    
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    
    pub fn rank(&self) -> usize {
        self.shape.len()
    }
}

impl<R: Rank> From<VulkanTensor<R>> for NDArrayNumericTensor<R> {
    fn from(value: VulkanTensor<R>) -> Self {
        value.to_ndarray()
    }
}