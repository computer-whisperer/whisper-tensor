use std::sync::Arc;
use vulkano::device::{Device, DeviceCreateInfo, Queue, QueueCreateInfo, QueueFlags};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::VulkanLibrary;
use crate::Error;

#[derive(Debug)]
pub struct VulkanContext {
    instance: Arc<Instance>,
    queue: Arc<Queue>,
    device: Arc<Device>,
    memory_allocator: Arc<StandardMemoryAllocator>,
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
