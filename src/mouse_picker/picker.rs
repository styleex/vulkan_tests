use vulkano::image::{AttachmentImage, ImageUsage};
use vulkano::buffer::{CpuAccessibleBuffer, BufferUsage};
use std::sync::Arc;

pub struct Picker {
    object_id_buffer: Arc<AttachmentImage>,
    object_id_cpu: Arc<CpuAccessibleBuffer<[u8]>>,
}

impl Picker {
    pub fn new() -> Picker {
        let obj_id_usage = ImageUsage {
            transfer_source: true, // This is necessary to copy to external buffer
            ..ImageUsage::none()
        };
        let object_id_buffer = AttachmentImage::with_usage(gfx_queue.device().clone(),
                                                           [1, 1],
                                                           Format::R8G8B8A8Unorm,
                                                           obj_id_usage).unwrap();

        let object_id_cpu = CpuAccessibleBuffer::from_iter(
            gfx_queue.device().clone(),
            BufferUsage::all(),
            false, (0..0).map(|_| 0u8),
        )
            .expect("Failed to create buffer");

        Picker{
            object_id_buffer,
            object_id_cpu
        }
    }

    pub fn draw() {

    }
}
