use vulkano::image::{AttachmentImage, ImageUsage, ImageAccess};
use vulkano::buffer::{CpuAccessibleBuffer, BufferUsage, BufferAccess};
use std::sync::Arc;
use vulkano::device::Queue;
use vulkano::framebuffer::{RenderPassAbstract, Framebuffer, Subpass, FramebufferAbstract};
use vulkano::sync::GpuFuture;
use cgmath::Matrix4;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBuffer};
use vulkano::format::Format;
use std::ops::Range;
use vulkano::descriptor::DescriptorSet;

pub struct Picker {
    // Queue to use to render everything.
    gfx_queue: Arc<Queue>,

    framebuffer: Arc<dyn FramebufferAbstract + Send + Sync>,

    // Render pass used for the drawing. See the `new` method for the actual render pass content.
    // We need to keep it in `FrameSystem` because we may want to recreate the intermediate buffers
    // in of a change in the dimensions.
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,

    object_id_buffer: Arc<AttachmentImage>,
    object_id_cpu: Arc<CpuAccessibleBuffer<[u8]>>,

    depth_buffer: Arc<AttachmentImage>,
}

fn get_entity_id(r: u8, g: u8, b: u8, a: u8) -> Option<u32> {
    if a == 0 {
        None
    } else {
        Some(((r as usize) | (g as usize) << 8 | (b as usize) << 16) as u32)
    }
}

impl Picker {
    pub fn new(gfx_queue: Arc<Queue>) -> Picker {
        let render_pass = Arc::new(
            vulkano::ordered_passes_renderpass!(gfx_queue.device().clone(),
            attachments: {
                // The image that will contain the final rendering (in this example the swapchain
                // image, but it could be another image).
                id_map: {
                    load: Clear,
                    store: Store,
                    format: Format::R8G8B8A8Unorm,
                    samples: 1,
                },
                // Will be bound to `self.depth_buffer`.
                depth: {
                    load: Clear,
                    store: DontCare,
                    format: Format::D32Sfloat,
                    samples: 1,
                }
            },
            passes: [
                // Write to the diffuse, normals and depth attachments.
                {
                    color: [id_map],
                    depth_stencil: {depth},
                    input: []
                }
            ]
        ).unwrap());

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
        ).expect("Failed to create buffer");

        let atch_usage = ImageUsage {
            transient_attachment: true,
            depth_stencil_attachment: true,
            ..ImageUsage::none()
        };

        let depth_buffer = AttachmentImage::with_usage(gfx_queue.device().clone(),
                                                       [1, 1],
                                                       Format::D32Sfloat,
                                                       atch_usage).unwrap();
        let framebuffer = Arc::new(Framebuffer::start(render_pass.clone())
            .add(object_id_buffer.clone()).unwrap()
            .add(depth_buffer.clone()).unwrap()
            .build().unwrap());

        Picker {
            gfx_queue,
            render_pass,
            framebuffer,
            object_id_buffer,
            object_id_cpu,
            depth_buffer,
        }
    }

    pub fn subpass(&self) -> Subpass<Arc<dyn RenderPassAbstract + Send + Sync>> {
        Subpass::from(self.render_pass.clone(), 0).unwrap()
    }

    pub fn draw<C>(&mut self, img_dims: [u32; 2], cmds: Vec<C>, x: u32, y: u32) -> Option<u32>
        where C: CommandBuffer + Send + Sync + 'static
    {
        if ImageAccess::dimensions(&self.object_id_buffer).width_height() != img_dims {
            let obj_id_usage = ImageUsage {
                transfer_source: true, // This is necessary to copy to external buffer
                ..ImageUsage::none()
            };
            self.object_id_buffer = AttachmentImage::with_usage(self.gfx_queue.device().clone(),
                                                                img_dims,
                                                                Format::R8G8B8A8Unorm,
                                                                obj_id_usage).unwrap();

            self.object_id_cpu = CpuAccessibleBuffer::from_iter(
                self.gfx_queue.device().clone(),
                BufferUsage::all(),
                false, (0..4).map(|_| 0u8),
            ).expect("Failed to create buffer");

            let atch_usage = ImageUsage {
                transient_attachment: true,
                depth_stencil_attachment: true,
                ..ImageUsage::none()
            };

            self.depth_buffer = AttachmentImage::with_usage(self.gfx_queue.device().clone(),
                                                            img_dims,
                                                            Format::D32Sfloat,
                                                            atch_usage).unwrap();

            self.framebuffer = Arc::new(Framebuffer::start(self.render_pass.clone())
                .add(self.object_id_buffer.clone()).unwrap()
                .add(self.depth_buffer.clone()).unwrap()
                .build().unwrap());
        }

        // Build the framebuffer. The image must be attached in the same order as they were defined
        // with the `ordered_passes_renderpass!` macro.

        // Start the command buffer builder that will be filled throughout the frame handling.
        let mut command_buffer =
            AutoCommandBufferBuilder::primary_one_time_submit(self.gfx_queue
                                                                  .device()
                                                                  .clone(),
                                                              self.gfx_queue.family())
                .unwrap()
                .begin_render_pass(self.framebuffer.clone(),
                                   true,
                                   vec![
                                       [0.0, 0.0, 0.0, 0.0].into(),
                                       1.0f32.into(),
                                   ])
                .unwrap();

        command_buffer = unsafe {
            command_buffer.execute_commands_from_vec(cmds).unwrap()
        };

        let cmd_buf = command_buffer
            .end_render_pass().unwrap()
            .copy_image_to_buffer_dimensions(
                self.object_id_buffer.clone(),
                self.object_id_cpu.clone(),
                [x, y, 0],
                [1, 1, 1],
                0, 1, 0,
            ).unwrap()
            .build().unwrap();

        cmd_buf.execute(self.gfx_queue.clone()).unwrap()
            .then_signal_fence_and_flush().unwrap()
            .wait(None).unwrap();

        let buffer_content = self.object_id_cpu.read().unwrap();
        get_entity_id(buffer_content[0], buffer_content[1], buffer_content[2], buffer_content[3])
    }

//    pub fn pick(&self, x: u32, y: u32) -> Option<u32> {
////        let dims = ImageAccess::dimensions(&self.object_id_buffer).width_height();
//    }
}


