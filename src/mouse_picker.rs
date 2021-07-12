use std::sync::Arc;

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, SecondaryCommandBuffer, SubpassContents, PrimaryCommandBuffer};
use vulkano::device::Queue;
use vulkano::format::Format;
use vulkano::image::{AttachmentImage, ImageAccess, ImageUsage};
use vulkano::image::view::ImageView;
use vulkano::render_pass::{Framebuffer, FramebufferAbstract, RenderPass, Subpass};
use vulkano::sync::GpuFuture;

pub struct Picker {
    // Queue to use to render everything.
    gfx_queue: Arc<Queue>,

    framebuffer: Arc<dyn FramebufferAbstract + Send + Sync>,

    // Render pass used for the drawing. See the `new` method for the actual render pass content.
    // We need to keep it in `FrameSystem` because we may want to recreate the intermediate buffers
    // in of a change in the dimensions.
    render_pass: Arc<RenderPass>,

    object_id_buffer: Arc<ImageView<Arc<AttachmentImage>>>,
    object_id_cpu: Arc<CpuAccessibleBuffer<[u8]>>,

    depth_buffer: Arc<ImageView<Arc<AttachmentImage>>>,
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
        let object_id_buffer = ImageView::new(
            AttachmentImage::with_usage(
                gfx_queue.device().clone(),
                [1, 1],
                Format::R8G8B8A8Unorm,
                obj_id_usage,
            )
                .unwrap()
        )
            .unwrap();

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

        let depth_buffer = ImageView::new(
            AttachmentImage::with_usage(
                gfx_queue.device().clone(),
                [1, 1],
                Format::D32Sfloat,
                atch_usage,
            )
                .unwrap()
        )
            .unwrap();

        let framebuffer = Arc::new(
            Framebuffer::start(render_pass.clone())
                .add(object_id_buffer.clone())
                .unwrap()
                .add(depth_buffer.clone())
                .unwrap()
                .build()
                .unwrap()
        );

        Picker {
            gfx_queue,
            render_pass,
            framebuffer,
            object_id_buffer,
            object_id_cpu,
            depth_buffer,
        }
    }
    pub fn subpass(&self) -> Subpass {
        Subpass::from(self.render_pass.clone(), 0).unwrap()
    }
    pub fn draw<C>(&mut self, img_dims: [u32; 2], cmds: Vec<C>, x: u32, y: u32) -> Option<u32>
        where C: SecondaryCommandBuffer + Send + Sync + 'static
    {
        // Recreate framebuffer
        if self.object_id_buffer.image().dimensions().width_height() != img_dims {
            let obj_id_usage = ImageUsage {
                transfer_source: true, // This is necessary to copy to external buffer
                ..ImageUsage::none()
            };
            self.object_id_buffer = ImageView::new(
                AttachmentImage::with_usage(
                    self.gfx_queue.device().clone(),
                    img_dims,
                    Format::R8G8B8A8Unorm,
                    obj_id_usage,
                )
                    .unwrap()
            )
                .unwrap();

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

            self.depth_buffer = ImageView::new(
                AttachmentImage::with_usage(
                    self.gfx_queue.device().clone(),
                    img_dims,
                    Format::D32Sfloat,
                    atch_usage,
                )
                    .unwrap()
            ).unwrap();

            self.framebuffer = Arc::new(
                Framebuffer::start(self.render_pass.clone())
                    .add(self.object_id_buffer.clone())
                    .unwrap()
                    .add(self.depth_buffer.clone())
                    .unwrap()
                    .build()
                    .unwrap()
            );
        }

        // Start the command buffer builder that will be filled throughout the frame handling.
        let mut command_buffer_builder =
            AutoCommandBufferBuilder::primary(self.gfx_queue.device().clone(),
                                              self.gfx_queue.family(),
                                              CommandBufferUsage::OneTimeSubmit).unwrap();

        command_buffer_builder.begin_render_pass(
            self.framebuffer.clone(),
            SubpassContents::SecondaryCommandBuffers,
            vec![[0.0, 0.0, 0.0, 0.0].into(), 1.0f32.into()],
        )
            .unwrap();

        command_buffer_builder.execute_commands_from_vec(cmds).unwrap();

        let dims = self.object_id_buffer.image().dimensions().width_height();
        if !(0..dims[0]).contains(&x) || !(0..dims[1]).contains(&y) {
            return None;
        }

        command_buffer_builder
            .end_render_pass().unwrap()
            .copy_image_to_buffer_dimensions(
                self.object_id_buffer.image().clone(),
                self.object_id_cpu.clone(),
                [x, y, 0],
                [1, 1, 1],
                0, 1, 0,
            ).unwrap();


        let cmd_buf = command_buffer_builder.build().unwrap();

        cmd_buf.execute(self.gfx_queue.clone()).unwrap()
            .then_signal_fence_and_flush().unwrap()
            .wait(None).unwrap();

        let buffer_content = self.object_id_cpu.read().unwrap();
        get_entity_id(buffer_content[0], buffer_content[1], buffer_content[2], buffer_content[3])
    }
}
