use std::ptr;
use std::sync::Arc;

use vulkano::{device, render_pass, sync};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryCommandBuffer, SubpassContents, SecondaryCommandBuffer, PrimaryAutoCommandBuffer};
use vulkano::device::Queue;
use vulkano::format::{Format, FormatTy};
use vulkano::image::{AttachmentImage, ImageLayout, ImageViewAbstract, SampleCount};
use vulkano::image::view::ImageView;
use vulkano::render_pass::{AttachmentDesc, AttachmentsList, FramebufferAbstract, FramebufferSys, LoadOp, StoreOp};
use vulkano::sync::GpuFuture;

struct FbWrapper {
    inner: render_pass::Framebuffer<Box<dyn AttachmentsList>>,
}

unsafe impl Send for FbWrapper {}

unsafe impl Sync for FbWrapper {}

unsafe impl FramebufferAbstract for FbWrapper {
    fn inner(&self) -> FramebufferSys {
        self.inner.inner()
    }

    /// Returns the width, height and array layers of the framebuffer.
    fn dimensions(&self) -> [u32; 3] {
        self.inner.dimensions()
    }

    /// Returns the render pass this framebuffer was created for.
    fn render_pass(&self) -> &Arc<render_pass::RenderPass> {
        self.inner.render_pass()
    }

    /// Returns the attachment of the framebuffer with the given index.
    ///
    /// If the `index` is not between `0` and `num_attachments`, then `None` should be returned.
    fn attached_image_view(&self, index: usize) -> Option<&dyn ImageViewAbstract> {
        self.inner.attached_image_view(index)
    }
}


#[allow(dead_code)]
pub struct Framebuffer {
    gfx_queue: Arc<Queue>,
    width: u32,
    height: u32,
    views: Vec<Arc<ImageView<Arc<AttachmentImage>>>>,
    framebuffer: Option<Arc<dyn render_pass::FramebufferAbstract + Sync + Send>>,
    render_pass: Option<Arc<render_pass::RenderPass>>,
}

#[allow(dead_code)]
impl Framebuffer {
    pub fn new(gfx_queue: Arc<Queue>, width: u32, height: u32) -> Framebuffer {
        Framebuffer {
            views: vec![],
            framebuffer: None,
            width,
            height,
            gfx_queue: gfx_queue.clone(),
            render_pass: None,
        }
    }

    pub fn view(&self, idx: usize) -> Arc<ImageView<Arc<AttachmentImage>>> {
        self.views.get(idx).unwrap().clone()
    }

    pub fn add_view(&mut self, format: Format, samples_count: SampleCount) {
        let view = ImageView::new(
            AttachmentImage::sampled_multisampled_input_attachment(
                self.gfx_queue.device().clone(),
                [self.width, self.height],
                samples_count,
                format,
            ).unwrap()
        ).unwrap();

        self.views.push(view);
    }

    pub fn create_framebuffer(&mut self) {
        let mut attachments: Vec<AttachmentDesc> = vec![];

        let mut color_attachments_refs: Vec<(usize, ImageLayout)> = vec![];
        let mut depth_attachment_ref: Option<(usize, ImageLayout)> = None;


        for (idx, view) in self.views.iter().enumerate() {
            let is_depth = match view.format().ty() {
                FormatTy::Depth => true,
                FormatTy::DepthStencil => true,
                FormatTy::Stencil => true,
                FormatTy::Compressed => panic!(),
                _ => false,
            };

            let final_layout = if is_depth {
                ImageLayout::DepthStencilAttachmentOptimal
            } else {
                ImageLayout::ColorAttachmentOptimal
            };

            attachments.push(AttachmentDesc {
                format: view.format(),
                samples: view.image().samples(),
                load: LoadOp::Clear,
                store: StoreOp::Store,
                stencil_load: LoadOp::DontCare,
                stencil_store: StoreOp::DontCare,
                initial_layout: ImageLayout::Undefined,
                final_layout,
            });

            if is_depth {
                depth_attachment_ref = Some((idx, final_layout));
            } else {
                color_attachments_refs.push((idx, final_layout));
            }
        }

        let subpass = render_pass::SubpassDesc {
            color_attachments: color_attachments_refs,
            depth_stencil: depth_attachment_ref,
            input_attachments: vec!(),
            resolve_attachments: vec!(),
            preserve_attachments: vec!(),
        };

        let subpass_dependencies = vec!(
            render_pass::SubpassDependencyDesc {
                source_subpass: !0,
                destination_subpass: 0,
                source_stages: sync::PipelineStages {
                    bottom_of_pipe: true,
                    ..sync::PipelineStages::none()
                },
                destination_stages: sync::PipelineStages {
                    color_attachment_output: true,
                    ..sync::PipelineStages::none()
                },
                source_access: sync::AccessFlags {
                    memory_read: true,
                    ..sync::AccessFlags::none()
                },
                destination_access: sync::AccessFlags {
                    color_attachment_read: true,
                    color_attachment_write: true,
                    ..sync::AccessFlags::none()
                },
                by_region: true,
            },
            render_pass::SubpassDependencyDesc {
                source_subpass: 0,
                destination_subpass: !0,
                source_stages: sync::PipelineStages {
                    color_attachment_output: true,
                    ..sync::PipelineStages::none()
                },
                destination_stages: sync::PipelineStages {
                    bottom_of_pipe: true,
                    ..sync::PipelineStages::none()
                },
                source_access: sync::AccessFlags {
                    color_attachment_read: true,
                    color_attachment_write: true,
                    ..sync::AccessFlags::none()
                },
                destination_access: sync::AccessFlags {
                    memory_read: true,
                    ..sync::AccessFlags::none()
                },
                by_region: true,
            },
        );

        let render_pass_desc = render_pass::RenderPassDesc::new(
            attachments,
            vec![subpass],
            subpass_dependencies,
        );

        let render_pass = Arc::new(
            render_pass::RenderPass::new(
                self.gfx_queue.device().clone(),
                render_pass_desc)
                .unwrap()
        );

        // Framebuffer<Box<dyn render_pass::AttachmentsList + Send + Sync>>
        let mut framebuffer_builder = render_pass::Framebuffer::start(render_pass.clone()).boxed();

        for view in self.views.iter() {
            framebuffer_builder = framebuffer_builder.add(view.clone()).unwrap().boxed();
        }

        let fb = framebuffer_builder.build().unwrap();

        // render_pass::Framebuffer::(fb);
        self.framebuffer = Some(
            Arc::new(
                FbWrapper {
                    inner: fb
                }
            )
        );

        self.render_pass = Some(render_pass);
    }

    pub fn framebuffer(&self) -> Arc<dyn render_pass::FramebufferAbstract + Sync + Send> {
        self.framebuffer.clone().unwrap()
    }

    pub fn subpass(&self) -> render_pass::Subpass{
        render_pass::Subpass::from(self.render_pass.clone().unwrap(), 0).unwrap()
    }
}

#[allow(dead_code)]
pub struct DeferredRenderer {}


pub fn render<F: FnMut(&mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>)>(gfx_queue: Arc<device::Queue>, framebuffer: &Framebuffer, mut f: F) {
    // Start the command buffer builder that will be filled throughout the frame handling.
    let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
        gfx_queue.device().clone(),
        gfx_queue.family(),
        CommandBufferUsage::OneTimeSubmit,
    )
        .unwrap();

    let mut clear = vec!();
    for view in framebuffer.views.iter() {
        let is_depth = match view.format().ty() {
            FormatTy::Depth => true,
            FormatTy::DepthStencil => true,
            FormatTy::Stencil => true,
            FormatTy::Compressed => panic!(),
            _ => false,
        };

        if is_depth {
            clear.push(1.0f32.into());
        } else {
            clear.push([0.0, 0.0, 0.0, 0.0].into());
        }
    }

    command_buffer_builder
        .begin_render_pass(
            framebuffer.framebuffer().clone(),
            SubpassContents::SecondaryCommandBuffers,
            clear,
        )
        .unwrap();

    f(&mut command_buffer_builder);
    command_buffer_builder.end_render_pass().unwrap();

    let cmd_buf = command_buffer_builder.build().unwrap();

    cmd_buf.execute(gfx_queue.clone()).unwrap()
        .then_signal_fence_and_flush().unwrap()
        .wait(None).unwrap();
}
