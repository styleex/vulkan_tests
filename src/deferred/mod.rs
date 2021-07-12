use std::sync::Arc;

use vulkano::{device, render_pass, sync};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, SubpassContents};
use vulkano::device::{Queue, Device};
use vulkano::format::{Format, FormatTy};
use vulkano::image::{AttachmentImage, ImageLayout, ImageViewAbstract, SampleCount};
use vulkano::image::view::ImageView;
use vulkano::render_pass::{AttachmentDesc, AttachmentsList, FramebufferAbstract, FramebufferSys, LoadOp, StoreOp};
use vulkano::sync::GpuFuture;

pub mod lighting_pass;


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

#[derive(Clone)]
pub struct RenderTargetDesc {
    pub format: Format,
    pub samples_count: SampleCount,
}


#[allow(dead_code)]
pub struct Framebuffer {
    gfx_queue: Arc<Queue>,
    descriptions: Vec<RenderTargetDesc>,

    views: Vec<Arc<ImageView<Arc<AttachmentImage>>>>,
    framebuffer: Option<Arc<dyn render_pass::FramebufferAbstract + Sync + Send>>,
    render_pass: Arc<render_pass::RenderPass>,
}

#[allow(dead_code)]
impl Framebuffer {
    pub fn new(gfx_queue: Arc<Queue>, targets: Vec<RenderTargetDesc>) -> Framebuffer {
        Framebuffer {
            gfx_queue: gfx_queue.clone(),
            descriptions: targets.clone(),
            views: vec![],
            framebuffer: None,
            render_pass: Self::_create_render_pass(gfx_queue.device().clone(), targets),
        }
    }

    pub fn view(&self, idx: usize) -> Arc<ImageView<Arc<AttachmentImage>>> {
        self.views.get(idx).unwrap().clone()
    }

    fn _create_render_pass(
        device: Arc<Device>,
        descriptions: Vec<RenderTargetDesc>,
    ) -> Arc<render_pass::RenderPass>
    {
        let mut attachments: Vec<AttachmentDesc> = vec![];

        let mut color_attachments_refs: Vec<(usize, ImageLayout)> = vec![];
        let mut depth_attachment_ref: Option<(usize, ImageLayout)> = None;

        for (idx, view) in descriptions.iter().enumerate() {
            let is_depth = match view.format.ty() {
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
                format: view.format,
                samples: view.samples_count,
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

        Arc::new(
            render_pass::RenderPass::new(
                device,
                render_pass_desc)
                .unwrap()
        )
    }

    pub fn resize_swapchain(&mut self, dimensions: [u32; 2]) {
        self.views = self.descriptions.iter().map(|desc| {
            ImageView::new(
                AttachmentImage::sampled_multisampled_input_attachment(
                    self.gfx_queue.device().clone(),
                    dimensions,
                    desc.samples_count,
                    desc.format,
                ).unwrap()
            ).unwrap()
        }).collect();

        let mut framebuffer_builder = render_pass::Framebuffer::start(
            self.render_pass.clone()
        ).boxed();

        for view in self.views.iter() {
            framebuffer_builder = framebuffer_builder.add(view.clone()).unwrap().boxed();
        }

        self.framebuffer = Some(
            Arc::new(
                FbWrapper {
                    inner: framebuffer_builder.build().unwrap()
                }
            )
        );
    }

    pub fn framebuffer(&self) -> Arc<dyn render_pass::FramebufferAbstract + Sync + Send> {
        self.framebuffer.clone().unwrap()
    }

    pub fn subpass(&self) -> render_pass::Subpass {
        render_pass::Subpass::from(self.render_pass.clone(), 0).unwrap()
    }
}

pub fn render_and_wait<F, Fn>(
    before_future: F,
    gfx_queue: Arc<device::Queue>,
    framebuffer: &Framebuffer,
    f: Fn,
) -> Box<dyn GpuFuture>
    where
        F: GpuFuture + 'static,
        Fn: FnOnce(&mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>)
{
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

    Box::new(before_future.then_execute(gfx_queue.clone(), cmd_buf).unwrap())
}
