use std::sync::Arc;
use vulkano::image::view::ImageView;
use vulkano::image::{AttachmentImage, SampleCount, ImageViewAbstract, ImageAccess, ImageLayout};
use vulkano::render_pass;
use vulkano::format::{Format, FormatTy};
use vulkano::device::Queue;
use vulkano::render_pass::{AttachmentDesc, SubpassDesc, SubpassDependencyDesc, LoadOp, StoreOp};


#[allow(dead_code)]
pub struct Framebuffer {
    gfx_queue: Arc<Queue>,
    width: u32,
    height: u32,
    views: Vec<Arc<ImageView<Arc<AttachmentImage>>>>,
    framebuffer: Option<Arc<dyn render_pass::FramebufferAbstract + Send + Sync>>,
    render_pass: Option<render_pass::RenderPass>,
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

    pub fn add_view(&mut self, format: Format, samples_count: SampleCount) {
        let view = ImageView::new(
            AttachmentImage::multisampled_input_attachment(
                self.gfx_queue.device().clone(),
                [self.width, self.height],
                samples_count,
                format,
            ).unwrap()
        ).unwrap();

        self.views.push(view);
    }

    // TODO: subpasses i.e.
    pub fn create_framebuffer(&mut self) {
        let mut attachments: Vec<AttachmentDesc> = vec![];
        let mut subpasses: Vec<SubpassDesc> = vec![];
        let mut dependencies: Vec<SubpassDependencyDesc> = vec![];

        for view in &self.views {
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

            let desc = AttachmentDesc {
                format: view.format(),
                samples: view.image().samples(),
                load: LoadOp::Clear,
                store: StoreOp::Store,
                stencil_load: LoadOp::DontCare,
                stencil_store: StoreOp::DontCare,
                initial_layout: ImageLayout::Undefined,
                final_layout,
            };
        }

        let desc = render_pass::RenderPassDesc::new(
            attachments,
            subpasses,
            dependencies,
        );

        let render_pass = render_pass::RenderPass::new(
            self.gfx_queue.device().clone(),
            desc)
            .unwrap();

        self.render_pass = Some(render_pass);
    }
}

#[allow(dead_code)]
pub struct DeferredRenderer {}
