use std::sync::Arc;

use imgui::{DrawCmd, DrawCmdParams, DrawVert, ImString, TextureId, Textures};
use imgui::internal::RawWrapper;
use vulkano::buffer::{BufferAccess, BufferUsage, CpuBufferPool};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, DynamicState, SubpassContents};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::device::{Device, Queue};
use vulkano::format::{ClearValue, Format};
use vulkano::image::{ImageDimensions, ImageViewAbstract, ImmutableImage};
use vulkano::image::view::ImageView;
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineAbstract};
use vulkano::pipeline::viewport::{Scissor, Viewport};
use vulkano::render_pass;
use vulkano::render_pass::Subpass;
use vulkano::sampler::Sampler;
use vulkano::sync::GpuFuture;

#[allow(dead_code)]
#[derive(Debug)]
pub enum RendererError {
    BadTexture(TextureId),
    BadImageDimensions(ImageDimensions),
}

pub type Texture = (Arc<dyn ImageViewAbstract + Send + Sync>, Arc<Sampler>);


#[allow(dead_code)]
pub struct GuiPass {
    gfx_queue: Arc<Queue>,
    pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
    pipeline_ms: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,

    vrt_buffer_pool: CpuBufferPool<Vertex>,
    idx_buffer_pool: CpuBufferPool<u16>,
    font_texture: Texture,
    pub textures: Textures<Texture>,

    render_pass: Arc<render_pass::RenderPass>,
}

#[allow(dead_code)]
impl GuiPass {
    pub fn new(ctx: &mut imgui::Context, gfx_queue: Arc<Queue>, output_format: vulkano::format::Format) -> GuiPass
    {
        let render_pass = Arc::new(
            vulkano::single_pass_renderpass!(
                gfx_queue.device().clone(),
                attachments: {
                    final_color: {
                        load: Load,
                        store: Store,
                        format: output_format,
                        samples: 1,
                    }
                },
                pass: {
                        color: [final_color],
                        depth_stencil: {}
                    }
            ).unwrap(),
        );

        let pipeline = {
            let vs = vs::Shader::load(gfx_queue.device().clone())
                .expect("failed to create shader module");
            let fs = fs::Shader::load(gfx_queue.device().clone())
                .expect("failed to create shader module");

            Arc::new(
                GraphicsPipeline::start()
                    .vertex_input_single_buffer::<Vertex>()
                    .vertex_shader(vs.main_entry_point(), ())
                    .triangle_list()
                    .viewports_scissors_dynamic(1)
                    .fragment_shader(fs.main_entry_point(), ())
                    .blend_alpha_blending()
                    .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
                    .build(gfx_queue.device().clone()).unwrap()
            )
        };

        let pipeline_ms = {
            let vs = vs::Shader::load(gfx_queue.device().clone())
                .expect("failed to create shader module");
            let fs = fs_multisampled::Shader::load(gfx_queue.device().clone())
                .expect("failed to create shader module");

            Arc::new(
                GraphicsPipeline::start()
                    .vertex_input_single_buffer::<Vertex>()
                    .vertex_shader(vs.main_entry_point(), ())
                    .triangle_list()
                    .viewports_scissors_dynamic(1)
                    .fragment_shader(fs.main_entry_point(), ())
                    .blend_alpha_blending()
                    .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
                    .build(gfx_queue.device().clone()).unwrap()
            )
        };

        let device = gfx_queue.device().clone();

        let vrt_buffer_pool = CpuBufferPool::new(device.clone(), BufferUsage::vertex_buffer_transfer_destination());
        let idx_buffer_pool = CpuBufferPool::new(device.clone(), BufferUsage::index_buffer_transfer_destination());

        let textures = Textures::new();
        let font_texture = Self::upload_font_texture(ctx.fonts(), device.clone(), gfx_queue.clone()).unwrap();
        ctx.set_renderer_name(Some(ImString::from(format!("imgui-vulkano-renderer {}", env!("CARGO_PKG_VERSION")))));

        GuiPass {
            gfx_queue,
            pipeline,
            pipeline_ms,
            textures,
            font_texture,
            vrt_buffer_pool,
            idx_buffer_pool,
            render_pass,
        }
    }

    pub fn draw<F, I>(
        &mut self,
        before_future: F,
        gfx_queue: Arc<Queue>,
        target_image: Arc<I>,
        viewport_dimensions: [u32; 2],
        draw_data: &imgui::DrawData,
    ) -> Box<dyn GpuFuture>
        where
            F: GpuFuture + 'static,
            I: ImageViewAbstract + Send + Sync + 'static
    {
        let framebuffer = Arc::new(
            render_pass::Framebuffer::start(self.render_pass.clone())
                .add(target_image.clone())
                .unwrap()
                .build()
                .unwrap()
        );

        let mut builder = AutoCommandBufferBuilder::primary(
            self.gfx_queue.device().clone(),
            self.gfx_queue.family(),
            CommandBufferUsage::OneTimeSubmit,
        )
            .unwrap();

        builder
            .begin_render_pass(
                framebuffer,
                SubpassContents::Inline,
                vec![
                    ClearValue::None,
                ],
            ).unwrap();

        for draw_list in draw_data.draw_lists() {
            let vertex_buffer = Arc::new(self.vrt_buffer_pool.chunk(draw_list.vtx_buffer().iter().map(|&v| Vertex::from(v))).unwrap());
            let index_buffer = Arc::new(self.idx_buffer_pool.chunk(draw_list.idx_buffer().iter().cloned()).unwrap());

            let fb_width = draw_data.display_size[0] * draw_data.framebuffer_scale[0];
            let fb_height = draw_data.display_size[1] * draw_data.framebuffer_scale[1];
            if !(fb_width > 0.0 && fb_height > 0.0) {
                panic!("imgui buffer size small is negative");
            }

            let left = draw_data.display_pos[0];
            let right = draw_data.display_pos[0] + draw_data.display_size[0];
            let top = draw_data.display_pos[1];
            let bottom = draw_data.display_pos[1] + draw_data.display_size[1];

            let pc = vs::ty::VertPC {
                matrix: [
                    [(2.0 / (right - left)), 0.0, 0.0, 0.0],
                    [0.0, (2.0 / (bottom - top)), 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [
                        (right + left) / (left - right),
                        (top + bottom) / (top - bottom),
                        0.0,
                        1.0,
                    ],
                ]
            };

            let mut dynamic_state = DynamicState::default();
            dynamic_state.viewports = Some(vec![
                Viewport {
                    origin: [0.0, 0.0],
                    dimensions: [viewport_dimensions[0] as f32, viewport_dimensions[1] as f32],
                    depth_range: 0.0..1.0,
                }
            ]);
            dynamic_state.scissors = Some(vec![
                Scissor::default()
            ]);

            let clip_off = draw_data.display_pos;
            let clip_scale = draw_data.framebuffer_scale;

            for cmd in draw_list.commands() {
                match cmd {
                    DrawCmd::Elements {
                        count,
                        cmd_params:
                        DrawCmdParams {
                            clip_rect,
                            texture_id,
                            // vtx_offset,
                            idx_offset,
                            ..
                        },
                    } => {
                        let clip_rect = [
                            (clip_rect[0] - clip_off[0]) * clip_scale[0],
                            (clip_rect[1] - clip_off[1]) * clip_scale[1],
                            (clip_rect[2] - clip_off[0]) * clip_scale[0],
                            (clip_rect[3] - clip_off[1]) * clip_scale[1],
                        ];

                        if clip_rect[0] < fb_width
                            && clip_rect[1] < fb_height
                            && clip_rect[2] >= 0.0
                            && clip_rect[3] >= 0.0
                        {
                            if let Some(ref mut scissors) = dynamic_state.scissors {
                                scissors[0] = Scissor {
                                    origin: [
                                        f32::max(0.0, clip_rect[0]).floor() as i32,
                                        f32::max(0.0, clip_rect[1]).floor() as i32
                                    ],
                                    dimensions: [
                                        (clip_rect[2] - clip_rect[0]).abs().ceil() as u32,
                                        (clip_rect[3] - clip_rect[1]).abs().ceil() as u32
                                    ],
                                };
                            }

                            let tex = self.lookup_texture(texture_id)
                                .unwrap();


                            let pipeline = if (tex.0.image().samples() as u32) == 1 {
                                self.pipeline.clone()
                            } else {
                                self.pipeline_ms.clone()
                            };

                            let layout = pipeline.layout().descriptor_set_layout(0).unwrap();

                            let set = Arc::new(
                                PersistentDescriptorSet::start(layout.clone())
                                    .add_sampled_image(tex.0.clone(), tex.1.clone())
                                    .unwrap()
                                    .build()
                                    .unwrap()
                            );

                            builder.draw_indexed(
                                pipeline,
                                &dynamic_state,
                                vec![vertex_buffer.clone()],
                                index_buffer.clone().into_buffer_slice().slice(idx_offset..(idx_offset + count)).unwrap(),
                                set,
                                pc,
                                vec![]).unwrap();
                        }
                    }
                    DrawCmd::ResetRenderState => (), // TODO
                    DrawCmd::RawCallback { callback, raw_cmd } => unsafe {
                        callback(draw_list.raw(), raw_cmd)
                    },
                }
            }
        }

        builder.end_render_pass().unwrap();

        let cmd_buf = builder.build().unwrap();
        Box::new(before_future.then_execute(gfx_queue.clone(), cmd_buf).unwrap())
    }

    fn upload_font_texture(
        mut fonts: imgui::FontAtlasRefMut,
        device: Arc<Device>,
        queue: Arc<Queue>,
    ) -> Result<Texture, Box<dyn std::error::Error>> {
        let texture = fonts.build_rgba32_texture();

        let (image, fut) = ImmutableImage::from_iter(
            texture.data.iter().cloned(),
            ImageDimensions::Dim2d {
                width: texture.width,
                height: texture.height,
                array_layers: 1,
            },
            vulkano::image::MipmapsCount::One,
            Format::R8G8B8A8Srgb,
            queue.clone(),
        )?;

        fut.then_signal_fence_and_flush()?
            .wait(None).unwrap();

        let sampler = Sampler::simple_repeat_linear(device.clone());

        fonts.tex_id = TextureId::from(usize::MAX);
        Ok((ImageView::new(image).unwrap(), sampler))
    }

    fn lookup_texture(&self, texture_id: TextureId) -> Result<&Texture, RendererError> {
        if texture_id.id() == usize::MAX {
            Ok(&self.font_texture)
        } else if let Some(texture) = self.textures.get(texture_id) {
            Ok(texture)
        } else {
            Err(RendererError::BadTexture(texture_id))
        }
    }
}

#[derive(Default, Debug, Clone)]
#[repr(C)]
struct Vertex {
    pub pos: [f32; 2],
    pub uv: [f32; 2],
    pub col: u32,
    // pub col: [u8; 4],
}
vulkano::impl_vertex!(Vertex, pos, uv, col);

impl From<DrawVert> for Vertex {
    fn from(v: DrawVert) -> Vertex {
        unsafe { std::mem::transmute(v) }
    }
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        bytes: "resources/shaders/imgui/render.vert.spv"
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        bytes: "resources/shaders/imgui/render.frag.spv"
    }
}

mod fs_multisampled {
    vulkano_shaders::shader! {
        ty: "fragment",
        bytes: "resources/shaders/imgui/render_multisampled.frag.spv"
    }
}
