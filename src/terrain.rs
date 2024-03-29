use std::io::Cursor;
use std::sync::Arc;

use cgmath::{InnerSpace, Matrix4, Vector3};
use vulkano::buffer::{BufferUsage, CpuBufferPool, ImmutableBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, DynamicState, SecondaryAutoCommandBuffer};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::device::Queue;
use vulkano::format::Format;
use vulkano::image::{ImageDimensions, ImmutableImage, MipmapsCount};
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineAbstract};
use vulkano::pipeline::viewport::Viewport;
use vulkano::render_pass::Subpass;
use vulkano::sampler::{Filter, MipmapMode, Sampler, SamplerAddressMode};
use vulkano::sync::GpuFuture;
use vulkano::image::view::ImageView;

#[allow(dead_code)]
pub struct HeightMap {
    pub w: u32,
    pub h: u32,
    height_fn: Box<dyn Fn(u32, u32) -> f32>,
}

#[allow(dead_code)]
impl HeightMap {
    pub fn from_png() -> HeightMap {
        let data = include_bytes!("static/heightmap.png").to_vec();
        let cursor = Cursor::new(data);
        let decoder = png::Decoder::new(cursor);

        let (info, mut reader) = decoder.read_info().unwrap();
        let mut image_data = Vec::new();
        image_data.resize((info.width * info.height * 4) as usize, 0);
        reader.next_frame(&mut image_data).unwrap();

        let w = info.width;
        HeightMap {
            w: info.width,
            h: info.height,
            height_fn: Box::new(move |x: u32, y: u32| -> f32 {
                4.0 * (image_data[(w * y * 4 + x * 4) as usize] as f32) / 255.0
            }),
        }
    }

    pub fn empty(w: u32, h: u32) -> HeightMap {
        HeightMap {
            w,
            h,
            height_fn: Box::new(|_, _| -> f32 { 0.0 }),
        }
    }

    pub fn get_height(&self, x: i32, y: i32) -> f32 {
        let clamp = |val: i32, min: i32, max: i32| -> i32 {
            if val < min {
                return min;
            }
            if val > max {
                return max;
            }

            return val;
        };

        let xx = clamp(x, 0, (self.w - 1) as i32);
        let yy = clamp(y, 0, (self.h - 1) as i32);
        let fn_ = &self.height_fn;

        -fn_(xx as u32, yy as u32)
    }
}

#[derive(Default, Debug, Clone)]
pub struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    texcoord: [f32; 2],
}
vulkano::impl_vertex!(Vertex, position, normal, texcoord);

#[allow(dead_code)]
pub struct Terrain {
    gfx_queue: Arc<Queue>,
    pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
    uniform_buffer: CpuBufferPool<vs::ty::Data>,

    texture: Arc<ImageView<Arc<ImmutableImage>>>,
    sampler: Arc<Sampler>,
    pub vertices: Arc<ImmutableBuffer<[Vertex]>>,
    pub indices: Arc<ImmutableBuffer<[u32]>>,

}

#[allow(dead_code)]
impl Terrain {
    pub fn new(gfx_queue: Arc<Queue>, height_map: HeightMap, subpass: Subpass) -> Terrain {
        let w = height_map.w;
        let h = height_map.h;

        let mut vertices = Vec::with_capacity((h * w) as usize);
        let mut indices = Vec::with_capacity((h * (w - 1) * 6) as usize);

        let get_pos = |x: i32, y: i32| -> Vector3<f32> {
            let height = height_map.get_height(x, y);
            Vector3::new((x as f32) * 0.1, height, -(y as f32) * 0.1)
        };

        for y in 0..(h as i32) {
            for x in 0..(w as i32) {
                let pos = get_pos(x, y);

                // Bottom left, Bottom right, Upper left
                let l = get_pos(x - 1, y) - pos;
                let t = get_pos(x, y + 1) - pos;
                let r = get_pos(x + 1, y) - pos;
                let b = get_pos(x, y - 1) - pos;

                let lb = l.cross(b).normalize();
                let br = b.cross(r).normalize();
                let rt = r.cross(t).normalize();
                let tl = t.cross(l).normalize();

                let normal = -(lb + br + rt + tl).normalize();

                vertices.push(Vertex {
                    position: pos.into(), //[(x as f32) * 0.1, height, -(y as f32) * 0.1],
                    normal: normal.into(),
                    texcoord: [x as f32, y as f32],
                });
            }
        }

        for y in 1..(h) {
            for x in 0..(w - 1) {
                indices.push((y - 1) * w + x);
                indices.push((y - 1) * w + x + 1);
                indices.push((y) * w + x);

                indices.push((y) * w + x);
                indices.push((y - 1) * w + x + 1);
                indices.push((y) * w + x + 1);
            }
        }

        let (bb, fut) = {
            ImmutableBuffer::from_iter(vertices.iter().cloned(), BufferUsage::vertex_buffer(), gfx_queue.clone()).unwrap()
        };

        let (ib, fut2) = {
            ImmutableBuffer::from_iter(indices.iter().cloned(), BufferUsage::index_buffer(), gfx_queue.clone()).unwrap()
        };

        fut2.join(fut).then_signal_fence_and_flush().unwrap().wait(None).unwrap();

        let pipeline = {
            let vs = vs::Shader::load(gfx_queue.device().clone())
                .expect("failed to create shader module");
            let fs = fs::Shader::load(gfx_queue.device().clone())
                .expect("failed to create shader module");

            Arc::new(GraphicsPipeline::start()
                .vertex_input_single_buffer::<Vertex>()
                .vertex_shader(vs.main_entry_point(), ())
                .triangle_list()
                .viewports_dynamic_scissors_irrelevant(1)
                .fragment_shader(fs.main_entry_point(), ())
                .render_pass(subpass)
                .cull_mode_back()
                .front_face_counter_clockwise()
//        .polygon_mode_line()
                .depth_stencil_simple_depth()
                .build(gfx_queue.device().clone())
                .unwrap())
        };

        let uniform_buffer = CpuBufferPool::<vs::ty::Data>::new(gfx_queue.device().clone(), BufferUsage::all());

        let (texture, tex_future) = {
            let png_bytes = include_bytes!("static/ground.png").to_vec();
            let cursor = Cursor::new(png_bytes);
            let decoder = png::Decoder::new(cursor);
            let (info, mut reader) = decoder.read_info().unwrap();
            let dimensions = ImageDimensions::Dim2d { width: info.width, height: info.height, array_layers: 0 }; // FIXME: check need array=0 or array=1?
            let mut image_data = Vec::new();
            image_data.resize((info.width * info.height * 4) as usize, 0);
            reader.next_frame(&mut image_data).unwrap();

            let (image, future) = ImmutableImage::from_iter(
                image_data.iter().cloned(),
                dimensions,
                MipmapsCount::One,
                Format::R8G8B8A8Srgb,
                gfx_queue.clone(),
            ).unwrap();

            (ImageView::new(image), future)
        };

        tex_future.then_signal_fence_and_flush().unwrap().wait(None).unwrap();

        let sampler = Sampler::new(gfx_queue.device().clone(), Filter::Linear, Filter::Linear,
                                   MipmapMode::Nearest, SamplerAddressMode::Repeat, SamplerAddressMode::Repeat,
                                   SamplerAddressMode::Repeat, 0.0, 5.0, 0.0, 0.0).unwrap();
        Terrain {
            gfx_queue,
            pipeline,
            uniform_buffer,
            sampler,
            texture: texture.unwrap(),
            vertices: bb,
            indices: ib,
        }
    }

    pub fn draw(&self, viewport_dimensions: [u32; 2], world: Matrix4<f32>, view: Matrix4<f32>, proj: Matrix4<f32>) -> SecondaryAutoCommandBuffer {
        let uniform_buffer_subbuffer = {
            let uniform_data = vs::ty::Data {
                world: world.into(),
                view: view.into(),
                proj: proj.into(),
            };

            self.uniform_buffer.next(uniform_data).unwrap()
        };


        let layout = self.pipeline.layout().descriptor_set_layout(0).unwrap();

        let set = Arc::new(
            PersistentDescriptorSet::start(layout.clone())
                .add_buffer(uniform_buffer_subbuffer)
                .unwrap()
                .add_sampled_image(self.texture.clone(), self.sampler.clone())
                .unwrap()
                .build()
                .unwrap()
        );

        let mut builder = AutoCommandBufferBuilder::secondary_graphics(self.gfx_queue.device().clone(),
                                                     self.gfx_queue.family(),
                                                     CommandBufferUsage::MultipleSubmit,
                                                     self.pipeline.subpass().clone()).unwrap();
        builder.draw_indexed(
                self.pipeline.clone(),
                &DynamicState {
                    viewports: Some(vec![Viewport {
                        origin: [0.0, 0.0],
                        dimensions: [viewport_dimensions[0] as f32,
                            viewport_dimensions[1] as f32],
                        depth_range: 0.0..1.0,
                    }]),
                    ..DynamicState::none()
                },
                vec![self.vertices.clone()],
                self.indices.clone(),
                set.clone(),
                (),
                vec![],
            )
            .unwrap();

        builder.build().unwrap()
    }
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        bytes: "resources/shaders/heightmap/terrain.vert.spv"
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        bytes: "resources/shaders/heightmap/terrain.frag.spv"
    }
}
