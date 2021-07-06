use std::sync::Arc;

use cgmath::Matrix4;
use vulkano::buffer::{BufferUsage, CpuBufferPool, ImmutableBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, DynamicState, SecondaryAutoCommandBuffer};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::device::Queue;
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineAbstract};
use vulkano::pipeline::viewport::Viewport;
use vulkano::render_pass::Subpass;
use vulkano::sync::GpuFuture;

#[derive(Default, Debug, Clone)]
pub struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    color: [f32; 3],
}
vulkano::impl_vertex!(Vertex, position, normal, color);

#[allow(dead_code)]
pub struct BlockRender {
    gfx_queue: Arc<Queue>,
    pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
    uniform_buffer: CpuBufferPool<vs::ty::Data>,

    pub vertices: Arc<ImmutableBuffer<[Vertex]>>,
    pub indices: Arc<ImmutableBuffer<[u32]>>,
}

#[allow(dead_code)]
impl BlockRender {
    pub fn new(gfx_queue: Arc<Queue>, subpass: Subpass) -> BlockRender
    {
        let h = 3.0_f32;

        let vertices = [
            // up
            Vertex { position: [0.0, -h, 0.0], normal: [0.0, 1.0, 0.0], color: [0.0, 1.0, 0.0] },
            Vertex { position: [0.0, -h, -1.0], normal: [0.0, 1.0, 0.0], color: [0.0, 1.0, 0.0] },
            Vertex { position: [1.0, -h, -1.0], normal: [0.0, 1.0, 0.0], color: [0.0, 1.0, 0.0] },
            Vertex { position: [1.0, -h, 0.0], normal: [0.0, 1.0, 0.0], color: [0.0, 1.0, 0.0] },

            // bottom
            Vertex { position: [0.0, 0.0, 0.0], normal: [0.0, -1.0, 0.0], color: [1.0, 1.0, 1.0] },
            Vertex { position: [0.0, 0.0, -1.0], normal: [0.0, -1.0, 0.0], color: [1.0, 1.0, 1.0] },
            Vertex { position: [1.0, 0.0, -1.0], normal: [0.0, -1.0, 0.0], color: [1.0, 1.0, 1.0] },
            Vertex { position: [1.0, 0.0, 0.0], normal: [0.0, -1.0, 0.0], color: [1.0, 1.0, 1.0] },

            // front
            Vertex { position: [0.0, 0.0, 0.0], normal: [0.0, 0.0, 1.0], color: [1.0, 0.0, 0.0] },
            Vertex { position: [0.0, -h, 0.0], normal: [0.0, 0.0, 1.0], color: [1.0, 0.0, 0.0] },
            Vertex { position: [1.0, -h, 0.0], normal: [0.0, 0.0, 1.0], color: [1.0, 0.0, 0.0] },
            Vertex { position: [1.0, 0.0, 0.0], normal: [0.0, 0.0, 1.0], color: [1.0, 0.0, 0.0] },

            // back
            Vertex { position: [0.0, 0.0, -1.0], normal: [0.0, 0.0, -1.0], color: [1.0, 1.0, 1.0] },
            Vertex { position: [0.0, -h, -1.0], normal: [0.0, 0.0, -1.0], color: [1.0, 1.0, 1.0] },
            Vertex { position: [1.0, -h, -1.0], normal: [0.0, 0.0, -1.0], color: [1.0, 1.0, 1.0] },
            Vertex { position: [1.0, 0.0, -1.0], normal: [0.0, 0.0, -1.0], color: [1.0, 1.0, 1.0] },

            // left
            Vertex { position: [0.0, 0.0, -1.0], normal: [-1.0, 0.0, 0.0], color: [0.0, 0.0, 1.0] },
            Vertex { position: [0.0, -h, -1.0], normal: [-1.0, 0.0, 0.0], color: [0.0, 0.0, 1.0] },
            Vertex { position: [0.0, -h, 0.0], normal: [-1.0, 0.0, 0.0], color: [0.0, 0.0, 1.0] },
            Vertex { position: [0.0, 0.0, 0.0], normal: [-1.0, 0.0, 0.0], color: [0.0, 0.0, 1.0] },

            // right
            Vertex { position: [1.0, 0.0, 0.0], normal: [1.0, 0.0, 0.0], color: [1.0, 1.0, 1.0] },
            Vertex { position: [1.0, -h, 0.0], normal: [1.0, 0.0, 0.0], color: [1.0, 1.0, 1.0] },
            Vertex { position: [1.0, -h, -1.0], normal: [1.0, 0.0, 0.0], color: [1.0, 1.0, 1.0] },
            Vertex { position: [1.0, 0.0, -1.0], normal: [1.0, 0.0, 0.0], color: [1.0, 1.0, 1.0] },
        ];

        let indices = [
            // top
            0, 3, 1, 1, 3, 2,

            // bottom
            7, 4, 6, 6, 4, 5,

            // front
            8, 11, 9, 9, 11, 10,

            // back
            15, 12, 14, 14, 12, 13,

            //left
            16, 19, 17, 17, 19, 18,

            //right
            20, 23, 21, 21, 23, 22,
        ];

//        let x: Vec<_> = Cube::new().vertex(|v| Vertex { position: v.pos.into(), normal: v.normal.into() }).triangulate().vertices().collect();


        let (bb, fut) = {
            ImmutableBuffer::from_iter(vertices.iter().cloned(), BufferUsage::vertex_buffer(), gfx_queue.clone()).unwrap()
        };

        let (ib, fut2) = {
            ImmutableBuffer::from_iter(indices.iter().cloned(), BufferUsage::index_buffer(), gfx_queue.clone()).unwrap()
        };

        fut.join(fut2).then_signal_fence_and_flush().unwrap().wait(None).unwrap();

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
                .polygon_mode_line()
                .depth_stencil_simple_depth()
                .build(gfx_queue.device().clone())
                .unwrap())
        };

        let uniform_buffer = CpuBufferPool::<vs::ty::Data>::new(gfx_queue.device().clone(), BufferUsage::all());

        BlockRender {
            gfx_queue,
            pipeline,
            uniform_buffer,
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
        let set = Arc::new(PersistentDescriptorSet::start(layout.clone())
            .add_buffer(uniform_buffer_subbuffer).unwrap()
            .build().unwrap()
        );

        let mut builder = AutoCommandBufferBuilder::secondary_graphics(
            self.gfx_queue.device().clone(),
            self.gfx_queue.family(),
            CommandBufferUsage::MultipleSubmit,
            self.pipeline.subpass().clone(),
        )
            .unwrap();

        builder.draw_indexed(self.pipeline.clone(),
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
        src: "
            #version 450

            layout(location = 0) in vec3 position;
            layout(location = 1) in vec3 normal;
            layout(location = 2) in vec3 color;

            layout(set = 0, binding = 0) uniform Data {
                mat4 world;
                mat4 view;
                mat4 proj;
            } uniforms;

            layout(location=1) out vec3 rnormal;
            layout(location=2) out vec3 rpos;
            layout(location=3) out vec3 out_color;
            void main() {
                mat4 worldview = uniforms.view;// * uniforms.world;
                gl_Position = uniforms.proj * worldview * vec4(position, 1.0);

                rpos = position;
                rnormal = normal;
                out_color = color;
            }
        "
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
            #version 450

            layout(location = 0) out vec4 f_color;
            layout(location = 1) in vec3 in_normal;
            layout(location = 2) in vec3 in_world;
            layout(location = 3) in vec3 in_color;

            void main() {
                vec3 light_pos = normalize(vec3(-0.0, 2.0, 1.0));
                float light_percent = max(-dot(light_pos, in_normal), 0.0);

                f_color = vec4(in_color, 1.0); vec4(1.0, 1.0, 0.0, 1.0);
            }
        "
    }
}
