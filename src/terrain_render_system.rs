use crate::terrain_game::{Map, TerrainBlock};
use crate::cube::{Cube, Vertex};
use crate::terrain::Terrain;
use std::sync::Arc;
use vulkano::buffer::{ImmutableBuffer, BufferUsage, CpuBufferPool};
use vulkano::device::Queue;
use vulkano::impl_vertex;
use vulkano::sync::GpuFuture;
use vulkano::framebuffer::{Subpass, RenderPassAbstract};
use vulkano::pipeline::{GraphicsPipelineAbstract, GraphicsPipeline};
use vulkano::pipeline::vertex::OneVertexOneInstanceDefinition;
use cgmath::Matrix4;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState, AutoCommandBuffer};
use vulkano::pipeline::viewport::Viewport;

#[derive(Default, Debug, Clone)]
struct InstanceData {
    position_offset: [f32; 2],
    object_id: [f32; 4],
}
impl_vertex!(InstanceData, position_offset, object_id);

pub struct TerrainRenderSystem {
    gfx_queue: Arc<Queue>,
    cube: Cube,

    pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
    uniform_buffer: CpuBufferPool<vs::ty::Data>,

    instance_data: Option<Arc<ImmutableBuffer<[InstanceData]>>>,
}

impl TerrainRenderSystem {
    pub fn new<R>(gfx_queue: Arc<Queue>, subpass: Subpass<R>) -> TerrainRenderSystem
        where R: RenderPassAbstract + Send + Sync + 'static
    {
        let pipeline = {
            let vs = vs::Shader::load(gfx_queue.device().clone())
                .expect("failed to create shader module");
            let fs = fs::Shader::load(gfx_queue.device().clone())
                .expect("failed to create shader module");

            Arc::new(GraphicsPipeline::start()
                .vertex_input(OneVertexOneInstanceDefinition::<Vertex, InstanceData>::new())
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


        TerrainRenderSystem {
            gfx_queue: gfx_queue.clone(),
            cube: Cube::new(gfx_queue.clone(), 1.0),
            instance_data: None,
            uniform_buffer,
            pipeline,
        }
    }

    pub fn render(&mut self, map: Arc<Map>, viewport_dimensions: [u32; 2], world: Matrix4<f32>, view: Matrix4<f32>, proj: Matrix4<f32>) -> AutoCommandBuffer {
        if self.instance_data.is_none() {
            self.instance_data = Some(self.rebuild_instance_data(map.blocks.clone()));
        }

        let uniform_buffer_subbuffer = {
            let uniform_data = vs::ty::Data {
                world: world.into(),
                view: view.into(),
                proj: proj.into(),
            };

            self.uniform_buffer.next(uniform_data).unwrap()
        };

        let layout = self.pipeline.descriptor_set_layout(0).unwrap();
        let set = Arc::new(PersistentDescriptorSet::start(layout.clone())
            .add_buffer(uniform_buffer_subbuffer).unwrap()
            .build().unwrap()
        );

        AutoCommandBufferBuilder::secondary_graphics(self.gfx_queue.device().clone(),
                                                     self.gfx_queue.family(),
                                                     self.pipeline.clone().subpass())
            .unwrap()
            .draw_indexed(self.pipeline.clone(),
                          &DynamicState {
                              viewports: Some(vec![Viewport {
                                  origin: [0.0, 0.0],
                                  dimensions: [viewport_dimensions[0] as f32,
                                      viewport_dimensions[1] as f32],
                                  depth_range: 0.0..1.0,
                              }]),
                              ..DynamicState::none()
                          },
                          vec!(self.cube.vertices.clone(), self.instance_data.clone().unwrap()), self.cube.indices.clone(), set.clone(), ())
            .unwrap()
            .build()
            .unwrap()
    }

    fn rebuild_instance_data(&self, blocks: Vec<TerrainBlock>) -> Arc<ImmutableBuffer<[InstanceData]>> {
        let mut instance_data = Vec::<InstanceData>::new();

        for block in blocks {
            let id = block.id;
            let x = [((id & 0xFF) as f32) / 255.0,
                ((id >> 8) & 0xFF) as f32 / 255.0,
                ((id >> 16) & 0xFF) as f32 / 255.0,
                1.0];

            instance_data.push(InstanceData {
                position_offset: [block.x as f32, block.y as f32],
                object_id: x,
            });
        }

        let (bb, fut) = {
            ImmutableBuffer::from_iter(instance_data.iter().cloned(), BufferUsage::vertex_buffer(), self.gfx_queue.clone()).unwrap()
        };

        fut.then_signal_fence_and_flush().unwrap().wait(None).unwrap();

        return bb;
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

            layout(location = 3) in vec2 position_offset;
            layout(location = 4) in vec4 object_id;

            layout(set = 0, binding = 0) uniform Data {
                mat4 world;
                mat4 view;
                mat4 proj;
            } uniforms;

            layout(location=1) out vec3 rnormal;
            layout(location=2) out vec3 rpos;
            layout(location=3) out vec3 out_color;
            layout(location=4) out vec4 out_object_id;
            void main() {
                mat4 worldview = uniforms.view;// * uniforms.world;
                vec3 s_pos = position;
                s_pos.x += position_offset.x;
                s_pos.z -= position_offset.y;
                gl_Position = uniforms.proj * worldview * vec4(s_pos, 1.0);

                rpos = s_pos;
                rnormal = normal;
                out_color = color;
                out_object_id = object_id;
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
            layout(location = 2) out vec4 f_object_id;


            layout(location = 1) in vec3 in_normal;
            layout(location = 2) in vec3 in_world;
            layout(location = 3) in vec3 in_color;
            layout(location = 4) in vec4 in_object_id;

            void main() {
                vec3 light_pos = normalize(vec3(-0.0, 2.0, 1.0));
                float light_percent = max(-dot(light_pos, in_normal), 0.0);

                f_color = vec4(in_color, 1.0); vec4(1.0, 1.0, 0.0, 1.0);
                f_object_id = in_object_id;
            }
        "
    }
}
