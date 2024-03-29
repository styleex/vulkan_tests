use std::sync::Arc;

use cgmath::{Angle, Deg, Matrix4, Rad};
use vulkano::buffer::{BufferUsage, CpuBufferPool};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, DynamicState, SecondaryAutoCommandBuffer};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::device::Queue;
use vulkano::impl_vertex;
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineAbstract};
use vulkano::pipeline::vertex::OneVertexOneInstanceDefinition;
use vulkano::pipeline::viewport::Viewport;
use vulkano::render_pass::Subpass;

use crate::cube::{Cube, Vertex};
use crate::terrain_game::{BlockState, Map, TerrainBlock};

#[allow(dead_code)]
pub enum RenderPipeline {
    ObjectIdMap,
    Diffuse,
    Shadows,
}

#[derive(Default, Debug, Clone)]
struct InstanceData {
    position_offset: [f32; 2],
    object_id: [f32; 4],
    highlight: [f32; 4],
}
impl_vertex!(InstanceData, position_offset, object_id, highlight);

pub struct TerrainRenderSystem {
    gfx_queue: Arc<Queue>,
    cube: Cube,

    object_map_pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
    main_pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,

    uniform_buffer: CpuBufferPool<vs::ty::Data>,
    instance_data: CpuBufferPool<InstanceData>,
}

impl TerrainRenderSystem {
    pub fn new(gfx_queue: Arc<Queue>, main_subpass: Subpass, object_map_subpass: Subpass) -> TerrainRenderSystem {
        let main_pipeline = {
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
                .render_pass(main_subpass)
                .cull_mode_back()
                .front_face_counter_clockwise()
                .depth_stencil_simple_depth()
                .build(gfx_queue.device().clone())
                .unwrap())
        };

        let object_map_pipeline = {
            let vs = vs_object_map::Shader::load(gfx_queue.device().clone())
                .expect("failed to create shader module");
            let fs = fs_object_map::Shader::load(gfx_queue.device().clone())
                .expect("failed to create shader module");

            Arc::new(GraphicsPipeline::start()
                .vertex_input(OneVertexOneInstanceDefinition::<Vertex, InstanceData>::new())
                .vertex_shader(vs.main_entry_point(), ())
                .triangle_list()
                .viewports_dynamic_scissors_irrelevant(1)
                .fragment_shader(fs.main_entry_point(), ())
                .render_pass(object_map_subpass)
                .cull_mode_back()
                .front_face_counter_clockwise()
                .depth_stencil_simple_depth()
                .build(gfx_queue.device().clone())
                .unwrap())
        };

        let uniform_buffer = CpuBufferPool::<vs::ty::Data>::new(gfx_queue.device().clone(), BufferUsage::all());

        let instance_data = CpuBufferPool::<InstanceData>::vertex_buffer(gfx_queue.device().clone());
        TerrainRenderSystem {
            gfx_queue: gfx_queue.clone(),
            cube: Cube::new(gfx_queue.clone(), 1.0),
            uniform_buffer,
            main_pipeline,
            object_map_pipeline,
            instance_data,
        }
    }

    pub fn render(&mut self, pipeline: RenderPipeline, map: &Map, viewport_dimensions: [u32; 2],
                  world: Matrix4<f32>, view: Matrix4<f32>, proj: Matrix4<f32>) -> SecondaryAutoCommandBuffer
    {
        let uniform_buffer_subbuffer = {
            let uniform_data = vs::ty::Data {
                world: world.into(),
                view: view.into(),
                proj: proj.into(),
            };

            self.uniform_buffer.next(uniform_data).unwrap()
        };

        let instance_data_subbuffer = {
            let inst_data = self.rebuild_instance_data(map.blocks.clone());
            self.instance_data.chunk(inst_data).unwrap()
        };

        let pipeline = match pipeline {
            RenderPipeline::Diffuse => self.main_pipeline.clone(),
            RenderPipeline::ObjectIdMap => self.object_map_pipeline.clone(),
            RenderPipeline::Shadows => unreachable!(),
        };

        let layout = pipeline.layout().descriptor_set_layout(0).unwrap();
        let set = Arc::new(PersistentDescriptorSet::start(layout.clone())
            .add_buffer(uniform_buffer_subbuffer).unwrap()
            .build().unwrap()
        );

        let mut builder = AutoCommandBufferBuilder::secondary_graphics(
            self.gfx_queue.device().clone(),
            self.gfx_queue.family(),
            CommandBufferUsage::MultipleSubmit,
            pipeline.subpass().clone())
            .unwrap();

        builder.draw_indexed(pipeline.clone(),
                             &DynamicState {
                                 viewports: Some(vec![Viewport {
                                     origin: [0.0, 0.0],
                                     dimensions: [viewport_dimensions[0] as f32,
                                         viewport_dimensions[1] as f32],
                                     depth_range: 0.0..1.0,
                                 }]),
                                 ..DynamicState::none()
                             },
                             vec!(self.cube.vertices.clone(),
                                  Arc::new(instance_data_subbuffer)),
                             self.cube.indices.clone(),
                             set.clone(),
                             (),
                             vec![],
        )
            .unwrap();

        builder.build().unwrap()
    }

    fn rebuild_instance_data(&self, blocks: Vec<TerrainBlock>) -> Vec<InstanceData> {
        let mut instance_data = Vec::<InstanceData>::new();

        for block in blocks {
            if block.state == BlockState::Cleared {
                continue;
            }
            let id = block.id;
            let x = [((id & 0xFF) as f32) / 255.0,
                ((id >> 8) & 0xFF) as f32 / 255.0,
                ((id >> 16) & 0xFF) as f32 / 255.0,
                1.0];

            let mut hightlight = [1.0, 1.0, 1.0, 1.0];

            if block.highlighted && !block.selected {
                hightlight[0] = 0.5 + (Rad::from(Deg(block.hightligh_start.elapsed().as_millis() as f32 / 8.0)).sin() / 4.0).abs();
            }

            if block.selected {
                hightlight[0] = 0.5;
            }

            instance_data.push(InstanceData {
                position_offset: [block.x as f32, block.y as f32],
                object_id: x,
                highlight: hightlight,
            });
        }

        return instance_data;
    }
}


mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        bytes: "resources/shaders/blocks_terrain/mrt.vert.spv"
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        bytes: "resources/shaders/blocks_terrain/mrt.frag.spv"
    }
}

mod vs_object_map {
    vulkano_shaders::shader! {
        ty: "vertex",
        bytes: "resources/shaders/blocks_terrain/object_id.vert.spv"
    }
}

mod fs_object_map {
    vulkano_shaders::shader! {
        ty: "fragment",
        bytes: "resources/shaders/blocks_terrain/object_id.frag.spv"
    }
}
