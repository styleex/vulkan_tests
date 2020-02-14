// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use vulkano::buffer::BufferUsage;
use vulkano::buffer::{CpuAccessibleBuffer, CpuBufferPool};
use vulkano::command_buffer::AutoCommandBuffer;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::command_buffer::DynamicState;
use vulkano::device::Queue;
use vulkano::framebuffer::RenderPassAbstract;
use vulkano::framebuffer::Subpass;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::GraphicsPipelineAbstract;
use vulkano::pipeline::viewport::Viewport;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use cgmath::{Matrix3, Matrix4, Point3, Vector3, Rad, SquareMatrix};


use std::sync::Arc;

pub struct TriangleDrawSystem {
    gfx_queue: Arc<Queue>,
    vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex]>>,
    pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
    uniform_buffer: CpuBufferPool<vs::ty::Data>,
}

impl TriangleDrawSystem {
    /// Initializes a triangle drawing system.
    pub fn new<R>(gfx_queue: Arc<Queue>, subpass: Subpass<R>) -> TriangleDrawSystem
        where R: RenderPassAbstract + Send + Sync + 'static
    {
        let vertex_buffer = {
            CpuAccessibleBuffer::from_iter(gfx_queue.device().clone(), BufferUsage::all(), false, [
                Vertex { position: [-0.5, -0.25, -1.0] },
                Vertex { position: [0.0, 0.5, -1.0] },
                Vertex { position: [0.25, -0.1, -1.0] }
            ].iter().cloned()).expect("failed to create buffer")
        };

        let uniform_buffer = CpuBufferPool::<vs::ty::Data>::new(gfx_queue.device().clone(), BufferUsage::all());

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
                .depth_stencil_simple_depth()
                .render_pass(subpass)
                .build(gfx_queue.device().clone())
                .unwrap()) as Arc<_>
        };

        TriangleDrawSystem {
            gfx_queue: gfx_queue,
            vertex_buffer: vertex_buffer,
            pipeline: pipeline,
            uniform_buffer: uniform_buffer,
        }
    }

    /// Builds a secondary command buffer that draws the triangle on the current subpass.
    pub fn draw(&self, viewport_dimensions: [u32; 2], world: Matrix4<f32>, view: Matrix4<f32>, proj: Matrix4<f32>) -> AutoCommandBuffer {
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
            .draw(self.pipeline.clone(),
                  &DynamicState {
                      viewports: Some(vec![Viewport {
                          origin: [0.0, 0.0],
                          dimensions: [viewport_dimensions[0] as f32,
                              viewport_dimensions[1] as f32],
                          depth_range: 0.0..1.0,
                      }]),
                      ..DynamicState::none()
                  },
                  vec![self.vertex_buffer.clone()], set.clone(), ())
            .unwrap()
            .build()
            .unwrap()
    }
}

#[derive(Default, Debug, Clone)]
struct Vertex {
    position: [f32; 3]
}
vulkano::impl_vertex!(Vertex, position);

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
#version 450

layout(location = 0) in vec3 position;

layout(set = 0, binding = 0) uniform Data {
    mat4 world;
    mat4 view;
    mat4 proj;
} uniforms;

void main() {
    mat4 worldview = uniforms.view * uniforms.world;
    gl_Position = uniforms.proj * worldview * vec4(position, 1.0);
}"
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
#version 450

layout(location = 0) out vec4 f_color;
layout(location = 1) out vec3 f_normal;

void main() {
    f_color = vec4(1.0, 1.0, 1.0, 1.0);
    f_normal = vec3(0.0, 0.0, 1.0);
}"
    }
}
