// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;

use cgmath::Matrix4;
use cgmath::Vector3;
use vulkano::buffer::BufferUsage;
use vulkano::buffer::CpuAccessibleBuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, SecondaryAutoCommandBuffer};
use vulkano::command_buffer::DynamicState;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::device::Queue;
use vulkano::image;
use vulkano::image::ImageViewAbstract;
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineAbstract};
use vulkano::pipeline::blend::{AttachmentBlend, BlendFactor, BlendOp};
use vulkano::pipeline::viewport::Viewport;
use vulkano::render_pass::Subpass;

#[allow(dead_code)]
pub struct PointLightingSystem {
    gfx_queue: Arc<Queue>,
    vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex]>>,
    pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
}

#[allow(dead_code)]
impl PointLightingSystem {
    /// Initializes the point lighting system.
    pub fn new(gfx_queue: Arc<Queue>, subpass: Subpass, samples_count: image::SampleCount) -> PointLightingSystem {
        // TODO: vulkano doesn't allow us to draw without a vertex buffer, otherwise we could
        //       hard-code these values in the shader
        let vertex_buffer = {
            CpuAccessibleBuffer::from_iter(gfx_queue.device().clone(), BufferUsage::all(), false, [
                Vertex { position: [-1.0, -1.0] },
                Vertex { position: [-1.0, 3.0] },
                Vertex { position: [3.0, -1.0] }
            ].iter().cloned()).expect("failed to create buffer")
        };

        let pipeline = {
            let vs = vs::Shader::load(gfx_queue.device().clone())
                .expect("failed to create shader module");
            let fs = fs::Shader::load(gfx_queue.device().clone())
                .expect("failed to create shader module");
            let spec_consts = fs::SpecializationConstants {
                NUM_SAMPLES: samples_count as i32,
            };

            Arc::new(GraphicsPipeline::start()
                .vertex_input_single_buffer::<Vertex>()
                .vertex_shader(vs.main_entry_point(), ())
                .triangle_list()
                .viewports_dynamic_scissors_irrelevant(1)
                .fragment_shader(fs.main_entry_point(), spec_consts)
                .blend_collective(AttachmentBlend {
                    enabled: true,
                    color_op: BlendOp::Add,
                    color_source: BlendFactor::One,
                    color_destination: BlendFactor::One,
                    alpha_op: BlendOp::Max,
                    alpha_source: BlendFactor::One,
                    alpha_destination: BlendFactor::One,
                    mask_red: true,
                    mask_green: true,
                    mask_blue: true,
                    mask_alpha: true,
                })
                .render_pass(subpass)
                .build(gfx_queue.device().clone())
                .unwrap()) as Arc<_>
        };

        PointLightingSystem {
            gfx_queue,
            vertex_buffer,
            pipeline,
        }
    }

    /// Builds a secondary command buffer that applies a point lighting.
    ///
    /// This secondary command buffer will read `depth_input` and rebuild the world position of the
    /// pixel currently being processed (modulo rounding errors). It will then compare this
    /// position with `position`, and process the lighting based on the distance and orientation
    /// (similar to the directional lighting system).
    ///
    /// It then writes the output to the current framebuffer with additive blending (in other words
    /// the value will be added to the existing value in the framebuffer, and not replace the
    /// existing value).
    ///
    /// Note that in a real-world application, you probably want to pass additional parameters
    /// such as some way to indicate the distance at which the lighting decrease. In this example
    /// this value is hardcoded in the shader.
    ///
    /// - `viewport_dimensions` contains the dimensions of the current framebuffer.
    /// - `color_input` is an image containing the albedo of each object of the scene. It is the
    ///   result of the deferred pass.
    /// - `normals_input` is an image containing the normals of each object of the scene. It is the
    ///   result of the deferred pass.
    /// - `depth_input` is an image containing the depth value of each pixel of the scene. It is
    ///   the result of the deferred pass.
    /// - `screen_to_world` is a matrix that turns coordinates from framebuffer space into world
    ///   space. This matrix is used alongside with `depth_input` to determine the world
    ///   coordinates of each pixel being processed.
    /// - `position` is the position of the spot light in world coordinates.
    /// - `color` is the color of the light.
    ///
    pub fn draw<C, N, P, D>(&self, viewport_dimensions: [u32; 2], color_input: C, normals_input: N,
                            positions_input: P, depth_input: D, screen_to_world: Matrix4<f32>,
                            position: Vector3<f32>, color: [f32; 3]) -> SecondaryAutoCommandBuffer
        where C: ImageViewAbstract + Send + Sync + 'static,
              N: ImageViewAbstract + Send + Sync + 'static,
              D: ImageViewAbstract + Send + Sync + 'static,
              P: ImageViewAbstract + Send + Sync + 'static,
    {
        let push_constants = fs::ty::PushConstants {
            screen_to_world: screen_to_world.into(),
            color: [color[0], color[1], color[2], 1.0],
            position: position.extend(0.0).into(),
        };

        let layout = self.pipeline.layout().descriptor_set_layout(0).unwrap();
        let descriptor_set = PersistentDescriptorSet::start(layout.clone())
            .add_image(color_input)
            .unwrap()
            .add_image(normals_input)
            .unwrap()
            .add_image(positions_input)
            .unwrap()
            .add_image(depth_input)
            .unwrap()
            .build()
            .unwrap();

        let dynamic_state = DynamicState {
            viewports: Some(vec![Viewport {
                origin: [0.0, 0.0],
                dimensions: [viewport_dimensions[0] as f32,
                    viewport_dimensions[1] as f32],
                depth_range: 0.0..1.0,
            }]),
            ..DynamicState::none()
        };

        let mut builder = AutoCommandBufferBuilder::secondary_graphics(
            self.gfx_queue.device().clone(),
            self.gfx_queue.family(),
            CommandBufferUsage::MultipleSubmit,
            self.pipeline.subpass().clone(),
        )
            .unwrap();

        builder.draw(
            self.pipeline.clone(),
            &dynamic_state,
            vec![self.vertex_buffer.clone()],
            descriptor_set,
            push_constants,
            vec![],
        )
            .unwrap();

        builder.build().unwrap()
    }
}

#[derive(Default, Debug, Clone)]
struct Vertex {
    position: [f32; 2],
}
vulkano::impl_vertex!(Vertex, position);

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
#version 450

layout(location = 0) in vec2 position;
layout(location = 0) out vec2 v_screen_coords;

void main() {
    v_screen_coords = position;
    gl_Position = vec4(position, 0.0, 1.0);
}"
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
#version 450

layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInputMS u_diffuse;
layout(input_attachment_index = 1, set = 0, binding = 1) uniform subpassInputMS u_normals;
layout(input_attachment_index = 2, set = 0, binding = 2) uniform subpassInputMS u_positions;
layout(input_attachment_index = 3, set = 0, binding = 3) uniform subpassInputMS u_depth;

layout(push_constant) uniform PushConstants {
    // The `screen_to_world` parameter of the `draw` method.
    mat4 screen_to_world;
    // The `color` parameter of the `draw` method.
    vec4 color;
    // The `position` parameter of the `draw` method.
    vec4 position;
} push_constants;

layout(location = 0) in vec2 v_screen_coords;
layout(location = 0) out vec4 f_color;

layout (constant_id = 0) const int NUM_SAMPLES = 8;

// Manual resolve for MSAA samples
vec4 resolve(subpassInputMS tex)
{
	vec4 result = vec4(0.0);
	for (int i = 0; i < NUM_SAMPLES; i++)
	{
		vec4 val = subpassLoad(tex, i);
		result += val;
	}
	// Average resolved samples
	return result / float(NUM_SAMPLES);
}

void main() {
    float in_depth = resolve(u_depth).x;

    // Any depth superior or equal to 1.0 means that the pixel has been untouched by the deferred
    // pass. We don't want to deal with them.
    if (in_depth >= 1.0) {
        discard;
    }

    vec3 ret = vec3(0.0);
    for (int i = 0; i < NUM_SAMPLES; i++)
	{
        vec4 world = subpassLoad(u_positions, i);
        vec3 in_normal = normalize(subpassLoad(u_normals, i).xyz);

        vec3 light_direction = normalize(push_constants.position.xyz - world.xyz);
        float light_percent = abs(dot(light_direction, in_normal));

        float light_distance = length(push_constants.position.xyz - world.xyz);
        light_percent = 1.0 / exp(light_distance / 10);

        vec3 in_diffuse = subpassLoad(u_diffuse, i).rgb;

        ret += push_constants.color.rgb * in_diffuse * vec3(light_percent * 4);
    }
    f_color.rgb = ret / float(NUM_SAMPLES); // vec3(light_percent); // push_constants.color.rgb * light_percent * in_diffuse;
    f_color.a = 1.0;
}"
    }
}
