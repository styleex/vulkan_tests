use std::sync::Arc;

use vulkano::{image, render_pass, sampler};
use vulkano::buffer::BufferUsage;
use vulkano::buffer::CpuAccessibleBuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, SubpassContents};
use vulkano::command_buffer::DynamicState;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::device::Queue;
use vulkano::image::ImageViewAbstract;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::GraphicsPipelineAbstract;
use vulkano::pipeline::viewport::Viewport;
use vulkano::render_pass::{RenderPass, Subpass};
use vulkano::sync::GpuFuture;


pub struct LightingPass {
    gfx_queue: Arc<Queue>,
    vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex]>>,
    pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
    sampler: Arc<sampler::Sampler>,

    render_pass: Arc<RenderPass>,
}

impl LightingPass {
    pub fn new(gfx_queue: Arc<Queue>, output_format: vulkano::format::Format, input_samples: image::SampleCount) -> LightingPass
    {
        let render_pass = Arc::new(
            vulkano::single_pass_renderpass!(
                gfx_queue.device().clone(),
                attachments: {
                    // The image that will contain the final rendering (in this example the swapchain
                    // image, but it could be another image).
                    final_color: {
                        load: Clear,
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
                NUM_SAMPLES: input_samples as i32,
            };

            Arc::new(GraphicsPipeline::start()
                .vertex_input_single_buffer::<Vertex>()
                .vertex_shader(vs.main_entry_point(), ())
                .triangle_list()
                .viewports_dynamic_scissors_irrelevant(1)
                .fragment_shader(fs.main_entry_point(), spec_consts)
                .blend_alpha_blending()
                .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
                .build(gfx_queue.device().clone())
                .unwrap()) as Arc<_>
        };

        let sampler = sampler::Sampler::new(
            gfx_queue.device().clone(),
            sampler::Filter::Linear,
            sampler::Filter::Linear,
            sampler::MipmapMode::Nearest,
            sampler::SamplerAddressMode::Repeat,
            sampler::SamplerAddressMode::Repeat,
            sampler::SamplerAddressMode::Repeat,
            1.0,
            1.0,
            0.0,
            100.0,
        ).unwrap();

        LightingPass {
            gfx_queue,
            vertex_buffer,
            pipeline,
            sampler,
            render_pass,
        }
    }

    pub fn draw<F, I, C>(&self,
                         before_future: F,
                         gfx_queue: Arc<Queue>,
                         target_image: Arc<I>,
                         color_input: C,
                         ambient_color: [f32; 3],
    ) -> Box<dyn GpuFuture>
        where
            F: GpuFuture + 'static,
            C: ImageViewAbstract + Send + Sync + 'static,
            I: ImageViewAbstract + Send + Sync + 'static
    {
        let framebuffer = Arc::new(
            render_pass::Framebuffer::start(self.render_pass.clone())
                .add(target_image.clone())
                .unwrap()
                .build()
                .unwrap()
        );

        let push_constants = fs::ty::PushConstants {
            color: [ambient_color[0], ambient_color[1], ambient_color[2], 1.0],
        };

        let layout = self.pipeline.layout().descriptor_set_layout(0).unwrap();
        let descriptor_set = PersistentDescriptorSet::start(layout.clone())
            .add_sampled_image(color_input, self.sampler.clone())
            .unwrap()
            .build()
            .unwrap();

        let viewport_dimensions = target_image.image().dimensions().width_height();
        let dynamic_state = DynamicState {
            viewports: Some(vec![Viewport {
                origin: [0.0, 0.0],
                dimensions: [viewport_dimensions[0] as f32,
                    viewport_dimensions[1] as f32],
                depth_range: 0.0..1.0,
            }]),
            ..DynamicState::none()
        };

        let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
            self.gfx_queue.device().clone(),
            self.gfx_queue.family(),
            CommandBufferUsage::OneTimeSubmit,
        ).unwrap();

        command_buffer_builder
            .begin_render_pass(
                framebuffer,
                SubpassContents::Inline,
                vec![
                    [0.0, 0.0, 0.0, 0.0].into(),
                ],
            ).unwrap();

        command_buffer_builder
            .draw(
                self.pipeline.clone(),
                &dynamic_state,
                vec![self.vertex_buffer.clone()],
                descriptor_set,
                push_constants,
                vec![],
            )
            .unwrap();

        command_buffer_builder.end_render_pass().unwrap();

        let cmd_buf = command_buffer_builder.build().unwrap();

        Box::new(before_future.then_execute(gfx_queue.clone(), cmd_buf).unwrap())
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
layout (location = 1) out vec2 outUV;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    outUV = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);

}"
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
#version 450

// The `color_input` parameter of the `draw` method.
layout(set = 0, binding = 0) uniform sampler2DMS u_diffuse;

layout(push_constant) uniform PushConstants {
    // The `ambient_color` parameter of the `draw` method.
    vec4 color;
} push_constants;

layout(location = 0) out vec4 f_color;

layout (constant_id = 0) const int NUM_SAMPLES = 8;

layout (location = 1) in vec2 inUV;

void main() {
	vec4 result = vec4(0.0);
	for (int i = 0; i < NUM_SAMPLES; i++)
	{
		vec4 val = texelFetch(u_diffuse, ivec2(gl_FragCoord.xy), i);
		result += val;
	}
	// Average resolved samples
	result = result / float(NUM_SAMPLES);

    f_color.rgb = push_constants.color.rgb * result.rgb;
    f_color.a = 1.0;
}"
    }
}
