use vulkano::buffer::{BufferUsage, CpuBufferPool};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::device::{Device, DeviceExtensions};
use vulkano::format::Format;
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, Subpass, RenderPassAbstract};
use vulkano::image::{AttachmentImage, ImageUsage, Dimensions, ImmutableImage};
use vulkano::image::SwapchainImage;
use vulkano::instance::{Instance, PhysicalDevice};
use vulkano::pipeline::GraphicsPipeline;
use vulkano::descriptor::pipeline_layout::PipelineLayoutAbstract;
use vulkano::pipeline::viewport::Viewport;
use vulkano::swapchain::{AcquireError, PresentMode, SurfaceTransform, Swapchain, SwapchainCreationError, ColorSpace, FullscreenExclusive};
use vulkano::swapchain;
use vulkano::sync::{GpuFuture, FlushError};
use vulkano::sync;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;

use vulkano_win::VkSurfaceBuild;
use winit::window::{WindowBuilder, Window};
use winit::event_loop::{EventLoop, ControlFlow};
use winit::event::{Event, WindowEvent};

use std::sync::Arc;
use crate::camera::Camera;
use winit::event::{ElementState, VirtualKeyCode};

mod terrain;
mod camera;

use cgmath::{Matrix4, SquareMatrix};
use crate::terrain::{Terrain, Vertex, HeightMap};
use std::io::Cursor;
use crate::block_render::BlockRender;
use crate::terrain_game::Map;
use crate::terrain_render_system::TerrainRenderSystem;

mod terrain_game;
mod block_render;
mod terrain_render_system;
mod cube;

fn main() {
    let required_extensions = vulkano_win::required_extensions();
    let instance = Instance::new(None, &required_extensions, None).unwrap();
    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();
    println!("Using device: {} (type: {:?})", physical.name(), physical.ty());

    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new().build_vk_surface(&event_loop, instance.clone()).unwrap();

    let queue_family = physical.queue_families().find(|&q| {
        // We take the first queue that supports drawing to our window.
        q.supports_graphics() && surface.is_supported(q).unwrap_or(false)
    }).unwrap();

    let device_ext = DeviceExtensions { khr_swapchain: true, ..DeviceExtensions::none() };
    let (device, mut queues) = Device::new(physical, physical.supported_features(), &device_ext,
                                           [(queue_family, 0.5)].iter().cloned()).unwrap();
    let queue = queues.next().unwrap();

    let mut cam = Camera::new();
    let (mut swapchain, images) = {
        let caps = surface.capabilities(physical).unwrap();
        let usage = caps.supported_usage_flags;

        let alpha = caps.supported_composite_alpha.iter().next().unwrap();
        let format = caps.supported_formats[0].0;

        let dimensions: [u32; 2] = surface.window().inner_size().into();

        cam.set_viewport(dimensions[0], dimensions[1]);
        Swapchain::new(device.clone(), surface.clone(), caps.min_image_count, format,
                       dimensions, 1, usage, &queue, SurfaceTransform::Identity, alpha,
                       PresentMode::Fifo, FullscreenExclusive::Default, true, ColorSpace::SrgbNonLinear).unwrap()
    };

    let render_pass = Arc::new(vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: swapchain.format(),
                samples: 1,
            },
            depth: {
                load: Clear,
                store: DontCare,
                format: Format::D16Unorm,
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {depth}
        }
    ).unwrap());

    let mut dynamic_state = DynamicState { line_width: None, viewports: None, scissors: None, compare_mask: None, write_mask: None, reference: None };

//    let terr = Terrain::new(queue.clone(), HeightMap::from_png(), Subpass::from(render_pass.clone(), 0).unwrap());
//    let cube = BlockRender::new(queue.clone(), Subpass::from(render_pass.clone(), 0).unwrap());


    let terrain_map = Arc::new(Map::new(10, 10));
    let mut terrain_rs = TerrainRenderSystem::new(queue.clone(), Subpass::from(render_pass.clone(), 0).unwrap());

    let mut framebuffers = window_size_dependent_setup(&images, render_pass.clone(), &mut dynamic_state, queue.clone());
    let mut recreate_swapchain = false;

    let mut previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<dyn GpuFuture>);
    event_loop.run(move |event, _, control_flow| {
        match &event {
            Event::WindowEvent { event, .. } => cam.handle_event(event),
            _ => (),
        }


        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent { event: WindowEvent::Resized(_), .. } => {
                recreate_swapchain = true;
            }
            Event::WindowEvent { event: WindowEvent::KeyboardInput { input, .. }, .. } => {
                if input.state == ElementState::Released {
                    if input.virtual_keycode == Some(VirtualKeyCode::Escape) {
                        *control_flow = ControlFlow::Exit;
                        return;
                    }
                }

                cam.handle_keyboard(input);
            }

            Event::RedrawEventsCleared => {
                previous_frame_end.as_mut().unwrap().cleanup_finished();

                if recreate_swapchain {
                    let dimensions: [u32; 2] = surface.window().inner_size().into();
                    let (new_swapchain, new_images) = match swapchain.recreate_with_dimensions(dimensions) {
                        Ok(r) => r,
                        Err(SwapchainCreationError::UnsupportedDimensions) => return,
                        Err(e) => panic!("Failed to recreate swapchain: {:?}", e)
                    };

                    swapchain = new_swapchain;
                    framebuffers = window_size_dependent_setup(&new_images, render_pass.clone(), &mut dynamic_state, queue.clone());
                    cam.set_viewport(dimensions[0], dimensions[1]);
                    recreate_swapchain = false;
                }

                let (image_num, suboptimal, acquire_future) = match swapchain::acquire_next_image(swapchain.clone(), None) {
                    Ok(r) => r,
                    Err(AcquireError::OutOfDate) => {
                        recreate_swapchain = true;
                        return;
                    }
                    Err(e) => panic!("Failed to acquire next image: {:?}", e)
                };

                if suboptimal {
                    recreate_swapchain = true;
                }

                let clear_values = vec!([0.0, 0.0, 0.0, 1.0].into(), 1.0f32.into());

                let world = Matrix4::identity();

                let dimensions: [u32; 2] = surface.window().inner_size().into();
//                let d = terr.draw(dimensions, world, cam.view_matrix(), cam.proj_matrix());
//
//                let d2 = cube.draw(dimensions, world, cam.view_matrix(), cam.proj_matrix());

                let d3 = terrain_rs.render(terrain_map.clone(), dimensions, world, cam.view_matrix(), cam.proj_matrix());
                let command_buffer = unsafe {
                    AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap()
                        .begin_render_pass(framebuffers[image_num].clone(), false, clear_values).unwrap()
//                        .execute_commands(d).unwrap()
//                        .execute_commands(d2).unwrap()
                        .execute_commands(d3).unwrap()
                        .end_render_pass().unwrap()
                        .build().unwrap()
                };

                let future = previous_frame_end.take().unwrap()
                    .join(acquire_future)
                    .then_execute(queue.clone(), command_buffer).unwrap()
                    .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
                    .then_signal_fence_and_flush();

                match future {
                    Ok(future) => {
                        previous_frame_end = Some(Box::new(future) as Box<_>);
                    }
                    Err(FlushError::OutOfDate) => {
                        recreate_swapchain = true;
                        previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<_>);
                    }
                    Err(e) => {
                        println!("Failed to flush future: {:?}", e);
                        previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<_>);
                    }
                }
            }
            _ => ()
        }
    });
}

/// This method is called once during initialization, then again whenever the window is resized
fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    dynamic_state: &mut DynamicState,
    gfx_queue: Arc<vulkano::device::Queue>,
) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>> {
    let dimensions = images[0].dimensions();

    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [dimensions[0] as f32, dimensions[1] as f32],
        depth_range: 0.0..1.0,
    };
    dynamic_state.viewports = Some(vec!(viewport));
    let atch_usage = ImageUsage {
        transient_attachment: true,
        input_attachment: true,
        ..ImageUsage::none()
    };
    let depth_buffer = AttachmentImage::with_usage(gfx_queue.device().clone(),
                                                   [dimensions[0], dimensions[1]],
                                                   Format::D16Unorm,
                                                   atch_usage)
        .unwrap();

    images.iter().map(|image| {
        Arc::new(
            Framebuffer::start(render_pass.clone())
                .add(image.clone()).unwrap()
                .add(depth_buffer.clone()).unwrap()
                .build().unwrap()
        ) as Arc<dyn FramebufferAbstract + Send + Sync>
    }).collect::<Vec<_>>()
}
