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
use winit::event::{Event, WindowEvent, MouseButton};

use std::sync::Arc;
use crate::camera::Camera;
use winit::event::{ElementState, VirtualKeyCode};

mod terrain;
mod camera;
mod deferred;

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

fn get_entity_id(r: u8, g: u8, b: u8, a: u8) -> Option<usize> {
    if a == 0 {
        None
    } else {
        Some((r as usize) | (g as usize) << 8 | (b as usize) << 16)
    }
}

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

    let mut frame_system = deferred::FrameSystem::new(queue.clone(), swapchain.format());
    let terrain_map = Arc::new(Map::new(10, 10));
    let mut terrain_rs = TerrainRenderSystem::new(queue.clone(), frame_system.deferred_subpass());

    let world = Matrix4::identity();
    let mut recreate_swapchain = false;
    let mut cursor_pos = [0, 0];

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
            Event::WindowEvent { event: WindowEvent::CursorMoved { position, .. }, ..} => {
                cursor_pos = [position.x as u32, position.y as u32];
            }
            Event::WindowEvent { event: WindowEvent::MouseInput { state, button, .. }, ..} => {
                if (state == ElementState::Pressed) && (button == MouseButton::Left) {
                    let buffer_content = frame_system.object_id_cpu.read().unwrap();
                    let buf_pos = 4 * (cursor_pos[1] * (1600) + cursor_pos[0]) as usize;

                    let entity_id = get_entity_id(
                        buffer_content[buf_pos],
                        buffer_content[buf_pos + 1],
                        buffer_content[buf_pos + 2],
                        buffer_content[buf_pos + 3],
                    );
                }
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

                let future = previous_frame_end.take().unwrap().join(acquire_future);
                let mut frame = frame_system.frame(future, images[image_num].clone(), Matrix4::identity());
                let mut after_future = None;
                while let Some(pass) = frame.next_pass() {
                    match pass {
                        deferred::Pass::Deferred(mut draw_pass) => {
                            let cb = terrain_rs.render(terrain_map.clone(), draw_pass.viewport_dimensions(),
                                                       world, cam.view_matrix(), cam.proj_matrix());
                            draw_pass.execute(cb);
                        }
                        deferred::Pass::Lighting(mut lighting) => {
                            lighting.ambient_light([1.0, 1.0, 1.0]);
//                            lighting.directional_light(Vector3::new(0.2, -0.1, -0.7), [0.6, 0.6, 0.6]);
//                            lighting.point_light(Vector3::new(0.5, -0.5, -0.1), [1.0, 0.0, 0.0]);
//                            lighting.point_light(Vector3::new(-0.9, 0.2, -0.15), [0.0, 1.0, 0.0]);
//                            lighting.point_light(Vector3::new(0.0, 0.5, -0.05), [0.0, 0.0, 1.0]);
                        }
                        deferred::Pass::Finished(af) => {
                            after_future = Some(af);
                        }
                    }
                }

                let future = after_future.unwrap()
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
