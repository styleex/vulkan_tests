use cgmath::{Matrix4, SquareMatrix};
use vulkano::{swapchain, Version};
use vulkano::device::{Device, DeviceExtensions};
use vulkano::image::{ImageAccess, ImageUsage};
use vulkano::image::view::ImageView;
use vulkano::instance::{Instance, PhysicalDevice};
use vulkano::swapchain::{AcquireError, Swapchain, SwapchainCreationError};
use vulkano::sync::{FlushError, GpuFuture};
use vulkano::sync;
use vulkano_win::VkSurfaceBuild;
use winit::event::{Event, MouseButton, WindowEvent};
use winit::event::{ElementState, VirtualKeyCode};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

use crate::camera::Camera;
use crate::terrain_game::Map;
use crate::terrain_render_system::{RenderPipeline, TerrainRenderSystem};

mod terrain;
mod camera;
mod deferred;

mod terrain_game;
mod block_render;
mod terrain_render_system;
mod cube;
mod mouse_picker;

fn main() {
    let required_extensions = vulkano_win::required_extensions();
    let instance = Instance::new(None, Version::V1_1, &required_extensions, None).unwrap();
    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();

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
    let (mut swapchain, mut images) = {
        let caps = surface.capabilities(physical).unwrap();
        let composite_alpha = caps.supported_composite_alpha.iter().next().unwrap();
        let format = caps.supported_formats[0].0;
        let dimensions: [u32; 2] = surface.window().inner_size().into();

        cam.set_viewport(dimensions[0], dimensions[1]);

        let (swapchain, images) = Swapchain::start(device.clone(), surface.clone())
            .num_images(caps.min_image_count)
            .format(format)
            .dimensions(dimensions)
            .usage(ImageUsage::color_attachment())
            .sharing_mode(&queue)
            .composite_alpha(composite_alpha)
            .build()
            .unwrap();

        let images = images
            .into_iter()
            .map(|image| ImageView::new(image.clone()).unwrap())
            .collect::<Vec<_>>();
        (swapchain, images)
    };

    let mut picker = mouse_picker::Picker::new(queue.clone());
    let mut frame_system = deferred::FrameSystem::new(queue.clone(), swapchain.format());

    let mut terrain_map = Map::new(40, 40);
    let mut terrain_rs = TerrainRenderSystem::new(queue.clone(),
                                                  frame_system.deferred_subpass(),
                                                  picker.subpass());

    let world = Matrix4::identity();
    let mut recreate_swapchain = false;

    let mut cursor_pos = [0, 0];
    let mut cursor_pos_changed = false;
    let mut entity_id: Option<u32> = None;

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
            Event::WindowEvent { event: WindowEvent::CursorMoved { position, .. }, .. } => {
                if cursor_pos != [position.x as u32, position.y as u32] {
                    cursor_pos = [position.x as u32, position.y as u32];
                    cursor_pos_changed = true;
                }
            }
            Event::WindowEvent { event: WindowEvent::MouseInput { state, button, .. }, .. } => {
                if (state == ElementState::Pressed) && (button == MouseButton::Left) {
                    println!("{:?}", entity_id);
                    terrain_map.select(entity_id);
                }
            }

            Event::RedrawEventsCleared => {
                previous_frame_end.as_mut().unwrap().cleanup_finished();

                terrain_map.update();
                if recreate_swapchain {
                    let dimensions: [u32; 2] = surface.window().inner_size().into();
                    let (new_swapchain, new_images) =
                        match swapchain.recreate().dimensions(dimensions).build() {
                            Ok(r) => r,
                            Err(SwapchainCreationError::UnsupportedDimensions) => return,
                            Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                        };

                    let new_images = new_images
                        .into_iter()
                        .map(|image| ImageView::new(image.clone()).unwrap())
                        .collect::<Vec<_>>();

                    swapchain = new_swapchain;
                    images = new_images;
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

                if cursor_pos_changed {

                    let dims = images[image_num].image().dimensions().width_height();

                    let cb = terrain_rs.render(RenderPipeline::ObjectIdMap,
                                               &terrain_map, dims,
                                               world, cam.view_matrix(), cam.proj_matrix());

                    entity_id = picker.draw(dims, vec!(cb), cursor_pos[0], cursor_pos[1]);
                    terrain_map.highlight(entity_id);

                    cursor_pos_changed = false;
                }

                let future = previous_frame_end.take().unwrap().join(acquire_future);
                let mut frame = frame_system.frame(future, images[image_num].clone(), Matrix4::identity());
                let mut after_future = None;
                while let Some(pass) = frame.next_pass() {
                    match pass {
                        deferred::Pass::Deferred(mut draw_pass) => {
                            let cb = terrain_rs.render(RenderPipeline::Diffuse,
                                                       &terrain_map, draw_pass.viewport_dimensions(),
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
                        future.wait(None).unwrap();
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
