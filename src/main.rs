use cgmath::{Matrix4, SquareMatrix, Vector3};
use imgui;
use imgui::{Condition, Context, FontConfig, FontGlyphRanges, FontSource, im_str, Window};
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use vulkano::{swapchain, Version, image};
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

    // IMGUI
    let mut imgui = Context::create();
    imgui.set_ini_filename(None);

    let mut platform = WinitPlatform::init(&mut imgui);
    platform.attach_window(imgui.io_mut(), &surface.window(), HiDpiMode::Rounded);

    let hidpi_factor = platform.hidpi_factor();
    let font_size = (13.0 * hidpi_factor) as f32;
    imgui.fonts().add_font(&[
        FontSource::DefaultFontData {
            config: Some(FontConfig {
                size_pixels: font_size,
                ..FontConfig::default()
            }),
        },
        FontSource::TtfData {
            data: include_bytes!("../resources/font/mplus-1p-regular.ttf"),
            size_pixels: font_size,
            config: Some(FontConfig {
                rasterizer_multiply: 1.75,
                glyph_ranges: FontGlyphRanges::japanese(),
                ..FontConfig::default()
            }),
        },
    ]);

    imgui.io_mut().font_global_scale = (1.0 / hidpi_factor) as f32;
    // /IMGUI

    let mut picker = mouse_picker::Picker::new(queue.clone());

    let mut frame_system = deferred::FrameSystem::new(queue.clone(), swapchain.format(), &mut imgui, image::SampleCount::Sample8);

    let mut terrain_map = Map::new(40, 40);
    let mut terrain_rs = TerrainRenderSystem::new(queue.clone(),
                                                  frame_system.deferred_subpass(),
                                                  picker.subpass());

    let world = Matrix4::identity();
    let mut recreate_swapchain = false;

    let mut cursor_pos = [0, 0];
    let mut cursor_pos_changed = false;
    let mut entity_id: Option<u32> = None;
    let mut imgui_hovered = false;

    let mut previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<dyn GpuFuture>);
    event_loop.run(move |event, _, control_flow| {
        if imgui_hovered {
            cam.clear_mouse();
        } else {
            match &event {
                Event::WindowEvent { event, .. } => cam.handle_event(event),
                _ => (),
            }
        }

        platform.handle_event(imgui.io_mut(), surface.window(), &event);

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
            Event::MainEventsCleared => {
                platform
                    .prepare_frame(imgui.io_mut(), &surface.window())
                    .expect("Failed to prepare frame");
                surface.window().request_redraw();
            }
            Event::RedrawRequested(_) => {
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
                let mut frame = frame_system.frame(future, images[image_num].clone(), world * cam.view_matrix() * cam.proj_matrix());
                let mut after_future = None;
                while let Some(pass) = frame.next_pass() {
                    match pass {
                        deferred::Pass::Deferred(mut draw_pass) => {
                            let cb = terrain_rs
                                .render(
                                    RenderPipeline::Diffuse,
                                    &terrain_map,
                                    draw_pass.viewport_dimensions(),
                                    world,
                                    cam.view_matrix(),
                                    cam.proj_matrix(),
                                );
                            draw_pass.execute(cb);
                        }
                        deferred::Pass::Lighting(mut lighting) => {
                            lighting.ambient_light([0.3, 0.3, 0.3]);
                            // lighting.directional_light(Vector3::new(-0.3, -1.0, -0.3), [0.6, 0.6, 0.6]);
                           lighting.point_light(Vector3::new(-5.0, 1.2, -5.0), [0.9, 0.9, 0.9]);
                           // lighting.point_light(Vector3::new(-0.9, 0.2, -0.15), [0.0, 1.0, 0.0]);
                           // lighting.point_light(Vector3::new(0.0, 0.5, -0.05), [0.0, 0.0, 1.0]);
                        }
                        deferred::Pass::UI(mut ui_pass) => {
                            ui_pass.draw(&mut imgui, ui_pass.viewport_dimensions(), |ui| {
                                Window::new(im_str!("Stats"))
                                    .size([100.0, 50.0], Condition::FirstUseEver)
                                    .position([0.0, 0.0], Condition::FirstUseEver)
                                    .build(&ui, || {
                                        ui.text(format!("FPS: ({:.1})", ui.io().framerate));
                                    });

                                platform.prepare_render(&ui, surface.window());
                                imgui_hovered = ui.is_any_item_active();
                            });
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
