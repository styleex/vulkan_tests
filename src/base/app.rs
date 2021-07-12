use std::sync::Arc;

use imgui::{Context, FontConfig, FontGlyphRanges, FontSource};
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use vulkano::{format, swapchain, sync, Version};
use vulkano::device::{Device, Queue};
use vulkano::device::DeviceExtensions;
use vulkano::image::{ImageUsage, ImageViewAbstract};
use vulkano::image::view::ImageView;
use vulkano::instance::{Instance, InstanceExtensions, PhysicalDevice};
use vulkano::instance::debug::{DebugCallback, MessageSeverity, MessageType};
use vulkano::swapchain::{AcquireError, Swapchain, SwapchainCreationError};
use vulkano::sync::{FlushError, GpuFuture};
use vulkano_win::VkSurfaceBuild;
use winit::event::{ElementState, Event, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

use super::imgui_pass::GuiPass;

pub trait App {
    fn resize_swapchain(&mut self, dimensions: [u32; 2]);
    fn render<F, I>(&mut self, before_future: F, dimensions: [u32; 2], image: Arc<I>) -> Box<dyn GpuFuture>
        where F: GpuFuture + 'static,
              I: ImageViewAbstract + Send + Sync + 'static;

    fn handle_event(&mut self, event: &WindowEvent);

    fn render_gui(&mut self, ui: &mut imgui::Ui);
}

pub fn run_app<F, A>(create_app: F)
    where F: Fn(Arc<Queue>, format::Format) -> A,
          A: App + 'static,
{
    let required_extensions = InstanceExtensions {
        ext_debug_utils: true,
        ..vulkano_win::required_extensions()
    };

    #[cfg(not(target_os = "macos"))]
        let layers = vec!["VK_LAYER_LUNARG_standard_validation"];

    #[cfg(target_os = "macos")]
        let layers = vec!["VK_LAYER_KHRONOS_validation"];

    let instance = Instance::new(None, Version::V1_1, &required_extensions, layers).unwrap();

    let severity = MessageSeverity {
        error: true,
        warning: true,
        information: false,
        verbose: false,
    };

    let ty = MessageType::all();

    let _debug_callback = DebugCallback::new(&instance, severity, ty, |msg| {
        let severity = if msg.severity.error {
            "error"
        } else if msg.severity.warning {
            "warning"
        } else if msg.severity.information {
            "information"
        } else if msg.severity.verbose {
            "verbose"
        } else {
            panic!("no-impl");
        };

        let ty = if msg.ty.general {
            "general"
        } else if msg.ty.validation {
            "validation"
        } else if msg.ty.performance {
            "performance"
        } else {
            panic!("no-impl");
        };

        println!("{} {} {}: {}",
                 msg.layer_prefix.unwrap_or("unknown"),
                 ty,
                 severity,
                 msg.description
        );
    }).ok();

    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();

    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new().build_vk_surface(&event_loop, instance.clone()).unwrap();

    let queue_family = physical.queue_families().find(|&q| {
        q.supports_graphics() && surface.is_supported(q).unwrap_or(false)
    }).unwrap();

    let device_ext = DeviceExtensions { khr_swapchain: true, ..DeviceExtensions::none() };
    let (device, mut queues) = Device::new(physical, physical.supported_features(), &device_ext,
                                           [(queue_family, 0.5)].iter().cloned()).unwrap();
    let queue = queues.next().unwrap();

    let (mut swapchain, mut swapchain_images) = {
        let caps = surface.capabilities(physical).unwrap();
        let composite_alpha = caps.supported_composite_alpha.iter().next().unwrap();
        let format = caps.supported_formats[0].0;
        let dimensions: [u32; 2] = surface.window().inner_size().into();

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

    // [IMGUI]
    let (mut imgui, mut imgui_platform) = {
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
                data: include_bytes!("../../resources/font/mplus-1p-regular.ttf"),
                size_pixels: font_size,
                config: Some(FontConfig {
                    rasterizer_multiply: 1.75,
                    glyph_ranges: FontGlyphRanges::cyrillic(),
                    ..FontConfig::default()
                }),
            },
        ]);

        imgui.io_mut().font_global_scale = (1.0 / hidpi_factor) as f32;

        (imgui, platform)
    };

    let mut imgui_render = GuiPass::new(&mut imgui, queue.clone(), swapchain.format());
    // [/IMGUI]

    let mut app = create_app(queue.clone(), swapchain.format());
    app.resize_swapchain(surface.window().inner_size().into());

    let mut recreate_swapchain = false;
    let mut previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<dyn GpuFuture>);
    event_loop.run(move |event, _, control_flow| {
        match &event {
            Event::WindowEvent { event, window_id: _ } => app.handle_event(event),
            _ => {}
        }

        imgui_platform.handle_event(imgui.io_mut(), surface.window(), &event);

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
            Event::MainEventsCleared => {
                imgui_platform.prepare_frame(imgui.io_mut(), surface.window()).unwrap();
                surface.window().request_redraw();
            }
            Event::RedrawRequested(_) => {
                previous_frame_end.as_mut().unwrap().cleanup_finished();
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

                    assert_eq!(swapchain.format(), new_swapchain.format());
                    swapchain = new_swapchain;
                    swapchain_images = new_images;

                    app.resize_swapchain(dimensions);
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

                let dims: [u32; 2] = surface.window().inner_size().into();
                let mut after_future = app.render(acquire_future, dims, swapchain_images[image_num].clone());

                // [IMGUI]
                let mut ui = imgui.frame();
                imgui_platform.prepare_render(&ui, surface.window());
                app.render_gui(&mut ui);

                let draw_data = ui.render();

                after_future = imgui_render.draw(
                    after_future,
                    queue.clone(),
                    swapchain_images[image_num].clone(),
                    dims,
                    draw_data,
                );
                // [/IMGUI]

                let frame_future = after_future
                    .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
                    .then_signal_fence_and_flush();

                match frame_future {
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
