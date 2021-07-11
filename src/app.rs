use std::sync::Arc;

use vulkano::{swapchain, Version, format, sync};
use vulkano::device::{Device, Queue};
use vulkano::image::{ImageUsage, ImageViewAbstract};
use vulkano::image::view::ImageView;
use vulkano::instance::{Instance, PhysicalDevice};
use vulkano::swapchain::{AcquireError, Swapchain, SwapchainCreationError};
use vulkano::sync::{FlushError, GpuFuture};
use vulkano_win::VkSurfaceBuild;
use winit::event::{ElementState, Event, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;
use vulkano::device::DeviceExtensions;


pub trait App {
    fn resize_swapchain(&mut self, format: format::Format, dimensions: [u32; 2]);
    fn render<F, I>(&mut self, before_future: F, dimensions: [u32; 2], image: Arc<I>) -> Box<dyn GpuFuture>
        where F: GpuFuture + 'static,
              I: ImageViewAbstract + Send + Sync + 'static;

    fn handle_event(&mut self, event: &WindowEvent);
}

pub fn run_app<F, A>(create_app: F)
    where F: Fn(Arc<Queue>) -> A,
          A: App + 'static,
{
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
    let mut app = create_app(queue.clone());
    app.resize_swapchain(swapchain.format(), surface.window().inner_size().into());

    let mut recreate_swapchain = false;
    let mut previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<dyn GpuFuture>);
    event_loop.run(move |event, _, control_flow| {
        match &event {
            Event::WindowEvent { event, window_id } => app.handle_event(event),
            _ => {},
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
            }
            Event::MainEventsCleared => {
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

                    swapchain = new_swapchain;
                    swapchain_images = new_images;

                    app.resize_swapchain(swapchain.format(), dimensions);
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
                let after_future = app.render(acquire_future, dims, swapchain_images[image_num].clone());

                // let after_future = sync::now(device.clone());

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
