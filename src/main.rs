use std::sync::Arc;

use cgmath::{Matrix4, SquareMatrix};
use imgui;
use imgui::{Condition, im_str, Window as ImguiWindow};
use vulkano::{format, sampler};
use vulkano::device::Queue;
use vulkano::format::Format;
use vulkano::image::{ImageViewAbstract, SampleCount};
use vulkano::sync::GpuFuture;
use winit::event::{ElementState, MouseButton, WindowEvent};

use crate::base::{app, imgui_pass};
use crate::camera::Camera;
use crate::deferred::{Framebuffer, lighting_pass, render_to_framebuffer, RenderTargetDesc};
use crate::terrain_game::Map;
use crate::terrain_render_system::{RenderPipeline, TerrainRenderSystem};

mod terrain;
mod camera;
mod deferred;

mod terrain_game;
mod terrain_render_system;
mod cube;
mod mouse_picker;
mod base;


struct MyApp {
    queue: Arc<Queue>,

    camera: Camera,
    gbuffer: Framebuffer,

    mouse_picker: mouse_picker::Picker,
    terrain_map: Map,
    terrain: TerrainRenderSystem,

    lighting_pass: Option<lighting_pass::LightingPass>,

    last_cursor_pos: [u32; 2],
    cursor_pos_changed: bool,
    last_selected_object_id: Option<u32>,

    normal_texture: Option<imgui::TextureId>,

    dims: [u32; 2],
}

impl MyApp {
    fn new(queue: Arc<Queue>, swapchain_format: format::Format) -> Self {
        let mouse_picker = mouse_picker::Picker::new(queue.clone());

        let gbuffer = deferred::Framebuffer::new(queue.clone(), vec!(
            RenderTargetDesc { format: Format::R8G8B8A8Unorm, samples_count: SampleCount::Sample4 },
            RenderTargetDesc { format: Format::R16G16B16A16Sfloat, samples_count: SampleCount::Sample4 },
            RenderTargetDesc { format: Format::R16G16B16A16Sfloat, samples_count: SampleCount::Sample4 },
            RenderTargetDesc { format: Format::D32Sfloat, samples_count: SampleCount::Sample4 },
        ));

        let terrain = TerrainRenderSystem::new(
            queue.clone(),
            gbuffer.subpass(),
            mouse_picker.subpass(),
        );

        let terrain_map = Map::new(40, 40);

        let lighting_pass = Some(deferred::lighting_pass::LightingPass::new(
            queue.clone(),
            swapchain_format,
            vulkano::image::SampleCount::Sample4,
        ));

        MyApp {
            camera: Camera::new(),
            queue: queue.clone(),
            gbuffer,

            mouse_picker,

            terrain,
            terrain_map,

            lighting_pass,

            last_cursor_pos: [0, 0],
            cursor_pos_changed: false,
            last_selected_object_id: None,

            normal_texture: None,
            dims: [0, 0],
        }
    }
}


impl app::App for MyApp {
    fn resize_swapchain(&mut self, dimensions: [u32; 2], textures: &mut imgui::Textures<imgui_pass::Texture>) {
        self.camera.set_viewport(dimensions[0], dimensions[1]);
        self.gbuffer.resize_swapchain(dimensions);

        let sampler = sampler::Sampler::simple_repeat_linear(self.queue.device().clone());

        self.normal_texture = Some(textures.insert((self.gbuffer.view(1).clone(), sampler)));
        self.dims = dimensions;
    }

    fn render<F, I>(&mut self, before_future: F, dimensions: [u32; 2], image: Arc<I>) -> Box<dyn GpuFuture>
        where F: GpuFuture + 'static,
              I: ImageViewAbstract + Send + Sync + 'static
    {
        self.terrain_map.update();
        if self.cursor_pos_changed {
            let cb = self.terrain.render(
                RenderPipeline::ObjectIdMap,
                &self.terrain_map,
                dimensions,
                Matrix4::identity(),
                self.camera.view_matrix(),
                self.camera.proj_matrix(),
            );

            let entity_id = self.mouse_picker.draw(dimensions, vec![cb], self.last_cursor_pos);
            self.terrain_map.highlight(entity_id);
            self.cursor_pos_changed = false;

            self.last_selected_object_id = entity_id;
        }

        let cb = self.terrain.render(
            RenderPipeline::Diffuse,
            &self.terrain_map,
            dimensions,
            Matrix4::identity(),
            self.camera.view_matrix(),
            self.camera.proj_matrix(),
        );

        let after_future = render_to_framebuffer(
            before_future,
            self.queue.clone(),
            &self.gbuffer,
            |cmd_buf| {
                cmd_buf.execute_commands(cb).unwrap();
            });

        self.lighting_pass.as_ref().unwrap().draw(
            after_future,
            self.queue.clone(),
            image,
            self.gbuffer.view(0).clone(),
            [1.0, 1.0, 1.0],
        )
    }

    fn handle_event(&mut self, event: &WindowEvent) {
        self.camera.handle_event(event);

        match event {
            WindowEvent::CursorMoved { position, .. } => {
                if self.last_cursor_pos != [position.x as u32, position.y as u32] {
                    self.last_cursor_pos = [position.x as u32, position.y as u32];
                    self.cursor_pos_changed = true;
                }
            }
            &WindowEvent::MouseInput { state, button, .. } => {
                if (state == ElementState::Pressed) && (button == MouseButton::Left) {
                    self.terrain_map.select(self.last_selected_object_id);
                }
            }
            _ => {}
        }
    }

    fn render_gui(&mut self, ui: &mut imgui::Ui) {
        ImguiWindow::new(im_str!("stats"))
            .title_bar(false)
            .size([100.0, 40.0], Condition::FirstUseEver)
            .position([0.0, 0.0], Condition::FirstUseEver)
            .build(&ui, || {
                ui.text(format!("FPS: ({:.1})", ui.io().framerate));
            });

        let w = 210.0;
        ImguiWindow::new(im_str!("gbuffer content"))
            .size([w, 240.0], Condition::FirstUseEver)
            .position([self.dims[0] as f32 - w, 0.0], Condition::Always)
            .collapsed(true, Condition::FirstUseEver)
            .build(&ui, || {
                imgui::Image::new(self.normal_texture.unwrap(), [200.0, 200.0]).build(&ui);
            });
    }
}

fn main() {
    app::run_app(|queue, swapchain_format| -> MyApp {
        MyApp::new(queue, swapchain_format)
    });
}
