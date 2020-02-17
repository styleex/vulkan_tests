use cgmath::{Vector3, Matrix4, SquareMatrix, Point3, vec3, Deg};
use cgmath::{Rad, Angle};
use winit::event::{KeyboardInput, ElementState, VirtualKeyCode, WindowEvent, MouseButton};
use cgmath::InnerSpace;
use std::f32;


pub struct Camera {
    position: Point3<f32>,
    proj: Matrix4<f32>,
    yaw: f32,
    pitch: f32,

    mouse_pressed: bool,
    last_mouse_position: [i32; 2],

    view_dir: Vector3<f32>,
    up_dir: Vector3<f32>,

    viewport: [u32; 2],
}

impl Camera {
    pub fn new() -> Camera {
        Camera {
            position: Point3::new(0.0, -2.0, 0.0),
            proj: Matrix4::identity(),
            mouse_pressed: false,
            last_mouse_position: [0, 0],
            viewport: [0, 0],
            view_dir: vec3(0.0, 0.0, -1.0),
            up_dir: vec3(0.0, -1.0, 0.0),
            yaw: -90.0,
            pitch: 0.0,
        }
    }

    pub fn set_viewport(&mut self, w: u32, h: u32) {
        println!("{:?} {:?}", w, h);
        self.viewport = [w, h];
        self.proj = cgmath::perspective(
            Rad::from(Deg(45.0)),
            w as f32 / h as f32,
            0.01,
            100.0);
    }

    pub fn view_matrix(&self) -> Matrix4<f32> {
//        let dir = rotation.transform_vector(vec3(0.0, 0.0, -1.0));

        return Matrix4::<f32>::look_at(self.position, self.position + self.view_dir, self.up_dir);
    }

    pub fn proj_matrix(&self) -> Matrix4<f32> {
        self.proj
    }

    pub fn handle_keyboard(&mut self, input: KeyboardInput) {
        if input.state == ElementState::Pressed {
            match input.virtual_keycode {
                Some(VirtualKeyCode::W) => self.position += self.view_dir * 0.1,
                Some(VirtualKeyCode::S) => self.position -= self.view_dir * 0.1,

                Some(VirtualKeyCode::A) => self.position -= self.view_dir.cross(self.up_dir) * 0.1,
                Some(VirtualKeyCode::D) => self.position += self.view_dir.cross(self.up_dir) * 0.1,
                Some(VirtualKeyCode::Space) => self.position.y += 0.1,
                Some(VirtualKeyCode::LShift) => self.position.y -= 0.1,
                _ => (),
            }
        }
    }

    pub fn handle_event(&mut self, event: &WindowEvent) {
        match event {
            &WindowEvent::MouseInput { state, button, .. } => {
                self.mouse_pressed = (state == ElementState::Pressed) && (button == MouseButton::Left);
            }
            &WindowEvent::CursorMoved { position, .. } => {
                if !self.mouse_pressed {
                    self.last_mouse_position = position.into();
                    return;
                }

                let pos: [i32; 2] = position.into();
                let sensitivity = 0.5;
                let dx = -(pos[0] - self.last_mouse_position[0]) as f32 * sensitivity;
                let dy = (pos[1] - self.last_mouse_position[1]) as f32 * sensitivity;
                self.last_mouse_position = position.into();

                self.yaw += dx;
                self.pitch += dy;

                if self.pitch > 89.0 {
                    self.pitch = 89.0;
                }

                if self.pitch < -89.0 {
                    self.pitch = -89.0;
                }

                self.view_dir = Vector3::new(
                    Rad::from(Deg(self.yaw)).cos() * Rad::from(Deg(self.pitch)).cos(),
                    Rad::from(Deg(self.pitch)).sin(),
                    Rad::from(Deg(self.yaw)).sin() * Rad::from(Deg(self.pitch)).cos(),
                ).normalize();
            }
            _ => (),
        }
    }
}