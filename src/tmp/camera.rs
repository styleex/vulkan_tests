use cgmath::{Vector3, Matrix4, SquareMatrix, Rotation3};
use cgmath::{perspective, Rad};
use winit::event::{KeyboardInput, ElementState, VirtualKeyCode, WindowEvent, MouseButton};
use winit::dpi::PhysicalPosition;
use std::f32;


pub struct Camera {
    position: Vector3<f32>,
    rotation: Vector3<f32>,

    proj: Matrix4<f32>,

    mouse_pressed: bool,
    last_mouse_position: [i32; 2],

    viewport: [u32; 2],
    rx: f32,
    ry: f32,
}

impl Camera {
    pub fn new() -> Camera {
        Camera {
            position: Vector3::new(0.0, 0.0, 0.0),
            rotation: Vector3::new(0.0, 0.0, 0.0),
            proj: Matrix4::identity(),
            mouse_pressed: false,
            last_mouse_position: [0, 0],
            viewport: [0, 0],
            rx: 0.0,
            ry: 0.0,
        }
    }

    pub fn set_viewport(&mut self, w: u32, h: u32) {
        self.viewport = [w, h];
        self.proj = cgmath::perspective(
            Rad(std::f32::consts::FRAC_PI_2),
            w as f32 / h as f32,
            0.01,
            100.0);
    }

    pub fn view_matrix(&self) -> Matrix4<f32> {
        let m1 = Matrix4::<f32>::from_translation(self.position);

        let rx = Matrix4::<f32>::from_angle_x(Rad(self.rotation.x + self.rx));
        let ry = Matrix4::<f32>::from_angle_y(Rad(self.rotation.y + self.ry));
        let rz = Matrix4::<f32>::from_angle_z(Rad(self.rotation.z));

        return m1 * (rx * ry * rz);
    }

    pub fn proj_matrix(&self) -> Matrix4<f32> {
        self.proj
    }

    pub fn handle_keyboard(&mut self, input: KeyboardInput) {
        if input.state == ElementState::Pressed {
            match input.virtual_keycode {
                Some(VirtualKeyCode::W) => self.position.z += 0.1,
                Some(VirtualKeyCode::S) => self.position.z -= 0.1,
                Some(VirtualKeyCode::A) => self.position.x += 0.1,
                Some(VirtualKeyCode::D) => self.position.x -= 0.1,
                Some(VirtualKeyCode::Space) => self.position.y += 0.1,
                Some(VirtualKeyCode::LShift) => self.position.y -= 0.1,
                _ => (),
            }
        }
    }

    pub fn handle_event(&mut self, event: &WindowEvent) {
        match event {
            &WindowEvent::MouseInput { state, button, .. } => {
                if self.mouse_pressed && (state == ElementState::Released) && (button == MouseButton::Left) {
                    self.rotation.y += self.ry;
                    self.rotation.x += self.rx;
                    self.rx = 0.0;
                    self.ry = 0.0;
                }

                self.mouse_pressed = (state == ElementState::Pressed) && (button == MouseButton::Left);
            }
            &WindowEvent::CursorMoved { position, .. } => {
                if !self.mouse_pressed {
                    self.last_mouse_position = position.into();
                    return;
                }

                let pos: [i32; 2] = position.into();
                let dx = pos[0] - self.last_mouse_position[0];
                let dy = pos[1] - self.last_mouse_position[1];
                println!("{:?} {:?}", dx as f32 / self.viewport[0] as f32, dy as f32 / self.viewport[1] as f32);

                let ax = (dx as f32 / self.viewport[0] as f32).asin();
                let ay = (dy as f32 / self.viewport[1] as f32).asin();
                self.ry = ax;
                self.rx = ay;
            }
            _ => (),
        }
    }
}
