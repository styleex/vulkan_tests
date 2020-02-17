use vulkano;
use vulkano::buffer::{ImmutableBuffer, BufferUsage, CpuAccessibleBuffer};
use std::sync::Arc;
use vulkano::sync::GpuFuture;
use std::io::Cursor;
use png::Limits;
use cgmath::{Vector3, InnerSpace};
use cgmath::num_traits::clamp;

type HeightFunc = fn (u32, u32) -> f32;

pub struct HeightMap {
    pub w: u32,
    pub h: u32,
    pub get_height: HeightFunc,
}

impl HeightMap {
    pub fn new() -> HeightMap {
        let data = include_bytes!("heightmap.png").to_vec();
        let cursor = Cursor::new(data);
        let decoder = png::Decoder::new(cursor);

        let (info, mut reader) = decoder.read_info().unwrap();
        let mut image_data = Vec::new();
        image_data.resize((info.width * info.height * 4) as usize, 0);
        reader.next_frame(&mut image_data).unwrap();

        let w = info.width;
        let h = info.height;
        HeightMap{
            w: info.width,
            h: info.height,
            get_height: |x: u32, y: u32| -> f32 {
                let xx = clamp(x, 0, w);
                let yy = clamp(y, 0, h);

                4.0 * (image_data[(w * yy * 4  + xx * 4) as usize] as f32) / 255.0
            },
        }
    }

    pub fn empty(w: u32, h: u32) -> HeightMap {
        HeightMap{
            w,
            h,
            get_height: |_, _| -> f32 { 0.0 }
        }
    }
}

#[derive(Default, Debug, Clone)]
pub struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    texcoord: [f32; 2],
}
vulkano::impl_vertex!(Vertex, position, normal, texcoord);

pub struct Terrain {
    pub vertices: Arc<ImmutableBuffer<[Vertex]>>,
    pub indices: Arc<ImmutableBuffer<[u32]>>,
}

impl Terrain {
    pub fn new(queue: Arc<vulkano::device::Queue>, height_map: HeightMap) -> Terrain {
        let w= height_map.w;
        let h = height_map.h;

        let mut vertices = Vec::with_capacity((h * w) as usize);
        let mut indices = Vec::with_capacity((h * (w - 1) * 6) as usize);

        // TODO: Triangle strip;
        for y in 0..h {
            for x in 0..w {
                let height = height_map.get_height(x, y);

                let l = height_map.get_height(x - 1, y);
                let r = height_map.get_height(x + 1, y);
                let t = height_map.get_height(x, y + 1);
                let b = height_map.get_height(x, y - 1);
                let normal = Vector3 { x: 2.0 * (l - r), y: 2.0 * (t - b), z: -4.0 }.normalize();

                vertices.push(Vertex {
                    position: [(x as f32) * 0.1, height, (y as f32) * 0.1],
                    normal: normal.into(),
                    texcoord: [x as f32, y as f32],
                });
            }
        }

        for y in 1..(h) {
            for x in 0..(w - 1) {
                indices.push((y - 1) * w + x);
                indices.push((y) * w + x);
                indices.push((y - 1) * w + x + 1);

                indices.push((y - 1) * w + x + 1);
                indices.push((y) * w + x);
                indices.push((y) * w + x + 1);
            }
        }

        let (bb, fut) = {
            ImmutableBuffer::from_iter(vertices.iter().cloned(), BufferUsage::vertex_buffer(), queue.clone()).unwrap()
        };

        let (ib, fut2) = {
            ImmutableBuffer::from_iter(indices.iter().cloned(), BufferUsage::index_buffer(), queue.clone()).unwrap()
        };

        fut2.join(fut).then_signal_fence_and_flush().unwrap().wait(None).unwrap();
        Terrain {
            vertices: bb,
            indices: ib,
        }
    }
}
