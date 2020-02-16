use vulkano;
use vulkano::buffer::{ImmutableBuffer, BufferUsage, CpuAccessibleBuffer};
use std::sync::Arc;
use vulkano::sync::GpuFuture;
use std::io::Cursor;
use png::Limits;
use cgmath::{Vector3, InnerSpace};

#[derive(Default, Debug, Clone)]
pub struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
}
vulkano::impl_vertex!(Vertex, position, normal);

pub struct Terrain {
    pub vertices: Arc<ImmutableBuffer<[Vertex]>>,
    pub indices: Arc<ImmutableBuffer<[u32]>>,
}

impl Terrain {
    pub fn new(queue: Arc<vulkano::device::Queue>) -> Terrain {
        let data = include_bytes!("heightmap.png").to_vec();
        let cursor = Cursor::new(data);
        let decoder = png::Decoder::new_with_limits(cursor, Limits{bytes: 20*1024*1024*1024});

        let (info, mut reader) = decoder.read_info().unwrap();
        let mut image_data = Vec::new();
        image_data.resize((info.width * info.height * 4) as usize, 0);
        reader.next_frame(&mut image_data).unwrap();

        let w = info.width;
        let h = info.height;

        let mut vertices = Vec::new();
        let mut indices = Vec::new();

//        vertices.push(Vertex { position: [-0.5, -0.25, -0.2] });
//        vertices.push(Vertex { position: [0.0, 0.5, -0.2] });
//        vertices.push(Vertex { position: [0.25, -0.1, -0.2] });
//
//        indices.push(2_u32);
//        indices.push(1_u32);
//        indices.push(0_u32);

        let get_height = |xx: u32, yy: u32| -> f32 {  -4.0*(image_data[(yy*4*w + 4*xx) as usize] as f32) / 255.0 };


        for y in 0..h {
            for x in 0..w {
                let height = get_height(x, y);

                let l = {
                    if (x == 0) {
                        get_height(x, y)
                    } else {
                        get_height(x - 1, y)
                    }
                };

                let r = {
                    if (x >= (w-1)) {
                        get_height(x, y)
                    } else {
                        get_height(x + 1, y)
                    }
                };

                let t = {
                    if (y >= (h-1)) {
                        get_height(x, y)
                    } else {
                        get_height(x, y + 1)
                    }
                };

                let b = {
                    if (y == 0) {
                        get_height(x, y)
                    } else {
                        get_height(x, y - 1)
                    }
                };

                let n = Vector3{x: 2.0*(l-r), y: 2.0*(t-b), z: -4.0};

                vertices.push(Vertex {
                    position: [(x as f32) * 0.1, height, (y as f32) * 0.1],
                    normal: n.normalize().into(),
                });
            }
        }

//		auto heightD = getPixelHeight(heightMapData, m_width, i, j - 1);
//		auto heightU = getPixelHeight(heightMapData, m_width, i, j + 1);
//		glm::vec3 normalVector = glm::normalize(glm::vec3(heightL - heightR, 1.0f, heightD - heightU))

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

        println!("{:?}", indices);

        let (bb, fut) = {
            ImmutableBuffer::from_iter(vertices.iter().cloned(), BufferUsage::vertex_buffer(), queue.clone()).unwrap()
        };

        fut.then_signal_fence_and_flush().unwrap().wait(None).unwrap();


        let (ib, fut2) = {
            ImmutableBuffer::from_iter(indices.iter().cloned(), BufferUsage::index_buffer(), queue.clone()).unwrap()
        };

        fut2.then_signal_fence_and_flush().unwrap().wait(None).unwrap();
        Terrain {
            vertices: bb,
            indices: ib,
        }
    }
}
