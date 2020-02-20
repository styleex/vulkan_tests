use vulkano::buffer::{ImmutableBuffer, BufferUsage, CpuBufferPool};
use std::sync::Arc;
use vulkano::sync::GpuFuture;
use std::io::Cursor;
use cgmath::{Vector3, InnerSpace, Matrix4};
use vulkano::pipeline::{GraphicsPipelineAbstract, GraphicsPipeline};
use vulkano::device::Queue;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::command_buffer::{AutoCommandBufferBuilder, AutoCommandBuffer, DynamicState};
use vulkano::pipeline::viewport::Viewport;
use vulkano::format::Format;

#[derive(Default, Debug, Clone)]
pub struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    color: [f32; 3],
}
vulkano::impl_vertex!(Vertex, position, normal, color);

pub struct Cube {
    pub vertices: Arc<ImmutableBuffer<[Vertex]>>,
    pub indices: Arc<ImmutableBuffer<[u32]>>,
}

impl Cube {
    pub fn new(gfx_queue: Arc<Queue>, h: f32) -> Cube {
        let vertices = [
            // up
            Vertex { position: [0.0, -h, 0.0], normal: [0.0, 1.0, 0.0], color: [0.0, 1.0, 0.0] },
            Vertex { position: [0.0, -h, -1.0], normal: [0.0, 1.0, 0.0], color: [0.0, 1.0, 0.0] },
            Vertex { position: [1.0, -h, -1.0], normal: [0.0, 1.0, 0.0], color: [0.0, 1.0, 0.0] },
            Vertex { position: [1.0, -h, 0.0], normal: [0.0, 1.0, 0.0], color: [0.0, 1.0, 0.0] },

            // bottom
            Vertex { position: [0.0, 0.0, 0.0], normal: [0.0, -1.0, 0.0], color: [1.0, 1.0, 1.0] },
            Vertex { position: [0.0, 0.0, -1.0], normal: [0.0, -1.0, 0.0], color: [1.0, 1.0, 1.0] },
            Vertex { position: [1.0, 0.0, -1.0], normal: [0.0, -1.0, 0.0], color: [1.0, 1.0, 1.0] },
            Vertex { position: [1.0, 0.0, 0.0], normal: [0.0, -1.0, 0.0], color: [1.0, 1.0, 1.0] },

            // front
            Vertex { position: [0.0, 0.0, 0.0], normal: [0.0, 0.0, 1.0], color: [1.0, 0.0, 0.0] },
            Vertex { position: [0.0, -h, 0.0], normal: [0.0, 0.0, 1.0], color: [1.0, 0.0, 0.0] },
            Vertex { position: [1.0, -h, 0.0], normal: [0.0, 0.0, 1.0], color: [1.0, 0.0, 0.0] },
            Vertex { position: [1.0, 0.0, 0.0], normal: [0.0, 0.0, 1.0], color: [1.0, 0.0, 0.0] },

            // back
            Vertex { position: [0.0, 0.0, -1.0], normal: [0.0, 0.0, -1.0], color: [1.0, 1.0, 1.0] },
            Vertex { position: [0.0, -h, -1.0], normal: [0.0, 0.0, -1.0], color: [1.0, 1.0, 1.0] },
            Vertex { position: [1.0, -h, -1.0], normal: [0.0, 0.0, -1.0], color: [1.0, 1.0, 1.0] },
            Vertex { position: [1.0, 0.0, -1.0], normal: [0.0, 0.0, -1.0], color: [1.0, 1.0, 1.0] },

            // left
            Vertex { position: [0.0, 0.0, -1.0], normal: [-1.0, 0.0, 0.0], color: [0.0, 0.0, 1.0] },
            Vertex { position: [0.0, -h, -1.0], normal: [-1.0, 0.0, 0.0], color: [0.0, 0.0, 1.0] },
            Vertex { position: [0.0, -h, 0.0], normal: [-1.0, 0.0, 0.0], color: [0.0, 0.0, 1.0] },
            Vertex { position: [0.0, 0.0, 0.0], normal: [-1.0, 0.0, 0.0], color: [0.0, 0.0, 1.0] },

            // right
            Vertex { position: [1.0, 0.0, 0.0], normal: [1.0, 0.0, 0.0], color: [1.0, 1.0, 1.0] },
            Vertex { position: [1.0, -h, 0.0], normal: [1.0, 0.0, 0.0], color: [1.0, 1.0, 1.0] },
            Vertex { position: [1.0, -h, -1.0], normal: [1.0, 0.0, 0.0], color: [1.0, 1.0, 1.0] },
            Vertex { position: [1.0, 0.0, -1.0], normal: [1.0, 0.0, 0.0], color: [1.0, 1.0, 1.0] },
        ];

        let indices = [
            // top
            0, 3, 1, 1, 3, 2,

            // bottom
            7, 4, 6, 6, 4, 5,

            // front
            8, 11, 9, 9, 11, 10,

            // back
            15, 12, 14, 14, 12, 13,

            //left
            16, 19, 17, 17, 19, 18,

            //right
            20, 23, 21, 21, 23, 22,
        ];

//        let x: Vec<_> = Cube::new().vertex(|v| Vertex { position: v.pos.into(), normal: v.normal.into() }).triangulate().vertices().collect();


        let (bb, fut) = {
            ImmutableBuffer::from_iter(vertices.iter().cloned(), BufferUsage::vertex_buffer(), gfx_queue.clone()).unwrap()
        };

        let (ib, fut2) = {
            ImmutableBuffer::from_iter(indices.iter().cloned(), BufferUsage::index_buffer(), gfx_queue.clone()).unwrap()
        };

        fut.join(fut2).then_signal_fence_and_flush().unwrap().wait(None).unwrap();

        Cube {
            vertices: bb,
            indices: ib,
        }
    }
}


