#[derive(Clone)]
enum BlockState {
    Normal,   // Стена
    Cleared,  // Есть проход
}

#[derive(Clone)]
pub struct TerrainBlock {
    x: u32,
    y: u32,
    state: BlockState,
}

pub struct Map {
    w: u32,
    h: u32,
    blocks: Vec<TerrainBlock>,
}

impl Map {
    pub fn new(w: u32, h: u32) -> Map {
        let mut blocks = Vec::new();

        for y in 0..h {
            for x in 0..w {
                blocks.push(TerrainBlock{x, y, state: BlockState::Normal});
            }
        }

        Map {
            w,
            h,
            blocks,
        }
    }
}
