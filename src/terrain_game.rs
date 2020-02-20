#[derive(Clone)]
pub enum BlockState {
    Normal,   // Стена
    Cleared,  // Есть проход
}

#[derive(Clone)]
pub struct TerrainBlock {
    pub id: u32,
    pub x: u32,
    pub y: u32,
    pub state: BlockState,
}

pub struct Map {
    pub w: u32,
    pub h: u32,
    pub blocks: Vec<TerrainBlock>,
}

impl Map {
    pub fn new(w: u32, h: u32) -> Map {
        let mut blocks = Vec::new();

        for y in 0..h {
            for x in 0..w {
                if x == 3 || y == 3{
                    continue;
                }

                blocks.push(TerrainBlock{x, y, state: BlockState::Normal, id: y * (w) + x});
            }
        }

        Map {
            w,
            h,
            blocks,
        }
    }
}
