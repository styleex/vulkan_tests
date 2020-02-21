#[derive(Clone, PartialEq)]
pub enum BlockState {
    Normal,
    // Стена
    Cleared,
    // Есть проход
    Highlighted, // Блок подсвечен мышкой
}

#[derive(Clone)]
pub struct TerrainBlock {
    pub id: u32,
    pub x: u32,
    pub y: u32,
    pub state: BlockState,
}

pub struct Map {
    pub changed: bool,
    pub w: u32,
    pub h: u32,
    pub blocks: Vec<TerrainBlock>,
}

impl Map {
    pub fn new(w: u32, h: u32) -> Map {
        let mut blocks = Vec::new();

        for y in 0..h {
            for x in 0..w {
                if x == 3 || y == 3 {
                    continue;
                }

                blocks.push(TerrainBlock { x, y, state: BlockState::Normal, id: y * (w) + x });
            }
        }

        Map {
            w,
            h,
            blocks,
            changed: false,
        }
    }

    pub fn highlight(&mut self, id: Option<u32>) {
        let mut changed = false;
        for block in self.blocks.iter_mut() {
            if id == Some(block.id) {
                if block.state != BlockState::Highlighted {
                    changed = true;
                }
                block.state = BlockState::Highlighted;
            } else {
                if block.state != BlockState::Normal {
                    changed = true;
                }
                block.state = BlockState::Normal;
            }
        }

        self.changed = changed;
    }
}
