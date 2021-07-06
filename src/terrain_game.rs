use std::time::Instant;

#[derive(Clone, PartialEq)]
#[allow(dead_code)]
pub enum BlockState {
    Normal, // Стена
    Cleared, // Есть проход
    Highlighted, // Блок подсвечен мышкой
}

#[derive(Clone)]
pub struct TerrainBlock {
    pub id: u32,
    pub x: u32,
    pub y: u32,

    pub selected: bool,
    pub selected_time: Instant,

    pub highlighted: bool,
    pub hightligh_start: Instant,

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

                blocks.push(TerrainBlock {
                    x,
                    y,
                    selected: false,
                    selected_time: Instant::now(),
                    highlighted: false,
                    hightligh_start: Instant::now(),
                    state: BlockState::Normal,
                    id: y * (w) + x,
                });
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
            let new_highlighted = Some(block.id) == id;

            if block.highlighted != new_highlighted {
                block.highlighted = new_highlighted;
                block.hightligh_start = Instant::now();
                changed = true;
            }
        }

        self.changed = changed;
    }

    pub fn select(&mut self, id: Option<u32>) {
        for block in self.blocks.iter_mut() {
            if Some(block.id) == id {
                block.selected = !block.selected;
                block.selected_time = Instant::now();
                break;
            }
        }

        self.changed = true;
    }

    pub fn update(&mut self) {
        for block in self.blocks.iter_mut() {
            if block.selected && block.selected_time.elapsed().as_millis() > 500 {
                block.selected = false;
                block.highlighted = false;
                block.state = BlockState::Cleared;
            }
        }
    }
}
