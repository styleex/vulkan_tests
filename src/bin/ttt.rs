use cgmath::{Rad, Deg, Angle};

fn main() {
    for i in 0..500 {
        println!("{:?}", Rad::from(Deg(i as f32)).sin());
    }
}
