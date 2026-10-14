//! Spiral accuracy by architecture with Adam. Run: cargo run --release -p nn_core --example spiral_depth
use nn_core::*;

fn main() {
    let data = Dataset::Spiral.points(7, 100);
    let acc = |net: &Mlp| data.iter().filter(|(x, y)| ((net.predict(x)[0] > 0.5) as i32 as f64) == *y).count() as f64 / data.len() as f64;
    let archs: Vec<Vec<usize>> = vec![vec![2, 4, 1], vec![2, 8, 1], vec![2, 16, 1], vec![2, 4, 4, 1], vec![2, 8, 8, 1], vec![2, 16, 16, 1]];
    println!("arch -> acc at 2k / 5k steps, seeds 42,1,2 (Adam lr 0.03)");
    for sizes in archs {
        let mut line = String::new();
        for seed in [42u64, 1, 2] {
            let mut net = Mlp::new(&sizes, &mut Rng::new(seed));
            let mut st = AdamState::new(&net);
            let mut a2 = 0.0;
            for t in 1..=5000 {
                net.adam_step(&data, &mut st, 0.03);
                if t == 2000 { a2 = acc(&net); }
            }
            line += &format!(" [{:.2} {:.2}]", a2, acc(&net));
        }
        println!("{:?} ->{line}", sizes);
    }
}
