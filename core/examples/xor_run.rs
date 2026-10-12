use nn_core::*;
fn main() {
    let mut rng = Rng::new(42);
    let mut net = Mlp::new(&[2, 4, 1], &mut rng);
    let data = xor_data();
    for step in 0..=8000 {
        let loss = net.sgd_step(&data, 0.5);
        if step % 1000 == 0 { println!("step {step:5}  loss {loss:.6}"); }
    }
    for (x, y) in &data {
        let p = net.predict(x)[0];
        println!("in={:?} target={} pred={:.4}", x, y, p);
    }
}
