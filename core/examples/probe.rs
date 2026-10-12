use nn_core::*;
fn acc(net: &Mlp, data: &[(Vec<f64>, f64)]) -> f64 {
    let ok = data.iter().filter(|(x, y)| ((net.predict(x)[0] > 0.5) as i32 as f64) == *y).count();
    ok as f64 / data.len() as f64
}
fn main() {
    for ds in [Dataset::Circles, Dataset::Spiral] {
        for hidden in [4usize, 8] {
            let data = ds.points(7, 100);
            let mut rng = Rng::new(42);
            let mut net = Mlp::new(&[2, hidden, 1], &mut rng);
            let mut checkpoints = vec![];
            for step in 1..=20000 {
                net.sgd_step(&data, 0.5);
                if step % 5000 == 0 { checkpoints.push(format!("{}:{:.3}", step, acc(&net, &data))); }
            }
            println!("{} hidden={} acc@ {}", ds.name(), hidden, checkpoints.join(" "));
        }
    }
}
