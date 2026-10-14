//! Compares optimizers and init gains on the spiral. Run: cargo run --release -p nn_core --example spiral_opt
use nn_core::*;

fn acc(net: &Mlp, data: &[(Vec<f64>, f64)]) -> f64 {
    let ok = data.iter().filter(|(x, y)| ((net.predict(x)[0] > 0.5) as i32 as f64) == *y).count();
    ok as f64 / data.len() as f64
}

/// Full-batch gradient (mean over data) plus loss.
fn batch_grad(net: &Mlp, data: &[(Vec<f64>, f64)]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n = data.len() as f64;
    let mut gw: Vec<Vec<f64>> = net.layers.iter().map(|l| vec![0.0; l.w.len()]).collect();
    let mut gb: Vec<Vec<f64>> = net.layers.iter().map(|l| vec![0.0; l.b.len()]).collect();
    for (x, y) in data {
        let (dw, db) = net.grads(x, *y);
        for k in 0..gw.len() {
            for j in 0..gw[k].len() { gw[k][j] += dw[k][j] / n; }
            for j in 0..gb[k].len() { gb[k][j] += db[k][j] / n; }
        }
    }
    (gw, gb)
}

fn run(opt: &str, hidden: usize, lr: f64, gain: f64, seed: u64, data: &[(Vec<f64>, f64)], marks: &[usize]) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    let mut net = Mlp::new(&[2, hidden, 1], &mut rng);
    for l in net.layers.iter_mut() { for w in l.w.iter_mut() { *w *= gain; } }
    let mut mw: Vec<Vec<f64>> = net.layers.iter().map(|l| vec![0.0; l.w.len()]).collect();
    let mut mb: Vec<Vec<f64>> = net.layers.iter().map(|l| vec![0.0; l.b.len()]).collect();
    let mut vw = mw.clone();
    let mut vb = mb.clone();
    let mut out = vec![];
    let last = *marks.last().unwrap();
    for t in 1..=last {
        if opt == "sgd" { net.sgd_step(data, lr); } else {
            let (gw, gb) = batch_grad(&net, data);
            for k in 0..net.layers.len() {
                for (params, g, m, v) in [(&mut net.layers[k].w, &gw[k], &mut mw[k], &mut vw[k]), ]  {
                    step(opt, params, g, m, v, lr, t);
                }
                let (params, g, m, v) = (&mut net.layers[k].b, &gb[k], &mut mb[k], &mut vb[k]);
                step(opt, params, g, m, v, lr, t);
            }
        }
        if marks.contains(&t) { out.push(acc(&net, data)); }
    }
    out
}

fn step(opt: &str, p: &mut [f64], g: &[f64], m: &mut [f64], v: &mut [f64], lr: f64, t: usize) {
    for j in 0..p.len() {
        if opt == "momentum" {
            m[j] = 0.9 * m[j] + g[j];
            p[j] -= lr * m[j];
        } else {
            m[j] = 0.9 * m[j] + 0.1 * g[j];
            v[j] = 0.999 * v[j] + 0.001 * g[j] * g[j];
            let mh = m[j] / (1.0 - 0.9f64.powi(t as i32));
            let vh = v[j] / (1.0 - 0.999f64.powi(t as i32));
            p[j] -= lr * mh / (vh.sqrt() + 1e-8);
        }
    }
}

fn main() {
    let data = Dataset::Spiral.points(7, 100);
    let marks = [2000, 5000, 10000];
    println!("opt,hidden,lr,gain,seed -> acc@2k,5k,10k");
    let cfgs: Vec<(&str, f64)> = vec![("sgd", 0.5), ("momentum", 0.05), ("momentum", 0.1), ("adam", 0.01), ("adam", 0.03), ("adam", 0.1)];
    for hidden in [4usize, 8] {
        for (opt, lr) in &cfgs {
            for gain in [1.0, 3.0] {
                let mut line = String::new();
                for seed in [42u64, 1, 2] {
                    let a = run(opt, hidden, *lr, gain, seed, &data, &marks);
                    line += &format!(" [{:.2} {:.2} {:.2}]", a[0], a[1], a[2]);
                }
                println!("{opt},{hidden},{lr},{gain} ->{line}");
            }
        }
    }
}
