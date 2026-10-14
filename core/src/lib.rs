//! Typed numeric kernels for the MLP. Hot math lives here (Phase 0 decision).

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Act { Tanh, Sigmoid }

impl Act {
    pub fn f(self, x: f64) -> f64 {
        match self { Act::Tanh => x.tanh(), Act::Sigmoid => 1.0 / (1.0 + (-x).exp()) }
    }
    /// derivative expressed in terms of the activation output y = f(x)
    pub fn df_from_y(self, y: f64) -> f64 {
        match self { Act::Tanh => 1.0 - y * y, Act::Sigmoid => y * (1.0 - y) }
    }
}

#[derive(Clone, Debug)]
pub struct Layer {
    pub n_in: usize,
    pub n_out: usize,
    /// row-major: w[o * n_in + i]
    pub w: Vec<f64>,
    pub b: Vec<f64>,
    pub act: Act,
}

#[derive(Clone, Debug)]
pub struct Mlp { pub layers: Vec<Layer> }

/// Tiny deterministic PRNG (xorshift64*) so seeds reproduce exactly.
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self { Rng(seed.max(1)) }
    pub fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 >> 12; self.0 ^= self.0 << 25; self.0 ^= self.0 >> 27;
        let v = self.0.wrapping_mul(0x2545F4914F6CDD1D);
        (v >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 { lo + (hi - lo) * self.next_f64() }
}

impl Mlp {
    /// sizes = [inputs, hidden..., outputs]; hidden layers tanh, output sigmoid.
    pub fn new(sizes: &[usize], rng: &mut Rng) -> Self {
        let mut layers = Vec::new();
        for k in 0..sizes.len() - 1 {
            let (n_in, n_out) = (sizes[k], sizes[k + 1]);
            let scale = (1.0 / n_in as f64).sqrt();
            let w = (0..n_in * n_out).map(|_| rng.uniform(-scale, scale)).collect();
            let act = if k + 2 == sizes.len() { Act::Sigmoid } else { Act::Tanh };
            layers.push(Layer { n_in, n_out, w, b: vec![0.0; n_out], act });
        }
        Mlp { layers }
    }

    /// Returns activations for every layer (index 0 = input).
    pub fn forward_all(&self, x: &[f64]) -> Vec<Vec<f64>> {
        let mut acts = vec![x.to_vec()];
        for l in &self.layers {
            let prev = acts.last().unwrap();
            let mut out = vec![0.0; l.n_out];
            for o in 0..l.n_out {
                let mut z = l.b[o];
                for i in 0..l.n_in { z += l.w[o * l.n_in + i] * prev[i]; }
                out[o] = l.act.f(z);
            }
            acts.push(out);
        }
        acts
    }

    pub fn predict(&self, x: &[f64]) -> Vec<f64> { self.forward_all(x).pop().unwrap() }

    /// Binary cross-entropy loss for a single sample with one sigmoid output.
    pub fn bce(p: f64, y: f64) -> f64 {
        let p = p.clamp(1e-12, 1.0 - 1e-12);
        -(y * p.ln() + (1.0 - y) * (1.0 - p).ln())
    }

    /// Analytic gradients for one sample under BCE (output must be sigmoid).
    /// Returns (dW per layer, dB per layer) as flat vectors.
    pub fn grads(&self, x: &[f64], y: f64) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let acts = self.forward_all(x);
        let nl = self.layers.len();
        let mut dws = vec![Vec::new(); nl];
        let mut dbs = vec![Vec::new(); nl];
        // dL/dz for output with BCE + sigmoid simplifies to (p - y)
        let p = acts[nl][0];
        let mut delta = vec![p - y];
        for k in (0..nl).rev() {
            let l = &self.layers[k];
            let a_prev = &acts[k];
            let mut dw = vec![0.0; l.n_out * l.n_in];
            for o in 0..l.n_out {
                for i in 0..l.n_in { dw[o * l.n_in + i] = delta[o] * a_prev[i]; }
            }
            dws[k] = dw;
            dbs[k] = delta.clone();
            if k > 0 {
                let mut next = vec![0.0; l.n_in];
                for i in 0..l.n_in {
                    let mut s = 0.0;
                    for o in 0..l.n_out { s += l.w[o * l.n_in + i] * delta[o]; }
                    next[i] = s * self.layers[k - 1].act.df_from_y(a_prev[i]);
                }
                delta = next;
            }
        }
        (dws, dbs)
    }

    /// Mean gradient over the batch, plus the mean loss: (dW per layer, dB per layer, loss).
    fn batch_grads(&self, batch: &[(Vec<f64>, f64)]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, f64) {
        let n = batch.len() as f64;
        let mut loss = 0.0;
        let mut gw: Vec<Vec<f64>> = self.layers.iter().map(|l| vec![0.0; l.w.len()]).collect();
        let mut gb: Vec<Vec<f64>> = self.layers.iter().map(|l| vec![0.0; l.b.len()]).collect();
        for (x, y) in batch {
            loss += Mlp::bce(self.predict(x)[0], *y);
            let (dw, db) = self.grads(x, *y);
            for k in 0..self.layers.len() {
                for j in 0..gw[k].len() { gw[k][j] += dw[k][j] / n; }
                for j in 0..gb[k].len() { gb[k][j] += db[k][j] / n; }
            }
        }
        (gw, gb, loss / n)
    }

    /// One full-batch gradient-descent step. Returns the mean loss before the update.
    pub fn sgd_step(&mut self, batch: &[(Vec<f64>, f64)], lr: f64) -> f64 {
        let (gw, gb, loss) = self.batch_grads(batch);
        for k in 0..self.layers.len() {
            for j in 0..self.layers[k].w.len() { self.layers[k].w[j] -= lr * gw[k][j]; }
            for j in 0..self.layers[k].b.len() { self.layers[k].b[j] -= lr * gb[k][j]; }
        }
        loss
    }

    /// One full-batch Adam step (beta1 0.9, beta2 0.999). Returns the mean loss before the update.
    pub fn adam_step(&mut self, batch: &[(Vec<f64>, f64)], st: &mut AdamState, lr: f64) -> f64 {
        let (gw, gb, loss) = self.batch_grads(batch);
        st.t += 1;
        let c1 = 1.0 - 0.9f64.powi(st.t as i32);
        let c2 = 1.0 - 0.999f64.powi(st.t as i32);
        for k in 0..self.layers.len() {
            adam_update(&mut self.layers[k].w, &gw[k], &mut st.mw[k], &mut st.vw[k], lr, c1, c2);
            adam_update(&mut self.layers[k].b, &gb[k], &mut st.mb[k], &mut st.vb[k], lr, c1, c2);
        }
        loss
    }
}

fn adam_update(p: &mut [f64], g: &[f64], m: &mut [f64], v: &mut [f64], lr: f64, c1: f64, c2: f64) {
    for j in 0..p.len() {
        m[j] = 0.9 * m[j] + 0.1 * g[j];
        v[j] = 0.999 * v[j] + 0.001 * g[j] * g[j];
        p[j] -= lr * (m[j] / c1) / ((v[j] / c2).sqrt() + 1e-8);
    }
}

/// Running moment estimates for Adam. Must be rebuilt whenever the network is.
#[derive(Clone, Debug)]
pub struct AdamState {
    mw: Vec<Vec<f64>>,
    mb: Vec<Vec<f64>>,
    vw: Vec<Vec<f64>>,
    vb: Vec<Vec<f64>>,
    t: u64,
}

impl AdamState {
    pub fn new(net: &Mlp) -> Self {
        let w: Vec<Vec<f64>> = net.layers.iter().map(|l| vec![0.0; l.w.len()]).collect();
        let b: Vec<Vec<f64>> = net.layers.iter().map(|l| vec![0.0; l.b.len()]).collect();
        AdamState { mw: w.clone(), vw: w, mb: b.clone(), vb: b, t: 0 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Optimizer { Sgd, Adam }

impl Optimizer {
    pub fn name(self) -> &'static str {
        match self { Optimizer::Sgd => "SGD", Optimizer::Adam => "Adam" }
    }
    /// A learning rate that works well for this optimizer on the built-in datasets.
    pub fn default_lr(self) -> f64 {
        match self { Optimizer::Sgd => 0.5, Optimizer::Adam => 0.03 }
    }
}

pub fn xor_data() -> Vec<(Vec<f64>, f64)> {
    vec![
        (vec![0.0, 0.0], 0.0),
        (vec![0.0, 1.0], 1.0),
        (vec![1.0, 0.0], 1.0),
        (vec![1.0, 1.0], 0.0),
    ]
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dataset { Xor, Circles, Spiral }

impl Dataset {
    pub fn name(self) -> &'static str {
        match self { Dataset::Xor => "XOR", Dataset::Circles => "circles", Dataset::Spiral => "spiral" }
    }
    /// Deterministic for a given seed. XOR ignores the seed.
    pub fn points(self, seed: u64, per_class: usize) -> Vec<(Vec<f64>, f64)> {
        match self {
            Dataset::Xor => xor_data(),
            Dataset::Circles => circles_data(seed, per_class),
            Dataset::Spiral => spiral_data(seed, per_class),
        }
    }
}

/// Two concentric rings in [-1,1]^2: inner class 0 (r<0.5), outer class 1 (r in [0.7,1.0]).
pub fn circles_data(seed: u64, per_class: usize) -> Vec<(Vec<f64>, f64)> {
    let mut rng = Rng::new(seed);
    let mut out = Vec::with_capacity(per_class * 2);
    for _ in 0..per_class {
        let a = rng.uniform(0.0, std::f64::consts::TAU);
        let r = rng.uniform(0.0, 0.5);
        out.push((vec![r * a.cos(), r * a.sin()], 0.0));
    }
    for _ in 0..per_class {
        let a = rng.uniform(0.0, std::f64::consts::TAU);
        let r = rng.uniform(0.7, 1.0);
        out.push((vec![r * a.cos(), r * a.sin()], 1.0));
    }
    out
}

/// Two interleaved arms of a spiral, in [-1,1]^2. Class 0 and class 1 are rotated 180 degrees.
pub fn spiral_data(seed: u64, per_class: usize) -> Vec<(Vec<f64>, f64)> {
    let mut rng = Rng::new(seed);
    let mut out = Vec::with_capacity(per_class * 2);
    for class in 0..2 {
        for i in 0..per_class {
            let t = i as f64 / per_class as f64; // 0..1
            let r = 0.05 + 0.9 * t;
            let theta = 3.0 * std::f64::consts::PI * t + class as f64 * std::f64::consts::PI;
            let jx = rng.uniform(-0.03, 0.03);
            let jy = rng.uniform(-0.03, 0.03);
            out.push((vec![r * theta.cos() + jx, r * theta.sin() + jy], class as f64));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradients_match_numeric() {
        let mut rng = Rng::new(7);
        let net = Mlp::new(&[2, 3, 2, 1], &mut rng);
        let x = vec![0.3, -0.8];
        let y = 1.0;
        let (dw, db) = net.grads(&x, y);
        let eps = 1e-6;
        for k in 0..net.layers.len() {
            for j in 0..net.layers[k].w.len() {
                let mut plus = net.clone(); plus.layers[k].w[j] += eps;
                let mut minus = net.clone(); minus.layers[k].w[j] -= eps;
                let lp = Mlp::bce(plus.predict(&x)[0], y);
                let lm = Mlp::bce(minus.predict(&x)[0], y);
                let num = (lp - lm) / (2.0 * eps);
                assert!((num - dw[k][j]).abs() < 1e-6, "w layer{k} idx{j}: num={num} ana={}", dw[k][j]);
            }
            for j in 0..net.layers[k].b.len() {
                let mut plus = net.clone(); plus.layers[k].b[j] += eps;
                let mut minus = net.clone(); minus.layers[k].b[j] -= eps;
                let num = (Mlp::bce(plus.predict(&x)[0], y) - Mlp::bce(minus.predict(&x)[0], y)) / (2.0 * eps);
                assert!((num - db[k][j]).abs() < 1e-6, "b layer{k} idx{j}");
            }
        }
    }

    #[test]
    fn xor_learns_with_fixed_seed() {
        let mut rng = Rng::new(42);
        let mut net = Mlp::new(&[2, 4, 1], &mut rng);
        let data = xor_data();
        let mut last = f64::MAX;
        for _ in 0..8000 { last = net.sgd_step(&data, 0.5); }
        assert!(last < 0.05, "final loss {last}");
        for (x, y) in &data {
            let p = net.predict(x)[0];
            assert_eq!((p > 0.5) as i32 as f64, *y, "x={x:?} p={p}");
        }
    }

    #[test]
    fn circles_reach_full_accuracy() {
        // Measured: 1.000 at 5000 steps for hidden=4 and hidden=8 (seed 7, 100/class).
        let data = Dataset::Circles.points(7, 100);
        let mut rng = Rng::new(42);
        let mut net = Mlp::new(&[2, 4, 1], &mut rng);
        for _ in 0..8000 { net.sgd_step(&data, 0.5); }
        let ok = data.iter().filter(|(x, y)| ((net.predict(x)[0] > 0.5) as i32 as f64) == *y).count();
        assert!(ok as f64 / data.len() as f64 >= 0.98, "circles acc {ok}/{}", data.len());
    }

    #[test]
    fn spiral_is_harder_than_circles_at_hidden_4() {
        // Measured: spiral with 4 hidden units stalls at 0.62-0.66 over 20000 steps.
        // This pins the known limit so a regression in either direction is visible.
        let data = Dataset::Spiral.points(7, 100);
        let mut rng = Rng::new(42);
        let mut net = Mlp::new(&[2, 4, 1], &mut rng);
        for _ in 0..20000 { net.sgd_step(&data, 0.5); }
        let ok = data.iter().filter(|(x, y)| ((net.predict(x)[0] > 0.5) as i32 as f64) == *y).count();
        let acc = ok as f64 / data.len() as f64;
        assert!(acc > 0.5 && acc < 0.9, "spiral hidden=4 acc {acc}");
    }

    #[test]
    fn datasets_are_deterministic_and_balanced() {
        for ds in [Dataset::Circles, Dataset::Spiral] {
            let a = ds.points(9, 50);
            let b = ds.points(9, 50);
            assert_eq!(a, b, "{} not deterministic", ds.name());
            let ones = a.iter().filter(|(_, y)| *y == 1.0).count();
            assert_eq!(ones, 50, "{} class balance", ds.name());
            assert!(a.iter().all(|(x, _)| x.iter().all(|v| v.abs() <= 1.2)), "{} out of domain", ds.name());
        }
    }

    #[test]
    fn no_nan_after_training() {
        let mut rng = Rng::new(3);
        let mut net = Mlp::new(&[2, 4, 1], &mut rng);
        for _ in 0..2000 { net.sgd_step(&xor_data(), 0.5); }
        for l in &net.layers {
            assert!(l.w.iter().chain(l.b.iter()).all(|v| v.is_finite()));
        }
    }

    #[test]
    fn adam_fits_spiral_far_faster_than_sgd() {
        let data = Dataset::Spiral.points(7, 100);
        let acc = |net: &Mlp| data.iter().filter(|(x, y)| ((net.predict(x)[0] > 0.5) as i32 as f64) == *y).count() as f64 / data.len() as f64;
        let mut net = Mlp::new(&[2, 8, 1], &mut Rng::new(42));
        let mut st = AdamState::new(&net);
        for _ in 0..3000 { net.adam_step(&data, &mut st, Optimizer::Adam.default_lr()); }
        assert!(acc(&net) > 0.95, "adam reached only {}", acc(&net));
        let mut sgd = Mlp::new(&[2, 8, 1], &mut Rng::new(42));
        for _ in 0..3000 { sgd.sgd_step(&data, Optimizer::Sgd.default_lr()); }
        assert!(acc(&sgd) < 0.8, "sgd unexpectedly reached {}", acc(&sgd));
    }
}
