use anyhow::Result;
use engine::app::{App, Context, Game};
use engine::sdl2::event::Event;
use engine::ui::EguiState;
use engine::ui::egui;
use nn_core::{Dataset, Mlp, Rng};

const GRID: usize = 48;
/// Plot domain per dataset. XOR lives in [0,1]^2; circles and spiral live in [-1,1]^2.
fn domain(ds: Dataset) -> (f64, f64) {
    match ds {
        Dataset::Xor => (-0.25, 1.25),
        Dataset::Circles | Dataset::Spiral => (-1.15, 1.15),
    }
}
const PER_CLASS: usize = 100;
const DATA_SEED: u64 = 7;

struct NnViz {
    net: Mlp,
    dataset: Dataset,
    hidden: usize,
    data: Vec<(Vec<f64>, f64)>,
    lr: f64,
    steps_per_frame: usize,
    step: usize,
    paused: bool,
    loss: f64,
    loss_hist: Vec<f64>,
    sel: usize,
    grid: Vec<f32>,
    grid_step: usize,
    ui: Option<EguiState>,
    shot_path: Option<String>,
    shot_at: usize,
}

fn fresh_net(hidden: usize) -> Mlp {
    let mut rng = Rng::new(42);
    Mlp::new(&[2, hidden, 1], &mut rng)
}

/// Blue (class 0) to orange (class 1), by network output p in [0,1].
fn heat(p: f32) -> egui::Color32 {
    let t = p.clamp(0.0, 1.0);
    let lerp = |a: f32, b: f32| (a + (b - a) * t).round() as u8;
    egui::Color32::from_rgb(lerp(50.0, 240.0), lerp(80.0, 140.0), lerp(200.0, 40.0))
}

/// Centre of node `node` (of `n_nodes`) in layer `layer` (of `n_layers`), inside `rect`.
fn node_pos(rect: egui::Rect, layer: usize, n_layers: usize, node: usize, n_nodes: usize) -> egui::Pos2 {
    let x = rect.left() + rect.width() * (layer as f32 + 0.5) / n_layers as f32;
    let y = rect.top() + rect.height() * (node as f32 + 0.5) / n_nodes as f32;
    egui::pos2(x, y)
}

/// Log10 range over the loss history, padded so the curve never touches the frame.
fn log_range(hist: &[f64]) -> (f64, f64) {
    let logs: Vec<f64> = hist.iter().map(|v| v.max(1e-12).log10()).collect();
    let lo = logs.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = logs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !lo.is_finite() || !hi.is_finite() || hi - lo < 1e-6 {
        return (lo.min(0.0) - 1.0, hi.max(0.0) + 1.0);
    }
    (lo - 0.1, hi + 0.1)
}


/// Dev override for capture runs: NN_VIZ_DATASET=xor|circles|spiral.
fn env_dataset() -> Dataset {
    match std::env::var("NN_VIZ_DATASET").as_deref() {
        Ok("circles") => Dataset::Circles,
        Ok("spiral") => Dataset::Spiral,
        _ => Dataset::Xor,
    }
}

/// Dev override for capture runs: NN_VIZ_HIDDEN=2|4|8.
fn env_hidden() -> usize {
    std::env::var("NN_VIZ_HIDDEN").ok().and_then(|s| s.parse().ok()).filter(|h| [2, 4, 8].contains(h)).unwrap_or(8)
}

/// Sidebar state the user can change. Copied in from NnViz each frame and written back after.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Controls {
    paused: bool,
    lr: f64,
    sel: usize,
    dataset: Dataset,
    hidden: usize,
}

/// Read-only numbers the sidebar displays.
struct SidebarInfo {
    step: usize,
    loss: f64,
    preds: Vec<(Vec<f64>, f64, f64)>,
}

/// Clicks produced by one frame of the sidebar. Rects are the buttons' on-screen positions,
/// so a test can aim a synthetic click at them.
#[derive(Default, Debug)]
struct Actions {
    step_once: bool,
    reset: bool,
    step_rect: Option<egui::Rect>,
    reset_rect: Option<egui::Rect>,
    dataset_rects: Vec<(Dataset, egui::Rect)>,
    hidden_rects: Vec<(usize, egui::Rect)>,
}

/// The left panel. Pure with respect to NnViz: it only reads `info` and edits `ctl`,
/// and reports clicks in the returned Actions.
fn sidebar(ui: &mut egui::Ui, ctl: &mut Controls, info: &SidebarInfo) -> Actions {
    let mut a = Actions::default();
    ui.heading(format!("nn_viz — {}", ctl.dataset.name()));
    ui.label(format!("step {}", info.step));
    ui.label(format!("loss {:.6}", info.loss));
    ui.add(egui::Slider::new(&mut ctl.lr, 0.01..=2.0).text("learning rate"));
    ui.checkbox(&mut ctl.paused, "paused");
    ui.horizontal(|ui| {
        ui.label("dataset:");
        for ds in [Dataset::Xor, Dataset::Circles, Dataset::Spiral] {
            let r = ui.selectable_label(ctl.dataset == ds, ds.name());
            a.dataset_rects.push((ds, r.rect));
            if r.clicked() { ctl.dataset = ds; }
        }
    });
    ui.horizontal(|ui| {
        ui.label("hidden:");
        for h in [2usize, 4, 8] {
            let r = ui.selectable_label(ctl.hidden == h, format!("{h}"));
            a.hidden_rects.push((h, r.rect));
            if r.clicked() { ctl.hidden = h; }
        }
    });
    let step_resp = ui.button("step (x1)");
    a.step_rect = Some(step_resp.rect);
    a.step_once = step_resp.clicked();
    let reset_resp = ui.button("reset");
    a.reset_rect = Some(reset_resp.rect);
    a.reset = reset_resp.clicked();
    ui.separator();
    let n = info.preds.len().max(1);
    ui.add(egui::Slider::new(&mut ctl.sel, 0..=n - 1).text("inspect sample"));
    ui.separator();
    if info.preds.len() <= 8 {
        for (x, y, p) in &info.preds {
            ui.label(format!("{:?} target {} -> {:.4}", x, y, p));
        }
    } else {
        let correct = info.preds.iter().filter(|(_, y, p)| ((*p > 0.5) as i32 as f64) == *y).count();
        ui.label(format!("accuracy {:.1}% ({} of {} points)", 100.0 * correct as f64 / info.preds.len() as f64, correct, info.preds.len()));
        let (x, y, p) = &info.preds[ctl.sel.min(info.preds.len() - 1)];
        ui.label(format!("sample {}: {:.3?} target {} -> {:.4}", ctl.sel, x, y, p));
    }
    a
}

impl NnViz {
    fn new() -> Self {
        NnViz {
            shot_path: std::env::var("NN_VIZ_SHOT").ok(),
            shot_at: std::env::var("NN_VIZ_SHOT_AT").ok().and_then(|s| s.parse().ok()).unwrap_or(2000),
            net: fresh_net(env_hidden()),
            dataset: env_dataset(),
            hidden: env_hidden(),
            data: env_dataset().points(DATA_SEED, PER_CLASS),
            lr: 0.5,
            steps_per_frame: 20,
            step: 0,
            paused: false,
            loss: f64::NAN,
            loss_hist: Vec::new(),
            sel: 0,
            grid: Vec::new(),
            grid_step: usize::MAX,
            ui: None,
        }
    }

    /// Applies one frame's clicks and control changes. Order matches the pre-refactor code:
    /// rebuild on dataset/hidden change, then reset, then a single manual step.
    fn apply(&mut self, ctl: &Controls, a: &Actions) {
        let arch_changed = ctl.hidden != self.hidden || ctl.dataset != self.dataset;
        self.dataset = ctl.dataset;
        self.hidden = ctl.hidden;
        if arch_changed {
            self.rebuild();
        }
        if a.reset {
            self.reset();
        }
        // Single-step always advances one iteration, even while running.
        if a.step_once {
            self.train(1);
        }
    }

    fn reset(&mut self) {
        self.net = fresh_net(self.hidden);
        self.step = 0;
        self.loss_hist.clear();
        self.loss = f64::NAN;
        self.grid_step = usize::MAX;
    }

    /// Switching dataset or architecture rebuilds everything, so no stale weights survive.
    fn rebuild(&mut self) {
        self.data = self.dataset.points(DATA_SEED, PER_CLASS);
        self.net = fresh_net(self.hidden);
        self.sel = 0;
        self.step = 0;
        self.loss_hist.clear();
        self.loss = f64::NAN;
        self.grid_step = usize::MAX;
    }

    fn train(&mut self, n: usize) {
        for _ in 0..n {
            self.loss = self.net.sgd_step(&self.data, self.lr);
            self.step += 1;
            if self.step % 10 == 0 {
                self.loss_hist.push(self.loss);
            }
        }
    }

    /// Recompute the decision-boundary grid only when the weights have changed.
    fn refresh_grid(&mut self) {
        if self.grid_step == self.step {
            return;
        }
        let (lo, hi) = domain(self.dataset);
        let span = hi - lo;
        self.grid = (0..GRID * GRID)
            .map(|k| {
                let gx = (k % GRID) as f64;
                let gy = (k / GRID) as f64;
                let x = lo + span * (gx + 0.5) / GRID as f64;
                let y = lo + span * (gy + 0.5) / GRID as f64;
                self.net.predict(&[x, y])[0] as f32
            })
            .collect();
        self.grid_step = self.step;
    }
}

impl Game for NnViz {
    fn init(&mut self, ctx: &mut Context) -> Result<()> {
        self.ui = Some(EguiState::new(ctx.platform.gl.clone())?);
        log::info!("nn_viz init: {} dataset, hidden={}", self.dataset.name(), self.hidden);
        Ok(())
    }

    fn handle_event(&mut self, _ctx: &mut Context, event: &Event) {
        if let Some(ui) = self.ui.as_mut() {
            ui.handle_event(event);
        }
    }

    fn update(&mut self, _ctx: &mut Context, _dt: f32) -> Result<()> {
        if !self.paused {
            let n = self.steps_per_frame;
            self.train(n);
        }
        Ok(())
    }

    fn render(&mut self, ctx: &mut Context) -> Result<()> {
        self.refresh_grid();
        let size = ctx.platform.drawable_size();
        let loss = self.loss;
        let step = self.step;
        let mut ctl = Controls {
            paused: self.paused,
            lr: self.lr,
            sel: self.sel,
            dataset: self.dataset,
            hidden: self.hidden,
        };
        let hist = self.loss_hist.clone();
        let grid = self.grid.clone();
        let net = self.net.clone();
        let data = self.data.clone();
        let preds: Vec<(Vec<f64>, f64, f64)> = data
            .iter()
            .map(|(x, y)| (x.clone(), *y, net.predict(x)[0]))
            .collect();
        let info = SidebarInfo { step, loss, preds: preds.clone() };
        let acts = net.forward_all(&data[ctl.sel.min(data.len() - 1)].0);

        let Some(ui) = self.ui.as_mut() else { return Ok(()) };
        let mut actions = Actions::default();
        let output = ui.run(size, |egui_ctx| {
            egui::SidePanel::left("controls").show(egui_ctx, |ui| {
                actions = sidebar(ui, &mut ctl, &info);
            });

            egui::CentralPanel::default().show(egui_ctx, |ui| {
                ui.label("network (inspected sample; green = +weight, red = -weight; node brightness = |activation|)");
                let (net_rect, _) = ui.allocate_exact_size(egui::vec2(560.0, 310.0), egui::Sense::hover());
                let painter = ui.painter_at(net_rect);
                // Inset so the bottom node and its value label stay inside the clip rect.
                let inner = net_rect.shrink2(egui::vec2(0.0, 34.0));
                let n_layers = acts.len();
                // Edges first, so nodes draw on top.
                for (k, layer) in net.layers.iter().enumerate() {
                    for o in 0..layer.n_out {
                        for i in 0..layer.n_in {
                            let w = layer.w[o * layer.n_in + i] as f32;
                            let a = node_pos(inner, k, n_layers, i, layer.n_in);
                            let b = node_pos(inner, k + 1, n_layers, o, layer.n_out);
                            let col = if w >= 0.0 {
                                egui::Color32::from_rgba_unmultiplied(80, 220, 120, 200)
                            } else {
                                egui::Color32::from_rgba_unmultiplied(240, 90, 90, 200)
                            };
                            let width = (w.abs() * 2.0).clamp(0.3, 4.0);
                            painter.line_segment([a, b], egui::Stroke::new(width, col));
                        }
                    }
                }
                for (layer_idx, layer_acts) in acts.iter().enumerate() {
                    let n = layer_acts.len();
                    for (node, act) in layer_acts.iter().enumerate() {
                        let pos = node_pos(inner, layer_idx, n_layers, node, n);
                        let g = (40.0 + 200.0 * act.abs().min(1.0)) as u8;
                        painter.circle_filled(pos, 14.0, egui::Color32::from_gray(g));
                        painter.circle_stroke(pos, 14.0, egui::Stroke::new(1.0_f32, egui::Color32::GRAY));
                        painter.text(
                            egui::pos2(pos.x, pos.y + 22.0),
                            egui::Align2::CENTER_TOP,
                            format!("{act:.2}"),
                            egui::FontId::proportional(11.0),
                            egui::Color32::LIGHT_GRAY,
                        );
                    }
                }

                ui.add_space(8.0);
                ui.label("decision boundary (background = network output; dots = training data, ring = inspected)");
                let (bd_rect, _) = ui.allocate_exact_size(egui::vec2(260.0, 260.0), egui::Sense::hover());
                let bp = ui.painter_at(bd_rect);
                let cell = bd_rect.width() / GRID as f32;
                for gy in 0..GRID {
                    for gx in 0..GRID {
                        let p = grid.get(gy * GRID + gx).copied().unwrap_or(0.5);
                        let r = egui::Rect::from_min_size(
                            egui::pos2(bd_rect.left() + gx as f32 * cell, bd_rect.bottom() - (gy as f32 + 1.0) * cell),
                            egui::vec2(cell + 0.5, cell + 0.5),
                        );
                        bp.rect_filled(r, 0.0, heat(p));
                    }
                }
                let (dlo, dhi) = domain(ctl.dataset);
                let to_px = |x: f64, y: f64| {
                    let span = (dhi - dlo) as f32;
                    egui::pos2(
                        bd_rect.left() + (x as f32 - dlo as f32) / span * bd_rect.width(),
                        bd_rect.bottom() - (y as f32 - dlo as f32) / span * bd_rect.height(),
                    )
                };
                for (i, (x, y)) in data.iter().map(|(x, y)| (x.clone(), *y)).enumerate() {
                    let pos = to_px(x[0], x[1]);
                    let fill = if y > 0.5 { egui::Color32::from_rgb(240, 140, 40) } else { egui::Color32::from_rgb(50, 80, 200) };
                    bp.circle_filled(pos, 6.0, fill);
                    bp.circle_stroke(pos, 6.0, egui::Stroke::new(1.5_f32, egui::Color32::WHITE));
                    if i == ctl.sel {
                        bp.circle_stroke(pos, 10.0, egui::Stroke::new(2.0_f32, egui::Color32::YELLOW));
                    }
                }

                ui.add_space(8.0);
                ui.label("loss history (log10 scale)");
                let (lo, hi) = log_range(&hist);
                let (lr_rect, _) = ui.allocate_exact_size(egui::vec2(560.0, 140.0), egui::Sense::hover());
                let lp = ui.painter_at(lr_rect);
                lp.rect_stroke(lr_rect, 0.0, egui::Stroke::new(1.0_f32, egui::Color32::DARK_GRAY));
                lp.text(lr_rect.left_top() + egui::vec2(4.0, 2.0), egui::Align2::LEFT_TOP,
                    format!("1e{:.1}", hi), egui::FontId::proportional(11.0), egui::Color32::GRAY);
                lp.text(lr_rect.left_bottom() + egui::vec2(4.0, -2.0), egui::Align2::LEFT_BOTTOM,
                    format!("1e{:.1}", lo), egui::FontId::proportional(11.0), egui::Color32::GRAY);
                if hist.len() > 1 {
                    let pts: Vec<egui::Pos2> = hist.iter().enumerate().map(|(i, v)| {
                        let fx = i as f32 / (hist.len() - 1) as f32;
                        let ly = v.max(1e-12).log10();
                        let fy = ((ly - lo) / (hi - lo)) as f32;
                        egui::pos2(lr_rect.left() + fx * lr_rect.width(), lr_rect.bottom() - fy * lr_rect.height())
                    }).collect();
                    lp.add(egui::Shape::line(pts, egui::Stroke::new(1.5_f32, egui::Color32::LIGHT_GREEN)));
                }
            });
        });
        ui.paint(size, output);

        self.paused = ctl.paused;
        self.lr = ctl.lr;
        self.sel = ctl.sel;
        self.apply(&ctl, &actions);

        if let Some(path) = self.shot_path.clone() {
            if self.step >= self.shot_at {
                let (w, h) = size;
                let gl = ctx.platform.gl.clone();
                let mut px = vec![0u8; (w * h * 4) as usize];
                unsafe {
                    use engine::glow::HasContext;
                    gl.bind_framebuffer(engine::glow::READ_FRAMEBUFFER, None);
                    gl.read_pixels(0, 0, w as i32, h as i32, engine::glow::RGBA, engine::glow::UNSIGNED_BYTE,
                        engine::glow::PixelPackData::Slice(px.as_mut_slice()));
                }
                let row = (w * 4) as usize;
                let mut flipped = Vec::with_capacity(px.len());
                for r in (0..h as usize).rev() {
                    flipped.extend_from_slice(&px[r * row..(r + 1) * row]);
                }
                image::save_buffer(&path, &flipped, w, h, image::ColorType::Rgba8)?;
                log::info!("SHOT saved {path} at step {} ({}x{})", self.step, w, h);
                ctx.should_quit = true;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Runs the sidebar once on a bare egui context. `events` are injected as raw input.
    fn run_sidebar(ctx: &egui::Context, ctl: &mut Controls, info: &SidebarInfo, events: Vec<egui::Event>) -> Actions {
        let mut out = Actions::default();
        let raw = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1200.0, 900.0))),
            events,
            ..Default::default()
        };
        let _ = ctx.run(raw, |c| {
            egui::SidePanel::left("controls").show(c, |ui| {
                out = sidebar(ui, ctl, info);
            });
        });
        out
    }

    fn click_at(pos: egui::Pos2) -> Vec<egui::Event> {
        vec![
            egui::Event::PointerMoved(pos),
            egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed: true, modifiers: Default::default() },
            egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed: false, modifiers: Default::default() },
        ]
    }

    /// Lay the sidebar out once with no input, to learn where its buttons are.
    fn layout(ctx: &egui::Context, ctl: &mut Controls, info: &SidebarInfo) -> Actions {
        run_sidebar(ctx, ctl, info, vec![])
    }

    fn info_for(a: &NnViz) -> SidebarInfo {
        let preds: Vec<(Vec<f64>, f64, f64)> = a.data.iter().map(|(x, y)| (x.clone(), *y, a.net.predict(x)[0])).collect();
        SidebarInfo { step: a.step, loss: a.loss, preds }
    }

    #[test]
    fn real_click_on_reset_reaches_the_handler() {
        let ctx = egui::Context::default();
        let mut a = app(Dataset::Circles, 4);
        a.train(30);
        let mut ctl = Controls { paused: false, lr: a.lr, sel: 0, dataset: a.dataset, hidden: a.hidden };
        let info = info_for(&a);
        let first = layout(&ctx, &mut ctl, &info);
        let pos = first.reset_rect.expect("reset button laid out").center();
        // Second frame: the click is delivered and must be seen by the button.
        let acts = run_sidebar(&ctx, &mut ctl, &info, click_at(pos));
        assert!(acts.reset, "a click on reset must set actions.reset");
        assert!(!acts.step_once, "a click on reset must not step");
        a.apply(&ctl, &acts);
        assert_eq!(a.step, 0, "reset click must clear training steps");
        assert!(a.loss_hist.is_empty());
    }

    #[test]
    fn real_click_on_step_advances_one_iteration() {
        let ctx = egui::Context::default();
        let mut a = app(Dataset::Xor, 4);
        let mut ctl = Controls { paused: true, lr: a.lr, sel: 0, dataset: a.dataset, hidden: a.hidden };
        let info = info_for(&a);
        let first = layout(&ctx, &mut ctl, &info);
        let pos = first.step_rect.expect("step button laid out").center();
        let acts = run_sidebar(&ctx, &mut ctl, &info, click_at(pos));
        assert!(acts.step_once && !acts.reset);
        a.apply(&ctl, &acts);
        assert_eq!(a.step, 1, "one click on step must advance exactly one iteration");
    }

    #[test]
    fn real_click_on_dataset_toggle_rebuilds() {
        let ctx = egui::Context::default();
        let mut a = app(Dataset::Xor, 4);
        let mut ctl = Controls { paused: false, lr: a.lr, sel: 0, dataset: a.dataset, hidden: a.hidden };
        let info = info_for(&a);
        let first = layout(&ctx, &mut ctl, &info);
        let (_, rect) = *first.dataset_rects.iter().find(|(d, _)| *d == Dataset::Spiral).expect("spiral toggle laid out");
        let acts = run_sidebar(&ctx, &mut ctl, &info, click_at(rect.center()));
        assert_eq!(ctl.dataset, Dataset::Spiral, "a click on the spiral toggle must change the control");
        a.apply(&ctl, &acts);
        assert_eq!(a.dataset, Dataset::Spiral);
        assert_eq!(a.data.len(), 200, "dataset click must rebuild the data");
    }

    #[test]
    fn real_click_on_hidden_toggle_rebuilds_network() {
        let ctx = egui::Context::default();
        let mut a = app(Dataset::Circles, 4);
        let mut ctl = Controls { paused: false, lr: a.lr, sel: 0, dataset: a.dataset, hidden: a.hidden };
        let info = info_for(&a);
        let first = layout(&ctx, &mut ctl, &info);
        let (_, rect) = *first.hidden_rects.iter().find(|(h, _)| *h == 8).expect("hidden=8 toggle laid out");
        let acts = run_sidebar(&ctx, &mut ctl, &info, click_at(rect.center()));
        assert_eq!(ctl.hidden, 8, "a click on the 8 toggle must change the control");
        a.apply(&ctl, &acts);
        assert_eq!(a.net.layers[0].n_out, 8, "hidden click must rebuild the first layer with 8 nodes");
        assert_eq!(a.net.layers[1].n_in, 8);
    }


    /// Build the app state without a window (no EguiState needed for these checks).
    fn app(ds: Dataset, hidden: usize) -> NnViz {
        let mut a = NnViz::new();
        a.dataset = ds;
        a.hidden = hidden;
        a.rebuild();
        a
    }

    #[test]
    fn switching_dataset_rebuilds_data_and_clears_state() {
        let mut a = app(Dataset::Xor, 4);
        a.train(50);
        assert!(a.step > 0 && !a.loss_hist.is_empty());
        a.dataset = Dataset::Spiral;
        a.rebuild();
        assert_eq!(a.data.len(), 200, "spiral should load 200 points");
        assert_eq!(a.step, 0, "step must reset on dataset switch");
        assert!(a.loss_hist.is_empty(), "loss history must clear");
        assert_eq!(a.sel, 0, "inspected sample resets so it stays in range");
    }

    #[test]
    fn switching_hidden_size_changes_architecture() {
        let mut a = app(Dataset::Circles, 4);
        assert_eq!(a.net.layers[0].n_out, 4);
        a.hidden = 8;
        a.rebuild();
        assert_eq!(a.net.layers[0].n_out, 8, "first layer should have 8 hidden nodes");
        assert_eq!(a.net.layers[1].n_in, 8, "output layer must match hidden size");
    }

    #[test]
    fn reset_keeps_dataset_and_architecture() {
        let mut a = app(Dataset::Circles, 2);
        a.train(30);
        a.reset();
        assert_eq!(a.dataset, Dataset::Circles);
        assert_eq!(a.hidden, 2);
        assert_eq!(a.net.layers[0].n_out, 2);
        assert_eq!(a.step, 0);
        assert!(a.loss_hist.is_empty());
    }

    #[test]
    fn training_records_loss_every_ten_steps() {
        let mut a = app(Dataset::Xor, 4);
        a.train(25);
        assert_eq!(a.step, 25);
        assert_eq!(a.loss_hist.len(), 2, "one entry per 10 steps");
    }

    #[test]
    fn grid_cache_follows_weights_and_is_invalidated_on_rebuild() {
        let mut a = app(Dataset::Xor, 4);
        a.refresh_grid();
        assert_eq!(a.grid.len(), GRID * GRID);
        assert_eq!(a.grid_step, a.step);
        a.train(10);
        a.refresh_grid();
        assert_eq!(a.grid_step, a.step, "grid must follow new weights");
        a.dataset = Dataset::Circles;
        a.rebuild();
        assert_eq!(a.grid_step, usize::MAX, "rebuild must invalidate the grid");
    }

    #[test]
    fn domain_covers_each_dataset() {
        // Every point of each dataset must sit inside its plot domain.
        for ds in [Dataset::Xor, Dataset::Circles, Dataset::Spiral] {
            let (lo, hi) = domain(ds);
            for (x, _) in ds.points(DATA_SEED, PER_CLASS) {
                assert!(x.iter().all(|v| *v >= lo && *v <= hi), "{} point {:?} outside [{lo},{hi}]", ds.name(), x);
            }
        }
    }


    #[test]
    fn node_positions_stay_inside_rect() {
        let rect = egui::Rect::from_min_size(egui::pos2(10.0, 20.0), egui::vec2(300.0, 200.0));
        for (layer, n_nodes) in [(0usize, 2usize), (1, 4), (2, 1)] {
            for node in 0..n_nodes {
                let p = node_pos(rect, layer, 3, node, n_nodes);
                assert!(rect.contains(p), "layer {layer} node {node} at {p:?} outside {rect:?}");
            }
        }
    }

    #[test]
    fn log_range_is_padded_and_ordered() {
        let (lo, hi) = log_range(&[1.0, 0.1, 0.001]);
        assert!(lo < -3.0 && hi > 0.0 && lo < hi);
        let (lo, hi) = log_range(&[0.5, 0.5]);
        assert!(hi > lo, "flat history must still give a non-empty range");
    }

    #[test]
    fn heat_endpoints() {
        assert_eq!(heat(0.0), egui::Color32::from_rgb(50, 80, 200));
        assert_eq!(heat(1.0), egui::Color32::from_rgb(240, 140, 40));
    }
}

fn main() -> Result<()> {
    env_logger::init();
    App::run("nn_viz", 1200, 900, NnViz::new())
}
