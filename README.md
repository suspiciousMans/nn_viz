# nn_viz

A small neural-network trainer with a live visualizer. Trains an MLP on XOR, circles, or spiral data and draws the network, the decision boundary, and the loss curve.

## Layout
- `core/` (`nn_core`): typed Rust MLP. Tanh hidden layers, sigmoid output, BCE loss, analytic backprop, seeded datasets. No window dependency.
- `app/` (`nn_viz`): the window and UI, built on the Jame engine (`engine` crate) with egui 0.29.

## Build and run
The app depends on the Jame engine as a git dependency pinned to a revision (see `app/Cargo.toml`), so no local checkout is needed. To develop against a local engine, temporarily point `engine` at a path.

```
cargo test                 # core + app tests
cargo run -p nn_viz        # open the window
```

Dev overrides for capture runs:
- `NN_VIZ_DATASET=xor|circles|spiral`
- `NN_VIZ_HIDDEN=2|4|8|16` (default 8)
- `NN_VIZ_SHOT=<file.png>` with `NN_VIZ_SHOT_AT=<steps>` saves one frame and exits.

## Controls
- `hidden`: 2, 4, 8 or 16 neurons per hidden layer. `hidden layers`: 1 or 2. Spiral needs hidden=8, or two layers of 4.
- `draw points`: left click on the decision-boundary plot adds a point of the chosen brush colour, right click removes the nearest one. Training continues with the current weights.
- Accuracy history is plotted under the loss curve.

## Optimizers
The sidebar switches between SGD and Adam (default Adam, lr 0.03; SGD defaults to lr 0.5). Switching restarts training so the two are comparable. On spiral with hidden=8, Adam reaches about 99% within 2k steps. Plain SGD takes about 40k (60% at 10k, 96.5% at 20k). `core/examples/spiral_opt.rs` has the comparison, including momentum and init gain.

## Known limits
- Hidden=4 with one layer cannot fit the spiral (about 70% at best, a capacity limit). Two layers of 4 reach 93–100% with Adam in the seeds tried (`core/examples/spiral_depth.rs`). Two layers of 8 can get stuck near 50% at lr 0.03; lower the learning rate or reset.
- The `.cargo/config.toml` `[env]` block is needed on any machine with a recent CMake (4.x) and a GCC that defaults to C23, to build the vendored SDL2. It was first added on Linux, not Windows.

## Status
Phases 0–6 done (Adam, hidden=16, two hidden layers, accuracy plot, draw points): core math, window and egui shell, visualizer, dataset and hidden-size controls. Click tests cover reset, step, the dataset and hidden toggles, the paused checkbox, and both sliders and the optimizer toggle.
