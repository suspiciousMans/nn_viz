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
- `NN_VIZ_HIDDEN=2|4|8` (default 8)
- `NN_VIZ_SHOT=<file.png>` with `NN_VIZ_SHOT_AT=<steps>` saves one frame and exits.

## Optimizers
The sidebar switches between SGD and Adam (default Adam, lr 0.03; SGD defaults to lr 0.5). Switching restarts training so the two are comparable. On spiral with hidden=8, Adam reaches about 99% within 2k steps. Plain SGD takes about 40k (60% at 10k, 96.5% at 20k). `core/examples/spiral_opt.rs` has the comparison, including momentum and init gain.

## Known limits
- Hidden=4 cannot fit the spiral with either optimizer: it tops out around 70–80%. That is a capacity limit of four tanh units, not a training-speed problem.
- The `.cargo/config.toml` `[env]` block is needed on Windows to build vendored SDL2 with a recent CMake.

## Status
Phases 0–5 done (Adam added): core math, window and egui shell, visualizer, dataset and hidden-size controls. Click tests cover reset, step, the dataset and hidden toggles, the paused checkbox, and both sliders and the optimizer toggle.
