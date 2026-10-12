# nn_viz

A small neural-network trainer with a live visualizer. Trains an MLP on XOR, circles, or spiral data and draws the network, the decision boundary, and the loss curve.

## Layout
- `core/` (`nn_core`): typed Rust MLP. Tanh hidden layers, sigmoid output, BCE loss, analytic backprop, seeded datasets. No window dependency.
- `app/` (`nn_viz`): the window and UI, built on the Jame engine (`engine` crate) with egui 0.29.

## Build and run
The app depends on the Jame engine by local path, `C:/Users/james/jame-engine/engine` (see `app/Cargo.toml`). Clone that checkout first or change the path.

```
cargo test                 # core + app tests
cargo run -p nn_viz        # open the window
```

Dev overrides for capture runs:
- `NN_VIZ_DATASET=xor|circles|spiral`
- `NN_VIZ_HIDDEN=2|4|8` (default 8)
- `NN_VIZ_SHOT=<file.png>` with `NN_VIZ_SHOT_AT=<steps>` saves one frame and exits.

## Known limits
- Spiral needs training time: about 60% at 10k steps, 96.5% at 20k, 99% at 40k (hidden=8, lr=0.5). Hidden=4 stalls near 60–71%.
- The `.cargo/config.toml` `[env]` block is needed on Windows to build vendored SDL2 with a recent CMake.

## Status
Phases 0–4 done: core math, window and egui shell, visualizer, dataset and hidden-size controls. Click tests cover reset, step, and the dataset and hidden toggles. Slider and checkbox clicks are not yet tested.
