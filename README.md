# Metal Orchard

This repository hosts the emerging **Tensor v0** framework for Apple Silicon.
The core implementation lives under `metal-tensor/` and builds into two
static libraries: `liborchard_core.a` and `liborchard_metal.a`.

The earlier PyTorch-centric approach is preserved under `experimental/` for
reference. It contains a forward-only FlashAttention kernel wired in via a
PyTorch extension and monkey‑patch. The kernel is correct but speedups only
appear at long sequence lengths and the backward path is still a CPU fallback.
Work has now pivoted to a **ground‑up Metal tensor stack** where both the
forward and eventual backward passes are implemented natively for Apple
Silicon, targeting performance beyond what PyTorch can deliver on this
hardware.

See `docs/project_status.md` for an overview of current progress and the
roadmap.

## Building

```bash
cmake -S . -B build
cmake --build build
```

Enable the optional PyTorch bridge by passing
`-DORCHARD_BUILD_EXPERIMENTAL=ON` to the first command.

## Tests

A test suite will live under `metal-tensor/tests`. Run it with:

```bash
cmake --build build --target test
```

## Contributing

See `AGENTS.md` for the full contributor guide and development workflow.
