# Metal Orchard

This repository hosts the emerging **Tensor v0** framework for Apple Silicon.
The codebase is rebuilding a tensor runtime and kernel stack from the ground up
for Metal. The core implementation lives under `metal-tensor/` and builds into
`liborchard_core.a` and `liborchard_metal.a`.

## Features
- rank‑8 shapes and explicit strides
- intrusive reference counted `Storage` with zero‑copy wrapping via
  `Tensor::fromData`
- cross‑device copies through `Tensor::to` with zero‑copy when devices match
- allocation profiling that logs to `/tmp/orchard_tensor_profile.log` and
  `dump_live_tensors()` for debugging

## Building
1. Install Xcode 15+ and the Metal 4 SDK
2. Configure the project:
   ```bash
   cmake -S . -B build
   ```
3. Build the libraries and tests:
   ```bash
   cmake --build build
   ```
4. Add `-DORCHARD_BUILD_EXPERIMENTAL=ON` during configuration to compile the
   legacy PyTorch bridge

## Testing
Run the test suite with:
```bash
cmake --build build --target test
```
Always attempt to configure and run tests before submitting a pull request,
even when the Metal toolchain is missing.

## Profiling
Enable allocation profiling by exporting `ORCHARD_TENSOR_PROFILE=1`. All
allocations, frees, and live tensor dumps append to
`/tmp/orchard_tensor_profile.log`.

## Next Steps
- Implement more tensor operators and begin autograd
- Replace CPU fallbacks with native Metal kernels
- Establish macOS CI that compiles and runs the full test suite
- Benchmark kernel performance and document results

## Contributing
Follow `AGENTS.md` for contributor guidelines and workflow.
