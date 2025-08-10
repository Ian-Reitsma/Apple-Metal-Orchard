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
- seeded autograd scaffold with `Tensor::requires_grad`, gradient storage, and
  a graph of `Node` objects driving `backward`
- Metal compute kernels beginning with elementwise add, falling back to CPU
  paths only when Metal is unavailable

## Building
1. Install Xcode 15+, the Metal 4 SDK, and the command line tools
2. Configure the project from the repository root:
   ```bash
   cmake -S . -B build -G Ninja
   ```
   The Ninja generator matches the GitHub Actions workflow.
3. Build the static libraries and unit tests:
   ```bash
   cmake --build build
   ```
4. Pass `-DORCHARD_BUILD_EXPERIMENTAL=ON` during configuration to compile the
   legacy PyTorch bridge

## Testing
Run the full test suite, which exercises CPU and Metal paths, with:
```bash
cmake --build build --target test
```
Always attempt to configure and run tests before submitting a pull request.
Even on systems lacking the Metal SDK, a failing configuration still provides
useful diagnostics.

## Autograd
Tensors opt into gradient tracking via `requires_grad`. Operations such as
`Tensor::add` register `Node` objects in a directed graph. Calling `backward`
traverses this graph and populates `Tensor::grad` tensors on leaf nodes. The
current implementation covers elementwise add; more operators will follow.

## Continuous Integration
GitHub Actions runs on `macos-latest` and mirrors the build instructions above.
Dependencies are installed via Homebrew, warnings are promoted to errors, and
tests must pass for the workflow to succeed. CMake build trees and ccache
objects are cached to accelerate incremental runs. Any build warning or test
failure causes the pipeline to fail.

## Profiling
Enable allocation profiling by exporting `ORCHARD_TENSOR_PROFILE=1`. All
allocations, frees, and live tensor dumps append to
`/tmp/orchard_tensor_profile.log`.

## Next Steps
- Broaden the operator set and autograd coverage beyond elementwise add
- Replace remaining CPU fallbacks with native Metal kernels
- Benchmark kernel performance and publish results

## Contributing
Follow `AGENTS.md` for contributor guidelines and workflow.
