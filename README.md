# Metal Orchard

`metal-orchard` is the incubation ground for Tensor v0, a tensor runtime and kernel stack engineered for Apple Silicon and the Metal application programming interface. The repository hosts every source file, test, and document required to construct the project; no external submodules are referenced.

## Repository Overview
- `metal-tensor/` contains the primary library. Its `metal/` tree defines `Storage`, `Tensor`, `Node`, and auxiliary infrastructure, while `tests/` verifies contiguity semantics, host and device copies through `Tensor::to`, allocation profiling via `dump_live_tensors`, and gradient propagation using the `backward` routine.
- `experimental/` carries the historical PyTorch bridge. Its `orchard_ops/` folder builds C++ and Python extension modules, `benchmarks/` and `tests/` exercise them under PyTorch, and `kernel_lib/` stores prebuilt FlashAttention binaries. The `data/` and `runs/` directories hold transient datasets and benchmark outputs and remain untracked by Git to avoid committing large artifacts. The bridge is disabled by default and only compiles when configuration passes -DORCHARD_BUILD_EXPERIMENTAL=ON and runtime sets USE_FLASH_ATTN to 2.
- `docs/` collects narrative material including design notes and project status reports.
- `.github/` defines the continuous integration pipeline in `workflows/macos.yml`.
- The repository root hosts `CMakeLists.txt` for configuring all targets and a `build/` directory is created by contributors to hold generated files.

## Feature Highlights
- Rank-eight shape representation with explicit stride control for advanced view and slice operations.
- Intrusive reference counted `Storage` objects that permit zero-copy wrapping of external buffers through `Tensor::fromData`.
- Host and device transfers mediated by `Tensor::to`, yielding zero-copy aliases when the destination `Device` matches the source.
- Allocation profiling managed by `metal/common/Profiling.h`. When `ORCHARD_TENSOR_PROFILE` is present in the environment, allocation and release events stream to `/tmp/orchard_tensor_profile.log`, and `dump_live_tensors` reports outstanding buffers.
- Autograd foundations supplied by the `requires_grad` flag, gradient accumulation in `Tensor::grad`, and a graph of `Node` and `Edge` objects traversed by `backward`.
- Initial Metal compute kernels, located under `metal-tensor/metal/kernels/`, implementing vector addition with automatic fallback to CPU code when Metal execution is unavailable.

## Building
1. Install Xcode 15+, the Metal 4 SDK, and the command line tools.
2. From the repository root run cmake -S . -B build -G Ninja to produce build files in the `build/` directory. The Ninja generator matches the GitHub Actions workflow.
3. Invoke cmake --build build to compile the static libraries and unit tests. Pass -DORCHARD_BUILD_EXPERIMENTAL=ON during configuration to compile the legacy PyTorch bridge.
4. Linux hosts lack the required toolchain; running the above commands still provides diagnostic output that must be included in pull requests.

## Testing
Run cmake --build build --target test to execute the suite under `metal-tensor/tests`. The tests cover CPU and Metal paths, queue reuse, profiling hooks, and autograd gradients. Always attempt to configure and run tests before submitting a pull request. Even on systems lacking the Metal SDK, failing output is still valuable and should be reported in the pull request.

## Autograd Notes
Tensors opt into gradient tracking through the requires_grad property. Operations such as Tensor::add register Node instances connected by Edge relationships. Calling backward performs a reverse traversal to populate Tensor::grad on leaf tensors. Only elementwise addition is currently implemented; further differentiable operators will expand the graph.

## Continuous Integration
The macOS workflow described in .github/workflows/macos.yml installs dependencies through Homebrew, configures the project with the Ninja generator, treats warnings as errors, and executes the full test suite. Build artifacts and ccache directories are cached to accelerate subsequent runs. Any warning or failing test causes the pipeline to halt.

## Profiling Guidance
Setting ORCHARD_TENSOR_PROFILE enables logging of allocation and deallocation events along with explicit dumps triggered by dump_live_tensors. Logs accumulate at /tmp/orchard_tensor_profile.log for offline inspection.

## Current Status
- Tensor v0 handles intrusive storage, host and device transfers, and validated gradients for elementwise addition.
- The PyTorch bridge under `experimental/` remains available for regression checks but is omitted from standard builds.
- macOS continuous integration enforces warnings-as-errors and executes the test suite; Linux hosts provide diagnostic failures only.
- Documentation covers design specifications, tensor features, and project status but evolves with each milestone.

## Milestones
1. Fused FlashAttention backward kernels with dropout support.
2. Removal of the PyTorch bridge once Tensor v0 reaches feature parity.
3. Public benchmarking harness with published baselines across Apple Silicon generations.
4. Versioned 0.1 release capturing the first stable API.

## Next Steps
1. Extend the differentiable operator set and broaden autograd coverage.
2. Replace CPU fallbacks with tuned Metal kernels and document performance gains.
3. Grow the test matrix for multi-device transfers, profiling scenarios, and stress tests.

## Contributing
All contributors must read and comply with AGENTS.md. It details repository expectations, commit formatting, the requirement to use `rg` for searches, and the mandatory build and test steps that precede every pull request.
