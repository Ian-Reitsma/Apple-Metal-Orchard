# AGENTS.md – Command and Coordination Manual

This manual governs every action within the `metal-orchard` repository. The project rebuilds a Metal-centric tensor runtime and kernel stack known internally as Tensor v0. All source code, tests, and documentation reside in this repository; no submodules or external trees exist. Every contributor must treat this document as the authoritative source of truth when modifying any file.

## Repository Cartography
The directory layout is intentionally shallow to make navigation unambiguous:

- `metal-tensor/` – hosts the core library. Inside this directory:
  - `metal/` contains the production source. It is subdivided into `common/` for utilities such as `Profiling.h`, `core/` for primary abstractions like `Tensor` and `Storage`, `kernels/` for Metal shader entry points, and `runtime/` for queue management and device utilities.
  - `tests/` includes the comprehensive suite exercising contiguity handling, device transfer paths, allocation profiling, and autograd validation. The primary entry file is `tensor_tests.cpp`.
  - `docs/` records design specifications such as `design_spec_tensor_v0.md`.
- `experimental/` – preserves the legacy PyTorch-based path. Subdirectories include `orchard_ops/` for C++ and Python extension modules, `benchmarks/` for performance scripts, `tests/` for PyTorch-driven verification, `kernel_lib/` for prebuilt FlashAttention binaries, and transient holders like `data/` and `runs/` which remain ignored by Git.
- `docs/` – houses project-wide narrative material.
- `.github/` – contains the continuous integration workflow `workflows/macos.yml` which configures the macOS builder.

## Component Highlights
- `Storage` implements intrusive reference counting to track underlying `MTLBuffer` allocations. The static factory `Tensor::fromData` wraps external memory without copying by accepting a raw pointer, explicit shape, data type, device, and optional deleter callback.
- `Tensor::to` performs host and device transfers. When the source and destination `Device` values match, the call resolves to a zero-copy view preserving the original storage.
- Allocation profiling is governed by `metal/common/Profiling.h`. Setting the environment variable `ORCHARD_TENSOR_PROFILE` enables logging of alloc, free, and live events. The diagnostic helper `dump_live_tensors` enumerates outstanding `Storage` instances to standard error for post-mortem analysis.
- Autograd is scaffolded through `Tensor::requires_grad`, the gradient tensor accessible via `Tensor::grad`, and a graph of `Node` and `Edge` objects culminating in the `backward` routine.
- Metal compute kernels cover elementwise add and multiply, matrix multiply, and reduce_sum. Shaders live under `metal/kernels/` and dispatch one thread per output element. CPU fallbacks in `metal/core/` activate when Metal is unavailable.

## Build Protocol
1. Obtain Xcode 15+ with the Metal 4 SDK; ensure command line tools are active.
2. From the repo root, configure with CMake into a `build/` dir (use Ninja). Optionally enable `-DORCHARD_BUILD_EXPERIMENTAL=ON` to compile the legacy PyTorch bridge under `experimental/`.
3. Build the default target. Outputs include `liborchard_core.a` and `liborchard_metal.a`.

## Test Protocol
1. With a configured build tree, run the `test` target. Tests live under `metal-tensor/tests/` and exercise CPU/Metal paths.
2. Run configure + tests before every PR and capture failure logs in the PR description when toolchains are missing.

## Benchmark Protocol
- Invoke `python benchmarks/run.py -o /tmp/bench` after building to record kernel timings and hardware metadata.
- Generated JSON results remain untracked; CI archives them as artifacts.

## Contribution Directives
- C++20 and ObjC++ only; format with `clang-format`.
- Do not commit artifacts larger than 5 MB. Large datasets belong under `experimental/data/` or `experimental/runs/` and remain untracked.
- Documentation/commits avoid code blocks; refer to identifiers by name (e.g., `Tensor::fromData`, `ORCHARD_TENSOR_PROFILE`, `flash_attn`).
- Always run configure + tests locally before PRs.
- Use `rg` for repository searches and avoid `ls -R` or `grep -R` to keep scans efficient.
- Commit messages use the imperative mood and include a short summary line only.
- Capture the output of `cmake -S . -B build -G Ninja` and `cmake --build build --target test` and report failures in the pull request.
- Reference touched files by path and line number in pull request descriptions.
- Work exclusively on the default branch and refrain from creating new branches within this repository.
- Keep the CI matrix green. macOS runners for `macos-13` and `macos-14` must pass; the Linux diagnostic job may fail but its logs require review before merging.

## Workflow Checklist
1. Run `cmake -S . -B build -G Ninja` from the repository root.
2. Invoke `cmake --build build --target test` and note any errors.
3. Stage changes with `git add` and create a single commit per task.
4. Formulate a pull request summarizing the intent, the files modified, and the test outcomes.

## Current Status
- Tensor v0 supplies intrusive storage, host and device transfer paths, basic autograd, and initial Metal kernels.
- The legacy PyTorch bridge persists under `experimental/` but is excluded from default builds.
- Continuous integration now covers `macos-13` (M1) and `macos-14` (M2) with Xcode 15.3 pinned and Homebrew updates disabled. A Linux job installs a clang-based Objective-C++ toolchain and is allowed to fail for diagnostics.
- Documentation outlines tensor internals, profiling hooks, and contributor expectations yet remains a living reference.

## Milestones
1. Phase out the PyTorch bridge once Tensor v0 covers FlashAttention and essential autograd features.
2. Ship a fully Metal-native backward pass with fused FlashAttention kernels.
3. Deliver a public benchmarking suite with reproducible configurations and published baselines.
4. Cut a 0.1 release that freezes the API and tags the first feature-complete snapshot.

## Next Steps
1. Expand differentiable operations beyond elementwise add to include matmul, reductions, and view transformations.
2. Replace remaining CPU fallbacks with tuned Metal kernels and delete redundant host code.
3. Grow the test matrix to cover multi-device transfers, profiling scenarios, and stress conditions.
4. Stand up a benchmarking harness that records hardware, runtime flags, and kernel timing for every commit.
