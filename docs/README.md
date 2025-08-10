# Documentation

The `docs` directory collects narrative material that spans the entire project. Each file explains the state of the effort or elaborates on a design facet.

## Contents
- `project_status.md` records milestones, active workstreams, and upcoming tasks for both the experimental PyTorch bridge and the Metal-native tensor path.
- `tensor.md` introduces Tensor v0, describes the toolchain requirements, and outlines features such as zero-copy transfers and allocation profiling.
- `../benchmarks` houses scripts that capture hardware details, runtime flags, and kernel timings.

## Current Status
- `project_status.md` and `tensor.md` are current as of August 2025 and chart both the experimental bridge and the Metal-native tensor implementation.
- Design specifications under `metal-tensor/docs/` provide deeper notes on kernels, runtime contexts, and autograd scaffolding.
- Documentation is actively maintained yet lacks full API references and architecture diagrams.

## Milestones
1. Publish a complete API reference for Tensor v0 covering public headers and usage semantics.
2. Add diagrams illustrating memory flows, command queues, and autograd graphs.
3. Establish a changelog that records major documentation updates alongside code milestones.

## Next Steps
1. Capture profiling and debugging tutorials that guide new contributors through common workflows.
2. Document the FlashAttention migration plan from the experimental bridge to Tensor v0.
3. Cross-link design notes and tutorials so related topics remain easy to navigate.

## Benchmarks
Scripts in `../benchmarks` record hardware information, runtime flags beginning with `ORCHARD_`, and kernel timings. Invoke `python benchmarks/run.py -o /tmp/bench` after building to generate a JSON file under `/tmp/bench/<commit>/benchmarks.json`.

## Guidelines
- Expand these documents whenever significant features land or roadmap items change.
- Reference identifiers with inline code and avoid fenced code blocks.
- Ensure any added file fits within the repository’s 5 MB artifact limit.
