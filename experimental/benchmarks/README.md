# Benchmarks

Benchmark and orchestration scripts for the experimental PyTorch path live here. Use them to measure FlashAttention or other prototype kernels on top of PyTorch. The scripts assume PyTorch and its dependencies are available and primarily serve regression comparisons against Tensor v0.

Results are not checked into source control. Each run should note the commit hash, input shapes, and whether dropout was enabled so performance changes can be correlated with code revisions.

## Next Steps
Update or remove these scripts once equivalent benchmarking exists for the Metal-native stack and the experimental path is retired.
