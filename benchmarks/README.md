# Benchmarks

Scripts here record hardware details, runtime flags, and kernel timings for
Tensor v0 operations.

## Usage
Run `python benchmarks/run.py -o /tmp/bench` after building to emit a JSON
file under `/tmp/bench/<commit>/benchmarks.json`. The harness exercises add,
mul, matmul, and reduce_sum kernels through the Tensor API.

Benchmark outputs are untracked; generate them locally as needed.
