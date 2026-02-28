# Launch Config Benchmarking

This repo includes lightweight benchmarking scaffolding to quantify CUDA kernel
launch overhead across three launch-config implementations (baseline, old
contextvar branch, and the new v2 branch).

## Status / Next Steps (Launch Config Work)
- LC-S plumbing is implemented in `dispatcher.py` and supporting files.
- CUDA LC-S tests have been run on GPU in this branch and are passing.
- There are uncommitted changes in `compiler.py`, `dispatcher.py`, and
  `scripts/bench-launch-overhead.py` that add an explicit LC-S API on
  `_LaunchConfiguration` and a compiler hook to honor it.
- cccl rewrite integration now uses the explicit LC-S API with a fallback to
  metadata for compatibility.
- Cross-process disk-cache behavior is covered by LC-S caching tests and passes.
- See `LAUNCH-CONFIG-TODO.md` for a detailed handoff checklist.

## What’s Included

### 1) `scripts/bench-launch-overhead.py`
A focused micro-benchmark that measures launch overhead (us/launch) for kernels
with 0..4 arguments, using a 1x1 launch. It:
- warms up each kernel
- runs `loops` iterations per kernel (default: 100k for 0–3 args, 10k for 4 args)
- repeats the measurement (default: 7 repeats)
- reports mean/stdev and deltas vs the first repo
- optionally writes JSON output

The benchmark is designed to compare multiple repos (or worktrees) in the same
Python environment.

### 2) `scripts/bench-against.py`
A helper that compares benchmarks between two git refs using a temporary
worktree, running the pixi benchmark tasks before and after.

### 3) Pixi tasks
Defined in `pixi.toml`:
- `bench-launch-overhead`: runs `scripts/bench-launch-overhead.py`
- `bench`: pytest benchmark suite (`numba.cuda.tests.benchmarks`)
- `benchcmp`: compare benchmark results from `bench`
- `bench-against`: runs `scripts/bench-against.py`

## Recommended Usage (Three-Way Compare)

Assuming you have three working trees for:
- **baseline** (main or a baseline ref)
- **contextvar** (old implementation)
- **v2** (new implementation)

Run the launch-overhead micro-benchmark:

```bash
pixi run -e cu-12-9-py312 bench-launch-overhead \
  --repo baseline=/path/to/numba-cuda-main \
  --repo contextvar=/path/to/numba-cuda-contextvar \
  --repo v2=/home/trentn/src/280-launch-config-v2
```

Notes:
- The script will `pip install -e` each repo by default. Use `--no-install`
  if you have already installed them and want to skip reinstalling.
- Use `--python` to point at a specific interpreter if needed.
- Use `--loops` to override the default loop counts, e.g. `--loops 200000,200000,200000,200000,20000`.
- Use `--output results.json` to persist the results.

## Example Output

```
Launch overhead (us/launch):
args  baseline            contextvar          v2
0     4.10 +/- 0.05       6.20 +/- 0.06       4.50 +/- 0.04
1     4.40 +/- 0.05       6.60 +/- 0.06       4.80 +/- 0.05
...

Deltas vs baseline:
args  contextvar          v2
0     2.10 (+51.2%)       0.40 (+9.8%)
1     2.20 (+50.0%)       0.40 (+9.1%)
...
```

## Benchmark Suite (Broader Coverage)

For more extensive benchmark coverage (not just launch overhead), use:

```bash
pixi run -e cu-12-9-py312 bench
```

To compare two git refs using a temporary worktree:

```bash
pixi run -e cu-12-9-py312 bench-against HEAD~ HEAD
```

This runs `bench` on the baseline ref and `benchcmp` on the proposed ref.

## Notes / Constraints

- Benchmarks require a real GPU (CUDA simulator is rejected).
- The micro-benchmark intentionally keeps kernels trivial to isolate launch
  overhead.
- The three-way comparison is the most direct way to capture the relative
  overhead introduced by launch-config state management.
