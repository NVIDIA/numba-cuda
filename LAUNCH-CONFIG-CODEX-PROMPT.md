# Codex Prompt: Launch Config Benchmarking (Baseline vs Contextvar vs V2)

You are working in the `numba-cuda` repo with three branches/worktrees:
- **baseline**: main (or a specific baseline ref) in `~/src/numba-cuda-main`
- **contextvar**: old implementation in `~/src/numba-cuda` (branch `280-launch-config-contextvar`)
- **v2**: new implementation in `~/src/280-launch-config-v2` (branch `280-launch-config-v2`)

Your goal is to benchmark CUDA kernel launch overhead across all three.

## Key Artifacts
- `scripts/bench-launch-overhead.py` (micro-benchmark for launch overhead)
- `scripts/bench-against.py` (two-ref compare helper)
- Pixi tasks in `pixi.toml`:
  - `bench-launch-overhead`
  - `bench`
  - `benchcmp`
  - `bench-against`

## Benchmarks to Run
### A) Launch Overhead Micro-Benchmark (Primary)
Run a three-way comparison using the micro-benchmark script. It measures kernel
launch overhead for 0..4 args and reports mean/stdev and deltas vs baseline.

From `~/src/280-launch-config-v2`:
```bash
pixi run -e cu-12-9-py312 bench-launch-overhead \
  --repo baseline=$HOME/src/numba-cuda-main \
  --repo contextvar=$HOME/src/numba-cuda \
  --repo v2=$HOME/src/280-launch-config-v2
```

Notes:
- The script will `pip install -e` each repo by default. If you already have
  them installed in the active environment, add `--no-install`.
- You can change the Python interpreter with `--python /path/to/python`.
- You can tune loops with `--loops 200000,200000,200000,200000,20000`.
- Use `--output results.json` to save output.

### B) Broader Benchmarks (Optional)
Use the full pytest benchmark suite for more coverage:

```bash
pixi run -e cu-12-9-py312 bench
```

If you want to compare two refs in a temporary worktree:

```bash
pixi run -e cu-12-9-py312 bench-against BASELINE_REF PROPOSED_REF
```

## Environment Expectations
- GPU available (the micro-benchmark refuses to run under CUDA simulator).
- Use the same CUDA toolkit + driver across all runs.
- Ensure `pixi` env `cu-12-9-py312` (or equivalent) is available.

## Data to Capture
- Full stdout from `bench-launch-overhead` (table + deltas)
- Optional JSON output for records
- Device info from the benchmark output (GPU name, CC, CUDA runtime version)
- Git SHAs for all three repos

## Interpretation Guidance
- Focus on delta vs baseline for each arg count.
- Large increases (e.g., +30–50%) are likely unacceptable for the new
  implementation if the contextvar branch is already known to be slower.
- If v2 overhead is close to baseline and materially below contextvar, that
  supports adopting v2.

## Cleanup / Reporting
- Summarize results in a short table:
  - mean ± stdev for each arg count
  - delta vs baseline for each branch
- Note any anomalies (e.g., variance, outliers, or errors)
- If results are noisy, increase repeats or loops and rerun.

## Troubleshooting
- If you see nvjitlink or compilation failures, verify that the active `pixi`
  environment has CUDA toolchain components compatible with the driver.
- If results are all identical, ensure each repo is actually installed (and
  not accidentally shadowed by an earlier editable install). Check `pip show
  numba-cuda` and `python -c "import numba_cuda,inspect;print(numba_cuda.__file__)"`.

## Deliverables
- A concise report of the measured overheads and deltas for baseline vs
  contextvar vs v2.
- A short conclusion on whether v2 overhead is acceptable.
