# Launch Config Sensitive (LC-S) plumbing

## Context / findings
- Current launchconfig capture is compile-time only. It is injected into the TLS only around `cuda_compile_only(...)` in `numba_cuda/numba/cuda/cext/_dispatcher.cpp` and read via `numba_cuda/numba/cuda/launchconfig.py`.
- This means `launchconfig.current_launch_config()` is only non-`None` while compilation is happening; cache hits do not update it.
- CUDA kernel compilation cache is keyed only by argument types. For rewrites that depend on launch config (e.g. cuda.coop using blockdim to select CUB algorithms or to size buffers), reusing a compiled kernel with a different grid/block config is unsafe.

## Test-first scaffolding added
- New test file: `numba_cuda/numba/cuda/tests/cudapy/test_launch_config_sensitive.py`.
- It registers a custom `Rewrite` for a single kernel name, and mimics cuda.coop behavior:
  - Reads launch config during rewrite via `launchconfig.ensure_current_launch_config()`.
  - Logs config to a global list (`LAUNCH_CONFIG_LOG`).
  - Sets `state.metadata["launch_config_sensitive"] = True` so later compilation can observe it.
- The test launches the same kernel twice with different block sizes and **expects two log entries** (i.e. a recompile per launch-config change).
- This test should FAIL before the fix (only first launch compiles/logs).

## ✅ Launch-config-sensitive plumbing (implemented)
- `_Kernel` now captures `launch_config_sensitive` from compile metadata.
- `CUDADispatcher` now:
  - tracks launch-config sensitivity
  - computes a launch-config key `(griddim, blockdim, sharedmem)`
  - routes calls for new configs to per-config sub-dispatchers
  - keeps the original dispatcher as the default config
- This avoids TypeManager conflicts and allows multiple compiled kernels per argtypes when launch config differs.
- Implemented in `numba_cuda/numba/cuda/dispatcher.py`.

## Remaining TODO
1. **Run the new test on a CUDA machine**
   - `python -m pytest numba_cuda/numba/cuda/tests/cudapy/test_launch_config_sensitive.py -k launch_config_sensitive`

2. **Disk cache interaction**
   - Disk cache key currently ignores launch config. For LC‑S kernels, decide:
     - disable caching, or
     - extend cache key to include launch config, or
     - use a separate cache path keyed by launch config.
   - Ensure kernel rebuild path preserves `launch_config_sensitive` if needed.

3. **Check for any additional call sites**
   - Make sure all kernel launch paths for CUDADispatcher go through `call(...)` so the dispatcher selection is effective.

## Files changed so far
- `docs/source/reference/kernel.rst`
- `numba_cuda/numba/cuda/cext/_dispatcher.cpp`
- `numba_cuda/numba/cuda/dispatcher.py`
- `numba_cuda/numba/cuda/launchconfig.py` (new)
- `numba_cuda/numba/cuda/tests/cudapy/test_dispatcher.py`
- `numba_cuda/numba/cuda/tests/cudapy/test_launch_config_sensitive.py` (new)
- `LAUNCH-CONFIG.md`
- `LAUNCH-CONFIG-CODEX-PROMPT.md`
- `LAUNCH-CONFIG-TODO.md`
- `pixi.toml`
- `scripts/bench-launch-overhead.py`
- `scripts/plot-launch-overhead.py`
- `plots/`
