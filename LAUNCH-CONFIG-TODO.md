# Launch Config Sensitive (LC-S) plumbing

Last updated: 2026-02-19

## Current status (summary)
- Launch-config TLS capture exists in C extension and is exposed via
  `numba_cuda/numba/cuda/launchconfig.py` (current/ensure/capture helpers).
- Dispatcher plumbing for LC-S is implemented:
  - `_Kernel` captures `launch_config_sensitive` from compile metadata.
  - `CUDADispatcher` tracks LC-S and routes to per-launch-config sub-dispatchers.
  - Disk cache includes a launch-config key and LC-S marker file (`.lcs`).
- Tests added:
  - `numba_cuda/numba/cuda/tests/cudapy/test_launch_config_sensitive.py`
  - `numba_cuda/numba/cuda/tests/cudapy/cache_launch_config_sensitive_usecases.py`
  - `numba_cuda/numba/cuda/tests/cudapy/test_caching.py` LC-S coverage
- Docs updated: `docs/source/reference/kernel.rst`.
- In cccl, `cuda/coop/_rewrite.py` now marks LC-S when accessing launch config.
  It uses the explicit LaunchConfiguration API when available, with fallback to
  `state.metadata["launch_config_sensitive"] = True` for compatibility.
- CUDA tests run and passing on GPU in this worktree:
  - `pixi run -e cu-12-9-py312 pytest testing --pyargs numba.cuda.tests.cudapy.test_launch_config_sensitive -k launch_config_sensitive`
  - `pixi run -e cu-12-9-py312 pytest testing --pyargs numba.cuda.tests.cudapy.test_caching -k launch_config_sensitive`

## Local working tree state (numba-cuda)
- Branch: `280-launch-config-v2`
- Modified files (uncommitted):
  - `numba_cuda/numba/cuda/compiler.py`
  - `numba_cuda/numba/cuda/dispatcher.py`
  - `scripts/bench-launch-overhead.py`
- Untracked: `PR.md`, `tags`

## New (uncommitted) LC-S API work
- `_LaunchConfiguration` gains explicit helpers:
  - `mark_kernel_as_launch_config_sensitive()`
  - `get_kernel_launch_config_sensitive()`
  - `is_kernel_launch_config_sensitive()`
- `CUDABackend` sets metadata when the launch config is explicitly marked.
  This provides an official path to mark LC-S without poking at `state.metadata`
  directly from rewrites.

## Remaining TODO
1. **Cleanup**
   - Remove or handle untracked `PR.md` and `tags` before committing.

## Completed checks (2026-02-19)
- **Cross-process disk-cache behavior**
  - Verified by:
    `pixi run -e cu-12-9-py312 pytest testing --pyargs numba.cuda.tests.cudapy.test_caching -k launch_config_sensitive`
  - `LaunchConfigSensitiveCachingTest.test_launch_config_sensitive_cache_keys`
    exercises cache reuse in a separate process and passed.
- **Launch path audit**
  - Python launch paths in `dispatcher.py` all route through
    `CUDADispatcher.call()`: `__getitem__()` -> `configure()` ->
    `_LaunchConfiguration.__call__()` -> `call()`, plus `ForAll.__call__()`.

## Notes
- Separate PR for cache invalidation on `numba_cuda.__version__` is already
  pushed; do not re-implement here.
