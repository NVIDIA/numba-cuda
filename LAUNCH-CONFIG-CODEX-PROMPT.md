# Codex Prompt: Launch Config Sensitive (LC-S) work for cuda.coop

You are resuming launch-config work in `numba-cuda` to support cuda.coop
single-phase. The cache invalidation change for `numba_cuda.__version__` is
already in a separate PR; do not redo that here.

## Repos / worktrees
- `numba-cuda` (current worktree): `/home/trentn/src/280-launch-config-v2`
  - branch: `280-launch-config-v2`
- `numba-cuda` main baseline: `/home/trentn/src/numba-cuda-main`
- cuda.coop repo: `/home/trentn/src/cccl/python/cuda_cccl`
  - see `SINGLE-PHASE-*.md` for context

## Current local state (numba-cuda)
Run:
- `git status -sb`
- `git diff`

Expected (uncommitted) changes in this worktree:
- `numba_cuda/numba/cuda/compiler.py`
  - CUDABackend sets `state.metadata["launch_config_sensitive"] = True`
    when the active launch config is explicitly marked.
- `numba_cuda/numba/cuda/dispatcher.py`
  - `_LaunchConfiguration` adds explicit API:
    `mark_kernel_as_launch_config_sensitive()`, `get_kernel_launch_config_sensitive()`,
    `is_kernel_launch_config_sensitive()`.
- `scripts/bench-launch-overhead.py`
  - import compatibility for `numba.cuda.core.config` vs `numba.core.config`.
- Untracked: `PR.md`, `tags` (clean up before commit).

## What is already implemented
- TLS-based launch-config capture in C extension, exposed via
  `numba_cuda/numba/cuda/launchconfig.py`.
- Dispatcher plumbing for LC-S (per-config specialization + cache keys + `.lcs` marker).
- Tests for LC-S recompile + cache coverage.
- Docs updated for launch-config introspection.
- In cccl: `cuda/coop/_rewrite.py` now marks LC-S when accessing launch config.
  It calls `mark_kernel_as_launch_config_sensitive()` when available, with
  fallback to `state.metadata["launch_config_sensitive"] = True`.

## Open decisions / tasks
1. **Explicit LC-S API decision: keep**
   - `_LaunchConfiguration` explicit LC-S API is retained.
   - Compiler hook in `CUDABackend` uses this API to set metadata.
   - cccl rewrite is updated to use the API when available.

2. **Run CUDA tests on a GPU**
   - `pixi run -e cu-12-9-py312 pytest testing --pyargs numba.cuda.tests.cudapy.test_launch_config_sensitive -k launch_config_sensitive`
   - `pixi run -e cu-12-9-py312 pytest testing --pyargs numba.cuda.tests.cudapy.test_caching -k launch_config_sensitive`
   - Status: both passing on GPU in this worktree.

3. **Validate disk-cache behavior across processes**
   - Ensure `.lcs` marker + launch-config cache keying behave correctly.
   - Status: covered by
     `LaunchConfigSensitiveCachingTest.test_launch_config_sensitive_cache_keys`
     in `test_caching.py` (passes, includes separate-process verification).

4. **Audit launch paths**
   - Confirm all kernel launch paths go through `CUDADispatcher.call()`.
   - Status: Python launch paths in `dispatcher.py` verified.

5. **Commit / cleanup**
   - Remove untracked `PR.md` and `tags`.
   - Prepare commit(s) for the launch-config work.

## Notes
- If you need to re-run the overhead micro-benchmark, see `LAUNCH-CONFIG.md`.
- Update `LAUNCH-CONFIG-TODO.md` with any new decisions or test results.
