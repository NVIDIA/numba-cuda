# Green Context Support in `numba-cuda`

This note summarizes the current state of green-context support in this repository, the implemented scope, and the remaining limitations.

## Current Status

`numba-cuda` now supports a limited, interop-first form of green-context support.

The supported model is:

- external code creates a CUDA green context,
- external code converts it to a `CUcontext` with `cuCtxFromGreenCtx()`,
- external code makes that context current,
- Numba borrows the active context and operates inside it.

This is intentionally narrower than full first-class green-context support.

## What Works In Phase 1

When a green-context-derived `CUcontext` is already active:

- `cuda.current_context()` can return a Numba context wrapper for it,
- `@cuda.require_context` APIs can use it,
- `cuda.device_array()` works in the active green context,
- `@cuda.jit` kernels can be loaded and launched in that context,
- `cuda.external_stream()` can wrap a stream created for that green context,
- CUDA Array Interface import and stream synchronization work in that context.

The implementation also keeps loaded CUDA functions context-aware so that a kernel loaded in one execution context is not reused incorrectly in another execution context on the same device.

## What Is Still Rejected

This change does not make all non-primary contexts valid.

The following is still intentionally rejected:

- ordinary non-primary contexts created with APIs such as `cuCtxCreate()`,
- any non-primary context that is not recognized as a green context.

The historical error:

```text
RuntimeError: Numba cannot operate on non-primary CUDA context
```

still applies to those unsupported contexts.

## Ownership Model

Green-context-derived contexts are treated as borrowed, externally managed contexts.

In practice this means:

- Numba can use the active green context,
- Numba does not claim ownership of creating or destroying that context,
- Numba does not store it as the device primary context,
- destructive subsystem reset is blocked while borrowed contexts are still live.

This prevents `cuda.close()` or `devices.reset()` from accidentally resetting or releasing state that was not created and owned by Numba.

## Context And Cache Behavior

The implementation now distinguishes between:

- the device primary context, and
- borrowed green-context-derived execution contexts.

Loaded CUDA function handles are cached by execution-context identity instead of only by `device.id`.

Context reset also advances a context-generation key so that handles tied to unloaded modules are not reused after `Context.reset()`.

The cached cubin path also recreates a fresh `ObjectCode` wrapper per load so that unloading one module does not leave later loads with a stale object handle.

## Explicit Phase 1 Limits

The following are still out of scope:

- public APIs to create green contexts from Numba,
- public APIs to select green contexts through `cuda.gpus[...]`,
- changing `cuda.select_device()` to choose green contexts,
- broad support for arbitrary multi-context-per-device workflows,
- multithreaded use of the same green context,
- treating all non-primary contexts as equivalent to green contexts.

`cuda.gpus[...]` and `cuda.select_device()` remain primary-context APIs.

## Practical Constraints

The current design is still mostly device-centric:

- each device still has one retained primary context managed by Numba,
- borrowed green contexts are attached by handle only when already active,
- reset and shutdown semantics are conservative whenever external ownership is involved.

This keeps the implementation compatible with the existing primary-context model while adding a narrow green-context interop path.

## Remaining Work For Broader Support

Full green-context support would still require additional work, including:

1. Public APIs for creating and managing green contexts.
2. A clearer execution-context abstraction across more subsystems.
3. Broader auditing of context-sensitive caches and resource ownership.
4. Defined behavior for multiple execution contexts on one device beyond the interop path.
5. Clear multithreading rules for borrowed green contexts.

## Bottom Line

Green-context interop is now supported when the green-context-derived `CUcontext` is created and activated externally first.

Full first-class green-context management is not implemented yet, and non-green non-primary contexts remain unsupported.
