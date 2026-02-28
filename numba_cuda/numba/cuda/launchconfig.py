from __future__ import annotations

from dataclasses import dataclass
from contextvars import ContextVar
from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    List,
    Tuple,
    Optional,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from numba.cuda.dispatcher import CUDADispatcher, _Kernel


@dataclass(frozen=True, slots=True)
class LaunchConfig:
    """
    Helper class used to encapsulate kernel launch configuration for storing
    and retrieving from a thread-local ContextVar.
    """

    dispatcher: "CUDADispatcher"
    args: Tuple[Any, ...]
    griddim: Tuple[int, int, int]
    blockdim: Tuple[int, int, int]
    stream: Any
    sharedmem: int
    pre_launch_callbacks: List[Callable[["_Kernel", "LaunchConfig"], None]]
    """
    List of functions to call before launching a kernel.  The functions are
    called with the kernel and the launch config as arguments.  This enables
    just-in-time modifications to the kernel's configuration prior to launch,
    such as appending extensions for dynamic types that were created after the
    @cuda.jit decorator appeared (i.e. as part of rewriting).
    """

    def __str__(self) -> str:
        a = ", ".join(map(str, self.args))
        g = "×".join(map(str, self.griddim))
        b = "×".join(map(str, self.blockdim))
        cb = ", ".join(map(str, self.pre_launch_callbacks))
        return (
            f"<LaunchConfig args=[{a}], grid={g}, block={b}, "
            f"stream={self.stream}, smem={self.sharedmem}B, "
            f"pre_launch_callbacks=[{cb}]>"
        )


_launch_config_var: ContextVar[Optional[LaunchConfig]] = ContextVar(
    "_launch_config_var",
    default=None,
)


def current_launch_config() -> Optional[LaunchConfig]:
    """
    Read the launch config visible in *this* thread/asyncio task.
    Returns None if no launch config is set.
    """
    return _launch_config_var.get()


def ensure_current_launch_config() -> LaunchConfig:
    """
    Ensure that a launch config is set for *this* thread/asyncio task.
    Returns the launch config.  Raises RuntimeError if no launch config is set.
    """
    launch_config = current_launch_config()
    if launch_config is None:
        raise RuntimeError("No launch config set for this thread/asyncio task")
    return launch_config


@contextmanager
def launch_config_ctx(
    *,
    dispatcher: "CUDADispatcher",
    args: Tuple[Any, ...],
    griddim: Tuple[int, int, int],
    blockdim: Tuple[int, int, int],
    stream: Any,
    sharedmem: int,
):
    """
    Install a LaunchConfig for the dynamic extent of the with-block.
    The previous value (if any) is restored automatically.
    """
    pre_launch_callbacks = []
    launch_config = LaunchConfig(
        dispatcher,
        args,
        griddim,
        blockdim,
        stream,
        sharedmem,
        pre_launch_callbacks,
    )
    token = _launch_config_var.set(launch_config)
    try:
        yield launch_config
    finally:
        _launch_config_var.reset(token)
