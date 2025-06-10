from __future__ import annotations

from dataclasses import dataclass
from contextvars import ContextVar
from contextlib import contextmanager
from typing import Any, Tuple, Optional


@dataclass(frozen=True, slots=True)
class LaunchConfig:
    """
    Helper class used to encapsulate kernel launch configuration for storing
    and retrieving from a thread-local ContextVar.
    """

    griddim: Tuple[int, int, int]
    blockdim: Tuple[int, int, int]
    stream: Any
    sharedmem: int

    def __str__(self) -> str:
        g = "×".join(map(str, self.griddim))
        b = "×".join(map(str, self.blockdim))
        return (
            f"<LaunchConfig grid={g}, block={b}, "
            f"stream={self.stream}, smem={self.sharedmem}B>"
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


@contextmanager
def launch_config_ctx(
    *,
    griddim: Tuple[int, int, int],
    blockdim: Tuple[int, int, int],
    stream: Any,
    sharedmem: int,
):
    """
    Install a LaunchConfig for the dynamic extent of the with-block.
    The previous value (if any) is restored automatically.
    """
    token = _launch_config_var.set(
        LaunchConfig(griddim, blockdim, stream, sharedmem)
    )
    try:
        yield
    finally:
        _launch_config_var.reset(token)
