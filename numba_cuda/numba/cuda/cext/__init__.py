# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import sys
import importlib
import importlib.util
import importlib.machinery
from pathlib import Path
from types import ModuleType
from importlib.machinery import ModuleSpec


def _load_ext_from_spec(
    spec: ModuleSpec, fullname: str, legacy_name: str
) -> ModuleType:
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[fullname] = module
    sys.modules[legacy_name] = (
        module  # Register under legacy name for C extensions
    )

    # Ensure parent modules exist for legacy name (e.g., numba_cuda for numba_cuda._devicearray)
    parts = legacy_name.split(".")
    for i in range(1, len(parts)):
        parent_name = ".".join(parts[:i])
        if parent_name not in sys.modules:
            parent_module = ModuleType(parent_name)
            sys.modules[parent_name] = parent_module

        # Set the child as an attribute of the parent
        parent_module = sys.modules[parent_name]
        child_name = parts[i]
        if i == len(parts) - 1:  # This is the final module
            setattr(parent_module, child_name, module)
        elif not hasattr(parent_module, child_name):
            # Create intermediate module if it doesn't exist
            intermediate_name = ".".join(parts[: i + 1])
            if intermediate_name not in sys.modules:
                intermediate_module = ModuleType(intermediate_name)
                sys.modules[intermediate_name] = intermediate_module
                setattr(parent_module, child_name, intermediate_module)

    spec.loader.exec_module(module)
    return module


def _find_in_dir(
    module_name: str, directory: Path | str | None
) -> ModuleSpec | None:
    if not directory:
        return None
    return importlib.machinery.PathFinder.find_spec(
        module_name, [str(directory)]
    )


def _load_cext_module(
    module_basename: str, required: bool = True
) -> ModuleType | None:
    fullname = f"numba.cuda.cext.{module_basename}"
    legacy_name = f"numba_cuda.{module_basename}"

    # 1) Try local numba_cuda directory (for development builds)
    local_numba_cuda = Path(__file__).parents[
        3
    ]  # Go up from cext/ to numba_cuda/
    spec = _find_in_dir(module_basename, local_numba_cuda)

    # 2) Fallback: scan sys.path for installed numba_cuda directory
    if spec is None:
        for entry in sys.path:
            numba_cuda_dir = Path(entry) / "numba_cuda/numba/cuda/cext"
            spec = _find_in_dir(module_basename, numba_cuda_dir)
            if spec is not None:
                break

    if spec is None:
        if required:
            raise ModuleNotFoundError(
                f"Could not find '{module_basename}' in numba_cuda directories"
            )
        return None

    return _load_ext_from_spec(spec, fullname, legacy_name)


# Load known cext modules (all required)
# Load _devicearray first since _dispatcher depends on it
_devicearray = _load_cext_module("_devicearray", required=True)
_dispatcher = _load_cext_module("_dispatcher", required=True)
mviewbuf = _load_cext_module("mviewbuf", required=True)

__all__ = ["mviewbuf", "_dispatcher", "_devicearray"]
