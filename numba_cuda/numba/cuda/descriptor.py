# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

from numba.cuda.core.options import TargetOptions
from .target import CUDATargetContext, CUDATypingContext


class CUDATargetOptions(TargetOptions):
    pass


class CUDATarget:
    def __init__(self, name):
        self.options = CUDATargetOptions
        # The typing and target contexts are initialized only when needed -
        # this prevents an attempt to load CUDA libraries at import time on
        # systems that might not have them present.
        self._typingctx = None
        self._targetctx = None
        self._target_name = name

    @property
    def typing_context(self):
        if self._typingctx is None:
            self._typingctx = CUDATypingContext()
        return self._typingctx

    @property
    def target_context(self):
        if self._targetctx is None:
            self._targetctx = CUDATargetContext(self._typingctx)
        return self._targetctx


cuda_target = CUDATarget("cuda")

# Monkey-patch numba's get_local_target and order_by_target_specificity for CUDATarget
try:
    from numba.core import target_extension
    from numba.cuda.utils import order_by_target_specificity
    from numba.core import utils as numba_utils

    def _is_cuda_context(obj):
        return (
            isinstance(obj, CUDATarget)
            or (hasattr(obj, "__class__") and "CUDA" in obj.__class__.__name__)
            or (hasattr(obj, "target") and isinstance(obj.target, CUDATarget))
        )

    def _patch_numba_for_cuda_target():
        _orig_get_local = target_extension.get_local_target

        def get_local_target_cuda(context):
            return (
                cuda_target
                if _is_cuda_context(context)
                else _orig_get_local(context)
            )

        target_extension.get_local_target = get_local_target_cuda
        numba_utils.order_by_target_specificity = order_by_target_specificity

    _patch_numba_for_cuda_target()

except ImportError:
    pass
