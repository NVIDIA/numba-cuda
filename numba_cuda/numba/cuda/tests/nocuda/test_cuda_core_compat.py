# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

from unittest.mock import patch

from numba.cuda._compat import (
    cuda_core_attr_value,
    make_cuda_core_launch_config,
)
from numba.cuda.testing import unittest


class TestCudaCoreCompat(unittest.TestCase):
    def test_cuda_core_attr_value_from_property(self):
        class Attrs:
            num_regs = 32

        self.assertEqual(cuda_core_attr_value(Attrs(), "num_regs"), 32)

    def test_cuda_core_attr_value_from_accessor(self):
        class Attrs:
            def num_regs(self):
                return 32

        self.assertEqual(cuda_core_attr_value(Attrs(), "num_regs"), 32)

    def test_make_cuda_core_launch_config_current_keyword(self):
        config = make_cuda_core_launch_config(
            grid=(1, 1, 1),
            block=(1, 1, 1),
            shmem_size=0,
            is_cooperative=False,
        )

        self.assertFalse(config.is_cooperative)

    def test_make_cuda_core_launch_config_legacy_keyword(self):
        class LegacyLaunchConfig:
            def __init__(self, **kwargs):
                if "is_cooperative" in kwargs:
                    raise TypeError(
                        "__init__() got an unexpected keyword argument "
                        "'is_cooperative'"
                    )
                self.kwargs = kwargs

        with patch("numba.cuda._compat.core.LaunchConfig", LegacyLaunchConfig):
            config = make_cuda_core_launch_config(
                grid=(1, 1, 1),
                block=(1, 1, 1),
                shmem_size=0,
                is_cooperative=False,
            )

        self.assertNotIn("is_cooperative", config.kwargs)
        self.assertFalse(config.kwargs["cooperative_launch"])
