# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import os
from numba import cuda
from numba.cuda.cudadrv.linkable_code import LinkableCode
from numba.cuda.testing import CUDATestCase

TEST_BIN_DIR = os.getenv("NUMBA_CUDA_TEST_BIN_DIR")
if TEST_BIN_DIR:
    test_device_functions_a = os.path.join(
        TEST_BIN_DIR, "test_device_functions.a"
    )
    test_device_functions_cubin = os.path.join(
        TEST_BIN_DIR, "test_device_functions.cubin"
    )
    test_device_functions_cu = os.path.join(
        TEST_BIN_DIR, "test_device_functions.cu"
    )
    test_device_functions_fatbin = os.path.join(
        TEST_BIN_DIR, "test_device_functions.fatbin"
    )
    test_device_functions_fatbin_multi = os.path.join(
        TEST_BIN_DIR, "test_device_functions_multi.fatbin"
    )
    test_device_functions_o = os.path.join(
        TEST_BIN_DIR, "test_device_functions.o"
    )
    test_device_functions_ptx = os.path.join(
        TEST_BIN_DIR, "test_device_functions.ptx"
    )
    test_device_functions_ltoir = os.path.join(
        TEST_BIN_DIR, "test_device_functions.ltoir"
    )


class TestLinkableCode(CUDATestCase):
    def test_linkable_code_from_path_or_obj(self):
        files_kind = [
            (test_device_functions_a, cuda.Archive),
            (test_device_functions_cubin, cuda.Cubin),
            (test_device_functions_cu, cuda.CUSource),
            (test_device_functions_fatbin, cuda.Fatbin),
            (test_device_functions_o, cuda.Object),
            (test_device_functions_ptx, cuda.PTXSource),
            (test_device_functions_ltoir, cuda.LTOIR),
        ]

        for path, kind in files_kind:
            obj = LinkableCode.from_path_or_obj(path)
            assert isinstance(obj, kind)

            # test identity of from_path_or_obj
            obj2 = LinkableCode.from_path_or_obj(obj)
            assert obj2 is obj
