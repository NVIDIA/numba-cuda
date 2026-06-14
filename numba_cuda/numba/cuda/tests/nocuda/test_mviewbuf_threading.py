# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

from concurrent.futures import ThreadPoolExecutor
import unittest

import numpy as np

from numba.cuda.cext import mviewbuf


class TestMViewBufThreading(unittest.TestCase):
    def test_memoryview_get_extents_info_bad_inputs_raise_cleanly(self):
        for args in (
            ((3,), (4,), 2, 4),
            ((3, 4), (16,), 2, 4),
            ((3,), (4,), 1, 0),
            ((3,), (4,), 1, -1),
        ):
            with self.assertRaises((ValueError, OverflowError, TypeError)):
                mviewbuf.memoryview_get_extents_info(*args)

    def test_concurrent_memoryview_extent_helpers(self):
        def check_extents_info(n):
            for k in range(n):
                ndim = k % 4
                shape = tuple(range(1, ndim + 1)) if ndim else ()
                strides = tuple(4 for _ in range(ndim))
                start, end = mviewbuf.memoryview_get_extents_info(
                    shape, strides, ndim, 4
                )
                self.assertGreaterEqual(end, start)

                start, end = mviewbuf.memoryview_get_extents_info(
                    (3, 4), (16, 4), 2, 4
                )
                self.assertEqual(end - start, 48)

        def check_extents(n):
            for k in range(n):
                ary = np.arange((k % 8) + 1, dtype=np.float64)
                start, end = mviewbuf.memoryview_get_extents(ary)
                self.assertGreaterEqual(end, start)
                self.assertEqual(end - start, ary.nbytes)

        def check_bad_inputs(n):
            for _ in range(n):
                self.test_memoryview_get_extents_info_bad_inputs_raise_cleanly()

        workers = (
            check_extents_info,
            check_extents,
            check_bad_inputs,
            check_extents_info,
            check_extents,
            check_bad_inputs,
        )
        with ThreadPoolExecutor(max_workers=len(workers)) as executor:
            futures = [executor.submit(worker, 500) for worker in workers]
            for future in futures:
                future.result()


if __name__ == "__main__":
    unittest.main()
