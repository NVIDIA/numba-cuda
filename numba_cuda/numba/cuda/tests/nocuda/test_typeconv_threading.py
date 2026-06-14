# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

from concurrent.futures import ThreadPoolExecutor
import unittest

from numba.cuda import types
from numba.cuda.typeconv import Conversion
from numba.cuda.typeconv.typeconv import TypeManager


class TestTypeConvThreading(unittest.TestCase):
    def test_concurrent_reads_and_writes(self):
        tm = TypeManager()
        i16 = types.int16
        i32 = types.int32
        i64 = types.int64
        f32 = types.float32
        f64 = types.float64

        tm.set_promote(i32, i64)
        tm.set_unsafe_convert(i32, f32)
        tm.set_promote(i16, i32)
        tm.set_safe_convert(f32, f64)

        sig = (i32, f32)
        overloads = (
            (i32, i32),
            (f32, f32),
            (i64, i64),
            (i16, i16),
        )

        def write_conversions():
            for _ in range(300):
                tm.set_promote(i32, i64)
                tm.set_unsafe_convert(i32, f32)
                tm.set_promote(i16, i32)
                tm.set_safe_convert(f32, f64)

        def read_conversions():
            for _ in range(300):
                self.assertEqual(
                    tm.check_compatible(i32, i64), Conversion.promote
                )
                self.assertEqual(
                    tm.check_compatible(i32, f32), Conversion.unsafe
                )
                self.assertEqual(
                    tm.check_compatible(i16, i32), Conversion.promote
                )
                self.assertEqual(tm.check_compatible(f32, f64), Conversion.safe)
                self.assertEqual(
                    tm.select_overload(sig, overloads, True, False), 1
                )

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(write_conversions) for _ in range(4)] + [
                executor.submit(read_conversions) for _ in range(4)
            ]
            for future in futures:
                future.result()


if __name__ == "__main__":
    unittest.main()
