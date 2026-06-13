# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

from concurrent.futures import ThreadPoolExecutor
import sysconfig
import threading
import unittest

import numpy as np

from numba.cuda.cext import _dispatcher


def sysconfig_var_is_true(name):
    value = sysconfig.get_config_var(name)
    return value is True or str(value).strip() == "1"


class TestTypeofThreading(unittest.TestCase):
    @unittest.skipUnless(
        sysconfig_var_is_true("Py_GIL_DISABLED"),
        "requires a free-threaded Python build",
    )
    def test_compute_fingerprint_with_mutating_containers(self):
        stop = threading.Event()
        list_value = [np.arange(4, dtype=np.int32)]
        set_value = {1}
        structured = np.zeros(1, dtype=[("a", np.int32), ("b", np.float64)])

        def mutate_list():
            while not stop.is_set():
                list_value[:] = [np.arange(4, dtype=np.int32)]
                list_value.append(np.arange(2, dtype=np.float64))
                list_value.pop()
                list_value.clear()
                list_value.append(np.arange(1, dtype=np.int16))

        def mutate_set():
            i = 0
            while not stop.is_set():
                set_value.clear()
                set_value.add(i)
                i += 1

        def fingerprint_values():
            benign_empty_container_races = 0
            values = (
                list_value,
                set_value,
                structured,
                structured[0],
                (1, 2.0, np.float32(3)),
            )
            for _ in range(1000):
                for value in values:
                    try:
                        fingerprint = _dispatcher.compute_fingerprint(value)
                        self.assertIsInstance(fingerprint, bytes)
                    except ValueError as e:
                        message = str(e)
                        if "empty list" in message or "empty set" in message:
                            benign_empty_container_races += 1
                        else:
                            raise
            return benign_empty_container_races

        mutators = [
            threading.Thread(target=mutate_list),
            threading.Thread(target=mutate_set),
        ]
        for mutator in mutators:
            mutator.start()

        try:
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [
                    executor.submit(fingerprint_values) for _ in range(8)
                ]
                for future in futures:
                    future.result()
        finally:
            stop.set()
            for mutator in mutators:
                mutator.join(timeout=5)

        self.assertFalse(any(mutator.is_alive() for mutator in mutators))


if __name__ == "__main__":
    unittest.main()
