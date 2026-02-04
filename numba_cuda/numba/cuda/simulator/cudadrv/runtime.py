# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""
The runtime API is unsupported in the simulator, but some stubs are
provided to allow tests to import correctly.
"""


class FakeRuntime:
    def get_version(self):
        return (-1, -1)

    def is_supported_version(self):
        return True

    @property
    def supported_versions(self):
        return ((-1, -1),)


runtime = FakeRuntime()
