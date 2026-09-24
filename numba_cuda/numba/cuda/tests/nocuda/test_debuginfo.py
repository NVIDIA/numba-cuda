# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

from unittest import mock

from numba.cuda import debuginfo


def test_runtime_version_unavailable_disables_polymorphic_debug_info():
    with mock.patch.object(
        debuginfo.runtime,
        "getLocalRuntimeVersion",
        side_effect=NotImplementedError,
    ):
        assert (
            debuginfo._check_polymorphic_debug_info_support()
            == (False, False)
        )
