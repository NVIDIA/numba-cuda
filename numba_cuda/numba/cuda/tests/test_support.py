# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import pytest

from .support import test_id_generator


class TestMonotonicallyIncreasingID:
    def test_id_starts_zero(self, test_id):
        assert test_id == 0

    def test_id_increments(self, test_id):
        assert test_id == 1


class TestIDResetsInSeparateClass:
    def test_id_starts_zero_again(self, test_id):
        assert test_id == 0

    def test_id_increments_again(self, test_id):
        assert test_id == 1
