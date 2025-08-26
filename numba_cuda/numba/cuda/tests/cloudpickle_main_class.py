# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# Expected to run this module as __main__


# Cloudpickle will think this is a dynamic class when this module is __main__
class Klass:
    classvar = None
