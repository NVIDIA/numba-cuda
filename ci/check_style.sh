#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION

set -euo pipefail

pip install pre-commit

# Run pre-commit checks
pre-commit run --all-files --show-diff-on-failure
