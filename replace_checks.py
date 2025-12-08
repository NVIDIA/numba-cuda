# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import re
import pathlib
import sys

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <file to rewrite>")
    sys.exit(1)

path = pathlib.Path(sys.argv[1])
text = path.read_text()

pattern = re.compile(r"(isinstance\([^,]+,\s*)ir\.([A-Z][A-Za-z0-9_]+)")


def repl(m):
    head = m.group(1)
    name = m.group(2).lower()
    return f"{head}ir.{name}_types"


new = pattern.sub(repl, text)
path.write_text(new)
