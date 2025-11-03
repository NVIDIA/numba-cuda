# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

try:
    from pygments.styles.default import DefaultStyle
except ImportError:
    msg = "Please install pygments to see highlighted dumps"
    raise ImportError(msg)

import numba.cuda.config
from pygments.styles.manni import ManniStyle
from pygments.styles.monokai import MonokaiStyle
from pygments.styles.native import NativeStyle

from pygments.token import Name

from pygments.style import Style


def by_colorscheme():
    """
    Get appropriate style for highlighting according to
    NUMBA_COLOR_SCHEME setting
    """
    styles = DefaultStyle.styles.copy()
    styles.update(
        {
            Name.Variable: "#888888",
        }
    )
    custom_default = type("CustomDefaultStyle", (Style,), {"styles": styles})

    style_map = {
        "no_color": custom_default,
        "dark_bg": MonokaiStyle,
        "light_bg": ManniStyle,
        "blue_bg": NativeStyle,
        "jupyter_nb": DefaultStyle,
    }

    return style_map[numba.cuda.config.COLOR_SCHEME]
