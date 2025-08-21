# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause


class LinkableCode:
    """An object that holds code to be linked from memory.

    :param data: A buffer containing the data to link.
    :param name: The name of the file to be referenced in any compilation or
                 linking errors that may be produced.
    """

    def __init__(self, data, name=None):
        self.data = data
        self._name = name

    @property
    def name(self):
        return self._name or self.default_name


class PTXSource(LinkableCode):
    """PTX source code in memory."""

    default_name = "<unnamed-ptx>"


class CUSource(LinkableCode):
    """CUDA C/C++ source code in memory."""

    default_name = "<unnamed-cu>"


class Fatbin(LinkableCode):
    """An ELF Fatbin in memory."""

    default_name = "<unnamed-fatbin>"


class Cubin(LinkableCode):
    """An ELF Cubin in memory."""

    default_name = "<unnamed-cubin>"


class Archive(LinkableCode):
    """An archive of objects in memory."""

    default_name = "<unnamed-archive>"


class Object(LinkableCode):
    """An object file in memory."""

    default_name = "<unnamed-object>"


class LTOIR(LinkableCode):
    """An LTOIR file in memory."""

    default_name = "<unnamed-ltoir>"
