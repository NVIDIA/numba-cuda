# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import io
import os
from typing import Union, Type

from .mappings import FILE_EXTENSION_MAP


class LinkableCode:
    """An object that holds code to be linked from memory.

    :param data: A buffer, StringIO or BytesIO containing the data to link.
                 If a file object is passed, the content in the object is
                 read when `data` property is accessed.
    :param name: The name of the file to be referenced in any compilation or
                 linking errors that may be produced.
    :param setup_callback: A function called prior to the launch of a kernel
                           contained within a module that has this code object
                           linked into it.
    :param teardown_callback: A function called just prior to the unloading of
                              a module that has this code object linked into
                              it.
    :param nrt: If True, assume this object contains NRT function calls and
                add NRT source code to the final link.
    """

    def __init__(
        self,
        data,
        name=None,
        setup_callback=None,
        teardown_callback=None,
        nrt=False,
    ):
        if setup_callback and not callable(setup_callback):
            raise TypeError("setup_callback must be callable")
        if teardown_callback and not callable(teardown_callback):
            raise TypeError("teardown_callback must be callable")

        self.nrt = nrt
        self._name = name
        self._data = data
        self.setup_callback = setup_callback
        self.teardown_callback = teardown_callback

    @property
    def name(self):
        return self._name or self.default_name

    @property
    def data(self):
        if isinstance(self._data, (io.StringIO, io.BytesIO)):
            return self._data.getvalue()
        return self._data

    @staticmethod
    def from_path(path: str):
        root, extension = os.path.splitext(path)
        basename = os.path.basename(root)
        if extension in (".cu", ".ptx"):
            mode = "r"
        else:
            mode = "rb"

        with open(path, mode) as f:
            data = f.read()

        cls = _extension_to_linkable_code_kind(extension)
        return cls(data, name=basename)

    @classmethod
    def from_path_or_obj(cls, path_or_obj: Union[str, "LinkableCode"]):
        if isinstance(path_or_obj, str):
            return cls.from_path(path_or_obj)
        return path_or_obj


class PTXSource(LinkableCode):
    """PTX source code in memory."""

    kind = FILE_EXTENSION_MAP["ptx"]
    default_name = "<unnamed-ptx>"


class CUSource(LinkableCode):
    """CUDA C/C++ source code in memory."""

    kind = "cu"
    default_name = "<unnamed-cu>"


class Fatbin(LinkableCode):
    """An ELF Fatbin in memory."""

    kind = FILE_EXTENSION_MAP["fatbin"]
    default_name = "<unnamed-fatbin>"


class Cubin(LinkableCode):
    """An ELF Cubin in memory."""

    kind = FILE_EXTENSION_MAP["cubin"]
    default_name = "<unnamed-cubin>"


class Archive(LinkableCode):
    """An archive of objects in memory."""

    kind = FILE_EXTENSION_MAP["a"]
    default_name = "<unnamed-archive>"


class Object(LinkableCode):
    """An object file in memory."""

    kind = FILE_EXTENSION_MAP["o"]
    default_name = "<unnamed-object>"


class LTOIR(LinkableCode):
    """An LTOIR file in memory."""

    kind = FILE_EXTENSION_MAP["ltoir"]
    default_name = "<unnamed-ltoir>"


def _extension_to_linkable_code_kind(extension: str) -> Type[LinkableCode]:
    if extension == ".cu":
        return CUSource
    elif extension == ".ptx":
        return PTXSource
    elif extension == ".fatbin":
        return Fatbin
    elif extension == ".cubin":
        return Cubin
    elif extension == ".a":
        return Archive
    elif extension == ".o":
        return Object
    elif extension == ".ltoir":
        return LTOIR
    else:
        raise ValueError(f"Unknown extension: {extension}")
