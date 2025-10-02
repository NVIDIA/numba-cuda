# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import io
import os
from typing import Union, Type

from numba.cuda.utils import cached_file_read

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
        debug=None,
        lineinfo=None,
        opt=None,
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

        self.debug = debug
        self.lineinfo = lineinfo
        self.opt = opt

    @property
    def name(self):
        return self._name or self.default_name

    @property
    def data(self):
        if isinstance(self._data, (io.StringIO, io.BytesIO)):
            return self._data.getvalue()
        return self._data

    @staticmethod
    def from_path(path: str, debug=None, lineinfo=None, opt=None):
        """
        Load a linkable code object from a file.

        Parameters
        ----------
        path : str
            The path to the file to load.

        Returns
        -------
        LinkableCode
            The linkable code object.

        Raises
        ------
        ValueError
            If the file extension is not supported.
        """
        root, extension = os.path.splitext(path)
        basename = os.path.basename(root)
        if extension in (".cu", ".ptx"):
            mode = "r"
        else:
            mode = "rb"

        data = cached_file_read(path, how=mode)

        cls = _extension_to_linkable_code_kind(extension)
        return cls(data, name=basename, debug=debug, lineinfo=lineinfo, opt=opt)

    @classmethod
    def from_path_or_obj(
        cls,
        path_or_obj: Union[str, "LinkableCode"],
        debug=None,
        lineinfo=None,
        opt=None,
    ):
        """
        Load a linkable code object from a file or a LinkableCode object.

        If a path is provided, the file is loaded and the LinkableCode object
        is returned. If a LinkableCode object is provided, it is returned as is.

        Parameters
        ----------
        path_or_obj : str or LinkableCode
            The path to the file or the LinkableCode object to load.

        Returns
        -------
        LinkableCode
            The linkable code object.

        Raises
        ------
        ValueError
            If the file extension is not supported.
        """
        if isinstance(path_or_obj, str):
            return cls.from_path(path_or_obj)

        if path_or_obj.debug is None:
            path_or_obj.debug = debug
        if path_or_obj.lineinfo is None:
            path_or_obj.lineinfo = lineinfo
        if path_or_obj.opt is None:
            path_or_obj.opt = opt

        return path_or_obj


class NonCompilableCode(LinkableCode):
    """A non-compilable code object."""

    def __init__(
        self,
        data,
        name=None,
        setup_callback=None,
        teardown_callback=None,
        nrt=False,
        debug=None,
        lineinfo=None,
        opt=None,
    ):
        if debug:
            raise ValueError(
                f"debug=True is not supported for {self.__class__.__name__} code"
            )
        if lineinfo:
            raise ValueError(
                f"lineinfo=True is not supported for {self.__class__.__name__} code"
            )
        if opt:
            raise ValueError(
                f"opt=True is not supported for {self.__class__.__name__} code"
            )
        super().__init__(
            data,
            name=name,
            setup_callback=setup_callback,
            teardown_callback=teardown_callback,
            nrt=nrt,
            debug=debug,
            lineinfo=lineinfo,
            opt=opt,
        )


class CompilableCode(LinkableCode):
    """A compilable code object.

    Only CUSource is a compilable code object.
    """

    def __init__(
        self,
        data,
        name=None,
        setup_callback=None,
        teardown_callback=None,
        nrt=False,
        debug=None,
        lineinfo=None,
        opt=None,
    ):
        super().__init__(
            data,
            name=name,
            setup_callback=setup_callback,
            teardown_callback=teardown_callback,
            nrt=nrt,
            debug=debug,
            lineinfo=lineinfo,
            opt=opt,
        )


class PTXSource(NonCompilableCode):
    """PTX source code in memory."""

    kind = FILE_EXTENSION_MAP["ptx"]
    default_name = "<unnamed-ptx>"


class CUSource(CompilableCode):
    """CUDA C/C++ source code in memory."""

    kind = "cu"
    default_name = "<unnamed-cu>"


class Fatbin(NonCompilableCode):
    """An ELF Fatbin in memory."""

    kind = FILE_EXTENSION_MAP["fatbin"]
    default_name = "<unnamed-fatbin>"


class Cubin(NonCompilableCode):
    """An ELF Cubin in memory."""

    kind = FILE_EXTENSION_MAP["cubin"]
    default_name = "<unnamed-cubin>"


class Archive(NonCompilableCode):
    """An archive of objects in memory."""

    kind = FILE_EXTENSION_MAP["a"]
    default_name = "<unnamed-archive>"


class Object(NonCompilableCode):
    """An object file in memory."""

    kind = FILE_EXTENSION_MAP["o"]
    default_name = "<unnamed-object>"


class LTOIR(NonCompilableCode):
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
