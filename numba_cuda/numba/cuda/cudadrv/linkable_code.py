from .mappings import FILE_EXTENSION_MAP


class LinkableCode:
    """An object that holds code to be linked from memory.

    :param data: A buffer containing the data to link.
    :param name: The name of the file to be referenced in any compilation or
                 linking errors that may be produced.
    """

    def __init__(self, data, name=None, nrt=False):
        self.data = data
        self.nrt = nrt
        self._name = name

    @property
    def name(self):
        return self._name or self.default_name


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

    kind = "ltoir"
    default_name = "<unnamed-ltoir>"
