from .mappings import FILE_EXTENSION_MAP

class LinkableCode:
    """An object that can be passed in the `link` list argument to `@cuda.jit`
    kernels to supply code to be linked from memory."""

    def __init__(self, data, name=None):
        self.data = data
        self._name = name

    @property
    def name(self):
        return self._name or self.default_name


class PTXSource(LinkableCode):
    """PTX Source code in memory"""

    kind = FILE_EXTENSION_MAP["ptx"]
    default_name = "<unnamed-ptx>"


class CUSource(LinkableCode):
    """CUDA C/C++ Source code in memory"""

    kind = "cu"
    default_name = "<unnamed-cu>"


class Fatbin(LinkableCode):
    """A fatbin ELF in memory"""

    kind = FILE_EXTENSION_MAP["fatbin"]
    default_name = "<unnamed-fatbin>"


class Cubin(LinkableCode):
    """A cubin ELF in memory"""

    kind = FILE_EXTENSION_MAP["cubin"]
    default_name = "<unnamed-cubin>"


class Archive(LinkableCode):
    """An archive of objects in memory"""

    kind = FILE_EXTENSION_MAP["a"]
    default_name = "<unnamed-archive>"


class Object(LinkableCode):
    """An object file in memory"""

    kind = FILE_EXTENSION_MAP["o"]
    default_name = "<unnamed-object>"


class LTOIR(LinkableCode):
    """An LTOIR file in memory"""

    kind = "ltoir"
    default_name = "<unnamed-ltoir>"
