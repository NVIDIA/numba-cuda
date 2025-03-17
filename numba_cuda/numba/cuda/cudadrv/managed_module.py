import weakref

from . import devices
from .driver import CtypesModule, USE_NV_BINDING


class _CuFuncProxy:
    def __init__(self, module, cufunc):
        self._module = module
        self._cufunc = cufunc

    def init_module(self, stream):
        self._module._init(stream)

    def lazy_finalize_module(self, stream):
        self._module._lazy_finalize(stream)

    def __getattr__(self, name):
        return getattr(self._cufunc, name)


class ManagedModule:
    def __init__(self, module, setup_callbacks, teardown_callbacks):
        # To be updated to object code
        if not isinstance(module, CtypesModule):
            raise TypeError("module must be a CtypesModule")

        if not isinstance(setup_callbacks, list):
            raise TypeError("setup_callbacks must be a list")
        if not isinstance(teardown_callbacks, list):
            raise TypeError("teardown_callbacks must be a list")

        for callback in setup_callbacks:
            if not callable(callback):
                raise TypeError("all callbacks must be callable")
        for callback in teardown_callbacks:
            if not callable(callback):
                raise TypeError("all callbacks must be callable")

        self._initialized = False
        self._module = module
        self._setup_callbacks = setup_callbacks
        self._teardown_callbacks = teardown_callbacks

    def _init(self, stream):
        for setup in self._setup_callbacks:
            setup(self._module, stream)

    def _lazy_finalize(self, stream):

        def lazy_callback(callbacks, module, stream):
            for teardown in callbacks:
                teardown(module, stream)

        ctx = devices.get_context()
        if USE_NV_BINDING:
            key = self._module.handle
        else:
            key = self._module.handle.value
        module_obj = ctx.modules.get(key, None)

        if module_obj is not None:
            weakref.finalize(
                module_obj,
                lazy_callback,
                self._teardown_callbacks,
                self._module,
                stream
            )

    def get_function(self, name):
        ctypesfunc = self._module.get_function(name)
        return _CuFuncProxy(self, ctypesfunc)

    def __getattr__(self, name):
        if name == "get_function":
            return getattr(self, "get_function")
        return getattr(self._module, name)
