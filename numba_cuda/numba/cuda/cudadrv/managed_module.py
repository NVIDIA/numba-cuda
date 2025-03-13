from .driver import CtypesModule


class ManagedModule:
    def __init__(self, module, setup_callbacks, teardown_callbacks):
        # To be updated to object code
        if not isinstance(module, CtypesModule):
            raise TypeError("module must be a CtypesModule")

        if not isinstance(setup_callbacks, list):
            raise TypeError("setup_callbacks must be a list")
        if not isinstance(teardown_callbacks, list):
            raise TypeError("teardown_callbacks must be a list")

        self._module = module
        self._setup_callbacks = setup_callbacks
        self._teardown_callbacks = teardown_callbacks

        for initialize in self._setup_callbacks:
            if not callable(initialize):
                raise TypeError("setup_callbacks must be callable")
            initialize(self._module)

    def __del__(self):
        for teardown in self._teardown_callbacks:
            if not callable(teardown):
                teardown(self._module)

    def __getattr__(self, name):
        return getattr(self._module, name)
