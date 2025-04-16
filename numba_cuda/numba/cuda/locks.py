from threading import Lock
from functools import wraps

# Thread safety guard for module initialization.
_module_init_lock = Lock()


def module_init_lock(func):
    """Decorator to make sure initialization is invoked once for all threads."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        with _module_init_lock:
            return func(*args, **kwargs)

    return wrapper
