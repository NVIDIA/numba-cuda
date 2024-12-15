import os
import warnings
import traceback


def _readenv(name, ctor, default):
    value = os.environ.get(name)
    if value is None:
        return default() if callable(default) else default
    try:
        if ctor is bool:
            return value.lower() in {'1', "true"}
        return ctor(value)
    except Exception:
        warnings.warn(
            f"Environment variable '{name}' is defined but its associated "
            f"value '{value}' could not be parsed.\n"
            "The parse failed with exception:\n"
            f"{traceback.format_exc()}",
            RuntimeWarning
        )
        return default
