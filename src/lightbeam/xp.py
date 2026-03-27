"""
Numerical backend selector for lightbeam.

Set the environment variable  LIGHTBEAM_BACKEND=cupy  to use CuPy (GPU).
Any other value (or absence of the variable) defaults to NumPy (CPU).

All other modules should do:

    from lightbeam.xp import xp

and then use  xp.array(...)  etc. everywhere instead of  np.array(...).
"""

import os

_backend = os.environ.get("LIGHTBEAM_BACKEND", "numpy").strip().lower()

if _backend == "cupy":
    try:
        import cupy as xp
        _GPU = True
    except ImportError as e:
        import warnings
        warnings.warn(
            "LIGHTBEAM_BACKEND=cupy requested but CuPy is not installed; "
            "falling back to NumPy. Original error: " + str(e)
        )
        import numpy as xp
        _GPU = False
else:
    import numpy as xp
    _GPU = False


def is_gpu() -> bool:
    """Return True when the active backend is CuPy (GPU)."""
    return _GPU


def to_cpu(arr):
    """
    Move *arr* to a plain NumPy array regardless of which backend is active.
    Safe to call even when the backend is already NumPy.
    """
    if _GPU:
        return xp.asnumpy(arr)
    return arr


def to_device(arr):
    """
    Move a NumPy array to the active device.
    No-op when the backend is NumPy.
    """
    if _GPU:
        return xp.asarray(arr)
    return arr
