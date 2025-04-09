"""
I/O functionality for NimbleNd arrays.
"""

# Try to import Zarr functionality
try:
    from .zarr import to_zarr, from_zarr
    __all__ = ["to_zarr", "from_zarr"]
except ImportError:
    __all__ = []

# Try to import IceChunk functionality
try:
    from .icechunk import to_icechunk
    __all__ = __all__ + ["to_icechunk"]
except ImportError:
    pass

# Try to import Series functionality
try:
    from .series import from_series, to_series
    __all__ = __all__ + ["from_series", "to_series"]
except ImportError:
    pass
