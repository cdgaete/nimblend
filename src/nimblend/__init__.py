"""
NimbleNd: Efficient Labeled N-Dimensional Arrays
"""

__version__ = "0.1.0"

from .core import Array, CoordinateMap
from .operations import concat, stack

# Import I/O functionality
try:
    from .io import from_zarr, to_zarr

    __io_imports = ["to_zarr", "from_zarr"]
except ImportError:
    __io_imports = []

# Import IceChunk functionality if available
try:
    from .io import from_icechunk, to_icechunk

    __icechunk_imports = ["to_icechunk", "from_icechunk"]
except ImportError:
    __icechunk_imports = []

# Import Series functionality if available
try:
    from .io import from_series, to_series

    __series_imports = ["from_series", "to_series"]
except ImportError:
    __series_imports = []

__all__ = (
    [
        "Array",
        "CoordinateMap",
        "concat",
        "stack",
    ]
    + __io_imports
    + __icechunk_imports
    + __series_imports
)
