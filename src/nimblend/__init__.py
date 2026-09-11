"""Labeled sparse N-dimensional arrays.

`SparseArray` and `DenseArray` implement the `Array` protocol. `SparseArray`
stores only the entries that are present. `from_long` builds one from one
label column per dimension and one value column. A `Domain` is a set of
coordinates over a tuple of dimensions. `combined_dims` returns the
dimensions of a binary result from the dimensions of the two operands.
"""

from collections.abc import Iterable, Mapping

import numpy as np
import numpy.typing as npt

from nimblend.buffer import EntryBuffer
from nimblend.coords import Coord, ProductCoord, StoredCoord, SubsetCoord
from nimblend.dense import DenseArray
from nimblend.domain import Domain
from nimblend.kernel import is_canonical
from nimblend.protocol import Array
from nimblend.sparse import SparseArray, combined_dims

__version__ = "0.2.0"

__all__ = [
    "Array",
    "DenseArray",
    "Domain",
    "EntryBuffer",
    "SparseArray",
    "combined_dims",
    "from_long",
    "from_dense",
    "is_canonical",
    "StoredCoord",
    "ProductCoord",
    "SubsetCoord",
    "__version__",
]


def from_long(
    dims: Iterable[str],
    coords: Mapping[str, Coord],
    labels: Mapping[str, npt.ArrayLike],
    values: npt.ArrayLike,
    absence: str = "empty",
) -> SparseArray:
    """Return an array from one label column per dimension and one value column.

    Each label column is converted to positions by the coordinate of its
    dimension. All columns have equal length. Raises ValueError for a missing
    coordinate or label column and for columns of different lengths. Raises
    KeyError for a label the coordinate does not contain.
    """
    dims = tuple(dims)
    values = np.asarray(values, dtype=np.float64)
    missing = [d for d in dims if d not in coords]
    if missing:
        raise ValueError(
            f"no coordinate for dimension(s) {missing}; pass a coordinate for "
            f"each dimension"
        )
    absent = [d for d in dims if d not in labels]
    if absent:
        raise ValueError(
            f"no label column for dimension(s) {absent}; pass a label column "
            f"for each dimension"
        )
    index = []
    for name in dims:
        column = np.asarray(labels[name])
        if column.size != values.size:
            raise ValueError(
                f"label column {name!r} has length {column.size} and the "
                f"value column has length {values.size}; pass columns of equal "
                f"length"
            )
        index.append(np.asarray(coords[name].to_position(column), dtype=np.int32))
    return SparseArray(np.stack(index), values, coords, dims, absence)


def from_dense(
    values: npt.ArrayLike,
    labels: Mapping[str, npt.ArrayLike],
    absence: str = "empty",
) -> SparseArray:
    """Return an array with every cell of `values` as an entry."""
    return SparseArray.from_dense(values, labels, absence=absence)
