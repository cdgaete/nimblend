"""Labeled sparse N-dimensional arrays.

`SparseArray` and `DenseArray` implement the `Array` protocol. `SparseArray`
stores only the entries that are present. `from_long` builds one from one
label column per dimension and one value column. `sum_arrays` adds several
arrays over one frame in one merge. A `Domain` is a set of
coordinates over a tuple of dimensions. `combined_dims` returns the
dimensions of a binary result from the dimensions of the two operands.
"""

from collections.abc import Iterable, Mapping

import numpy as np
import numpy.typing as npt

from nimblend import kernel
from nimblend.buffer import EntryBuffer
from nimblend.coords import (
    Coord,
    ProductCoord,
    StoredCoord,
    SubsetCoord,
    label_positions,
)
from nimblend.dense import DenseArray
from nimblend.domain import Domain
from nimblend.frame import combined_dims
from nimblend.protocol import Array
from nimblend.sparse import SparseArray, is_canonical, sum_arrays

__version__ = "0.20260923.0"

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
    "sum_arrays",
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
    on_duplicate: str = "raise",
) -> SparseArray:
    """Return an array from one label column per dimension and one value column.

    Each label column is converted to positions by the coordinate of its
    dimension. A label column of a `ProductCoord` or a `SubsetCoord` is an
    index matrix with one column per label. Every column has one label per
    value. `on_duplicate` is "raise" or "sum". Under "raise" a repeated
    coordinate raises ValueError. Under "sum" the rows of a repeated
    coordinate keep their input order, `np.add.reduceat` adds them, and the
    result has one entry for that coordinate. Raises ValueError for a missing
    coordinate or label column, for columns of different lengths and for
    another `on_duplicate`. Raises KeyError for a label the coordinate does
    not contain.
    """
    dims = tuple(dims)
    values = np.asarray(values, dtype=np.float64)
    index = label_positions(dims, coords, labels, values.size)
    if on_duplicate == "raise":
        return SparseArray(index, values, coords, dims, absence)
    shape = tuple(len(coords[d]) for d in dims)
    index, values = kernel.canonicalize(
        index.astype(np.int32, copy=False), values, shape, on_duplicate=on_duplicate
    )
    return SparseArray.from_canonical(index, values, coords, dims, absence)


def from_dense(
    values: npt.ArrayLike,
    labels: Mapping[str, npt.ArrayLike],
    absence: str = "empty",
) -> SparseArray:
    """Return an array with every cell of `values` as an entry."""
    return SparseArray.from_dense(values, labels, absence=absence)
