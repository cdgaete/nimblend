"""Labeled sparse N-dimensional arrays.

`Array` is the contract and is never constructed. `SparseArray` holds only
the entries it carries; `from_long` builds one from the label-and-value
columns a columnar store holds. A `Domain` is the set of coordinates an
array carries over a tuple of its dimensions, and answers with an array of
its own: `array` values its members, `identity` pairs each with its position
along a new dimension. `combined_dims` names the frame a binary operator's
result carries, which a caller reads before materialising either operand.
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

__version__ = "0.1.0"

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
    """An array from one label column per dimension and one value column.

    Each label resolves through its dimension's own coordinate, so a caller
    already holding one states its entries in the labels a columnar store
    carries rather than in positions it would resolve twice. The columns name
    the same entries and are read in step, so they have equal length.
    """
    dims = tuple(dims)
    values = np.asarray(values, dtype=np.float64)
    missing = [d for d in dims if d not in coords]
    if missing:
        raise ValueError(f"no coordinate for dimension(s) {missing}")
    absent = [d for d in dims if d not in labels]
    if absent:
        raise ValueError(f"no label column for dimension(s) {absent}")
    index = []
    for name in dims:
        column = np.asarray(labels[name])
        if column.size != values.size:
            raise ValueError(
                f"label column {name!r} has length {column.size} and the "
                f"value column has length {values.size}; they name the same "
                f"entries"
            )
        index.append(np.asarray(coords[name].to_position(column), dtype=np.int32))
    return SparseArray(np.stack(index), values, coords, dims, absence)


def from_dense(
    values: npt.ArrayLike,
    labels: Mapping[str, npt.ArrayLike],
    absence: str = "empty",
) -> SparseArray:
    """An array holding every cell of `values` as an entry."""
    return SparseArray.from_dense(values, labels, absence=absence)
