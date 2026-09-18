"""Operations on the frame of an array, shared by both implementations.

Each function takes a `SparseArray` or a `DenseArray` and calls only the
methods of the `Array` protocol.
"""

from collections.abc import Iterable, Mapping
from typing import Any

from nimblend.coords import Coord, unique_dims


def broadcast(array: Any, dims: Iterable[str], coords: Mapping[str, Coord]) -> Any:
    """Return `array` over exactly `dims`, in that order.

    The array is replicated across each dimension of `dims` it does not have.
    Raises ValueError for a dimension of the array not in `dims`, a repeated
    dimension, and a dimension of `dims` that neither the array nor `coords`
    contains.
    """
    dims = unique_dims(dims)
    extra = [d for d in array.dims if d not in dims]
    if extra:
        raise ValueError(
            f"dimension(s) {extra} of the array over {array.dims} are not in "
            f"{dims}; pass dims that contain every dimension of the array"
        )
    missing = tuple(d for d in dims if d not in array.dims)
    if missing:
        array = array.expand(missing, coords)
    return array if array.dims == dims else array.transpose(*dims)
