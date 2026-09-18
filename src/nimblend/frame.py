"""Operations on the frame of an array, shared by both implementations.

Each function takes a `SparseArray` or a `DenseArray` and calls only the
methods of the `Array` protocol.
"""

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from nimblend.coords import Coord, known_dims, unique_dims
from nimblend.kernel import Values


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


def weights_along(
    array: Any, dim: str, weights: npt.ArrayLike, skip: bool | None
) -> tuple[int, Values]:
    """Return the axis of `dim` in `array` and `weights` as a float64 vector.

    Raises ValueError for a `dim` the array does not have, a `skip` other
    than True or None, no `skip=True` under absence "unknown", and weights
    of a shape other than the extent of `dim`.
    """
    known_dims("weighted_sum", (dim,), array.dims)
    if skip is not None and skip is not True:
        raise ValueError(f"skip is True or None; got {skip!r}")
    if array.absence == "unknown" and skip is None:
        raise ValueError(
            "absence is 'unknown' and no reduction policy is given; pass "
            "skip=True to sum the present entries"
        )
    axis = array.dims.index(dim)
    weights = np.asarray(weights, dtype=np.float64)
    extent = array.shape[axis]
    if weights.shape != (extent,):
        raise ValueError(
            f"weights have shape {weights.shape} and dimension {dim!r} has extent "
            f"{extent}; pass one weight per position of {dim!r}"
        )
    return axis, weights
