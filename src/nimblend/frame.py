"""Checks and operations on the frame of an array, shared by both implementations.

Each function takes a `SparseArray` or a `DenseArray` and reads only the
members of the `Array` protocol.
"""

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from nimblend.coords import Coord, known_dims, same_extents, same_labels, unique_dims
from nimblend.kernel import Values, span
from nimblend.protocol import same_absence


def combined_dims(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
    """Return the dimensions of a binary result from the operands' dimensions.

    Equal frames keep their order. A frame nested in the other gives the
    wider frame. Overlapping frames give the left dimensions, then those only
    the right has. Raises ValueError for frames that share no dimension.
    """
    if left == right:
        return left
    if set(left) <= set(right):
        return right
    if set(right) <= set(left):
        return left
    if set(left) & set(right):
        return left + tuple(d for d in right if d not in left)
    raise ValueError(
        f"frames {left} and {right} share no dimension; pass operands that share "
        f"a dimension"
    )


def nested_operands(left: Any, right: Any) -> tuple[Any, Any]:
    """Return the operands ordered narrower, wider, checked to broadcast.

    Raises ValueError for dimensions of the narrower operand that are not a
    subset of the wider one, for different absence, for a shared dimension
    of differing extent, and for a shared dimension with different labels.
    """
    narrow, wide = (left, right)
    if len(narrow.dims) > len(wide.dims):
        narrow, wide = wide, narrow
    if not set(narrow.dims) <= set(wide.dims):
        raise ValueError(
            f"dimensions {narrow.dims} are not a subset of {wide.dims}; pass "
            f"operands whose frames nest"
        )
    same_absence(narrow, wide)
    shared_shape = tuple(wide.shape[wide.dims.index(d)] for d in narrow.dims)
    same_extents(narrow.dims, narrow.shape, shared_shape)
    same_labels(narrow.dims, narrow.coords, wide.coords)
    return narrow, wide


def axes_of(array: Any, dims: Iterable[str], what: str) -> list[int]:
    """Return the axis of each dimension of `dims` in `array`.

    Raises ValueError for a dimension the array does not have.
    """
    dims = tuple(dims)
    known_dims(what, dims, array.dims)
    return [array.dims.index(name) for name in dims]


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


def conformed(left: Any, right: Any) -> tuple[Any, Any]:
    """Return both operands over the frame from `combined_dims`.

    An operand without a dimension of that frame is replicated across it.
    Raises ValueError for operands with different absence.
    """
    dims = combined_dims(left.dims, right.dims)
    same_absence(left, right)
    return left.broadcast(dims, right.coords), right.broadcast(dims, left.coords)


def same_frame(left: Any, right: Any) -> None:
    """Raise ValueError unless both arrays have one frame and one absence.

    The dimensions, their order and their labels must be equal.
    """
    if left.dims != right.dims:
        raise ValueError(
            f"dimensions {left.dims} and {right.dims} differ; conform one to the "
            f"other first"
        )
    same_labels(left.dims, left.coords, right.coords)
    same_absence(left, right)


def renamed_frame(
    array: Any, names: Mapping[str, str]
) -> tuple[tuple[str, ...], dict[str, Coord]]:
    """Return the array's dimensions and coordinates with `names` applied.

    Raises ValueError for a key of `names` that is not a dimension of the
    array, and for two dimensions mapped to one name.
    """
    known_dims("rename", names, array.dims)
    dims = tuple(names.get(d, d) for d in array.dims)
    if len(set(dims)) != len(dims):
        raise ValueError(
            f"rename maps two dimensions onto one name in {dims}; map each "
            f"dimension to a distinct name"
        )
    coords = {names.get(d, d): array.coords[d] for d in array.dims}
    return dims, coords


def reduction_policy(
    array: Any, skip: bool | None, fill: float | None, takes_fill: bool = True
) -> None:
    """Raise ValueError for a reduction policy the absence of `array` rejects.

    `skip` is True or None, and `skip` and `fill` are not given together.
    Under absence "unknown" one of them is required. `takes_fill` is False for
    a reduction without a `fill` parameter.
    """
    if skip is not None and skip is not True:
        raise ValueError(f"skip is True or None; got {skip!r}")
    if skip is not None and fill is not None:
        raise ValueError("skip= and fill= are given together; pass one of them")
    if array.absence == "unknown" and skip is None and fill is None:
        action = "pass skip=True to reduce the present entries"
        if takes_fill:
            action += ", or fill=<value> to include the absent coordinates"
        raise ValueError(
            f"absence is 'unknown' and no reduction policy is given; {action}"
        )


def some_values(array: Any, op: str, fill: float | None) -> None:
    """Raise ValueError for `min`, `max` or `mean` of a whole array of no values.

    Without `fill` the values are the entries. With `fill` every coordinate of
    the frame has a value. A `sum` over no values is 0.0 and does not raise.
    """
    if op == "sum":
        return
    count = array.nnz if fill is None else span(array.shape)
    if not count:
        raise ValueError(
            f"{op}() over an array with no values has no result; pass "
            f"fill=<value> or reduce an array with entries"
        )


def weights_along(
    array: Any, dim: str, weights: npt.ArrayLike, skip: bool | None
) -> tuple[int, Values]:
    """Return the axis of `dim` in `array` and `weights` as a float64 vector.

    Raises ValueError for a `dim` the array does not have, a reduction policy
    the absence rejects, and weights of a shape other than the extent of `dim`.
    """
    known_dims("weighted_sum", (dim,), array.dims)
    reduction_policy(array, skip, None, takes_fill=False)
    axis = array.dims.index(dim)
    weights = np.asarray(weights, dtype=np.float64)
    extent = array.shape[axis]
    if weights.shape != (extent,):
        raise ValueError(
            f"weights have shape {weights.shape} and dimension {dim!r} has extent "
            f"{extent}; pass one weight per position of {dim!r}"
        )
    return axis, weights
