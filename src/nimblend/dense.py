"""A dense labeled array, with absence stored as NaN or as a boolean mask."""

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from nimblend import frame, kernel
from nimblend.coords import (
    Coord,
    StoredCoord,
    distinct_labels,
    known_dims,
    require_coords,
    same_extents,
    same_labels,
    unique_dims,
)
from nimblend.domain import Domain
from nimblend.frame import combined_dims
from nimblend.protocol import ABSENCE, same_absence
from nimblend.sparse import SparseArray

type Grid = npt.NDArray[np.float64]
type Mask = npt.NDArray[np.bool_]
type Scalar = int | float | np.number[Any]
type Operand = DenseArray | SparseArray | Scalar
type Result = DenseArray | SparseArray
type Binary = Callable[[Any, Any], Any]


class DenseArray:
    """An ndarray over labeled dimensions, distinguishing absence from zero.

    An `"unknown"` array stores NaN at each absent coordinate, and `mask` is
    None. An `"empty"` array stores a boolean `mask`, true at each present
    coordinate. A stored 0.0 is present under both declarations. The
    constructor raises ValueError for a mask under `"unknown"` and for values
    or a mask of another shape.
    """

    def __init__(
        self,
        data: npt.ArrayLike,
        coords: Mapping[str, Coord],
        dims: Iterable[str],
        absence: str = "empty",
        mask: npt.ArrayLike | None = None,
    ) -> None:
        if absence not in ABSENCE:
            raise ValueError(f"absence is 'empty' or 'unknown'; got {absence!r}")
        self.dims = unique_dims(dims)
        self.absence = absence
        require_coords(self.dims, coords)
        self.coords = {d: coords[d] for d in self.dims}

        self.data = np.array(data, dtype=np.float64)
        if self.data.shape != self.shape:
            raise ValueError(
                f"the values have shape {self.data.shape} and the dimensions "
                f"{self.dims} have extents {self.shape}; pass values of shape "
                f"{self.shape}"
            )
        if absence == "unknown":
            if mask is not None:
                raise ValueError(
                    "a mask is not supported with absence 'unknown'; set each "
                    "absent coordinate to NaN in the values"
                )
            self.mask = None
        else:
            self.mask = (
                np.ones(self.shape, dtype=bool)
                if mask is None
                else np.array(mask, dtype=bool)
            )
            if self.mask.shape != self.shape:
                raise ValueError(
                    f"the mask has shape {self.mask.shape} and the dimensions "
                    f"{self.dims} have extents {self.shape}; pass a mask of "
                    f"shape {self.shape}"
                )

    @classmethod
    def from_dense(
        cls,
        values: npt.ArrayLike,
        labels: Mapping[str, npt.ArrayLike],
        absence: str = "empty",
        mask: npt.ArrayLike | None = None,
    ) -> "DenseArray":
        """Return an array over `values`, with one label array per dimension."""
        coords = {
            name: StoredCoord(np.asarray(given)) for name, given in labels.items()
        }
        return cls(values, coords, tuple(labels), absence, mask)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the extent of each dimension."""
        return tuple(len(self.coords[d]) for d in self.dims)

    @property
    def nnz(self) -> int:
        """Return the number of present coordinates."""
        return int(self.present.sum())

    @property
    def present(self) -> Mask:
        """Return a boolean array, true at each present coordinate."""
        return ~np.isnan(self.data) if self.mask is None else self.mask

    def __repr__(self) -> str:
        return (
            f"DenseArray({self.dims}, shape={self.shape}, "
            f"{self.nnz} present, absence={self.absence!r})"
        )

    def _like(
        self,
        data: npt.ArrayLike,
        mask: Mask | None = None,
        dims: Iterable[str] | None = None,
        coords: Mapping[str, Coord] | None = None,
        absence: str | None = None,
    ) -> "DenseArray":
        return DenseArray(
            data,
            self.coords if coords is None else coords,
            self.dims if dims is None else dims,
            self.absence if absence is None else absence,
            mask,
        )

    def _tagged(
        self, data: Grid, present: Mask, absence: str
    ) -> tuple[Grid, Mask | None]:
        """Return `data` and its mask, with absence stored as `absence` declares."""
        if absence == "unknown":
            out = np.where(present, data, np.nan)
            return out, None
        return np.where(present, data, 0.0), present

    def as_empty(self) -> "DenseArray":
        """Return this array with absence "empty".

        An absent coordinate then contributes nothing.
        """
        if self.absence == "empty":
            return self
        present = self.present
        return DenseArray(
            np.where(present, self.data, 0.0), self.coords, self.dims, "empty", present
        )

    def as_unknown(self) -> "DenseArray":
        """Return this array with absence "unknown".

        An absent coordinate is then not modeled.
        """
        if self.absence == "unknown":
            return self
        return DenseArray(
            np.where(self.present, self.data, np.nan), self.coords, self.dims, "unknown"
        )

    def _sparse(self) -> SparseArray:
        """Return the present coordinates of this array as a `SparseArray`."""
        return SparseArray.from_canonical(
            self.coordinates(), self.values(), self.coords, self.dims, self.absence
        )

    def coordinates(self, dims: Iterable[str] | None = None) -> kernel.Index:
        """Return the multi-index of each present coordinate, over `dims`."""
        names = self.dims if dims is None else unique_dims(dims)
        index = np.stack(np.nonzero(self.present)).astype(np.int32)
        if names == self.dims:
            return index
        return index[frame.axes_of(self, names, "coordinates")]

    def values(self) -> kernel.Values:
        """Return the values of the present coordinates, in canonical order."""
        return np.array(self.data[self.present], dtype=np.float64)

    def domain(self, dims: Iterable[str] | None = None) -> Domain:
        """Return the domain of the present coordinates over `dims`."""
        names = self.dims if dims is None else unique_dims(dims)
        known_dims("domain", names, self.dims)
        shape = tuple(len(self.coords[d]) for d in names)
        keys = kernel.ravel(self.coordinates(names), shape)
        coords = {d: self.coords[d] for d in names}
        return Domain(kernel.distinct(keys), names, coords, shape)

    def _spread(self, domain: Domain) -> Mask:
        """Return a boolean over this frame, true at the coordinates in `domain`."""
        lacking = [d for d in domain.dims if d not in self.dims]
        if lacking:
            raise ValueError(
                f"dimension(s) {lacking} of the domain are not in the array over "
                f"{self.dims}; pass a domain over dimensions of the array"
            )
        members = np.zeros(domain.shape, dtype=bool)
        members[tuple(domain.coordinates())] = True
        axes = frame.axes_of(self, domain.dims, "restrict")
        held = [1] * len(self.dims)
        for position, axis in enumerate(axes):
            held[axis] = domain.shape[position]
        ordered = np.transpose(members, np.argsort(axes)).reshape(held)
        return np.broadcast_to(ordered, self.shape)

    def restrict(self, domain: Domain) -> "DenseArray":
        """Return the present coordinates of this array that are in `domain`.

        Raises ValueError for a domain over a dimension the array does not have.
        """
        keep = self.present & self._spread(domain)
        data, mask = self._tagged(self.data, keep, self.absence)
        return self._like(data, mask)

    def expand(self, dims: Iterable[str], coords: Mapping[str, Coord]) -> "DenseArray":
        """Return every value replicated across the full extent of `dims`.

        The new dimensions are appended, as in `SparseArray.expand`;
        `transpose` reorders them. Raises ValueError for a dimension the array
        already has or one without a coordinate, and for a repeated dimension.
        """
        dims = unique_dims(dims)
        clash = [name for name in dims if name in self.dims]
        if clash:
            raise ValueError(
                f"the array already has dimension(s) {clash}; pass dimensions "
                f"it does not have"
            )
        require_coords(dims, coords)
        widened = self.shape + tuple(len(coords[name]) for name in dims)
        held = self.shape + (1,) * len(dims)
        data = np.broadcast_to(self.data.reshape(held), widened).copy()
        mask = (
            None
            if self.mask is None
            else np.broadcast_to(self.mask.reshape(held), widened).copy()
        )
        merged = dict(self.coords)
        merged.update({name: coords[name] for name in dims})
        return self._like(data, mask, self.dims + dims, merged)

    def broadcast(
        self, dims: Iterable[str], coords: Mapping[str, Coord]
    ) -> "DenseArray":
        """Return the array over exactly `dims`, in that order.

        The array is replicated across each dimension of `dims` it does not
        have, with the coordinate in `coords` for that dimension. `coords` is
        read only for those dimensions. An array over exactly `dims` is
        returned as it is. Raises ValueError for a dimension of the array not
        in `dims`, a repeated dimension, and a dimension of `dims` that neither
        the array nor `coords` contains.
        """
        return frame.broadcast(self, dims, coords)

    def to_dense(self, fill: float | None = None) -> Grid:
        """Return a numpy array with `fill` at each absent coordinate.

        Without `fill`, an absent coordinate is 0.0 under absence "empty".
        Under absence "unknown", an absent coordinate without `fill` raises
        ValueError.
        """
        if self.absence == "unknown" and fill is None:
            if self.nnz < self.data.size:
                raise ValueError(
                    f"absence is 'unknown' and the array has no value at "
                    f"{self.data.size - self.nnz} of {self.data.size} "
                    f"coordinates; pass fill=<value> to to_dense()"
                )
            return np.array(self.data, dtype=np.float64)
        return np.where(self.present, self.data, 0.0 if fill is None else fill)

    def rename(self, names: Mapping[str, str]) -> "DenseArray":
        """Return the array with its dimensions renamed.

        Raises ValueError for a key that is not a dimension of the array, and
        when two dimensions map to one name.
        """
        known_dims("rename", names, self.dims)
        dims = tuple(names.get(d, d) for d in self.dims)
        if len(set(dims)) != len(dims):
            raise ValueError(
                f"rename maps two dimensions onto one name in {dims}; map each "
                f"dimension to a distinct name"
            )
        coords = {names.get(d, d): self.coords[d] for d in self.dims}
        return self._like(self.data, self.mask, dims, coords)

    def transpose(self, *dims: str) -> "DenseArray":
        """Return the array with its dimensions in the given order.

        Without arguments the order is reversed. Raises ValueError unless
        `dims` contains each dimension once.
        """
        dims = tuple(reversed(self.dims)) if not dims else tuple(dims)
        if Counter(dims) != Counter(self.dims):
            raise ValueError(
                f"transpose requires each dimension of {self.dims} once; got {dims}"
            )
        axes = frame.axes_of(self, dims, "transpose")
        mask = None if self.mask is None else np.transpose(self.mask, axes)
        return self._like(np.transpose(self.data, axes), mask, dims)

    def sel(self, indexers: Mapping[str, Any]) -> "DenseArray":
        """Return the values at the given labels, without the selected dimensions.

        Raises KeyError for a label the coordinate does not contain. Raises
        ValueError for a dimension the array does not have.
        """
        known_dims("sel", indexers, self.dims)
        data, mask, dims = self.data, self.mask, list(self.dims)
        coords = dict(self.coords)
        for name, label in indexers.items():
            axis = dims.index(name)
            at = int(coords[name].to_position(np.asarray([label]))[0])
            data = np.take(data, at, axis=axis)
            if mask is not None:
                mask = np.take(mask, at, axis=axis)
            dims.pop(axis)
            del coords[name]
        return self._like(data, mask, tuple(dims), coords)

    def shift(self, shifts: Mapping[str, int], mode: str = "drop") -> "DenseArray":
        """Return the values moved along each dimension in `shifts`.

        `mode` is "drop" or "wrap". Under "drop" a value moved outside the
        frame is removed. A shift of 0 removes no value. Raises ValueError for
        another `mode`, and for a dimension the array does not have.
        """
        if mode not in ("drop", "wrap"):
            raise ValueError(f"mode is 'drop' or 'wrap'; got {mode!r}")
        known_dims("shift", shifts, self.dims)
        data, mask = self.data, self.present
        for name, amount in shifts.items():
            axis = self.dims.index(name)
            data = np.roll(data, amount, axis=axis)
            mask = np.roll(mask, amount, axis=axis)
            if mode == "drop" and amount:
                leaving = [slice(None)] * len(self.dims)
                leaving[axis] = slice(0, amount) if amount > 0 else slice(amount, None)
                mask[tuple(leaving)] = False
        data, mask = self._tagged(data, mask, self.absence)
        return self._like(data, mask)

    def roll(self, shifts: Mapping[str, int]) -> "DenseArray":
        """Return the values shifted along each dimension, wrapping at the ends.

        Raises ValueError for a dimension the array does not have.
        """
        known_dims("roll", shifts, self.dims)
        return self.shift(shifts, mode="wrap")

    def group(
        self,
        dims: Iterable[str],
        into: str,
        domain: Domain | None = None,
        coord: Coord | None = None,
        start: int = 0,
        out: kernel.Block | None = None,
    ) -> SparseArray:
        """Return `dims` collapsed into one dimension `into`, numbered by a domain.

        The result is a `SparseArray`, computed by `SparseArray.group` from the
        present coordinates.
        """
        return self._sparse().group(dims, into, domain, coord, start, out)

    def conform(
        self, dims: Iterable[str], labels: Mapping[str, npt.ArrayLike]
    ) -> "DenseArray":
        """Return the array read at exactly `labels`, over `dims` in that order.

        Raises KeyError for a label the coordinate does not contain. Raises
        ValueError for a label given twice.
        """
        data, mask = self.data, self.present
        coords = {}
        for name in self.dims:
            axis = self.dims.index(name)
            wanted = np.asarray(labels[name])
            at = self.coords[name].to_position(wanted)
            distinct_labels(name, wanted, at)
            data = np.take(data, at, axis=axis)
            mask = np.take(mask, at, axis=axis)
            coords[name] = StoredCoord(wanted)
        data, mask = self._tagged(data, mask, self.absence)
        arr = DenseArray(data, coords, self.dims, self.absence, mask)
        return arr.transpose(*dims) if tuple(dims) != arr.dims else arr

    def _scalar(self, value: Scalar, op: Binary) -> "DenseArray":
        data = op(self.data, float(value))
        mask = None if self.mask is None else self.mask.copy()
        return self._like(np.where(self.present, data, self.data), mask)

    def _additive(self, other: Operand, op: Binary) -> Result:
        """Return a sum or a difference of this array and `other`.

        Under absence "empty" an absent coordinate adds nothing. Under absence
        "unknown" an absent coordinate in either operand is absent in the
        result.
        """
        if isinstance(other, (int, float, np.number)):
            return self._scalar(other, op)
        if isinstance(other, SparseArray):
            return op(self._sparse(), other)
        if self.dims != other.dims:
            left, right = frame.conformed(self, other)
            return left._additive(right, op)
        frame.same_frame(self, other)
        if self.absence == "unknown":
            return self._like(op(self.data, other.data))
        mine, theirs = self.present, other.present
        data = op(np.where(mine, self.data, 0.0), np.where(theirs, other.data, 0.0))
        return self._like(np.where(mine | theirs, data, 0.0), mine | theirs)

    def __add__(self, other: Operand) -> Result:
        return self._additive(other, np.add)

    def __radd__(self, other: Operand) -> Result:
        if isinstance(other, SparseArray):
            return other + self._sparse()
        return self._additive(other, np.add)

    def __sub__(self, other: Operand) -> Result:
        return self._additive(other, np.subtract)

    def __rsub__(self, other: Scalar) -> "DenseArray":
        return self._scalar(other, lambda a, b: np.subtract(b, a))

    def __rtruediv__(self, other: Scalar) -> "DenseArray":
        """Return a number divided by each present value.

        A stored zero gives infinity. An absent coordinate stays absent.
        """
        if not isinstance(other, (int, float, np.number)):
            return NotImplemented
        with np.errstate(divide="ignore", invalid="ignore"):
            return self._scalar(other, lambda a, b: np.true_divide(b, a))

    def __neg__(self) -> "DenseArray":
        return self._like(-self.data, None if self.mask is None else self.mask.copy())

    def __pow__(self, other: Scalar) -> "DenseArray":
        """Return each present value raised to a number.

        An absent coordinate stays absent.
        """
        if not isinstance(other, (int, float, np.number)):
            return NotImplemented
        with np.errstate(divide="ignore", invalid="ignore"):
            return self._scalar(other, np.power)

    def __mul__(self, other: Operand) -> Result | tuple[str, ...]:
        """Return the product of this array and `other`.

        A coordinate absent from either operand is absent from the product. A
        frame nested in the other broadcasts over the wider frame. Frames that
        share some dimensions align on them and multiply out the rest. Raises
        ValueError for frames that share no dimension.
        """
        if isinstance(other, (int, float, np.number)):
            return self._scalar(other, np.multiply)
        if isinstance(other, SparseArray):
            return self._sparse() * other
        if self.dims == other.dims:
            frame.same_frame(self, other)
            if self.absence == "unknown":
                return self._like(self.data * other.data)
            both = self.present & other.present
            return self._like(np.where(both, self.data * other.data, 0.0), both)
        if set(self.dims) <= set(other.dims) or set(other.dims) <= set(self.dims):
            return self._broadcast_mul(other)
        if set(self.dims) & set(other.dims):
            return self._overlap_mul(other)
        return combined_dims(self.dims, other.dims)

    def __rmul__(self, other: Operand) -> Result | tuple[str, ...]:
        if isinstance(other, SparseArray):
            return other * self._sparse()
        return self.__mul__(other)

    def _spread_to(self, dims: tuple[str, ...]) -> tuple[Grid, Mask]:
        """Return the values and the presence, shaped to broadcast over `dims`.

        The axes follow the order of `dims`. A dimension this array does not
        have is an axis of extent 1.
        """
        axes = [dims.index(name) for name in self.dims]
        order = list(np.argsort(axes))
        held = [1] * len(dims)
        for at in order:
            held[axes[at]] = self.shape[at]
        data = np.transpose(self.data, order).reshape(held)
        present = np.transpose(self.present, order).reshape(held)
        return data, present

    def _product_over(
        self, other: "DenseArray", dims: tuple[str, ...], coords: Mapping[str, Coord]
    ) -> "DenseArray":
        """Return the product of two operands over `dims`.

        A coordinate absent from either operand is absent from the product.
        """
        mine, mine_present = self._spread_to(dims)
        theirs, theirs_present = other._spread_to(dims)
        data, mask = self._tagged(
            mine * theirs, mine_present & theirs_present, self.absence
        )
        return DenseArray(data, coords, dims, self.absence, mask)

    def _broadcast_mul(self, other: "DenseArray") -> "DenseArray":
        """Return the product of two arrays with nested frames, over the wider.

        The result is over the wider frame in either operand order.
        """
        narrow, wide = (self, other)
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
        return wide._product_over(narrow, wide.dims, wide.coords)

    def _overlap_mul(self, other: "DenseArray") -> "DenseArray":
        """Return the product of two arrays whose frames share some dimensions.

        The shared dimensions align and the others multiply out. The result is
        over the dimensions of this array, then those only `other` has.
        """
        shared = tuple(d for d in self.dims if d in other.dims)
        extra = tuple(d for d in other.dims if d not in self.dims)
        same_absence(self, other)
        same_labels(shared, self.coords, other.coords)
        dims = self.dims + extra
        coords = dict(self.coords)
        coords.update({d: other.coords[d] for d in extra})
        return self._product_over(other, dims, coords)

    def __truediv__(self, other: Operand) -> Result:
        """Return the quotient of this array by `other`.

        A stored zero in the denominator gives infinity, or NaN over a zero
        numerator. Raises ValueError where the numerator has a value and the
        denominator is absent.
        """
        if isinstance(other, (int, float, np.number)):
            return self._scalar(other, np.true_divide)
        if isinstance(other, SparseArray):
            return self._sparse() / other
        if self.dims != other.dims:
            left, right = frame.conformed(self, other)
            return left / right
        frame.same_frame(self, other)
        wanted = self.present & ~other.present
        if wanted.any():
            raise ValueError(
                f"the denominator is absent at {int(wanted.sum())} coordinate(s) "
                f"where the numerator has a value; restrict the numerator to the "
                f"domain of the denominator"
            )
        with np.errstate(invalid="ignore", divide="ignore"):
            data = self.data / other.data
        if self.absence == "unknown":
            return self._like(data)
        return self._like(np.where(self.present, data, 0.0), self.present.copy())

    _IDENTITY = {"sum": 0.0, "mean": 0.0, "min": np.inf, "max": -np.inf}

    def _reduce(
        self, dim: str | None, op: str, skip: bool | None, fill: float | None
    ) -> "DenseArray | float":
        if dim is not None:
            known_dims(op, (dim,), self.dims)
        frame.reduction_policy(self, skip, fill)
        if dim is None:
            frame.some_values(self, op, fill)
        present = self.present
        if fill is None:
            filled = np.where(present, self.data, self._IDENTITY[op])
        else:
            filled = np.where(present, self.data, float(fill))
            present = np.ones(self.shape, dtype=bool)

        if dim is None:
            if op == "mean":
                return float(filled.sum() / int(present.sum()))
            return float(getattr(np, op)(filled))

        axis = self.dims.index(dim)
        if op == "mean":
            counted = present.sum(axis=axis)
            with np.errstate(invalid="ignore"):
                reduced = filled.sum(axis=axis) / counted
            standing = counted > 0
        else:
            reduced = getattr(np, op)(filled, axis=axis)
            standing = present.any(axis=axis)
        dims = tuple(d for d in self.dims if d != dim)
        coords = {d: self.coords[d] for d in dims}
        data, mask = self._tagged(reduced, standing, self.absence)
        return DenseArray(data, coords, dims, self.absence, mask)

    def sum(
        self,
        dim: str | None = None,
        skip: bool | None = None,
        fill: float | None = None,
    ) -> "DenseArray | float":
        """Return the sum over `dim`, or over the whole array when `dim` is None.

        With `fill` each absent coordinate counts as `fill`. With `skip=True`
        only the present coordinates count. Raises ValueError for a `skip`
        other than True or None, when both are given, and when neither is given
        under absence "unknown".
        """
        return self._reduce(dim, "sum", skip, fill)

    def mean(
        self,
        dim: str | None = None,
        skip: bool | None = None,
        fill: float | None = None,
    ) -> "DenseArray | float":
        """Return the mean over `dim`, or over the whole array when `dim` is None.

        With `fill` each absent coordinate counts as `fill`. With `skip=True`
        only the present coordinates count. Raises ValueError for a `skip`
        other than True or None, when both are given, and when neither is given
        under absence "unknown".
        """
        return self._reduce(dim, "mean", skip, fill)

    def min(
        self,
        dim: str | None = None,
        skip: bool | None = None,
        fill: float | None = None,
    ) -> "DenseArray | float":
        """Return the minimum over `dim`, or over the whole array when `dim` is None.

        With `fill` each absent coordinate counts as `fill`. With `skip=True`
        only the present coordinates count. Raises ValueError for a `skip`
        other than True or None, when both are given, and when neither is given
        under absence "unknown".
        """
        return self._reduce(dim, "min", skip, fill)

    def max(
        self,
        dim: str | None = None,
        skip: bool | None = None,
        fill: float | None = None,
    ) -> "DenseArray | float":
        """Return the maximum over `dim`, or over the whole array when `dim` is None.

        With `fill` each absent coordinate counts as `fill`. With `skip=True`
        only the present coordinates count. Raises ValueError for a `skip`
        other than True or None, when both are given, and when neither is given
        under absence "unknown".
        """
        return self._reduce(dim, "max", skip, fill)

    def weighted_sum(
        self, dim: str, weights: npt.ArrayLike, skip: bool | None = None
    ) -> "DenseArray":
        """Return the sum over `dim` of each value times the weight at its position.

        The result equals `(self * w).sum(dim)` for an array `w` of `weights`
        over `dim`.
        `weights` has one value per position along `dim`. With `skip=True`
        only the present coordinates count. Raises ValueError for a `dim` the array does
        not have, weights of another shape, a `skip` other than True or None,
        and no `skip=True` under absence "unknown".
        """
        axis, weights = frame.weights_along(self, dim, weights, skip)
        present = self.present
        view = [1] * len(self.dims)
        view[axis] = weights.size
        filled = np.where(present, self.data, 0.0) * weights.reshape(view)
        dims = tuple(d for d in self.dims if d != dim)
        coords = {d: self.coords[d] for d in dims}
        data, mask = self._tagged(
            filled.sum(axis=axis), present.any(axis=axis), self.absence
        )
        return DenseArray(data, coords, dims, self.absence, mask)
