"""A sparse labeled array that stores only its entries, in canonical order."""

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from nimblend import display, frame, kernel
from nimblend.coords import (
    Coord,
    StoredCoord,
    distinct_labels,
    known_dims,
    numbered_from,
    require_coords,
    same_extents,
    same_labels,
    unique_dims,
)
from nimblend.domain import Domain
from nimblend.frame import combined_dims
from nimblend.protocol import ABSENCE, same_absence

if TYPE_CHECKING:
    from nimblend.dense import DenseArray

type Scalar = int | float | np.number[Any]
type Operand = SparseArray | Scalar
type Binary = Callable[[Any, Any], Any]


def is_canonical(index: npt.ArrayLike, shape: Iterable[int]) -> bool:
    """Return True when `index` ascends by raveled key with no repeated key.

    `index` is an integer index matrix with one row per extent of `shape`.
    Raises TypeError for an index that is not integer. Raises ValueError for
    an index that is not 2-D, a row count other than the number of extents,
    and a position outside the extent of its row.
    """
    index = np.asarray(index)
    shape = tuple(int(size) for size in shape)
    if index.dtype.kind not in "iu":
        raise TypeError(f"index has dtype {index.dtype}; pass an integer index matrix")
    if index.ndim != 2:
        raise ValueError(
            f"index has {index.ndim} dimension(s); pass a 2-D index matrix with "
            f"one row per dimension"
        )
    if index.shape[0] != len(shape):
        raise ValueError(
            f"index has {index.shape[0]} row(s) and shape {shape} has "
            f"{len(shape)} extent(s); pass one row per extent"
        )
    if index.shape[1]:
        for axis, extent in enumerate(shape):
            low, high = int(index[axis].min()), int(index[axis].max())
            if low < 0 or high >= extent:
                raise ValueError(
                    f"row {axis} of the index has positions from {low} to "
                    f"{high} and extent {extent}; pass positions from 0 to "
                    f"{extent - 1}"
                )
    return kernel.is_canonical(index, shape)


class SparseArray:
    """Entries in canonical order, under a coordinate per dimension.

    A coordinate without an entry is absent. `absence` declares its meaning:
    `"empty"` means it contributes nothing, and `"unknown"` means it is not
    modeled. The constructor sorts the entries and raises ValueError for a
    repeated index.
    """

    def __init__(
        self,
        index: npt.ArrayLike,
        data: npt.ArrayLike,
        coords: Mapping[str, Coord],
        dims: Iterable[str],
        absence: str = "empty",
    ) -> None:
        self._set_frame(coords, dims, absence)
        index, data = kernel.canonicalize(
            np.asarray(index, dtype=np.int32),
            np.asarray(data, dtype=np.float64),
            self.shape,
            on_duplicate="raise",
        )
        self.index = index
        self.data = data

    def _set_frame(
        self, coords: Mapping[str, Coord], dims: Iterable[str], absence: str
    ) -> None:
        if absence not in ABSENCE:
            raise ValueError(f"absence is 'empty' or 'unknown'; got {absence!r}")
        self.dims = unique_dims(dims)
        self.absence = absence
        require_coords(self.dims, coords)
        self.coords = {d: coords[d] for d in self.dims}

    @classmethod
    def from_canonical(
        cls,
        index: kernel.Index,
        data: kernel.Values,
        coords: Mapping[str, Coord],
        dims: Iterable[str],
        absence: str = "empty",
    ) -> "SparseArray":
        """Return an array over buffers in canonical order, without a copy.

        `index` must ascend by raveled key with no repeat. `is_canonical`
        checks this; this method does not. The array shares memory with
        `index` and `data`.
        """
        self = cls.__new__(cls)
        self._set_frame(coords, dims, absence)
        self.index = index
        self.data = data
        return self

    @classmethod
    def from_dense(
        cls,
        values: npt.ArrayLike,
        labels: Mapping[str, npt.ArrayLike],
        absence: str = "empty",
    ) -> "SparseArray":
        """Return an array with every cell of `values` as an entry."""
        values = np.asarray(values, dtype=np.float64)
        dims = tuple(labels)
        coords = {name: StoredCoord(labels[name]) for name in dims}
        index = np.indices(values.shape, dtype=np.int32).reshape(len(dims), values.size)
        return cls(index, values.ravel(), coords, dims, absence=absence)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the extent of each dimension."""
        return tuple(len(self.coords[d]) for d in self.dims)

    def __repr__(self) -> str:
        return display.one_line(
            "SparseArray",
            self.dims,
            self.shape,
            nnz=self.nnz,
            absence=repr(self.absence),
        )

    def _repr_html_(self) -> str:
        """Return the first entries as an HTML table of labels."""
        n = min(self.nnz, display.HEAD)
        columns = [
            display.as_text(self.coords[name].to_index(self.index[axis, :n]))
            for axis, name in enumerate(self.dims)
        ]
        values = [repr(float(value)) for value in self.data[:n]]
        return display.table(
            "SparseArray",
            self.dims,
            self.shape,
            (*self.dims, "value"),
            list(zip(*columns, values, strict=True)),
            self.nnz,
            nnz=self.nnz,
            absence=repr(self.absence),
        )

    def _sub_index(self, axes: list[int]) -> kernel.Index:
        if axes == list(range(len(axes))):
            return self.index[: len(axes)]
        return self.index[axes]

    def domain(self, dims: Iterable[str] | None = None) -> Domain:
        """Return the domain of the entries over `dims`."""
        dims = self.dims if dims is None else unique_dims(dims)
        axes = frame.axes_of(self, dims, "domain")
        shape = tuple(self.shape[axis] for axis in axes)
        keys = kernel.ravel(self._sub_index(axes), shape)
        coords = {name: self.coords[name] for name in dims}
        return Domain(kernel.distinct(keys), dims, coords, shape)

    def coordinates(self, dims: Iterable[str] | None = None) -> kernel.Index:
        """Return the multi-index of each entry over `dims`, as a copy."""
        dims = self.dims if dims is None else unique_dims(dims)
        return np.array(
            self._sub_index(frame.axes_of(self, dims, "coordinates")), dtype=np.int32
        )

    def values(self) -> kernel.Values:
        """Return the value of each entry, as a copy, in canonical order.

        The order matches `coordinates()`. A stored zero is a value. An absent
        coordinate has no entry and no value.
        """
        return np.array(self.data, dtype=np.float64)

    def restrict(self, domain: Domain) -> "SparseArray":
        """Return the entries whose coordinate over `domain.dims` is in `domain`.

        Raises ValueError for a domain over a dimension the array does not have.
        """
        lacking = [d for d in domain.dims if d not in self.dims]
        if lacking:
            raise ValueError(
                f"dimension(s) {lacking} of the domain are not in the array over "
                f"{self.dims}; pass a domain over dimensions of the array"
            )
        at = domain.positions_of(self)
        keep = at >= 0
        if bool(keep.all()):
            return self
        return SparseArray.from_canonical(
            self.index[:, keep], self.data[keep], self.coords, self.dims, self.absence
        )

    def expand(self, dims: Iterable[str], coords: Mapping[str, Coord]) -> "SparseArray":
        """Return every entry replicated across the full extent of `dims`.

        The new dimensions are appended; `transpose` reorders them. A dimension
        of extent `k` multiplies the entry count by `k`. Raises ValueError for
        a dimension the array already has or one without a coordinate, and
        for a repeated dimension.
        """
        dims = unique_dims(dims)
        clash = [name for name in dims if name in self.dims]
        if clash:
            raise ValueError(
                f"the array already has dimension(s) {clash}; pass dimensions "
                f"it does not have"
            )
        require_coords(dims, coords)
        sizes = [len(coords[name]) for name in dims]
        total = 1
        for size in sizes:
            total *= int(size)
        held = len(self.dims)
        index = np.empty((held + len(dims), self.nnz * total), dtype=np.int32)
        for axis in range(held):
            index[axis] = np.repeat(self.index[axis], total)
        grid = kernel.unravel(np.arange(total, dtype=np.int64), sizes)
        for at in range(len(dims)):
            index[held + at] = np.tile(grid[at], self.nnz)
        data = np.repeat(self.data, total)
        out_coords = dict(self.coords)
        out_coords.update({name: coords[name] for name in dims})
        return SparseArray.from_canonical(
            index, data, out_coords, self.dims + dims, self.absence
        )

    def broadcast(
        self, dims: Iterable[str], coords: Mapping[str, Coord]
    ) -> "SparseArray":
        """Return the array over exactly `dims`, in that order.

        The array is replicated across each dimension of `dims` it does not
        have, with the coordinate in `coords` for that dimension. `coords` is
        read only for those dimensions. An array over exactly `dims` is
        returned as it is. Raises ValueError for a dimension of the array not
        in `dims`, a repeated dimension, and a dimension of `dims` that neither
        the array nor `coords` contains.
        """
        return frame.broadcast(self, dims, coords)

    @property
    def nnz(self) -> int:
        """Return the number of entries."""
        return int(self.data.size)

    def to_dense(self, fill: float | None = None) -> npt.NDArray[np.float64]:
        """Return a numpy array with `fill` at each absent coordinate.

        Without `fill`, an absent coordinate is 0.0 under absence "empty".
        Under absence "unknown", an absent coordinate without `fill` raises
        ValueError.
        """
        if self.absence == "unknown" and fill is None:
            total = 1
            for size in self.shape:
                total *= size
            if self.nnz < total:
                raise ValueError(
                    f"absence is 'unknown' and the array has no value at "
                    f"{total - self.nnz} of {total} coordinates; pass "
                    f"fill=<value> to to_dense()"
                )
        return self._filled(0.0 if fill is None else fill)

    def rename(self, names: Mapping[str, str]) -> "SparseArray":
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
        return SparseArray(self.index, self.data, coords, dims, self.absence)

    def transpose(self, *dims: str) -> "SparseArray":
        """Return the array with its dimensions in the given order.

        Without arguments the order is reversed. Raises ValueError unless
        `dims` contains each dimension once.
        """
        dims = tuple(reversed(self.dims)) if not dims else tuple(dims)
        if Counter(dims) != Counter(self.dims):
            raise ValueError(
                f"transpose requires each dimension of {self.dims} once; got {dims}"
            )
        order = [self.dims.index(d) for d in dims]
        return SparseArray(
            self.index[order], self.data, self.coords, dims, self.absence
        )

    def sel(self, indexers: Mapping[str, Any]) -> "SparseArray":
        """Return the entries at the given labels, without the selected dimensions.

        Raises KeyError for a label the coordinate does not contain. Raises
        ValueError for a dimension the array does not have.
        """
        known_dims("sel", indexers, self.dims)
        index, data = self.index, self.data
        dims = list(self.dims)
        coords = dict(self.coords)
        for name, label in indexers.items():
            axis = dims.index(name)
            at = int(coords[name].to_position(np.asarray([label]))[0])
            keep = np.flatnonzero(index[axis] == at)
            index = np.delete(index[:, keep], axis, axis=0)
            data = data[keep]
            dims.pop(axis)
            del coords[name]
        return SparseArray(index, data, coords, tuple(dims), self.absence)

    def as_empty(self) -> "SparseArray":
        """Return this array with absence "empty".

        An absent coordinate then contributes nothing.
        """
        return SparseArray(self.index, self.data, self.coords, self.dims, "empty")

    def as_unknown(self) -> "SparseArray":
        """Return this array with absence "unknown".

        An absent coordinate is then not modeled.
        """
        return SparseArray(self.index, self.data, self.coords, self.dims, "unknown")

    def _scalar(self, value: Scalar, op: Binary) -> "SparseArray":
        return SparseArray(
            self.index, op(self.data, value), self.coords, self.dims, self.absence
        )

    def _assemble(
        self,
        other: "SparseArray",
        op: Binary,
        merged: kernel.Keys,
        take_a: kernel.Positions,
        take_b: kernel.Positions,
    ) -> "SparseArray":
        left = np.zeros(merged.size, dtype=np.float64)
        right = np.zeros(merged.size, dtype=np.float64)
        has_a = take_a >= 0
        has_b = take_b >= 0
        left[has_a] = self.data[take_a[has_a]]
        right[has_b] = other.data[take_b[has_b]]
        index = kernel.unravel(merged, self.shape)
        return SparseArray.from_canonical(
            index, op(left, right), self.coords, self.dims, self.absence
        )

    def _combine(self, other: "SparseArray", op: Binary, how: str) -> "SparseArray":
        frame.same_frame(self, other)
        keys_a = kernel.ravel(self.index, self.shape)
        keys_b = kernel.ravel(other.index, other.shape)
        return self._assemble(other, op, *kernel.align(keys_a, keys_b, how))

    def _additive(self, other: Operand, op: Binary) -> "SparseArray":
        if isinstance(other, (int, float, np.number)):
            return self._scalar(other, op)
        if not isinstance(other, SparseArray):
            return NotImplemented
        if self.dims != other.dims:
            left, right = frame.conformed(self, other)
            return left._additive(right, op)
        how = "union" if self.absence == "empty" else "intersect"
        return self._combine(other, op, how)

    def __add__(self, other: Operand) -> "SparseArray":
        return self._additive(other, np.add)

    def __radd__(self, other: Operand) -> "SparseArray":
        return self._additive(other, np.add)

    def __sub__(self, other: Operand) -> "SparseArray":
        return self._additive(other, np.subtract)

    def __rsub__(self, other: Scalar) -> "SparseArray":
        return self._scalar(other, lambda a, b: b - a)

    def __mul__(self, other: "Operand | DenseArray") -> "SparseArray | tuple[str, ...]":
        if isinstance(other, (int, float, np.number)):
            return self._scalar(other, np.multiply)
        if not isinstance(other, SparseArray):
            return NotImplemented
        if self.dims == other.dims:
            return self._combine(other, np.multiply, "intersect")
        if set(self.dims) <= set(other.dims) or set(other.dims) <= set(self.dims):
            return self._broadcast_mul(other)
        if set(self.dims) & set(other.dims):
            return self._overlap_mul(other)
        return combined_dims(self.dims, other.dims)

    def _broadcast_mul(self, other: "SparseArray") -> "SparseArray":
        """Return the product of two arrays with nested frames, over the wider.

        An entry of the wider operand is dropped where the narrower operand has
        no entry at its shared coordinate.
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
        axes = [wide.dims.index(d) for d in narrow.dims]
        shared_shape = tuple(wide.shape[a] for a in axes)
        same_extents(narrow.dims, narrow.shape, shared_shape)
        same_labels(narrow.dims, narrow.coords, wide.coords)
        probe = kernel.ravel(wide.index[axes], shared_shape)
        take = kernel.lookup(kernel.ravel(narrow.index, narrow.shape), probe)
        hit = take >= 0
        index = wide.index[:, hit]
        data = wide.data[hit] * narrow.data[take[hit]]
        return SparseArray.from_canonical(
            index, data, wide.coords, wide.dims, wide.absence
        )

    def _overlap_mul(self, other: "SparseArray") -> "SparseArray":
        """Return the product of two arrays whose frames share some dimensions.

        Each result entry pairs one entry of each operand with the same shared
        coordinate. The result is over the dimensions of this array, then those
        only `other` has. An entry with no pair in the other operand is
        dropped.
        """
        same_absence(self, other)
        shared = tuple(d for d in self.dims if d in other.dims)
        extra = tuple(d for d in other.dims if d not in self.dims)
        same_labels(shared, self.coords, other.coords)
        mine = [self.dims.index(d) for d in shared]
        theirs = [other.dims.index(d) for d in shared]
        shape = tuple(self.shape[a] for a in mine)
        keys = kernel.ravel(self.index[mine], shape)
        against = kernel.ravel(other.index[theirs], shape)
        order = np.argsort(against, kind="stable")
        sorted_against = against[order]
        lo = np.searchsorted(sorted_against, keys, "left")
        counts = np.searchsorted(sorted_against, keys, "right") - lo
        total = int(counts.sum())
        take_mine = np.repeat(np.arange(keys.size), counts)
        offsets = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
        take_theirs = order[np.repeat(lo, counts) + offsets]
        index = np.empty((len(self.dims) + len(extra), total), dtype=np.int32)
        index[: len(self.dims)] = self.index[:, take_mine]
        for row, dim in enumerate(extra, start=len(self.dims)):
            index[row] = other.index[other.dims.index(dim)][take_theirs]
        coords = dict(self.coords)
        coords.update({d: other.coords[d] for d in extra})
        return SparseArray(
            index,
            self.data[take_mine] * other.data[take_theirs],
            coords,
            self.dims + extra,
            self.absence,
        )

    def __rmul__(
        self, other: "Operand | DenseArray"
    ) -> "SparseArray | tuple[str, ...]":
        return self.__mul__(other)

    def __truediv__(self, other: "Operand | DenseArray") -> "SparseArray":
        """Return the quotient of this array by `other`.

        A stored zero in the denominator gives infinity, or NaN over a zero
        numerator. Raises ValueError where the numerator has an entry and the
        denominator is absent.
        """
        if isinstance(other, (int, float, np.number)):
            with np.errstate(divide="ignore", invalid="ignore"):
                return self._scalar(other, np.divide)
        if not isinstance(other, SparseArray):
            return NotImplemented
        if set(other.dims) < set(self.dims):
            return self._broadcast_div(other)
        if self.dims != other.dims:
            left, right = frame.conformed(self, other)
            return left / right
        frame.same_frame(self, other)
        keys_a = kernel.ravel(self.index, self.shape)
        keys_b = kernel.ravel(other.index, other.shape)
        merged, take_a, take_b = kernel.align(keys_a, keys_b, "intersect")
        if merged.size < keys_a.size:
            raise ValueError(
                f"the denominator is absent at {keys_a.size - merged.size} "
                f"coordinate(s) where the numerator has a value; restrict the "
                f"numerator to the domain of the denominator"
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            return self._assemble(other, np.divide, merged, take_a, take_b)

    def _broadcast_div(self, other: "SparseArray") -> "SparseArray":
        """Return the quotient by an array over a subset of these dimensions.

        Each entry is divided by the denominator at its coordinate over the
        shared dimensions. The denominator is not replicated.
        """
        same_absence(self, other)
        axes = frame.axes_of(self, other.dims, "align")
        shared_shape = tuple(self.shape[axis] for axis in axes)
        same_extents(other.dims, other.shape, shared_shape)
        same_labels(other.dims, other.coords, self.coords)
        probe = kernel.ravel(self.index[axes], shared_shape)
        take = kernel.lookup(kernel.ravel(other.index, other.shape), probe)
        absent = int((take < 0).sum())
        if absent:
            raise ValueError(
                f"the denominator is absent at {absent} coordinate(s) where the "
                f"numerator has a value; restrict the numerator to the domain "
                f"of the denominator"
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            data = self.data / other.data[take]
        return SparseArray.from_canonical(
            self.index, data, self.coords, self.dims, self.absence
        )

    def __rtruediv__(self, other: Scalar) -> "SparseArray":
        """Return a number divided by each entry.

        A stored zero gives infinity. An absent coordinate stays absent.
        """
        if not isinstance(other, (int, float, np.number)):
            return NotImplemented
        with np.errstate(divide="ignore", invalid="ignore"):
            return self._scalar(other, lambda a, b: b / a)

    def __neg__(self) -> "SparseArray":
        return self._scalar(0.0, lambda a, _: -a)

    def __pow__(self, other: Scalar) -> "SparseArray":
        """Return each entry raised to a number.

        An absent coordinate stays absent.
        """
        if not isinstance(other, (int, float, np.number)):
            return NotImplemented
        with np.errstate(divide="ignore", invalid="ignore"):
            return self._scalar(other, np.power)

    def _filled(self, value: float) -> npt.NDArray[np.float64]:
        """Return every cell of the frame, with `value` at each absent coordinate.

        A frame over no dimensions is a single cell.
        """
        out = np.full(self.shape, value, dtype=np.float64)
        if not self.nnz:
            return out
        if not self.dims:
            out[()] = self.data[0]
            return out
        out[tuple(self.index)] = self.data
        return out

    def _reduce(
        self, dim: str | None, op: str, skip: bool | None, fill: float | None
    ) -> "SparseArray | float":
        if dim is not None:
            known_dims(op, (dim,), self.dims)
        frame.reduction_policy(self, skip, fill)
        if dim is None:
            frame.some_values(self, op, fill)
            dense = self.data if fill is None else self._filled(fill)
            return float(getattr(np, op)(dense))
        axis = self.dims.index(dim)
        if fill is not None:
            reduced = getattr(np, op)(self._filled(fill), axis=axis)
            dims = tuple(d for d in self.dims if d != dim)
            labels = {
                d: self.coords[d].to_index(np.arange(len(self.coords[d]))) for d in dims
            }
            return SparseArray.from_dense(reduced, labels, self.absence)
        index, data = kernel.reduce_axis(
            self.index,
            self.data,
            axis,
            self.shape,
            op="sum" if op == "mean" else op,
        )
        if op == "mean":
            counts = kernel.reduce_axis(
                self.index, np.ones_like(self.data), axis, self.shape, op="sum"
            )[1]
            data = data / counts
        dims = tuple(d for d in self.dims if d != dim)
        coords = {d: self.coords[d] for d in dims}
        return SparseArray(index, data, coords, dims, self.absence)

    def sum(
        self,
        dim: str | None = None,
        skip: bool | None = None,
        fill: float | None = None,
    ) -> "SparseArray | float":
        """Return the sum over `dim`, or over the whole array when `dim` is None.

        With `fill` each absent coordinate counts as `fill`. With `skip=True`
        only the entries count. Raises ValueError for a `skip` other than True
        or None, when both are given, and when neither is given under absence
        "unknown".
        """
        return self._reduce(dim, "sum", skip, fill)

    def mean(
        self,
        dim: str | None = None,
        skip: bool | None = None,
        fill: float | None = None,
    ) -> "SparseArray | float":
        """Return the mean over `dim`, or over the whole array when `dim` is None.

        With `fill` each absent coordinate counts as `fill`. With `skip=True`
        only the entries count. Raises ValueError for a `skip` other than True
        or None, when both are given, and when neither is given under absence
        "unknown".
        """
        return self._reduce(dim, "mean", skip, fill)

    def min(
        self,
        dim: str | None = None,
        skip: bool | None = None,
        fill: float | None = None,
    ) -> "SparseArray | float":
        """Return the minimum over `dim`, or over the whole array when `dim` is None.

        With `fill` each absent coordinate counts as `fill`. With `skip=True`
        only the entries count. Raises ValueError for a `skip` other than True
        or None, when both are given, and when neither is given under absence
        "unknown".
        """
        return self._reduce(dim, "min", skip, fill)

    def max(
        self,
        dim: str | None = None,
        skip: bool | None = None,
        fill: float | None = None,
    ) -> "SparseArray | float":
        """Return the maximum over `dim`, or over the whole array when `dim` is None.

        With `fill` each absent coordinate counts as `fill`. With `skip=True`
        only the entries count. Raises ValueError for a `skip` other than True
        or None, when both are given, and when neither is given under absence
        "unknown".
        """
        return self._reduce(dim, "max", skip, fill)

    def weighted_sum(
        self, dim: str, weights: npt.ArrayLike, skip: bool | None = None
    ) -> "SparseArray":
        """Return the sum over `dim` of each entry times the weight at its position.

        The result equals `(self * w).sum(dim)` for an array `w` of `weights`
        over `dim`. No temporary of the size of the array is allocated.
        `weights` has one value per position along `dim`. With `skip=True`
        only the entries count. Raises ValueError for a `dim` the array does
        not have, weights of another shape, a `skip` other than True or None,
        and no `skip=True` under absence "unknown".
        """
        axis, weights = frame.weights_along(self, dim, weights, skip)
        index, data = kernel.weighted_sum_axis(
            self.index, self.data, axis, weights, self.shape
        )
        dims = tuple(d for d in self.dims if d != dim)
        coords = {d: self.coords[d] for d in dims}
        return SparseArray.from_canonical(index, data, coords, dims, self.absence)

    def shift(self, shifts: Mapping[str, int], mode: str = "drop") -> "SparseArray":
        """Return the entries moved along each dimension in `shifts`.

        `mode` is "drop" or "wrap". Under "drop" an entry moved outside the
        frame is removed. Raises ValueError for another `mode`, and for a
        dimension the array does not have.
        """
        if mode not in ("drop", "wrap"):
            raise ValueError(f"mode is 'drop' or 'wrap'; got {mode!r}")
        known_dims("shift", shifts, self.dims)
        index, data = self.index, self.data
        for name, amount in shifts.items():
            axis = self.dims.index(name)
            index, data = kernel.shift_axis(
                index, data, axis, amount, self.shape[axis], mode=mode
            )
        return SparseArray(index, data, self.coords, self.dims, self.absence)

    def roll(self, shifts: Mapping[str, int]) -> "SparseArray":
        """Return the entries shifted along each dimension, wrapping at the ends.

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
    ) -> "SparseArray":
        """Return `dims` collapsed into one dimension `into`, numbered by a domain.

        An entry is placed along `into` at the rank of its member in `domain`
        plus `start`. `domain` defaults to `self.domain(dims)`. `coord` is the
        coordinate of `into` and defaults to `domain.as_coord()`. Its extent can
        exceed the member count when several arrays share one numbering. Entries
        outside `domain` are dropped. With `out` the entries are written into
        `out`. Raises ValueError for a `dims` that is not a leading prefix, an
        `into` among the remaining dimensions, a negative `start`, or positions
        outside the extent of `coord`.
        """
        dims = unique_dims(dims)
        axes = frame.axes_of(self, dims, "group")
        if axes != list(range(len(dims))):
            raise ValueError(
                f"dimensions {dims} are at axes {axes} of {self.dims}, not a "
                f"leading prefix; transpose the array to put them first"
            )
        rest = self.dims[len(dims) :]
        if into in rest:
            raise ValueError(
                f"dimension {into!r} is among the remaining dimensions {rest}; pass "
                f"another name as into"
            )
        if domain is None:
            domain = self.domain(dims)
        if coord is None:
            coord = domain.as_coord()
        start = numbered_from(domain.size, into, coord, start)
        at = domain.positions_of(self)
        keep = at >= 0
        n = int(keep.sum())
        if out is None:
            out_index = np.empty((1 + len(rest), n), dtype=np.int32)
            out_data = np.empty(n, dtype=np.float64)
        else:
            out_index, out_data = out[0][:, :n], out[1][:n]
        np.add(at[keep], np.int32(start), out=out_index[0], casting="unsafe")
        for at_rest, axis in enumerate(range(len(dims), len(self.dims))):
            np.compress(keep, self.index[axis], out=out_index[1 + at_rest])
        np.compress(keep, self.data, out=out_data)
        coords = {into: coord}
        coords.update({name: self.coords[name] for name in rest})
        return SparseArray.from_canonical(
            out_index, out_data, coords, (into,) + rest, self.absence
        )

    def to_csr(self) -> tuple[kernel.Index, kernel.Values, kernel.Index]:
        """Return the array as CSR triplets `(indices, values, indptr)`.

        The column indices and the values are views of the entry buffers. Only
        the row pointer is computed. Raises ValueError for an array without two
        dimensions, and for an entry whose position along the first dimension is
        outside its extent.
        """
        if len(self.dims) != 2:
            raise ValueError(f"to_csr requires two dimensions; got {self.dims}")
        rows = self.shape[0]
        if self.nnz and (self.index[0, 0] < 0 or self.index[0, -1] >= rows):
            raise ValueError(
                f"row positions range from {int(self.index[0, 0])} to "
                f"{int(self.index[0, -1])} and dimension {self.dims[0]!r} has "
                f"extent {rows}; call to_csr on an array with row positions from 0 to "
                f"{rows - 1}"
            )
        return kernel.to_csr(self.index, self.data, self.shape)

    def conform(
        self, dims: Iterable[str], labels: Mapping[str, npt.ArrayLike]
    ) -> "SparseArray":
        """Return the array read at exactly `labels`, over `dims` in that order.

        Raises KeyError for a label the coordinate does not contain. Raises
        ValueError for a label given twice.
        """
        index, data = self.index, self.data
        coords = {}
        for name in self.dims:
            axis = self.dims.index(name)
            wanted_labels = np.asarray(labels[name])
            wanted = self.coords[name].to_position(wanted_labels)
            distinct_labels(name, wanted_labels, wanted)
            index, data = kernel.gather(index, data, axis, wanted, self.shape[axis])
            coords[name] = StoredCoord(wanted_labels)
        arr = SparseArray(index, data, coords, self.dims, self.absence)
        return arr.transpose(*dims) if tuple(dims) != arr.dims else arr
