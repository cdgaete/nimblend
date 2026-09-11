"""A sparse labeled array that stores only its entries, in canonical order."""

from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from nimblend import display, kernel
from nimblend.coords import Coord, StoredCoord, SubsetCoord, python_value
from nimblend.domain import Domain
from nimblend.protocol import ABSENCE

if TYPE_CHECKING:
    from nimblend.dense import DenseArray

type Scalar = int | float | np.number[Any]
type Operand = SparseArray | Scalar
type Binary = Callable[[Any, Any], Any]


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
        self.dims = tuple(dims)
        self.absence = absence
        missing = [d for d in self.dims if d not in coords]
        if missing:
            raise ValueError(
                f"no coordinate for dimension(s) {missing}; pass a coordinate "
                f"for each dimension"
            )
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

    def _axes_of(self, dims: Iterable[str]) -> list[int]:
        return [self.dims.index(name) for name in dims]

    def _sub_index(self, axes: list[int]) -> kernel.Index:
        if axes == list(range(len(axes))):
            return self.index[: len(axes)]
        return self.index[axes]

    def domain(self, dims: Iterable[str] | None = None) -> Domain:
        """Return the domain of the entries over `dims`."""
        dims = self.dims if dims is None else tuple(dims)
        axes = self._axes_of(dims)
        shape = tuple(self.shape[axis] for axis in axes)
        keys = kernel.ravel(self._sub_index(axes), shape)
        coords = {name: self.coords[name] for name in dims}
        return Domain(kernel.distinct(keys), dims, coords, shape)

    def coordinates(self, dims: Iterable[str] | None = None) -> kernel.Index:
        """Return the multi-index of each entry over `dims`, as a copy."""
        dims = self.dims if dims is None else tuple(dims)
        return np.array(self._sub_index(self._axes_of(dims)), dtype=np.int32)

    def values(self) -> kernel.Values:
        """Return the value of each entry, as a copy, in canonical order.

        The order matches `coordinates()`. A stored zero is a value. An absent
        coordinate has no entry and no value.
        """
        return np.array(self.data, dtype=np.float64)

    def restrict(self, domain: Domain) -> "SparseArray":
        """Return the entries whose coordinate over `domain.dims` is in `domain`."""
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
        a dimension the array already has or one without a coordinate.
        """
        dims = tuple(dims)
        clash = [name for name in dims if name in self.dims]
        if clash:
            raise ValueError(
                f"the array already has dimension(s) {clash}; pass dimensions "
                f"it does not have"
            )
        missing = [name for name in dims if name not in coords]
        if missing:
            raise ValueError(
                f"no coordinate for dimension(s) {missing}; pass a coordinate "
                f"for each dimension"
            )
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

        Raises ValueError when two dimensions map to one name.
        """
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
        `dims` contains every dimension of the array.
        """
        dims = tuple(reversed(self.dims)) if not dims else tuple(dims)
        if set(dims) != set(self.dims):
            raise ValueError(
                f"transpose requires every dimension of {self.dims}; got {dims}"
            )
        order = [self.dims.index(d) for d in dims]
        return SparseArray(
            self.index[order], self.data, self.coords, dims, self.absence
        )

    def sel(self, indexers: Mapping[str, Any]) -> "SparseArray":
        """Return the entries at the given labels, without the selected dimensions.

        Raises KeyError for a label the coordinate does not contain.
        """
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

    def _conform(self, other: "SparseArray") -> tuple["SparseArray", "SparseArray"]:
        """Return both operands over the frame from `combined_dims`.

        An operand without a dimension of that frame is replicated across it.
        Raises ValueError for operands with different absence.
        """
        dims = combined_dims(self.dims, other.dims)
        if self.absence != other.absence:
            raise ValueError(
                f"one array declares absence {self.absence!r} and the other "
                f"{other.absence!r}; convert one with as_empty() or as_unknown()"
            )
        return self._widen(dims, other), other._widen(dims, self)

    def _widen(self, dims: tuple[str, ...], other: "SparseArray") -> "SparseArray":
        """Return this array over `dims`, expanded by the dimensions of `other`."""
        missing = tuple(d for d in dims if d not in self.dims)
        if not missing:
            return self if self.dims == dims else self.transpose(*dims)
        widened = self.expand(missing, {d: other.coords[d] for d in missing})
        return widened.transpose(*dims)

    def _same_frame(self, other: "SparseArray") -> None:
        if self.dims != other.dims:
            raise ValueError(
                f"dimensions {self.dims} and {other.dims} differ; conform one "
                f"to the other first"
            )
        differing = [d for d in self.dims if self.coords[d] != other.coords[d]]
        if differing:
            raise ValueError(
                f"dimension(s) {differing} have different labels in the two "
                f"arrays; conform one to the other first"
            )
        if self.absence != other.absence:
            raise ValueError(
                f"one array declares absence {self.absence!r} and the other "
                f"{other.absence!r}; convert one with as_empty() or as_unknown()"
            )

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
        self._same_frame(other)
        keys_a = kernel.ravel(self.index, self.shape)
        keys_b = kernel.ravel(other.index, other.shape)
        return self._assemble(other, op, *kernel.align(keys_a, keys_b, how))

    def _additive(self, other: Operand, op: Binary) -> "SparseArray":
        if isinstance(other, (int, float, np.number)):
            return self._scalar(other, op)
        if not isinstance(other, SparseArray):
            return NotImplemented
        if self.dims != other.dims:
            left, right = self._conform(other)
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
        if narrow.absence != wide.absence:
            raise ValueError(
                f"one array declares absence {narrow.absence!r} and the other "
                f"{wide.absence!r}; convert one with as_empty() or as_unknown()"
            )
        axes = [wide.dims.index(d) for d in narrow.dims]
        shared_shape = tuple(wide.shape[a] for a in axes)
        if shared_shape != narrow.shape:
            raise ValueError(
                f"shared dimensions {narrow.dims} have size {narrow.shape} in "
                f"one operand and {shared_shape} in the other; conform one to "
                f"the other first"
            )
        differing = [d for d in narrow.dims if wide.coords[d] != narrow.coords[d]]
        if differing:
            raise ValueError(
                f"shared dimension(s) {differing} have different labels in the "
                f"two operands; conform one to the other first"
            )
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
        if self.absence != other.absence:
            raise ValueError(
                f"one array declares absence {self.absence!r} and the other "
                f"{other.absence!r}; convert one with as_empty() or as_unknown()"
            )
        shared = tuple(d for d in self.dims if d in other.dims)
        extra = tuple(d for d in other.dims if d not in self.dims)
        differing = [d for d in shared if self.coords[d] != other.coords[d]]
        if differing:
            raise ValueError(
                f"shared dimension(s) {differing} have different labels in the "
                f"two operands; conform one to the other first"
            )
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
            left, right = self._conform(other)
            return left / right
        self._same_frame(other)
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
        if self.absence != other.absence:
            raise ValueError(
                f"one array declares absence {self.absence!r} and the other "
                f"{other.absence!r}; convert one with as_empty() or as_unknown()"
            )
        axes = self._axes_of(other.dims)
        shared_shape = tuple(self.shape[axis] for axis in axes)
        if shared_shape != other.shape:
            raise ValueError(
                f"shared dimensions {other.dims} have size {other.shape} in "
                f"one operand and {shared_shape} in the other; conform one to "
                f"the other first"
            )
        differing = [d for d in other.dims if self.coords[d] != other.coords[d]]
        if differing:
            raise ValueError(
                f"shared dimension(s) {differing} have different labels in the "
                f"two operands; conform one to the other first"
            )
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

    def _policy(self, skip: bool | None, fill: float | None) -> None:
        if skip is not None and skip is not True:
            raise ValueError(f"skip is True or None; got {skip!r}")
        if skip is not None and fill is not None:
            raise ValueError("skip= and fill= are given together; pass one of them")
        if self.absence == "unknown" and skip is None and fill is None:
            raise ValueError(
                "absence is 'unknown' and no reduction policy is given; pass "
                "skip=True to reduce the present entries, or fill=<value> to "
                "include the absent coordinates"
            )

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
        self._policy(skip, fill)
        if dim is None:
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

    def shift(self, shifts: Mapping[str, int], mode: str = "drop") -> "SparseArray":
        """Return the entries moved along each dimension in `shifts`.

        `mode` is "drop" or "wrap". Under "drop" an entry moved outside the
        frame is removed.
        """
        index, data = self.index, self.data
        for name, amount in shifts.items():
            axis = self.dims.index(name)
            index, data = kernel.shift_axis(
                index, data, axis, amount, self.shape[axis], mode=mode
            )
        return SparseArray(index, data, self.coords, self.dims, self.absence)

    def roll(self, shifts: Mapping[str, int]) -> "SparseArray":
        """Return the entries shifted along each dimension, wrapping at the ends."""
        return self.shift(shifts, mode="wrap")

    def group(
        self,
        dims: Iterable[str],
        into: str,
        domain: Domain | None = None,
        offset: int = 0,
        out: kernel.Block | None = None,
    ) -> "SparseArray":
        """Return `dims` collapsed into one dimension `into`, numbered by a domain.

        An entry is placed along `into` at the position of its member in `domain`
        plus `offset`. `domain` defaults to `self.domain(dims)`. Entries outside
        `domain` are dropped. With `out` the entries are written into `out`.
        `to_csr` raises where a non-zero `offset` places an entry beyond the extent
        of `into`. Raises ValueError for a negative `offset`, a `dims` that is not
        a leading prefix, or an `into` among the remaining dimensions.
        """
        dims = tuple(dims)
        offset = int(offset)
        if offset < 0:
            raise ValueError(
                f"offset {offset} is negative; pass an offset of 0 or more"
            )
        axes = self._axes_of(dims)
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
        at = domain.positions_of(self)
        keep = at >= 0
        n = int(keep.sum())
        if out is None:
            out_index = np.empty((1 + len(rest), n), dtype=np.int32)
            out_data = np.empty(n, dtype=np.float64)
        else:
            out_index, out_data = out[0][:, :n], out[1][:n]
        np.add(at[keep], np.int32(offset), out=out_index[0], casting="unsafe")
        for at_rest, axis in enumerate(range(len(dims), len(self.dims))):
            np.compress(keep, self.index[axis], out=out_index[1 + at_rest])
        np.compress(keep, self.data, out=out_data)
        coords = {into: SubsetCoord(domain.codes, domain.shape, start=offset)}
        coords.update({name: self.coords[name] for name in rest})
        return SparseArray.from_canonical(
            out_index, out_data, coords, (into,) + rest, self.absence
        )

    def to_csr(self) -> tuple[kernel.Index, kernel.Values, kernel.Index]:
        """Return the array as CSR triplets `(indices, values, indptr)`.

        The column indices and the values are views of the entry buffers. Only
        the row pointer is computed. Raises ValueError for an array without two
        dimensions, and for row positions outside the extent of the first.
        """
        if len(self.dims) != 2:
            raise ValueError(f"to_csr requires two dimensions; got {self.dims}")
        rows = self.shape[0]
        if self.nnz and (self.index[0, 0] < 0 or self.index[0, -1] >= rows):
            raise ValueError(
                f"row positions range from {int(self.index[0, 0])} to "
                f"{int(self.index[0, -1])} and dimension {self.dims[0]!r} has "
                f"extent {rows}; pass an array with row positions from 0 to "
                f"{rows - 1}"
            )
        return kernel.to_csr(self.index, self.data, self.shape)

    def _distinct_labels(
        self, name: str, labels: npt.NDArray[Any], positions: kernel.Positions
    ) -> None:
        at = kernel.first_repeat(positions)
        if at >= 0:
            raise ValueError(
                f"label {python_value(labels[at])!r} appears twice for dimension "
                f"{name!r}; pass each label once"
            )

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
            self._distinct_labels(name, wanted_labels, wanted)
            index, data = kernel.gather(index, data, axis, wanted, self.shape[axis])
            coords[name] = StoredCoord(wanted_labels)
        arr = SparseArray(index, data, coords, self.dims, self.absence)
        return arr.transpose(*dims) if tuple(dims) != arr.dims else arr
