"""A labeled array holding only the entries it carries."""

from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from nimblend import display, kernel
from nimblend.coords import Coord, StoredCoord, SubsetCoord
from nimblend.domain import Domain
from nimblend.protocol import ABSENCE

if TYPE_CHECKING:
    from nimblend.dense import DenseArray

type Scalar = int | float | np.number[Any]
type Operand = SparseArray | Scalar
type Binary = Callable[[Any, Any], Any]


def combined_dims(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
    """The frame a binary result carries, from the two operands' dimensions.

    Equal frames keep their order; a frame nesting inside another takes the
    wider; frames that overlap take the left operand's dimensions followed by
    the dimensions only the right carries. Frames sharing no dimension have
    nothing to align on, and their combination is an outer product no caller
    asked for.
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
        f"frames {left} and {right} share no dimension; there is nothing to "
        f"align them on"
    )


class SparseArray:
    """Entries in canonical order, under a coordinate per dimension.

    An entry that is not stored is not carried. What that means is stated by
    `absence`: `"empty"` that the coordinate contributes nothing, `"unknown"`
    that it was not modelled.
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
            raise ValueError(f"no coordinate for dimension(s) {missing}")
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
        """An array over buffers already in canonical order, taking no copy.

        The caller states that `index` is sorted by ravel key with no repeat;
        `nimblend.is_canonical` answers that question where a caller cannot.
        Verifying it here would cost the ravel this path exists to avoid, and
        the array shares memory with the buffers it is handed.
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
        """Every cell of `values` as a stored entry."""
        values = np.asarray(values, dtype=np.float64)
        dims = tuple(labels)
        coords = {name: StoredCoord(labels[name]) for name in dims}
        index = np.indices(values.shape, dtype=np.int32).reshape(len(dims), values.size)
        return cls(index, values.ravel(), coords, dims, absence=absence)

    @property
    def shape(self) -> tuple[int, ...]:
        """Size of each dimension."""
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
        """The head of the entries carried, as a table of labels."""
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
        """The distinct coordinates this array covers over `dims`."""
        dims = self.dims if dims is None else tuple(dims)
        axes = self._axes_of(dims)
        shape = tuple(self.shape[axis] for axis in axes)
        keys = kernel.ravel(self._sub_index(axes), shape)
        coords = {name: self.coords[name] for name in dims}
        return Domain(kernel.distinct(keys), dims, coords, shape)

    def coordinates(self, dims: Iterable[str] | None = None) -> kernel.Index:
        """Each entry's multi-index over `dims`, as a copy."""
        dims = self.dims if dims is None else tuple(dims)
        return np.array(self._sub_index(self._axes_of(dims)), dtype=np.int32)

    def values(self) -> kernel.Values:
        """The value each entry carries, as a copy, in canonical order.

        The partner of `coordinates()`: together they answer the entries an
        array holds without handing out the buffer it owns. A stored zero is
        a value and is answered; an absent coordinate has no entry and no
        value here.
        """
        return np.array(self.data, dtype=np.float64)

    def restrict(self, domain: Domain) -> "SparseArray":
        """The entries whose coordinate over the domain's dimensions it carries."""
        at = domain.positions_of(self)
        keep = at >= 0
        if bool(keep.all()):
            return self
        return SparseArray.from_canonical(
            self.index[:, keep], self.data[keep], self.coords, self.dims, self.absence
        )

    def expand(self, dims: Iterable[str], coords: Mapping[str, Coord]) -> "SparseArray":
        """Every entry replicated across the full extent of the named dimensions.

        The new dimensions are appended, which keeps the result canonical; a
        different order is reached with `transpose`. Adding a dimension of
        size `k` multiplies the entry count by `k`, so a caller states the
        replication rather than an operator implying it.
        """
        dims = tuple(dims)
        clash = [name for name in dims if name in self.dims]
        if clash:
            raise ValueError(f"dimension(s) {clash} are already carried")
        missing = [name for name in dims if name not in coords]
        if missing:
            raise ValueError(f"no coordinate for dimension(s) {missing}")
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
        """Number of entries carried."""
        return int(self.data.size)

    def to_dense(self, fill: float | None = None) -> npt.NDArray[np.float64]:
        """A dense array with absent coordinates carrying `fill`.

        An array declaring absence `"empty"` densifies to zero where it
        carries no entry, because that is what the coordinate contributes.
        One declaring `"unknown"` has no value to place there and states
        `fill`, unless it carries every coordinate of its frame and there is
        nowhere to place one.
        """
        if self.absence == "unknown" and fill is None:
            total = 1
            for size in self.shape:
                total *= size
            if self.nnz < total:
                raise ValueError(
                    "this array declares absence 'unknown' and does not carry "
                    "every coordinate of its frame, so densifying must state "
                    "fill=<value> to place at the rest"
                )
        return self._filled(0.0 if fill is None else fill)

    def rename(self, names: Mapping[str, str]) -> "SparseArray":
        """The array with dimensions renamed."""
        dims = tuple(names.get(d, d) for d in self.dims)
        if len(set(dims)) != len(dims):
            raise ValueError(f"rename maps two dimensions onto one name: {dims}")
        coords = {names.get(d, d): self.coords[d] for d in self.dims}
        return SparseArray(self.index, self.data, coords, dims, self.absence)

    def transpose(self, *dims: str) -> "SparseArray":
        """The array with its dimensions in the given order."""
        dims = tuple(reversed(self.dims)) if not dims else tuple(dims)
        if set(dims) != set(self.dims):
            raise ValueError(
                f"transpose needs every dimension of {self.dims}; got {dims}"
            )
        order = [self.dims.index(d) for d in dims]
        return SparseArray(
            self.index[order], self.data, self.coords, dims, self.absence
        )

    def sel(self, indexers: Mapping[str, Any]) -> "SparseArray":
        """Entries at the given labels, dropping each dimension named once."""
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
        """The array declaring that an absent coordinate contributes nothing."""
        return SparseArray(self.index, self.data, self.coords, self.dims, "empty")

    def as_unknown(self) -> "SparseArray":
        """The array declaring that an absent coordinate was not modelled."""
        return SparseArray(self.index, self.data, self.coords, self.dims, "unknown")

    def _conform(self, other: "SparseArray") -> tuple["SparseArray", "SparseArray"]:
        """Both operands over the frame their dimensions combine to.

        An operand missing a dimension of that frame gains one entry per
        coordinate of it, which is the replication a wider result carries.
        """
        dims = combined_dims(self.dims, other.dims)
        if self.absence != other.absence:
            raise ValueError(
                f"one array declares absence {self.absence!r} and the other "
                f"{other.absence!r}; state which the result carries with "
                f"as_empty() or as_unknown()"
            )
        return self._widen(dims, other), other._widen(dims, self)

    def _widen(self, dims: tuple[str, ...], other: "SparseArray") -> "SparseArray":
        """This array over `dims`, taking any missing coordinate from `other`."""
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
                f"dimension(s) {differing} carry different labels in the two "
                f"arrays; an entry is aligned by its label, so conform one to "
                f"the other first"
            )
        if self.absence != other.absence:
            raise ValueError(
                f"one array declares absence {self.absence!r} and the other "
                f"{other.absence!r}; state which the result carries with "
                f"as_empty() or as_unknown()"
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
        """The product of two arrays whose dimensions nest, over the wider frame.

        The operand carrying fewer dimensions supplies a factor for every entry
        of the wider one sharing its coordinate. An entry whose shared
        coordinate the narrower operand does not carry has no factor and does
        not survive, which is intersection extended over the extra dimensions.
        """
        narrow, wide = (self, other)
        if len(narrow.dims) > len(wide.dims):
            narrow, wide = wide, narrow
        if not set(narrow.dims) <= set(wide.dims):
            raise ValueError(
                f"dimensions {narrow.dims} are not a subset of {wide.dims}; a "
                f"broadcast product needs one frame to nest inside the other"
            )
        if narrow.absence != wide.absence:
            raise ValueError(
                f"one array declares absence {narrow.absence!r} and the other "
                f"{wide.absence!r}; state which the result carries with "
                f"as_empty() or as_unknown()"
            )
        axes = [wide.dims.index(d) for d in narrow.dims]
        shared_shape = tuple(wide.shape[a] for a in axes)
        if shared_shape != narrow.shape:
            raise ValueError(
                f"shared dimensions {narrow.dims} have size {narrow.shape} in "
                f"one operand and {shared_shape} in the other"
            )
        differing = [d for d in narrow.dims if wide.coords[d] != narrow.coords[d]]
        if differing:
            raise ValueError(
                f"shared dimension(s) {differing} carry different labels in "
                f"the two operands; an entry is aligned by its label"
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
        """The product of two arrays whose frames share some dimensions.

        The shared dimensions align and the rest multiply out, so an entry of
        the result pairs one entry of each operand agreeing on the shared
        coordinate. The result carries this array's dimensions followed by
        the dimensions only the other carries. An entry whose shared
        coordinate the other operand does not carry has no factor and does
        not survive.
        """
        if self.absence != other.absence:
            raise ValueError(
                f"one array declares absence {self.absence!r} and the other "
                f"{other.absence!r}; state which the result carries with "
                f"as_empty() or as_unknown()"
            )
        shared = tuple(d for d in self.dims if d in other.dims)
        extra = tuple(d for d in other.dims if d not in self.dims)
        differing = [d for d in shared if self.coords[d] != other.coords[d]]
        if differing:
            raise ValueError(
                f"shared dimension(s) {differing} carry different labels in "
                f"the two operands; an entry is aligned by its label"
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
        """The quotient over the coordinates both operands carry.

        A stored zero is a value the array carries, so dividing by one
        answers what the arithmetic answers: infinity, or nan where the
        numerator is zero too. An absent denominator is not a value and has
        nothing to divide by, so it raises. A caller that refuses a
        non-finite result states that rule itself.
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
                f"coordinate(s) the numerator carries; a quotient there is not "
                f"zero and not one, so it is refused"
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            return self._assemble(other, np.divide, merged, take_a, take_b)

    def _broadcast_div(self, other: "SparseArray") -> "SparseArray":
        """The quotient of this array by one whose dimensions it carries.

        Each entry reads the denominator at its own coordinate over the
        shared dimensions, so the result carries this array's entries and the
        denominator is never replicated across the wider frame.
        """
        if self.absence != other.absence:
            raise ValueError(
                f"one array declares absence {self.absence!r} and the other "
                f"{other.absence!r}; state which the result carries with "
                f"as_empty() or as_unknown()"
            )
        axes = self._axes_of(other.dims)
        shared_shape = tuple(self.shape[axis] for axis in axes)
        if shared_shape != other.shape:
            raise ValueError(
                f"shared dimensions {other.dims} have size {other.shape} in "
                f"one operand and {shared_shape} in the other"
            )
        differing = [d for d in other.dims if self.coords[d] != other.coords[d]]
        if differing:
            raise ValueError(
                f"shared dimension(s) {differing} carry different labels in "
                f"the two operands; an entry is aligned by its label"
            )
        probe = kernel.ravel(self.index[axes], shared_shape)
        take = kernel.lookup(kernel.ravel(other.index, other.shape), probe)
        absent = int((take < 0).sum())
        if absent:
            raise ValueError(
                f"the denominator is absent at {absent} coordinate(s) the "
                f"numerator carries; a quotient there is not zero and not "
                f"one, so it is refused"
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            data = self.data / other.data[take]
        return SparseArray.from_canonical(
            self.index, data, self.coords, self.dims, self.absence
        )

    def __rtruediv__(self, other: Scalar) -> "SparseArray":
        """A number divided by every entry this array carries.

        A stored zero divides to infinity, which is what the arithmetic
        answers; an absent coordinate has no entry and stays absent.
        """
        if not isinstance(other, (int, float, np.number)):
            return NotImplemented
        with np.errstate(divide="ignore", invalid="ignore"):
            return self._scalar(other, lambda a, b: b / a)

    def __neg__(self) -> "SparseArray":
        return self._scalar(0.0, lambda a, _: -a)

    def __pow__(self, other: Scalar) -> "SparseArray":
        """Every entry raised to a number.

        An absent coordinate stays absent, as it does under a scalar product:
        it carries no value to raise.
        """
        if not isinstance(other, (int, float, np.number)):
            return NotImplemented
        with np.errstate(divide="ignore", invalid="ignore"):
            return self._scalar(other, np.power)

    def _policy(self, skip: bool | None, fill: float | None) -> None:
        if skip is not None and skip is not True:
            raise ValueError(
                f"skip states that the entries present are the whole of the "
                f"reduction and is True; got {skip!r}"
            )
        if skip is not None and fill is not None:
            raise ValueError("state skip= or fill=, not both")
        if self.absence == "unknown" and skip is None and fill is None:
            raise ValueError(
                "this array declares absence 'unknown', so a reduction must "
                "state skip=True to use present entries only, or fill=<value> "
                "to count absences as that value"
            )

    def _filled(self, value: float) -> npt.NDArray[np.float64]:
        """Every cell of the frame, with absent coordinates carrying `value`.

        A frame over no dimensions is a single cell, which the one entry an
        array can carry there fills whole.
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
        """Total over `dim`, or over the whole array when `dim` is None.

        With `fill` every coordinate of the frame counts, carrying that
        value where the array holds no entry; with `skip` only the
        entries held count.
        """
        return self._reduce(dim, "sum", skip, fill)

    def mean(
        self,
        dim: str | None = None,
        skip: bool | None = None,
        fill: float | None = None,
    ) -> "SparseArray | float":
        """Mean over `dim`, or over the whole array when `dim` is None.

        With `fill` every coordinate of the frame counts, carrying that
        value where the array holds no entry; with `skip` only the
        entries held count.
        """
        return self._reduce(dim, "mean", skip, fill)

    def min(
        self,
        dim: str | None = None,
        skip: bool | None = None,
        fill: float | None = None,
    ) -> "SparseArray | float":
        """Minimum over `dim`, or over the whole array when `dim` is None.

        With `fill` every coordinate of the frame counts, carrying that
        value where the array holds no entry; with `skip` only the
        entries held count.
        """
        return self._reduce(dim, "min", skip, fill)

    def max(
        self,
        dim: str | None = None,
        skip: bool | None = None,
        fill: float | None = None,
    ) -> "SparseArray | float":
        """Maximum over `dim`, or over the whole array when `dim` is None.

        With `fill` every coordinate of the frame counts, carrying that
        value where the array holds no entry; with `skip` only the
        entries held count.
        """
        return self._reduce(dim, "max", skip, fill)

    def shift(self, shifts: Mapping[str, int], mode: str = "drop") -> "SparseArray":
        """Entries moved along each named dimension."""
        index, data = self.index, self.data
        for name, amount in shifts.items():
            axis = self.dims.index(name)
            index, data = kernel.shift_axis(
                index, data, axis, amount, self.shape[axis], mode=mode
            )
        return SparseArray(index, data, self.coords, self.dims, self.absence)

    def roll(self, shifts: Mapping[str, int]) -> "SparseArray":
        """Entries moved along each named dimension, wrapping at the ends."""
        return self.shift(shifts, mode="wrap")

    def group(
        self,
        dims: Iterable[str],
        into: str,
        domain: Domain | None = None,
        offset: int = 0,
        out: kernel.Block | None = None,
    ) -> "SparseArray":
        """`dims` collapsed into one dimension numbered by a domain.

        A member's position in the domain, plus `offset`, is its index along
        `into`. The grouped dimensions must be a leading prefix of the
        canonical order, which is what makes the result canonical as written:
        their positions ascend with the coordinate and the remaining
        dimensions ascend within each group. An entry whose coordinate the
        domain does not carry is not emitted.

        A non-zero `offset` numbers the result into an extent wider than its
        own members span, which is what lets several results share one
        destination buffer and one numbering. Such a result is a fragment of
        that extent: reading it as a standalone array is what `to_csr`
        refuses.
        """
        dims = tuple(dims)
        offset = int(offset)
        if offset < 0:
            raise ValueError(
                f"offset {offset} numbers the first member below zero; a "
                f"position along {into!r} is not negative"
            )
        axes = self._axes_of(dims)
        if axes != list(range(len(dims))):
            raise ValueError(
                f"dimensions {dims} sit at axes {axes} of {self.dims}; "
                f"grouping reads a leading prefix of the canonical order"
            )
        rest = self.dims[len(dims) :]
        if into in rest:
            raise ValueError(f"dimension {into!r} is already carried")
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
        """The array as CSR triplets `(indices, values, indptr)`.

        Canonical order is sorted by row and then column, which is CSR's own
        requirement, so the column indices and values are views and only the
        row pointer is built.
        """
        if len(self.dims) != 2:
            raise ValueError(
                f"CSR is a two-dimensional form; this array has dimensions {self.dims}"
            )
        rows = self.shape[0]
        if self.nnz and (self.index[0, 0] < 0 or self.index[0, -1] >= rows):
            raise ValueError(
                f"rows run from {int(self.index[0, 0])} to "
                f"{int(self.index[0, -1])} and dimension {self.dims[0]!r} "
                f"spans {rows}; a row pointer over that extent would not "
                f"reach these entries"
            )
        return kernel.to_csr(self.index, self.data, self.shape)

    def _distinct_labels(
        self, name: str, labels: npt.NDArray[Any], positions: kernel.Positions
    ) -> None:
        at = kernel.first_repeat(positions)
        if at >= 0:
            raise ValueError(
                f"label {labels[at]!r} is named twice for dimension {name!r}; "
                f"conform reads each position of a dimension once"
            )

    def conform(
        self, dims: Iterable[str], labels: Mapping[str, npt.ArrayLike]
    ) -> "SparseArray":
        """The array read at exactly `labels`, laid out over `dims`.

        Each label is named once: a repeat would ask one position to occupy
        two, and `gather` renumbers a position to a single destination.
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
