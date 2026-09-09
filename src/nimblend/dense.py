"""A dense labeled array, whose presence is carried the way it declares."""

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from nimblend import kernel
from nimblend.coords import Coord, StoredCoord
from nimblend.domain import Domain
from nimblend.protocol import ABSENCE
from nimblend.sparse import SparseArray, combined_dims

type Grid = npt.NDArray[np.float64]
type Mask = npt.NDArray[np.bool_]
type Scalar = int | float | np.number[Any]
type Operand = DenseArray | SparseArray | Scalar
type Result = DenseArray | SparseArray
type Binary = Callable[[Any, Any], Any]


class DenseArray:
    """An ndarray over labeled dimensions, distinguishing absence from zero.

    How presence is carried follows the absence declaration, because the two
    declarations want opposite things from an operator. An `"unknown"` array
    tags absence with NaN, which propagates through arithmetic at no cost: an
    addition of two 2000x2000 arrays runs 2.3 ms against 14.0 ms for a mask,
    and the tag needs no storage beside the values. An `"empty"` array carries
    a boolean mask, because absence is the additive identity there and
    substituting it costs 10.0 ms against 34.8 ms for NaN, at 12.5% over
    float64.

    Absence and zero stay distinct under both: a stored 0.0 is a coordinate
    that is present and worth nothing.
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
        self.dims = tuple(dims)
        self.absence = absence
        missing = [d for d in self.dims if d not in coords]
        if missing:
            raise ValueError(f"no coordinate for dimension(s) {missing}")
        self.coords = {d: coords[d] for d in self.dims}

        self.data = np.array(data, dtype=np.float64)
        if self.data.shape != self.shape:
            raise ValueError(
                f"the values have shape {self.data.shape} and the dimensions "
                f"{self.dims} have extents {self.shape}"
            )
        if absence == "unknown":
            if mask is not None:
                raise ValueError(
                    "an array declaring absence 'unknown' tags an absent "
                    "coordinate with NaN and carries no mask"
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
                    f"{self.dims} have extents {self.shape}"
                )

    @classmethod
    def from_dense(
        cls,
        values: npt.ArrayLike,
        labels: Mapping[str, npt.ArrayLike],
        absence: str = "empty",
        mask: npt.ArrayLike | None = None,
    ) -> "DenseArray":
        """An array over `values`, labelled per dimension."""
        coords = {
            name: StoredCoord(np.asarray(given)) for name, given in labels.items()
        }
        return cls(values, coords, tuple(labels), absence, mask)

    @property
    def shape(self) -> tuple[int, ...]:
        """The extent of each dimension."""
        return tuple(len(self.coords[d]) for d in self.dims)

    @property
    def nnz(self) -> int:
        """Number of coordinates this array carries a value at."""
        return int(self.present.sum())

    @property
    def present(self) -> Mask:
        """Where this array carries a value, read from how it stores absence."""
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
        """`data` carrying its absence the way `absence` says to."""
        if absence == "unknown":
            out = np.where(present, data, np.nan)
            return out, None
        return np.where(present, data, 0.0), present

    def as_empty(self) -> "DenseArray":
        """The array declaring that an absent coordinate contributes nothing."""
        if self.absence == "empty":
            return self
        present = self.present
        return DenseArray(
            np.where(present, self.data, 0.0), self.coords, self.dims, "empty", present
        )

    def as_unknown(self) -> "DenseArray":
        """The array declaring that an absent coordinate was not modelled."""
        if self.absence == "unknown":
            return self
        return DenseArray(
            np.where(self.present, self.data, np.nan), self.coords, self.dims, "unknown"
        )

    def _axes_of(self, dims: Iterable[str]) -> list[int]:
        return [self.dims.index(name) for name in dims]

    def _sparse(self) -> SparseArray:
        """This array's present coordinates as a `SparseArray`.

        A product intersects presence, so a mixed product carries at most the
        entries the sparse operand holds and is answered as a `SparseArray`.
        """
        return SparseArray.from_canonical(
            self.coordinates(), self.values(), self.coords, self.dims, self.absence
        )

    def coordinates(self, dims: Iterable[str] | None = None) -> kernel.Index:
        """The multi-index of each present coordinate, over `dims`."""
        names = self.dims if dims is None else tuple(dims)
        index = np.stack(np.nonzero(self.present)).astype(np.int32)
        if names == self.dims:
            return index
        return index[self._axes_of(names)]

    def values(self) -> kernel.Values:
        """The values the present coordinates carry, in canonical order."""
        return np.array(self.data[self.present], dtype=np.float64)

    def domain(self, dims: Iterable[str] | None = None) -> Domain:
        """The distinct coordinates this array covers over `dims`."""
        names = self.dims if dims is None else tuple(dims)
        shape = tuple(len(self.coords[d]) for d in names)
        keys = kernel.ravel(self.coordinates(names), shape)
        coords = {d: self.coords[d] for d in names}
        return Domain(kernel.distinct(keys), names, coords, shape)

    def _spread(self, domain: Domain) -> Mask:
        """A boolean over this array's frame, true where `domain` reaches."""
        lacking = [d for d in domain.dims if d not in self.dims]
        if lacking:
            raise ValueError(
                f"this array is over {self.dims} and does not carry {lacking}"
            )
        carried = np.zeros(domain.shape, dtype=bool)
        carried[tuple(domain.coordinates())] = True
        axes = self._axes_of(domain.dims)
        held = [1] * len(self.dims)
        for position, axis in enumerate(axes):
            held[axis] = domain.shape[position]
        ordered = np.transpose(carried, np.argsort(axes)).reshape(held)
        return np.broadcast_to(ordered, self.shape)

    def restrict(self, domain: Domain) -> "DenseArray":
        """The coordinates this array carries that the domain carries too."""
        keep = self.present & self._spread(domain)
        data, mask = self._tagged(self.data, keep, self.absence)
        return self._like(data, mask)

    def expand(self, dims: Iterable[str], coords: Mapping[str, Coord]) -> "DenseArray":
        """Every value replicated across the full extent of the named dimensions.

        The new dimensions are appended, which is where `SparseArray.expand`
        puts them; a different order is reached with `transpose`.
        """
        dims = tuple(dims)
        clash = [name for name in dims if name in self.dims]
        if clash:
            raise ValueError(f"dimension(s) {clash} are already carried")
        missing = [name for name in dims if name not in coords]
        if missing:
            raise ValueError(f"no coordinate for dimension(s) {missing}")
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

    def to_dense(self, fill: float | None = None) -> Grid:
        """A dense array with absent coordinates carrying `fill`."""
        if self.absence == "unknown" and fill is None:
            if self.nnz < self.data.size:
                raise ValueError(
                    "this array declares absence 'unknown' and does not carry "
                    "every coordinate of its frame, so densifying must state "
                    "fill=<value> to place at the rest"
                )
            return np.array(self.data, dtype=np.float64)
        return np.where(self.present, self.data, 0.0 if fill is None else fill)

    def rename(self, names: Mapping[str, str]) -> "DenseArray":
        """The array with dimensions renamed."""
        dims = tuple(names.get(d, d) for d in self.dims)
        if len(set(dims)) != len(dims):
            raise ValueError(f"rename maps two dimensions onto one name: {dims}")
        coords = {names.get(d, d): self.coords[d] for d in self.dims}
        return self._like(self.data, self.mask, dims, coords)

    def transpose(self, *dims: str) -> "DenseArray":
        """The array with its dimensions in the order given, or reversed."""
        dims = tuple(reversed(self.dims)) if not dims else tuple(dims)
        if sorted(dims) != sorted(self.dims):
            raise ValueError(f"this array is over {self.dims}; got {dims}")
        axes = self._axes_of(dims)
        mask = None if self.mask is None else np.transpose(self.mask, axes)
        return self._like(np.transpose(self.data, axes), mask, dims)

    def sel(self, indexers: Mapping[str, Any]) -> "DenseArray":
        """Values at the given labels, dropping each dimension named once."""
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
        """Values moved along each named dimension.

        A shift of nothing moves nothing out of the frame, so it drops
        nothing.
        """
        data, mask = self.data, self.present
        for name, amount in shifts.items():
            if mode not in ("drop", "wrap"):
                raise ValueError(f"mode is 'drop' or 'wrap'; got {mode!r}")
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
        """Values moved along each named dimension, wrapping at the ends."""
        return self.shift(shifts, mode="wrap")

    def group(
        self,
        dims: Iterable[str],
        into: str,
        domain: Domain | None = None,
        offset: int = 0,
        out: kernel.Block | None = None,
    ) -> SparseArray:
        """`dims` collapsed into one dimension numbered by a domain.

        The result holds the entries that survive rather than a grid: an
        offset numbers them into an extent wider than their own members
        span, which no dense frame states, so it is a `SparseArray`.
        """
        return self._sparse().group(dims, into, domain, offset, out)

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
    ) -> "DenseArray":
        """The array read at exactly `labels`, laid out over `dims`.

        Each label is named once: a repeat would ask one position to occupy
        two, which is not a reading the contract offers.
        """
        data, mask = self.data, self.present
        coords = {}
        for name in self.dims:
            axis = self.dims.index(name)
            wanted = np.asarray(labels[name])
            at = self.coords[name].to_position(wanted)
            self._distinct_labels(name, wanted, at)
            data = np.take(data, at, axis=axis)
            mask = np.take(mask, at, axis=axis)
            coords[name] = StoredCoord(wanted)
        data, mask = self._tagged(data, mask, self.absence)
        arr = DenseArray(data, coords, self.dims, self.absence, mask)
        return arr.transpose(*dims) if tuple(dims) != arr.dims else arr

    def _conform(self, other: "DenseArray") -> tuple["DenseArray", "DenseArray"]:
        """Both operands over the frame their dimensions combine to.

        An operand missing a dimension of that frame carries its values at
        every coordinate of it, which is the replication a wider result
        carries.
        """
        dims = combined_dims(self.dims, other.dims)
        if self.absence != other.absence:
            raise ValueError(
                f"one array declares absence {self.absence!r} and the other "
                f"{other.absence!r}; state which the result carries with "
                f"as_empty() or as_unknown()"
            )
        return self._widen(dims, other), other._widen(dims, self)

    def _widen(self, dims: tuple[str, ...], other: "DenseArray") -> "DenseArray":
        """This array over `dims`, taking any missing coordinate from `other`."""
        missing = tuple(d for d in dims if d not in self.dims)
        if not missing:
            return self if self.dims == dims else self.transpose(*dims)
        widened = self.expand(missing, {d: other.coords[d] for d in missing})
        return widened.transpose(*dims)

    def _same_frame(self, other: "DenseArray") -> None:
        if self.dims != other.dims:
            raise ValueError(
                f"dimensions {self.dims} and {other.dims} differ; conform one "
                f"to the other first"
            )
        if self.absence != other.absence:
            raise ValueError(
                f"one array declares absence {self.absence!r} and the other "
                f"{other.absence!r}; state which the result carries with "
                f"as_empty() or as_unknown()"
            )
        differing = [d for d in self.dims if self.coords[d] != other.coords[d]]
        if differing:
            raise ValueError(
                f"dimension(s) {differing} carry different labels; conform one "
                f"to the other first"
            )

    def _scalar(self, value: Scalar, op: Binary) -> "DenseArray":
        data = op(self.data, float(value))
        mask = None if self.mask is None else self.mask.copy()
        return self._like(np.where(self.present, data, self.data), mask)

    def _additive(self, other: Operand, op: Binary) -> Result:
        """A sum or difference: absence is the identity, or it propagates."""
        if isinstance(other, (int, float, np.number)):
            return self._scalar(other, op)
        if isinstance(other, SparseArray):
            return op(self._sparse(), other)
        if self.dims != other.dims:
            left, right = self._conform(other)
            return left._additive(right, op)
        self._same_frame(other)
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
        """A number divided by every value this array carries.

        A stored zero divides to infinity, which is what the arithmetic
        answers; an absent coordinate has no value and stays absent.
        """
        if not isinstance(other, (int, float, np.number)):
            return NotImplemented
        with np.errstate(divide="ignore", invalid="ignore"):
            return self._scalar(other, lambda a, b: np.true_divide(b, a))

    def __neg__(self) -> "DenseArray":
        return self._like(-self.data, None if self.mask is None else self.mask.copy())

    def __pow__(self, other: Scalar) -> "DenseArray":
        """Every value raised to a number.

        An absent coordinate stays absent, as it does under a scalar product:
        it carries no value to raise.
        """
        if not isinstance(other, (int, float, np.number)):
            return NotImplemented
        with np.errstate(divide="ignore", invalid="ignore"):
            return self._scalar(other, np.power)

    def __mul__(self, other: Operand) -> Result | tuple[str, ...]:
        """A product: an absent operand takes the coordinate out either way.

        Frames that differ multiply the way `SparseArray` multiplies them: one
        nesting inside the other broadcasts over the wider, and frames that
        share some dimensions align on those and multiply the rest out.
        """
        if isinstance(other, (int, float, np.number)):
            return self._scalar(other, np.multiply)
        if isinstance(other, SparseArray):
            return self._sparse() * other
        if self.dims == other.dims:
            self._same_frame(other)
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
        """This array's values and presence, shaped to broadcast over `dims`.

        The dimensions are moved into the order `dims` names them in and the
        ones this array does not carry enter as extents of one, which is what
        lets numpy replicate a narrower operand across a wider frame without
        materialising the replication.
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
        """The product of two operands read over `dims`.

        A coordinate either operand does not carry has no factor and does not
        survive, which is what an intersection means over a wider frame.
        """
        mine, mine_present = self._spread_to(dims)
        theirs, theirs_present = other._spread_to(dims)
        data, mask = self._tagged(
            mine * theirs, mine_present & theirs_present, self.absence
        )
        return DenseArray(data, coords, dims, self.absence, mask)

    def _shared_frame(self, other: "DenseArray", shared: Iterable[str]) -> None:
        """Refuse operands that disagree on absence or on a shared label."""
        if self.absence != other.absence:
            raise ValueError(
                f"one array declares absence {self.absence!r} and the other "
                f"{other.absence!r}; state which the result carries with "
                f"as_empty() or as_unknown()"
            )
        differing = [d for d in shared if self.coords[d] != other.coords[d]]
        if differing:
            raise ValueError(
                f"shared dimension(s) {differing} carry different labels in "
                f"the two operands; an entry is aligned by its label"
            )

    def _broadcast_mul(self, other: "DenseArray") -> "DenseArray":
        """The product of two arrays whose dimensions nest, over the wider frame.

        The operand carrying fewer dimensions supplies a factor for every
        coordinate of the wider one sharing its own, so the result is over the
        wider frame however the operands were ordered.
        """
        narrow, wide = (self, other)
        if len(narrow.dims) > len(wide.dims):
            narrow, wide = wide, narrow
        if not set(narrow.dims) <= set(wide.dims):
            raise ValueError(
                f"dimensions {narrow.dims} are not a subset of {wide.dims}; a "
                f"broadcast product needs one frame to nest inside the other"
            )
        narrow._shared_frame(wide, narrow.dims)
        shared_shape = tuple(wide.shape[wide.dims.index(d)] for d in narrow.dims)
        if shared_shape != narrow.shape:
            raise ValueError(
                f"shared dimensions {narrow.dims} have size {narrow.shape} in "
                f"one operand and {shared_shape} in the other"
            )
        return wide._product_over(narrow, wide.dims, wide.coords)

    def _overlap_mul(self, other: "DenseArray") -> "DenseArray":
        """The product of two arrays whose frames share some dimensions.

        The shared dimensions align and the rest multiply out. The result
        carries this array's dimensions followed by the dimensions only the
        other carries, which is the order `SparseArray` answers with.
        """
        shared = tuple(d for d in self.dims if d in other.dims)
        extra = tuple(d for d in other.dims if d not in self.dims)
        self._shared_frame(other, shared)
        dims = self.dims + extra
        coords = dict(self.coords)
        coords.update({d: other.coords[d] for d in extra})
        return self._product_over(other, dims, coords)

    def __truediv__(self, other: Operand) -> Result:
        """A quotient: an absent denominator is refused, not treated as zero."""
        if isinstance(other, (int, float, np.number)):
            return self._scalar(other, np.true_divide)
        if isinstance(other, SparseArray):
            return self._sparse() / other
        if self.dims != other.dims:
            left, right = self._conform(other)
            return left / right
        self._same_frame(other)
        wanted = self.present & ~other.present
        if wanted.any():
            raise ValueError(
                f"the denominator is absent at {int(wanted.sum())} coordinate(s) "
                f"the numerator carries; a quotient there is not zero and not "
                f"one, so it is refused"
            )
        with np.errstate(invalid="ignore", divide="ignore"):
            data = self.data / other.data
        if self.absence == "unknown":
            return self._like(data)
        return self._like(np.where(self.present, data, 0.0), self.present.copy())

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

    _IDENTITY = {"sum": 0.0, "mean": 0.0, "min": np.inf, "max": -np.inf}

    def _reduce(
        self, dim: str | None, op: str, skip: bool | None, fill: float | None
    ) -> "DenseArray | float":
        self._policy(skip, fill)
        present = self.present
        if fill is None:
            filled = np.where(present, self.data, self._IDENTITY[op])
        else:
            filled = np.where(present, self.data, float(fill))
            present = np.ones(self.shape, dtype=bool)

        if dim is None:
            if op == "mean":
                counted = int(present.sum())
                return float(filled.sum() / counted) if counted else float("nan")
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
        """Total over `dim`, or over the whole array when `dim` is None."""
        return self._reduce(dim, "sum", skip, fill)

    def mean(
        self,
        dim: str | None = None,
        skip: bool | None = None,
        fill: float | None = None,
    ) -> "DenseArray | float":
        """Mean over `dim`, or over the whole array when `dim` is None."""
        return self._reduce(dim, "mean", skip, fill)

    def min(
        self,
        dim: str | None = None,
        skip: bool | None = None,
        fill: float | None = None,
    ) -> "DenseArray | float":
        """Minimum over `dim`, or over the whole array when `dim` is None."""
        return self._reduce(dim, "min", skip, fill)

    def max(
        self,
        dim: str | None = None,
        skip: bool | None = None,
        fill: float | None = None,
    ) -> "DenseArray | float":
        """Maximum over `dim`, or over the whole array when `dim` is None."""
        return self._reduce(dim, "max", skip, fill)
