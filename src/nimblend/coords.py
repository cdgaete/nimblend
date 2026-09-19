"""The mapping between the labels and the positions of a dimension.

`StoredCoord` stores its labels in an array. `ProductCoord` and `SubsetCoord`
store no labels. Their labels are multi-indices, computed on each lookup.
Two coordinates are equal when they map the same labels to the same
positions.
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from nimblend import kernel
from nimblend.kernel import Positions

type Labels = npt.NDArray[Any]


def unique_dims(dims: Iterable[str]) -> tuple[str, ...]:
    """Return `dims` as a tuple.

    Raises ValueError for a dimension that appears more than once.
    """
    dims = tuple(dims)
    repeated = list(dict.fromkeys(d for d in dims if dims.count(d) > 1))
    if repeated:
        raise ValueError(
            f"dimension(s) {repeated} appear more than once in {dims}; pass each "
            f"dimension once"
        )
    return dims


def known_dims(what: str, requested: Iterable[Any], dims: tuple[str, ...]) -> None:
    """Raise ValueError for a requested dimension that `dims` does not contain."""
    missing = [d for d in requested if d not in dims]
    if missing:
        raise ValueError(
            f"{what} dimension(s) {missing} are not in the array over {dims}; "
            f"pass dimensions of the array"
        )


def python_value(value: Any) -> Any:
    """Return a numpy scalar as the equivalent Python value, for a message.

    A datetime64 or timedelta64 scalar is returned as its string form.
    """
    if isinstance(value, (np.datetime64, np.timedelta64)):
        return str(value)
    return value.item() if isinstance(value, np.generic) else value


class StoredCoord:
    """A coordinate that stores its labels in an array."""

    def __init__(self, labels: npt.ArrayLike) -> None:
        self.labels = np.asarray(labels)
        self._order = None
        self._sorted = None

    def __len__(self) -> int:
        return int(self.labels.size)

    def __repr__(self) -> str:
        return f"StoredCoord({len(self)} labels)"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StoredCoord):
            return NotImplemented
        return bool(np.array_equal(self.labels, other.labels))

    def _lookup_order(self) -> tuple[Positions, Labels]:
        """Return the sort order and the sorted labels, computed on first use.

        The two arrays are cached on the instance. A coordinate with no label
        lookup does not allocate them.
        """
        order, sorted_labels = self._order, self._sorted
        if order is None or sorted_labels is None:
            order = np.argsort(self.labels, kind="stable")
            sorted_labels = self.labels[order]
            self._order, self._sorted = order, sorted_labels
        return order, sorted_labels

    def to_position(self, labels: npt.ArrayLike) -> Positions:
        """Return the position of each label.

        Raises KeyError for a label the coordinate does not contain.
        """
        labels = np.asarray(labels)
        order, sorted_labels = self._lookup_order()
        at = np.searchsorted(sorted_labels, labels)
        probe = np.minimum(at, sorted_labels.size - 1)
        miss = sorted_labels[probe] != labels
        if miss.any():
            label = python_value(labels[miss][0])
            raise KeyError(
                f"label {label!r} is not in the coordinate; pass only labels "
                f"the coordinate contains"
            )
        return order[at]

    def to_index(self, positions: Positions) -> Labels:
        """Return the label at each position."""
        return self.labels[positions]


class ProductCoord:
    """A coordinate over the full product of axis sizes, numbered from `start`."""

    def __init__(self, sizes: Sequence[int], start: int = 0) -> None:
        self.sizes = tuple(int(s) for s in sizes)
        self.start = int(start)

    def __len__(self) -> int:
        total = 1
        for size in self.sizes:
            total *= size
        return total

    def __repr__(self) -> str:
        return f"ProductCoord(sizes={self.sizes}, start={self.start})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ProductCoord):
            return NotImplemented
        return self.sizes == other.sizes and self.start == other.start

    def to_position(self, index: npt.NDArray[Any]) -> kernel.Keys:
        """Return the position of each column of an index matrix."""
        return kernel.ravel(index, self.sizes) + self.start

    def to_index(self, positions: Positions) -> kernel.Index:
        """Return the index matrix at the given positions."""
        return kernel.unravel(np.asarray(positions) - self.start, self.sizes)


class SubsetCoord:
    """A coordinate over a subset of a product, in code order from `start`.

    `codes` are the raveled keys of the members and must ascend without
    repeats. The position of a member is its rank among the codes plus
    `start`.
    """

    def __init__(
        self, codes: npt.ArrayLike, sizes: Sequence[int], start: int = 0
    ) -> None:
        self.codes = np.asarray(codes, dtype=np.int64)
        self.sizes = tuple(int(s) for s in sizes)
        self.start = int(start)

    def __len__(self) -> int:
        return int(self.codes.size)

    def __repr__(self) -> str:
        return f"SubsetCoord({len(self)} of {self.sizes}, start={self.start})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SubsetCoord):
            return NotImplemented
        return (
            self.sizes == other.sizes
            and self.start == other.start
            and bool(np.array_equal(self.codes, other.codes))
        )

    def to_position(self, index: npt.NDArray[Any]) -> Positions:
        """Return the position of each column of an index matrix.

        Raises KeyError for a cell the subset does not contain.
        """
        at = kernel.lookup(self.codes, kernel.ravel(index, self.sizes))
        miss = at < 0
        if miss.any():
            raise KeyError(
                f"cell {tuple(int(v) for v in index[:, np.flatnonzero(miss)[0]])} "
                f"is not in the subset; pass only cells the subset contains"
            )
        return at + self.start

    def to_index(self, positions: Positions) -> kernel.Index:
        """Return the index matrix at the given positions."""
        return kernel.unravel(
            self.codes[np.asarray(positions) - self.start], self.sizes
        )


type Coord = StoredCoord | ProductCoord | SubsetCoord


def numbered_from(size: int, into: str, coord: Coord, start: int) -> int:
    """Return `start` as an int for `size` positions numbered inside `coord`.

    Raises ValueError for a negative `start`, and for positions from `start`
    that end beyond the extent of `coord`.
    """
    start = int(start)
    if start < 0:
        raise ValueError(f"start {start} is negative; pass a start of 0 or more")
    extent = len(coord)
    if start + size > extent:
        raise ValueError(
            f"{size} member(s) numbered from {start} end at position "
            f"{start + size - 1}, and dimension {into!r} has extent {extent}; "
            f"pass a smaller start or a larger coord"
        )
    return start


def require_coords(dims: Iterable[str], coords: Mapping[str, Coord]) -> None:
    """Raise ValueError for a dimension of `dims` without a coordinate in `coords`."""
    missing = [d for d in dims if d not in coords]
    if missing:
        raise ValueError(
            f"no coordinate for dimension(s) {missing}; pass a coordinate for "
            f"each dimension"
        )


def same_labels(
    dims: Iterable[str],
    left: Mapping[str, Coord],
    right: Mapping[str, Coord],
    operands: str = "arrays",
    action: str = "conform one to the other first",
) -> None:
    """Raise ValueError for a dimension of `dims` with unequal coordinates.

    `left` and `right` map each dimension of `dims` to a coordinate.
    """
    differing = [d for d in dims if left[d] != right[d]]
    if differing:
        raise ValueError(
            f"dimension(s) {differing} have different labels in the two "
            f"{operands}; {action}"
        )


def same_extents(
    dims: tuple[str, ...], left: tuple[int, ...], right: tuple[int, ...]
) -> None:
    """Raise ValueError for two shapes of `dims` that differ."""
    if left != right:
        raise ValueError(
            f"shared dimensions {dims} have size {left} in one operand and "
            f"{right} in the other; conform one to the other first"
        )


def distinct_labels(name: str, labels: Labels, positions: Positions) -> None:
    """Raise ValueError for a label of dimension `name` given twice."""
    at = kernel.first_repeat(positions)
    if at >= 0:
        raise ValueError(
            f"label {python_value(labels[at])!r} appears twice for dimension "
            f"{name!r}; pass each label once"
        )


def label_at(coord: Coord, position: int) -> Any:
    """Return the label at `position` as a Python value, for a message.

    The label of a `ProductCoord` or a `SubsetCoord` is a tuple of ints.
    """
    label = coord.to_index(np.array([position]))
    if label.ndim == 1:
        return python_value(label[0])
    return tuple(int(v) for v in label[:, 0])


def label_positions(
    dims: tuple[str, ...],
    coords: Mapping[str, Coord],
    labels: Mapping[str, npt.ArrayLike],
    length: int | None = None,
) -> kernel.Index:
    """Return the position of each label, one row per dimension of `dims`.

    The coordinate of each dimension converts its label column. A label
    column of a `ProductCoord` or a `SubsetCoord` is an index matrix with one
    column per label. With `length` every column has `length` labels. Raises
    ValueError for a missing coordinate or label column, a column that does
    not convert to one position per label, and columns with different label
    counts. Raises KeyError for a label the coordinate does not contain.
    """
    require_coords(dims, coords)
    absent = [d for d in dims if d not in labels]
    if absent:
        raise ValueError(
            f"no label column for dimension(s) {absent}; pass a label column "
            f"for each dimension"
        )
    rows = []
    for name in dims:
        column = np.asarray(labels[name])
        at = np.asarray(coords[name].to_position(column))
        if at.ndim != 1:
            raise ValueError(
                f"the coordinate of dimension {name!r} converts a label column "
                f"of shape {column.shape} to positions of shape {at.shape}; pass "
                f"one label per position"
            )
        rows.append(at.astype(np.int32))
    lengths = {name: int(at.size) for name, at in zip(dims, rows, strict=True)}
    counts = set(lengths.values())
    if length is None and len(counts) > 1:
        raise ValueError(
            f"label columns have different lengths {lengths}; pass columns of "
            f"equal length"
        )
    if length is not None and counts - {length}:
        raise ValueError(
            f"label columns have lengths {lengths} and the value column has "
            f"length {length}; pass columns of equal length"
        )
    if not rows:
        return np.empty((0, length or 0), dtype=np.int32)
    return np.stack(rows)
