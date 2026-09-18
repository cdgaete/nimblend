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

    def to_position(self, index: npt.NDArray[Any]) -> kernel.Keys:
        """Return the position of each column of an index matrix.

        Raises KeyError for a cell the subset does not contain.
        """
        keys = kernel.ravel(index, self.sizes)
        at = np.searchsorted(self.codes, keys)
        probe = np.minimum(at, self.codes.size - 1)
        miss = self.codes[probe] != keys
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
    dims: Iterable[str], left: Mapping[str, Coord], right: Mapping[str, Coord]
) -> None:
    """Raise ValueError for a dimension of `dims` with unequal coordinates.

    `left` and `right` map each dimension of `dims` to a coordinate.
    """
    differing = [d for d in dims if left[d] != right[d]]
    if differing:
        raise ValueError(
            f"dimension(s) {differing} have different labels in the two arrays; "
            f"conform one to the other first"
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
