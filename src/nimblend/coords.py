"""The mapping between the labels and the positions of a dimension.

`StoredCoord` stores its labels in an array. `ProductCoord` and `SubsetCoord`
store no labels. Their labels are multi-indices, computed on each lookup.
Two coordinates are equal when they map the same labels to the same
positions.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from nimblend import kernel
from nimblend.kernel import Positions

type Labels = npt.NDArray[Any]


def python_value(value: Any) -> Any:
    """Return a numpy scalar as the equivalent Python value, for a message."""
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
