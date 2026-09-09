"""What a dimension's positions are named by.

A coordinate answers both directions of the label question. A stored one
holds an array of labels; a generated one computes the answer, so a
dimension spanning millions of positions costs nothing to carry.

Two coordinates are equal when they name the same positions, which is what
lets an operation refuse operands whose labels differ.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from nimblend import kernel
from nimblend.kernel import Positions

type Labels = npt.NDArray[Any]


class StoredCoord:
    """Labels held as an array."""

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
        """The sorted labels and their positions, built on the first lookup.

        A coordinate carried only to name a dimension's extent is never asked
        where a label sits, so the permutation and the sorted copy it needs —
        together twice the labels themselves — are built when one is.
        """
        order, sorted_labels = self._order, self._sorted
        if order is None or sorted_labels is None:
            order = np.argsort(self.labels, kind="stable")
            sorted_labels = self.labels[order]
            self._order, self._sorted = order, sorted_labels
        return order, sorted_labels

    def to_position(self, labels: npt.ArrayLike) -> Positions:
        """Positions the given labels occupy."""
        labels = np.asarray(labels)
        order, sorted_labels = self._lookup_order()
        at = np.searchsorted(sorted_labels, labels)
        probe = np.minimum(at, sorted_labels.size - 1)
        miss = sorted_labels[probe] != labels
        if miss.any():
            raise KeyError(f"label {labels[miss][0]!r} is not carried")
        return order[at]

    def to_index(self, positions: Positions) -> Labels:
        """Labels at the given positions."""
        return self.labels[positions]


class ProductCoord:
    """Positions of a full product of axis sizes, numbered from `start`."""

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
        """Positions the given index matrix occupies."""
        return kernel.ravel(index, self.sizes) + self.start

    def to_index(self, positions: Positions) -> kernel.Index:
        """The index matrix the given positions stand for."""
        return kernel.unravel(np.asarray(positions) - self.start, self.sizes)


class SubsetCoord:
    """Positions of a subset of a product, numbered from `start` in code order.

    An entry's position is its rank among the codes, so a block already in
    canonical order needs no lookup at all.
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
        """Positions the given index matrix occupies."""
        keys = kernel.ravel(index, self.sizes)
        at = np.searchsorted(self.codes, keys)
        probe = np.minimum(at, self.codes.size - 1)
        miss = self.codes[probe] != keys
        if miss.any():
            raise KeyError(
                f"cell {tuple(int(v) for v in index[:, np.flatnonzero(miss)[0]])} "
                f"is not carried by this subset"
            )
        return at + self.start

    def to_index(self, positions: Positions) -> kernel.Index:
        """The index matrix the given positions stand for."""
        return kernel.unravel(
            self.codes[np.asarray(positions) - self.start], self.sizes
        )


type Coord = StoredCoord | ProductCoord | SubsetCoord
