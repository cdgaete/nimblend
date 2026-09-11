"""A preallocated COO buffer that blocks are written into."""

import numpy as np

from nimblend.coords import Coord
from nimblend.kernel import Block, Index, Values
from nimblend.sparse import SparseArray


class EntryBuffer:
    """A fixed index and value buffer, reserved in successive slices.

    `reserve` returns views into the buffer. A method given such a view as
    `out`, for example `SparseArray.group`, writes its entries into the view.
    """

    def __init__(self, ndim: int, capacity: int) -> None:
        self.index = np.empty((ndim, capacity), dtype=np.int32)
        self.data = np.empty(capacity, dtype=np.float64)
        self.capacity = int(capacity)
        self.at = 0

    def __repr__(self) -> str:
        return (
            f"EntryBuffer(ndim={self.index.shape[0]}, "
            f"capacity={self.capacity}, at={self.at})"
        )

    def reserve(self, n: int) -> Block:
        """Return views `(index, data)` for the next `n` entries.

        The cursor advances by `n`. Raises ValueError when the reservation
        exceeds the capacity.
        """
        n = int(n)
        if self.at + n > self.capacity:
            raise ValueError(
                f"reserving {n} entries at position {self.at} exceeds the "
                f"capacity {self.capacity}; reserve at most "
                f"{self.capacity - self.at} entries"
            )
        dest = (self.index[:, self.at : self.at + n], self.data[self.at : self.at + n])
        self.at += n
        return dest

    def written(self) -> tuple[Index, Values]:
        """Return views of the reserved entries."""
        return self.index[:, : self.at], self.data[: self.at]

    def array(
        self,
        coords: dict[str, Coord],
        dims: tuple[str, ...],
        absence: str = "empty",
    ) -> SparseArray:
        """Return the reserved entries as a `SparseArray`, without a copy.

        The entries must be in canonical order; `is_canonical` checks this.
        """
        index, data = self.written()
        return SparseArray.from_canonical(index, data, coords, dims, absence)
