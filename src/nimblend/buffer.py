"""A preallocated COO buffer that blocks are written into."""

import numpy as np

from nimblend.coords import Coord
from nimblend.kernel import Block, Index, Values
from nimblend.sparse import SparseArray


class EntryBuffer:
    """An index and value buffer, reserved in successive slices.

    `reserve` returns views into the buffers. A method given such a view as
    `out`, for example `SparseArray.group`, writes its entries into the view.
    With `resizable=True` a reservation beyond the capacity allocates larger
    buffers and copies the reserved entries into them. A view returned before
    such an allocation refers to the buffers before it.
    """

    def __init__(self, ndim: int, capacity: int, resizable: bool = False) -> None:
        self.ndim = int(ndim)
        self.index = np.empty((self.ndim, capacity), dtype=np.int32)
        self.data = np.empty(capacity, dtype=np.float64)
        self.capacity = int(capacity)
        self.resizable = bool(resizable)
        self.at = 0

    def __repr__(self) -> str:
        return f"EntryBuffer(ndim={self.ndim}, capacity={self.capacity}, at={self.at})"

    def _allocate(self, n: int) -> None:
        capacity = max(self.at + n, 2 * self.capacity)
        index = np.empty((self.ndim, capacity), dtype=np.int32)
        data = np.empty(capacity, dtype=np.float64)
        index[:, : self.at] = self.index[:, : self.at]
        data[: self.at] = self.data[: self.at]
        self.index = index
        self.data = data
        self.capacity = capacity

    def reserve(self, n: int) -> Block:
        """Return views `(index, data)` for the next `n` entries.

        The cursor advances by `n`. With `resizable=True` a reservation beyond
        the capacity allocates buffers of `max(at + n, 2 * capacity)` entries
        and copies the reserved entries into them. With `resizable=False` a
        reservation beyond the capacity raises ValueError.
        """
        n = int(n)
        if self.at + n > self.capacity:
            if not self.resizable:
                raise ValueError(
                    f"reserving {n} entries at position {self.at} exceeds the "
                    f"capacity {self.capacity}; reserve at most "
                    f"{self.capacity - self.at} entries"
                )
            self._allocate(n)
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
