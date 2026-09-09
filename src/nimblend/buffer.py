"""One preallocated COO destination that blocks are computed into."""

import numpy as np

from nimblend.coords import Coord
from nimblend.kernel import Block, Index, Values
from nimblend.sparse import SparseArray


class EntryBuffer:
    """A fixed index and value buffer handing out successive slices.

    A block computed into a reserved slice never exists as a separate object,
    so assembling several of them holds one copy of the result rather than
    one copy per block plus the result.
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
        """A destination `(index, data)` for `n` entries, advancing the cursor."""
        n = int(n)
        if self.at + n > self.capacity:
            raise ValueError(
                f"reserving {n} entries at {self.at} exceeds the capacity "
                f"{self.capacity}"
            )
        dest = (self.index[:, self.at : self.at + n], self.data[self.at : self.at + n])
        self.at += n
        return dest

    def written(self) -> tuple[Index, Values]:
        """Views of the entries reserved so far."""
        return self.index[:, : self.at], self.data[: self.at]

    def array(
        self,
        coords: dict[str, Coord],
        dims: tuple[str, ...],
        absence: str = "empty",
    ) -> SparseArray:
        """The written prefix as an array, taking no copy."""
        index, data = self.written()
        return SparseArray.from_canonical(index, data, coords, dims, absence)
