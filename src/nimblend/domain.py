"""The coordinates an array carries over a tuple of its dimensions."""

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from nimblend import display, kernel
from nimblend.coords import Coord, SubsetCoord
from nimblend.kernel import Index, Keys, Positions

if TYPE_CHECKING:
    from nimblend.sparse import SparseArray


class Domain:
    """A sorted, unique set of multi-indices over named dimensions.

    A coordinate answers the label question for one dimension; a domain
    answers it for a tuple of them: which multi-indices are carried, what
    position each occupies, and which multi-index sits at a position. It
    states which coordinates are carried and nothing about where they are
    numbered from.

    `dims`, `shape`, `coords` and `size` are the frame a caller reads. The
    members themselves are held as `codes`, the ravelled keys of the layer
    below, and each question a caller asks of them has a reader above them:
    `coordinates` and `labels` say which are carried,
    `positions_of_coordinates` says where one sits, `as_coord` numbers them,
    and `array` and `identity` answer with an array over them.
    """

    def __init__(
        self,
        codes: npt.ArrayLike,
        dims: Iterable[str],
        coords: Mapping[str, Coord],
        shape: Iterable[int],
    ) -> None:
        self._set_frame(dims, coords, shape)
        codes = np.asarray(codes, dtype=np.int64)
        if codes.size > 1:
            step = np.diff(codes)
            if not bool(np.all(step > 0)):
                at = int(np.flatnonzero(step <= 0)[0])
                raise ValueError(
                    f"the codes of a domain ascend without repeating; "
                    f"{int(codes[at])} at position {at} is followed by "
                    f"{int(codes[at + 1])}"
                )
        self.codes = codes

    def _set_frame(
        self, dims: Iterable[str], coords: Mapping[str, Coord], shape: Iterable[int]
    ) -> None:
        """Take the dimensions, their extents and their coordinates."""
        self.dims = tuple(dims)
        self.shape = tuple(int(s) for s in shape)
        if len(self.dims) != len(self.shape):
            raise ValueError(
                f"dimensions {self.dims} and shape {self.shape} name a "
                f"different number of axes"
            )
        missing = [d for d in self.dims if d not in coords]
        if missing:
            raise ValueError(f"no coordinate for dimension(s) {missing}")
        self.coords = {d: coords[d] for d in self.dims}

    @classmethod
    def _over(
        cls,
        codes: Keys,
        dims: Iterable[str],
        coords: Mapping[str, Coord],
        shape: Iterable[int],
    ) -> "Domain":
        """A domain over codes this class generated, taking no copy.

        The ascending-without-repeat check the constructor runs costs an
        array the size of the codes. A range, and a subsequence of codes
        already checked, are ascending by construction, so the domains built
        from them skip it.
        """
        self = cls.__new__(cls)
        self._set_frame(dims, coords, shape)
        self.codes = codes
        return self

    @classmethod
    def from_coordinates(
        cls, dims: Iterable[str], coords: Mapping[str, Coord], index: npt.ArrayLike
    ) -> "Domain":
        """A domain from an index matrix of one row per dimension.

        Each column names one member. A caller holding positions states them
        directly rather than resolving labels it would only resolve back.
        """
        dims = tuple(dims)
        if not dims:
            raise ValueError("a domain is over at least one dimension")
        missing = [d for d in dims if d not in coords]
        if missing:
            raise ValueError(f"no coordinate for dimension(s) {missing}")
        index = np.asarray(index, dtype=np.int32)
        if index.ndim != 2 or index.shape[0] != len(dims):
            raise ValueError(
                f"an index matrix over {dims} has {len(dims)} rows; got shape "
                f"{index.shape}"
            )
        shape = tuple(len(coords[d]) for d in dims)
        keys = kernel.ravel(index, shape)
        codes = kernel.distinct(keys)
        if codes.size != keys.size:
            at = kernel.first_repeat(keys)
            member = tuple(int(index[axis, at]) for axis in range(len(dims)))
            raise ValueError(
                f"member {member} is named twice; a domain names each member once"
            )
        return cls(codes, dims, coords, shape)

    @classmethod
    def from_labels(
        cls,
        dims: Iterable[str],
        coords: Mapping[str, Coord],
        labels: Mapping[str, npt.ArrayLike],
    ) -> "Domain":
        """A domain from one label column per dimension.

        Each column resolves through its dimension's own coordinate, so a
        member is stated in the labels a caller holds rather than in codes.
        The columns name the same members and are read in step, so they have
        equal length. A member named twice raises: a coordinate set built
        from labels holds a label coordinate's contract, where a repeat is
        the caller's error.
        """
        dims = tuple(dims)
        if not dims:
            raise ValueError("a domain is over at least one dimension")
        missing = [d for d in dims if d not in coords]
        if missing:
            raise ValueError(f"no coordinate for dimension(s) {missing}")
        absent = [d for d in dims if d not in labels]
        if absent:
            raise ValueError(f"no label column for dimension(s) {absent}")
        columns = {d: np.asarray(labels[d]) for d in dims}
        lengths = {d: int(column.size) for d, column in columns.items()}
        if len(set(lengths.values())) > 1:
            raise ValueError(
                f"label columns have differing length {lengths}; each names "
                f"the same members"
            )
        n = lengths[dims[0]]
        shape = tuple(len(coords[d]) for d in dims)
        rows = []
        for d in dims:
            at = np.asarray(coords[d].to_position(columns[d]))
            if at.shape != (n,):
                raise ValueError(
                    f"the coordinate of dimension {d!r} answered "
                    f"{at.shape} for {n} label(s); a domain built from "
                    f"labels reads one position per label"
                )
            rows.append(at.astype(np.int32))
        keys = kernel.ravel(np.stack(rows), shape)
        codes = kernel.distinct(keys)
        if codes.size != keys.size:
            at = kernel.first_repeat(keys)
            member = tuple(columns[d][at] for d in dims)
            raise ValueError(
                f"member {member} is named twice; a domain built from labels "
                f"names each member once"
            )
        return cls(codes, dims, coords, shape)

    @classmethod
    def full(cls, dims: Iterable[str], coords: Mapping[str, Coord]) -> "Domain":
        """Every coordinate of the product `dims` spans.

        The domain an array over a full product carries, named without
        enumerating it: a caller holding the coordinates need not build the
        index matrix to say which members exist.
        """
        dims = tuple(dims)
        missing = [d for d in dims if d not in coords]
        if missing:
            raise ValueError(f"no coordinate for dimension(s) {missing}")
        shape = tuple(len(coords[d]) for d in dims)
        total = 1
        for size in shape:
            total *= size
        return cls._over(np.arange(total, dtype=np.int64), dims, coords, shape)

    @property
    def size(self) -> int:
        """Number of members."""
        return int(self.codes.size)

    @property
    def is_full(self) -> bool:
        """Whether every coordinate of the product the dimensions span is carried.

        The question a caller asks before reading the members as a grid: a
        full domain has a member at every cell of its shape, so values
        ordered by member reshape into it.
        """
        span = 1
        for size in self.shape:
            span *= size
        return self.size == span

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return display.one_line("Domain", self.dims, self.shape, size=self.size)

    def _repr_html_(self) -> str:
        """The head of the members carried, as a table of labels."""
        n = min(self.size, display.HEAD)
        index = kernel.unravel(self.codes[:n], self.shape)
        columns = [
            display.as_text(self.coords[name].to_index(index[axis]))
            for axis, name in enumerate(self.dims)
        ]
        return display.table(
            "Domain",
            self.dims,
            self.shape,
            self.dims,
            list(zip(*columns, strict=True)),
            self.size,
            size=self.size,
        )

    def coordinates(self) -> Index:
        """The multi-index of each member, as a fresh int32 index matrix.

        One row per dimension and one column per member, in the order the
        members are carried.
        """
        return kernel.unravel(self.codes, self.shape)

    def labels(self) -> dict[str, npt.NDArray[Any]]:
        """Each member's label, per dimension."""
        index = self.coordinates()
        return {
            name: self.coords[name].to_index(index[axis])
            for axis, name in enumerate(self.dims)
        }

    def _same_frame(self, other: "Domain") -> None:
        if self.dims != other.dims or self.shape != other.shape:
            raise ValueError(
                f"a domain over {self.dims} of shape {self.shape} and one "
                f"over {other.dims} of shape {other.shape} do not describe "
                f"the same coordinates"
            )
        differing = [d for d in self.dims if self.coords[d] != other.coords[d]]
        if differing:
            raise ValueError(
                f"dimension(s) {differing} carry different labels in the two "
                f"domains; a member is named by its label"
            )

    def intersect(self, other: "Domain") -> "Domain":
        """The members both domains carry."""
        self._same_frame(other)
        codes, _, _ = kernel.align(self.codes, other.codes, "intersect")
        return Domain(codes, self.dims, self.coords, self.shape)

    def union(self, other: "Domain") -> "Domain":
        """The members either domain carries."""
        self._same_frame(other)
        codes, _, _ = kernel.align(self.codes, other.codes, "union")
        return Domain(codes, self.dims, self.coords, self.shape)

    def difference(self, other: "Domain") -> "Domain":
        """The members this domain carries and `other` does not."""
        self._same_frame(other)
        at = kernel.lookup(other.codes, self.codes)
        return Domain._over(self.codes[at < 0], self.dims, self.coords, self.shape)

    def positions_of(self, array: "SparseArray") -> Positions:
        """Each entry of `array` as its position here, -1 where absent."""
        axes = [array.dims.index(name) for name in self.dims]
        sizes = tuple(array.shape[axis] for axis in axes)
        if sizes != self.shape:
            raise ValueError(
                f"dimensions {self.dims} have shape {sizes} in the array and "
                f"{self.shape} in this domain"
            )
        if axes == list(range(len(axes))):
            index = array.index[: len(axes)]
        else:
            index = array.index[axes]
        return kernel.lookup(self.codes, kernel.ravel(index, self.shape))

    def positions_of_coordinates(self, index: npt.ArrayLike) -> Positions:
        """Each column of `index` as its position here, -1 where absent.

        The partner of `positions_of`, for a caller holding an index matrix
        rather than an array: the rows are read in this domain's own
        dimension order, and a column the domain does not carry answers -1.
        A caller asking only whether a member is carried reads the answer as
        `>= 0` rather than building an array to ask with.
        """
        index = np.asarray(index, dtype=np.int32)
        if index.ndim != 2 or index.shape[0] != len(self.dims):
            raise ValueError(
                f"an index matrix over {self.dims} has {len(self.dims)} rows; "
                f"got shape {index.shape}"
            )
        return kernel.lookup(self.codes, kernel.ravel(index, self.shape))

    def expand(self, dims: Iterable[str], coords: Mapping[str, Coord]) -> "Domain":
        """Every member crossed with the full extent of the named dimensions.

        The new dimensions are appended, which keeps the members ascending;
        a different order is reached with `transpose`. Adding a dimension of
        size `k` multiplies the member count by `k`, so a caller states the
        replication rather than an operator implying it.

        A member's code is its own scaled by the extent appended, plus each
        position within it, so the cross product is arithmetic on the codes
        and no index matrix is built to hold it.
        """
        dims = tuple(dims)
        clash = [name for name in dims if name in self.dims]
        if clash:
            raise ValueError(f"dimension(s) {clash} are already carried")
        missing = [name for name in dims if name not in coords]
        if missing:
            raise ValueError(f"no coordinate for dimension(s) {missing}")
        shape = self.shape + tuple(int(len(coords[name])) for name in dims)
        span = 1
        for size in shape:
            span *= size
        if span > np.iinfo(np.int64).max:
            raise OverflowError(
                f"shape {shape} exceeds the int64 range a ravelled index key holds"
            )
        total = 1
        for name in dims:
            total *= int(len(coords[name]))
        held = dict(self.coords)
        held.update({name: coords[name] for name in dims})
        codes = (
            self.codes[:, None] * total + np.arange(total, dtype=np.int64)
        ).reshape(-1)
        return Domain._over(codes, self.dims + dims, held, shape)

    def transpose(self, *dims: str) -> "Domain":
        """The same members, over the dimensions in the order given.

        A permutation of the axes is a bijection on the members, so sorting
        the codes it produces leaves them ascending without repeating and the
        constructor's check is skipped.
        """
        dims = tuple(dims)
        if sorted(dims) != sorted(self.dims):
            raise ValueError(
                f"a transpose names the dimensions {self.dims} once each; got {dims}"
            )
        if dims == self.dims:
            return self
        axes = [self.dims.index(name) for name in dims]
        shape = tuple(self.shape[axis] for axis in axes)
        keys = kernel.ravel(self.coordinates()[axes], shape)
        keys.sort()
        return Domain._over(keys, dims, self.coords, shape)

    def as_coord(self, start: int = 0) -> SubsetCoord:
        """This domain read as a coordinate, its members numbered from `start`.

        A member's position is its rank among the members carried, so a
        domain states the numbering of a dimension it spans without the
        caller reaching for how the members are held.
        """
        return SubsetCoord(self.codes, self.shape, start)

    def array(self, values: npt.ArrayLike, absence: str = "empty") -> "SparseArray":
        """The members of this domain, valued by `values`.

        One value per member, read in the order the members are carried, so a
        caller holding a vector ordered by member states an array without
        building an index for it. The members ascend, so the entries are
        canonical as written, and the array shares the value buffer it is
        handed rather than copying it.
        """
        from nimblend.sparse import SparseArray

        values = np.asarray(values, dtype=np.float64)
        if values.shape != (self.size,):
            raise ValueError(
                f"a domain of {self.size} member(s) takes one value each, as "
                f"a column of that length; got shape {values.shape}"
            )
        return SparseArray.from_canonical(
            self.coordinates(), values, self.coords, self.dims, absence
        )

    def identity(
        self,
        into: str,
        coord: Coord,
        start: int = 0,
        absence: str = "empty",
    ) -> "SparseArray":
        """Each member paired with its own position along `into`, valued 1.0.

        A member's position is its rank here plus `start`, which is the
        numbering `as_coord(start)` states, so an array built here and a
        coordinate built there place a member alike. `coord` is the
        coordinate of `into` and spans the whole extent the positions are
        numbered into, which is wider than these members where several
        domains share one numbering.

        The new dimension is appended and the members ascend, so the entries
        are canonical as written.
        """
        from nimblend.sparse import SparseArray

        if into in self.dims:
            raise ValueError(f"dimension {into!r} is already carried")
        start = int(start)
        if start < 0:
            raise ValueError(
                f"start {start} numbers the first member below zero; a "
                f"position along {into!r} is not negative"
            )
        extent = len(coord)
        if start + self.size > extent:
            raise ValueError(
                f"{self.size} member(s) numbered from {start} reach position "
                f"{start + self.size - 1}, and dimension {into!r} spans {extent}"
            )
        index = np.empty((len(self.dims) + 1, self.size), dtype=np.int32)
        index[:-1] = self.coordinates()
        index[-1] = np.arange(start, start + self.size, dtype=np.int32)
        held = dict(self.coords)
        held[into] = coord
        return SparseArray.from_canonical(
            index,
            np.ones(self.size, dtype=np.float64),
            held,
            self.dims + (into,),
            absence,
        )
