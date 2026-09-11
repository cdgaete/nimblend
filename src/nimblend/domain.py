"""A set of coordinates over a tuple of dimensions."""

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from nimblend import display, kernel
from nimblend.coords import Coord, SubsetCoord, python_value, unique_dims
from nimblend.kernel import Index, Keys, Positions

if TYPE_CHECKING:
    from nimblend.sparse import SparseArray


class Domain:
    """A sorted, unique set of multi-indices over a tuple of dimensions.

    `dims`, `shape` and `coords` define the frame, and `size` is the number of
    members. The members are stored in `codes`, the raveled C-order keys of
    their multi-indices. The constructor raises ValueError for codes that do
    not ascend without repeats.
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
                    f"domain code {int(codes[at])} at position {at} is followed "
                    f"by {int(codes[at + 1])}; pass codes that ascend without "
                    f"repeating"
                )
        self.codes = codes

    def _set_frame(
        self, dims: Iterable[str], coords: Mapping[str, Coord], shape: Iterable[int]
    ) -> None:
        """Set the dimensions, the shape and the coordinates of the frame."""
        self.dims = unique_dims(dims)
        self.shape = tuple(int(s) for s in shape)
        if len(self.dims) != len(self.shape):
            raise ValueError(
                f"dimensions {self.dims} and shape {self.shape} have a "
                f"different number of axes; pass one extent per dimension"
            )
        missing = [d for d in self.dims if d not in coords]
        if missing:
            raise ValueError(
                f"no coordinate for dimension(s) {missing}; pass a coordinate "
                f"for each dimension"
            )
        self.coords = {d: coords[d] for d in self.dims}

    @classmethod
    def _over(
        cls,
        codes: Keys,
        dims: Iterable[str],
        coords: Mapping[str, Coord],
        shape: Iterable[int],
    ) -> "Domain":
        """Return a domain over `codes`, with no check and no copy.

        `codes` must ascend without repeats.
        """
        self = cls.__new__(cls)
        self._set_frame(dims, coords, shape)
        self.codes = codes
        return self

    @classmethod
    def from_coordinates(
        cls, dims: Iterable[str], coords: Mapping[str, Coord], index: npt.ArrayLike
    ) -> "Domain":
        """Return a domain from an index matrix with one row per dimension.

        Each column is one member. Raises ValueError for no dimensions, a
        missing coordinate, an index matrix of the wrong shape and a repeated
        member.
        """
        dims = tuple(dims)
        if not dims:
            raise ValueError("no dimension is given; pass at least one dimension")
        missing = [d for d in dims if d not in coords]
        if missing:
            raise ValueError(
                f"no coordinate for dimension(s) {missing}; pass a coordinate "
                f"for each dimension"
            )
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
                f"member {member} appears twice in the index matrix; pass each "
                f"member once"
            )
        return cls(codes, dims, coords, shape)

    @classmethod
    def from_labels(
        cls,
        dims: Iterable[str],
        coords: Mapping[str, Coord],
        labels: Mapping[str, npt.ArrayLike],
    ) -> "Domain":
        """Return a domain from one label column per dimension.

        Each label column is converted to positions by the coordinate of its
        dimension. All columns have equal length. Raises ValueError for no
        dimensions, a missing coordinate or label column, columns of different
        lengths and a repeated member. Raises KeyError for a label the
        coordinate does not contain.
        """
        dims = tuple(dims)
        if not dims:
            raise ValueError("no dimension is given; pass at least one dimension")
        missing = [d for d in dims if d not in coords]
        if missing:
            raise ValueError(
                f"no coordinate for dimension(s) {missing}; pass a coordinate "
                f"for each dimension"
            )
        absent = [d for d in dims if d not in labels]
        if absent:
            raise ValueError(
                f"no label column for dimension(s) {absent}; pass a label column "
                f"for each dimension"
            )
        columns = {d: np.asarray(labels[d]) for d in dims}
        lengths = {d: int(column.size) for d, column in columns.items()}
        if len(set(lengths.values())) > 1:
            raise ValueError(
                f"label columns have different lengths {lengths}; pass columns "
                f"of equal length"
            )
        n = lengths[dims[0]]
        shape = tuple(len(coords[d]) for d in dims)
        rows = []
        for d in dims:
            at = np.asarray(coords[d].to_position(columns[d]))
            if at.shape != (n,):
                raise ValueError(
                    f"the coordinate of dimension {d!r} returns shape "
                    f"{at.shape} for {n} label(s); pass a coordinate that "
                    f"returns one position per label"
                )
            rows.append(at.astype(np.int32))
        keys = kernel.ravel(np.stack(rows), shape)
        codes = kernel.distinct(keys)
        if codes.size != keys.size:
            at = kernel.first_repeat(keys)
            member = tuple(python_value(columns[d][at]) for d in dims)
            raise ValueError(
                f"member {member} appears twice in the label columns; pass each "
                f"member once"
            )
        return cls(codes, dims, coords, shape)

    @classmethod
    def full(cls, dims: Iterable[str], coords: Mapping[str, Coord]) -> "Domain":
        """Return the domain of every cell in the product of the extents of `dims`.

        Raises ValueError for a dimension without a coordinate.
        """
        dims = tuple(dims)
        missing = [d for d in dims if d not in coords]
        if missing:
            raise ValueError(
                f"no coordinate for dimension(s) {missing}; pass a coordinate "
                f"for each dimension"
            )
        shape = tuple(len(coords[d]) for d in dims)
        total = 1
        for size in shape:
            total *= size
        return cls._over(np.arange(total, dtype=np.int64), dims, coords, shape)

    @property
    def size(self) -> int:
        """Return the number of members."""
        return int(self.codes.size)

    @property
    def is_full(self) -> bool:
        """Return True when the domain contains every cell of its shape.

        For a full domain, values in member order reshape into `shape`.
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
        """Return the first members as an HTML table of labels."""
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
        """Return the multi-index of each member as a new int32 index matrix.

        The matrix has one row per dimension and one column per member, in
        member order.
        """
        return kernel.unravel(self.codes, self.shape)

    def labels(self) -> dict[str, npt.NDArray[Any]]:
        """Return the label of each member, per dimension."""
        index = self.coordinates()
        return {
            name: self.coords[name].to_index(index[axis])
            for axis, name in enumerate(self.dims)
        }

    def _same_frame(self, other: "Domain") -> None:
        if self.dims != other.dims or self.shape != other.shape:
            raise ValueError(
                f"domains over {self.dims} of shape {self.shape} and over "
                f"{other.dims} of shape {other.shape} differ in frame; combine "
                f"domains over the same frame"
            )
        differing = [d for d in self.dims if self.coords[d] != other.coords[d]]
        if differing:
            raise ValueError(
                f"dimension(s) {differing} have different labels in the two "
                f"domains; combine domains over the same labels"
            )

    def intersect(self, other: "Domain") -> "Domain":
        """Return the members that both domains contain.

        Raises ValueError for domains with different frames or labels.
        """
        self._same_frame(other)
        codes, _, _ = kernel.align(self.codes, other.codes, "intersect")
        return Domain(codes, self.dims, self.coords, self.shape)

    def union(self, other: "Domain") -> "Domain":
        """Return the members that either domain contains.

        Raises ValueError for domains with different frames or labels.
        """
        self._same_frame(other)
        codes, _, _ = kernel.align(self.codes, other.codes, "union")
        return Domain(codes, self.dims, self.coords, self.shape)

    def difference(self, other: "Domain") -> "Domain":
        """Return the members of this domain that `other` does not contain.

        Raises ValueError for domains with different frames or labels.
        """
        self._same_frame(other)
        at = kernel.lookup(other.codes, self.codes)
        return Domain._over(self.codes[at < 0], self.dims, self.coords, self.shape)

    def positions_of(self, array: "SparseArray") -> Positions:
        """Return the position of each entry of `array` here, -1 where absent.

        `array` must have every dimension of the domain. Raises ValueError for
        dimensions with different extents in the array and the domain.
        """
        axes = [array.dims.index(name) for name in self.dims]
        sizes = tuple(array.shape[axis] for axis in axes)
        if sizes != self.shape:
            raise ValueError(
                f"dimensions {self.dims} have shape {sizes} in the array and "
                f"{self.shape} in this domain; pass an array over the "
                f"coordinates of the domain"
            )
        if axes == list(range(len(axes))):
            index = array.index[: len(axes)]
        else:
            index = array.index[axes]
        return kernel.lookup(self.codes, kernel.ravel(index, self.shape))

    def positions_of_coordinates(self, index: npt.ArrayLike) -> Positions:
        """Return the position of each column of `index` here, -1 where absent.

        The rows of `index` are in the dimension order of the domain. A column
        with a position of 0 or more is a member. Raises ValueError for an
        index matrix with a different number of rows.
        """
        index = np.asarray(index, dtype=np.int32)
        if index.ndim != 2 or index.shape[0] != len(self.dims):
            raise ValueError(
                f"an index matrix over {self.dims} has {len(self.dims)} rows; "
                f"got shape {index.shape}"
            )
        return kernel.lookup(self.codes, kernel.ravel(index, self.shape))

    def expand(self, dims: Iterable[str], coords: Mapping[str, Coord]) -> "Domain":
        """Return every member crossed with the full extent of `dims`.

        The new dimensions are appended; `transpose` reorders them. A dimension
        of extent `k` multiplies the member count by `k`. Raises ValueError for
        a dimension the domain already has or one without a coordinate, and
        for a repeated dimension. Raises OverflowError when the product of the
        extents exceeds the int64 range.
        """
        dims = unique_dims(dims)
        clash = [name for name in dims if name in self.dims]
        if clash:
            raise ValueError(
                f"the domain already has dimension(s) {clash}; pass dimensions "
                f"it does not have"
            )
        missing = [name for name in dims if name not in coords]
        if missing:
            raise ValueError(
                f"no coordinate for dimension(s) {missing}; pass a coordinate "
                f"for each dimension"
            )
        shape = self.shape + tuple(int(len(coords[name])) for name in dims)
        span = 1
        for size in shape:
            span *= size
        if span > np.iinfo(np.int64).max:
            raise OverflowError(
                f"shape {shape} exceeds the int64 range of a raveled index key; "
                f"reduce the number or the extent of the dimensions"
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
        """Return the same members over the dimensions in the given order.

        Raises ValueError unless `dims` contains each dimension once.
        """
        dims = tuple(dims)
        if Counter(dims) != Counter(self.dims):
            raise ValueError(
                f"transpose requires each dimension of {self.dims} once; got {dims}"
            )
        if dims == self.dims:
            return self
        axes = [self.dims.index(name) for name in dims]
        shape = tuple(self.shape[axis] for axis in axes)
        keys = kernel.ravel(self.coordinates()[axes], shape)
        keys.sort()
        return Domain._over(keys, dims, self.coords, shape)

    def as_coord(self, start: int = 0) -> SubsetCoord:
        """Return this domain as a `SubsetCoord`, its members numbered from `start`.

        The position of a member is its rank in the domain plus `start`.
        """
        return SubsetCoord(self.codes, self.shape, start)

    def array(self, values: npt.ArrayLike, absence: str = "empty") -> "SparseArray":
        """Return an array over the members of this domain, valued by `values`.

        `values` has one value per member, in member order. A float64 `values`
        array is used without a copy. Raises ValueError for values of another
        shape.
        """
        from nimblend.sparse import SparseArray

        values = np.asarray(values, dtype=np.float64)
        if values.shape != (self.size,):
            raise ValueError(
                f"a domain of {self.size} member(s) requires values of shape "
                f"({self.size},); got shape {values.shape}"
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
        """Return each member paired with its position along `into`, valued 1.0.

        The position of a member is its rank here plus `start`, as in
        `as_coord(start)`. `coord` is the coordinate of `into`. Its extent can
        exceed the member count when several domains share one numbering.
        Raises ValueError for an `into` the domain already has, a negative
        `start`, or positions outside the extent of `coord`.
        """
        from nimblend.sparse import SparseArray

        if into in self.dims:
            raise ValueError(
                f"the domain already has dimension {into!r}; pass another name as into"
            )
        start = int(start)
        if start < 0:
            raise ValueError(f"start {start} is negative; pass a start of 0 or more")
        extent = len(coord)
        if start + self.size > extent:
            raise ValueError(
                f"{self.size} member(s) numbered from {start} end at position "
                f"{start + self.size - 1}, and dimension {into!r} has extent "
                f"{extent}; pass a smaller start or a larger coord"
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
