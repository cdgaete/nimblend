# Changelog

Every change a user of `nimblend` can observe is listed here, under the
release that carries it. Work that has landed and is not yet released is
listed under `Unreleased`; a release renames that section to its version and
date and opens a new `Unreleased` above it. `tests/test_changelog.py` holds
the latest released section to the version the package states, so a release
without a section here, or a section without the version bump, fails the
suite.

The format is the one at <https://keepachangelog.com/en/1.1.0/>.

A version is `MAJOR.YYYYMMDD.PATCH`. `MAJOR` is 0 for a package that is not
stable, and any release of it can contain a breaking change. `MAJOR` is 1 for
the first stable release and increases with every breaking change. `YYYYMMDD`
is the date of a release that adds features. `PATCH` counts the releases that
only correct defects of that release, from 0. The releases up to 0.2.2 follow
<https://semver.org/spec/v2.0.0.html>.

## Unreleased

### Added

- `sum_arrays(arrays)` returns the sum of several arrays over one frame,
  computed in one merge. The result equals adding the arrays in order with
  `+`, including the order in which the values at a coordinate are added.
  It raises `ValueError` for no arrays and for arrays with different
  dimensions, labels or absence, and `TypeError` for an array that is not a
  `SparseArray`.
- `Domain.project(dims)` returns the distinct coordinates of the members over
  `dims`, in the order given.
- `group` takes `reserve=`, a function it calls with the number of entries
  it writes and that returns the destination. `EntryBuffer.reserve` is one.

### Changed

- `Domain.positions_of`, `Domain.intersect` and the other lookups of positions
  among sorted keys read the positions from a table over the range of the
  keys. The table is used when that range plus the number of keys is at most
  four times the number of probes, and a binary search otherwise. A lookup of
  1,401,520 probes among 700,720 keys takes 2.7 ms instead of 30.0 ms.

### Removed

- `ProductCoord`, `SubsetCoord` and `Domain.as_coord` take no `start`. Every
  coordinate numbers its positions from 0 to its extent less one. `group` and
  `identity` keep `start`, the position of the first member inside a wider
  `coord`.

### Fixed

- `group` raises `ValueError` for an `out` or a reservation whose size
  differs from the number of entries it writes, and writes nothing into it.
- The `SparseArray` constructor raises `ValueError` for an index that is not
  2-D, an index without one row per dimension, a position outside the extent
  of its dimension, including a negative position, and an index without one
  value per column.
- `Domain` raises `ValueError` for a code outside the cells of its shape.
  `Domain.from_coordinates` raises `ValueError` for a position outside the
  extent of its dimension.
- `ProductCoord.to_position` and `SubsetCoord.to_position` raise `KeyError`
  for a cell outside their sizes, and `ValueError` for an index without one
  row per axis. `from_long` raises the same `KeyError` for such a cell.
- `sum`, `mean`, `min` and `max` with `dim` and `fill=` return an array over
  the coordinates of the remaining dimensions, with their extents.

## 0.20260921.0 - 2026-09-21

### Added

- `Domain.labels` and `Domain.coordinates` take `positions` and return the
  labels or the multi-indices of the members at those positions alone.

## 0.20260919.0 - 2026-09-19

### Added

- `SparseArray.broadcast(dims, coords)` and `DenseArray.broadcast(dims,
  coords)` return the array over exactly `dims`, in that order, replicated
  across each dimension of `dims` it does not have. They raise `ValueError`
  for a dimension of the array not in `dims`, and for a dimension without a
  coordinate. The `Array` protocol includes `broadcast`.
- `SparseArray.weighted_sum(dim, weights)` and `DenseArray.weighted_sum(dim,
  weights)` return the sum over `dim` of each entry times the weight at its
  position along `dim`. The sparse implementation calls the kernel function
  `weighted_sum_axis`. It reads the entries in blocks and allocates no
  temporary of the size of the array. Under absence "unknown" both require
  `skip=True`. The `Array` protocol includes `weighted_sum`.
- `Domain.symmetric_difference(other)` returns the members that exactly one of
  the two domains contains. It raises `ValueError` for domains with different
  frames or labels, as `difference` does.
- `Domain.cross(other)` returns every member of the domain paired with every
  member of `other`, over `self.dims + other.dims`. The members of the domain
  come first, in their order, each paired with the members of `other` in
  their order. It raises `ValueError` for a dimension both domains have, and
  `OverflowError` when the product of the extents exceeds the int64 range.

### Changed

- `Domain.transpose()` with no arguments returns the domain over the
  reversed dimensions.
- `group` takes `coord=` and `start=` in place of `offset=`. `coord` is the
  coordinate of the new dimension and defaults to the domain numbered from 0.
  An entry is placed at the rank of its member plus `start`. `group` raises
  `ValueError` where the positions end beyond the extent of `coord`. Every
  position of a grouped array is inside its coordinate: `to_dense`,
  `domain().labels()` and `restrict` return the grouped entries.
- Operands whose shared dimensions have different labels raise `ValueError`
  with one message in every operator of both implementations: `dimension(s)
  [...] have different labels in the two arrays`.
- `nimblend.is_canonical` checks its arguments before it reads the order. It
  raises `TypeError` for an index that is not integer. It raises `ValueError`
  for an index that is not 2-D, a row count other than the number of extents
  in `shape`, and a position outside the extent of its row. It reads an index
  given as a nested list.

### Fixed

- `Domain.full`, `Domain.is_full` and `len()` of a `ProductCoord` raise
  `OverflowError` for a shape with more cells than the int64 range.
- A shape with an extent of 0 has 0 cells in every position of that extent.
  `SparseArray.domain` returns the domain with no members for such a shape,
  and raises no `OverflowError` for the extents of the other dimensions.
- `SparseArray.expand` raises `OverflowError` for a shape with more cells than
  the int64 range.
- `min()`, `max()` and `mean()` with `fill=`, and `to_dense()` under absence
  "unknown", raise `OverflowError` for a shape with more cells than the int64
  range.
- `min()`, `max()` and `mean()` over the whole of an array with no values
  raise `ValueError` in both implementations. `sum()` over no values returns
  0.0. With `fill=` every coordinate of the frame has a value.
- `from_long` and `Domain.from_labels` convert every label column with one
  function. A label column of a `ProductCoord` or a `SubsetCoord` is an index
  matrix with one column per label, and its length is that label count. Both
  raise `ValueError` for columns of different lengths, and the message
  reports the label count of each column. `from_long` raises `ValueError`
  where the label count differs from the value count. `from_long` over no
  dimensions returns an array over no dimension.
- `SubsetCoord.to_position` over a subset with no member raises `KeyError`
  for any cell, as it does for a cell outside a non-empty subset.

## 0.2.2 - 2026-09-11

### Changed

- The README opens on the PyPI install and is written in technical English.

### Fixed

- `shift` and `roll` raise `ValueError` for a `mode` other than "drop" or
  "wrap" when `shifts` is empty.
- Every method that takes dimension names raises `ValueError` for a name the
  array does not have: `sel`, `sum`, `mean`, `min`, `max`, `domain`,
  `coordinates`, `group`, `rename`, `shift` and `roll`. The message
  identifies the dimensions of the array. `rename` raises for a key that is
  not a dimension of the array. `SparseArray.restrict` raises for a domain
  over a dimension the array does not have, as `DenseArray.restrict` does.
- `SparseArray`, `DenseArray` and `Domain` raise `ValueError` for a repeated
  dimension name at construction and in `expand`. `SparseArray.domain` and
  `DenseArray.domain` raise it for a repeated name in `dims`.
- `transpose` raises `ValueError` for a repeated dimension and for a name that
  is not a dimension, in both implementations and in `Domain`.

## 0.2.1 - 2026-09-11

### Changed

- The error messages and docstrings are written in technical English. Each
  message reports the condition, then the action to take. The text of
  several messages changed. A label in a message prints as its value, as in
  the `KeyError` message `label 'zz' is not in the coordinate`. A datetime64
  or timedelta64 label prints as its string form.

## 0.2.0 - 2026-09-11

### Changed

- The package supports Python 3.12, 3.13 and 3.14, and numpy from 2.3.

## 0.1.1 - 2026-09-09

### Changed

- The README and the description on the package index name the documentation
  site.

## 0.1.0 - 2026-09-08

### Added

- A `Domain` of labeled dimensions, and sparse and dense arrays over it, with
  alignment, arithmetic, reduction and shifting by label.
- A kernel of module-level functions over plain numpy buffers that carries
  every operation, so that a module satisfying the same signatures replaces
  it wholesale.
