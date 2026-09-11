# Changelog

Every change a user of `nimblend` can observe is listed here, under the
release that carries it. Work that has landed and is not yet released is
listed under `Unreleased`; a release renames that section to its version and
date and opens a new `Unreleased` above it. `tests/test_changelog.py` holds
the latest released section to the version the package states, so a release
without a section here, or a section without the version bump, fails the
suite.

The format is the one at <https://keepachangelog.com/en/1.1.0/>, and the
versions follow <https://semver.org/spec/v2.0.0.html>.

## Unreleased

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
