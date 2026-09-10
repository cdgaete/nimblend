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
