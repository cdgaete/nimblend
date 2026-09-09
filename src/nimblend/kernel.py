"""Index and value operations over plain buffers.

Every function takes and returns numpy arrays and knows nothing of labels or
dimensions. The signatures are the seam a compiled implementation replaces.
"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

type Index = npt.NDArray[np.int32]
type Values = npt.NDArray[np.float64]
type Keys = npt.NDArray[np.int64]
type Positions = npt.NDArray[np.integer]
type Block = tuple[Index, Values]

_INT64_MAX = 2**63 - 1


def ravel(idx: Index | Sequence[Index], shape: Sequence[int]) -> Keys:
    """`idx` as one int64 key per entry, in C order.

    A single key is what lets a sort over several dimensions be one argsort
    rather than a lexsort.

    Over no dimensions the product of the sizes is one, so every entry lands
    on the single coordinate that product holds.
    """
    if not len(shape):
        return np.zeros(np.shape(idx)[1], dtype=np.int64)
    total = 1
    for size in shape:
        total *= int(size)
        if total > _INT64_MAX:
            raise OverflowError(
                f"shape {tuple(shape)} exceeds the int64 range a ravelled "
                f"index key holds"
            )
    keys = idx[0].astype(np.int64)
    for axis in range(1, len(shape)):
        keys *= int(shape[axis])
        keys += idx[axis]
    return keys


def unravel(keys: Positions, shape: Sequence[int]) -> Index:
    """The `(ndim, n)` index matrix a set of C-order keys stands for."""
    out = np.empty((len(shape), keys.size), dtype=np.int32)
    rest = keys
    for axis in range(len(shape) - 1, -1, -1):
        size = int(shape[axis])
        out[axis] = rest % size
        rest = rest // size
    return out


def distinct(keys: Keys) -> Keys:
    """Sorted unique keys, read off in one pass where they already ascend.

    Keys ravelled from a canonical block over a leading prefix of its
    dimensions arrive in order and the repeats a shared coordinate leaves
    are adjacent, so neither a sort nor a hash is needed to answer them.
    """
    if keys.size < 2:
        return keys
    step = np.diff(keys)
    if bool(np.all(step > 0)):
        return keys
    if bool(np.all(step >= 0)):
        first = np.empty(keys.size, dtype=bool)
        first[0] = True
        np.not_equal(keys[1:], keys[:-1], out=first[1:])
        return keys[first]
    return np.unique(keys)


def first_repeat(keys: Positions) -> int:
    """The position of the second key to repeat a value, or -1 when distinct.

    A stable argsort brings equal keys together, so the earliest repeat in
    sorted order is the earliest a caller has to name.
    """
    if keys.size < 2:
        return -1
    order = np.argsort(keys, kind="stable")
    repeat = keys[order][1:] == keys[order][:-1]
    hit = repeat.nonzero()[0]
    return -1 if not hit.size else int(order[1:][hit[0]])


def _emit(idx: Index, data: Values, out: Block | None) -> Block:
    """`idx` and `data` in `out`'s leading positions, or unchanged without one."""
    if out is None:
        return idx, data
    n = data.size
    out_idx, out_data = out[0][:, :n], out[1][:n]
    out_idx[:] = idx
    out_data[:] = data
    return out_idx, out_data


def canonicalize(
    idx: Index,
    data: Values,
    shape: Sequence[int],
    on_duplicate: str = "sum",
    out: Block | None = None,
) -> Block:
    """`idx` and `data` sorted by index with no repeated entry.

    Sorting on the ravelled key is one argsort rather than a lexsort over
    every dimension. What that sort costs is not the ordering but applying
    it: permuting the index and the data gathers them at random, so a block
    whose keys already ascend is copied straight through. Entries carrying
    zero are kept: a stored zero states that a coordinate is present.

    The result never shares memory with what it was given;
    `SparseArray.from_canonical` is the door that does.
    """
    if data.size == 0:
        return _emit(idx, data, out)
    keys = ravel(idx, shape)
    if keys.size > 1 and not bool(np.all(keys[1:] >= keys[:-1])):
        order = np.argsort(keys, kind="stable")
        keys = keys[order]
        idx = idx[:, order]
        data = data[order]
    elif out is None:
        idx = idx.copy()
        data = data.copy()

    first = np.empty(keys.size, dtype=bool)
    first[0] = True
    np.not_equal(keys[1:], keys[:-1], out=first[1:])
    if first.all():
        return _emit(idx, data, out)

    if on_duplicate == "raise":
        at = int(np.flatnonzero(~first)[0])
        raise ValueError(
            f"index {tuple(int(v) for v in idx[:, at])} repeats; a coordinate "
            f"names one entry"
        )
    if on_duplicate != "sum":
        raise ValueError(f"on_duplicate is 'sum' or 'raise'; got {on_duplicate!r}")

    starts = np.flatnonzero(first)
    return _emit(idx[:, starts], np.add.reduceat(data, starts), out)


def align(keys_a: Keys, keys_b: Keys, how: str) -> tuple[Keys, Positions, Positions]:
    """Merge two sorted unique key sets, with each operand's take-vector.

    `take_a[i]` is the position in `keys_a` supplying merged entry `i`, or
    -1 where `keys_a` does not carry it. A binary search over sorted keys is
    what makes this a merge rather than a hash join.
    """
    if how == "intersect":
        if keys_a.size == 0 or keys_b.size == 0:
            empty_i = np.empty(0, dtype=np.int64)
            return np.empty(0, dtype=np.int64), empty_i, empty_i.copy()
        pos = np.searchsorted(keys_b, keys_a)
        probe = np.minimum(pos, keys_b.size - 1)
        hit = keys_b[probe] == keys_a
        return keys_a[hit], np.flatnonzero(hit), pos[hit]

    if how != "union":
        raise ValueError(f"how is 'union' or 'intersect'; got {how!r}")

    if keys_a.size == 0:
        return keys_b, np.full(keys_b.size, -1, np.int64), np.arange(keys_b.size)
    if keys_b.size == 0:
        return keys_a, np.arange(keys_a.size), np.full(keys_a.size, -1, np.int64)

    pos = np.searchsorted(keys_a, keys_b)
    probe = np.minimum(pos, keys_a.size - 1)
    hit = keys_a[probe] == keys_b
    fresh = keys_b[~hit]

    merged = np.empty(keys_a.size + fresh.size, dtype=keys_a.dtype)
    slot = np.searchsorted(keys_a, fresh) + np.arange(fresh.size)
    is_new = np.zeros(merged.size, dtype=bool)
    is_new[slot] = True
    merged[is_new] = fresh
    merged[~is_new] = keys_a

    a_at = np.flatnonzero(~is_new)
    take_a = np.full(merged.size, -1, dtype=np.int64)
    take_a[a_at] = np.arange(keys_a.size)
    take_b = np.full(merged.size, -1, dtype=np.int64)
    take_b[slot] = np.flatnonzero(~hit)
    take_b[a_at[pos[hit]]] = np.flatnonzero(hit)
    return merged, take_a, take_b


def gather(
    idx: Index,
    data: Values,
    axis: int,
    take: Positions,
    axis_len: int,
    out: Block | None = None,
) -> Block:
    """Entries whose position along `axis` is in `take`, renumbered to it.

    Positions absent from `take` are dropped. `take` names each position at
    most once, so the result carries no repeated coordinate.
    """
    lookup = np.full(axis_len, -1, dtype=np.int32)
    lookup[take] = np.arange(len(take), dtype=np.int32)
    moved = lookup[idx[axis]]
    keep = np.flatnonzero(moved >= 0)

    n = keep.size
    if out is None:
        out_idx = np.empty((idx.shape[0], n), dtype=np.int32)
        out_data = np.empty(n, dtype=np.float64)
    else:
        out_idx, out_data = out[0][:, :n], out[1][:n]
    np.take(idx, keep, axis=1, out=out_idx)
    np.take(data, keep, out=out_data)
    out_idx[axis] = moved[keep]
    return out_idx, out_data


_REDUCERS = {"sum": np.add, "min": np.minimum, "max": np.maximum}


def reduce_axis(
    idx: Index,
    data: Values,
    axis: int,
    shape: Sequence[int],
    op: str = "sum",
    out: Block | None = None,
) -> Block:
    """`axis` removed, entries sharing the remaining coordinate combined.

    Entries that carried distinct positions on the other axes do not meet,
    so such a reduction leaves the entry count unchanged and only drops the
    axis. Where no two entries meet the reduction is the identity on the kept
    axes, and the entries are copied across with no sort at all. Reducing the
    last axis leaves no axis to sort on: the product of no sizes is one, so
    every entry meets on the single coordinate it holds and the result is one
    entry. With `out`
    the result is written into it directly and the block never exists as a
    second object; the working set that remains is the sort-merge, which
    scales with the input.
    """
    reducer = _REDUCERS.get(op)
    if reducer is None:
        raise ValueError(f"op is one of sum, min, max; got {op!r}")

    kept = [a for a in range(idx.shape[0]) if a != axis]
    sub_shape = [shape[a] for a in kept]
    rows = [idx[a] for a in kept]
    if data.size == 0:
        return idx[kept], data

    if not kept:
        return _emit(
            np.empty((0, 1), dtype=np.int32),
            np.array([reducer.reduce(data)], dtype=np.float64),
            out,
        )

    keys = ravel(rows, sub_shape)
    if bool(np.all(keys[1:] > keys[:-1])):
        del keys
        if out is None:
            return np.stack(rows), data
        out_idx, out_data = out[0][:, : data.size], out[1][: data.size]
        for at, row in enumerate(rows):
            np.copyto(out_idx[at], row)
        np.copyto(out_data, data)
        return out_idx, out_data

    sub = np.stack(rows)
    del rows
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    first = np.empty(keys.size, dtype=bool)
    first[0] = True
    np.not_equal(keys[1:], keys[:-1], out=first[1:])
    del keys
    starts = np.flatnonzero(first)
    del first

    n = starts.size
    if out is None:
        return sub[:, order][:, starts], reducer.reduceat(data[order], starts)
    out_idx, out_data = out[0][:, :n], out[1][:n]
    np.take(sub, order[starts], axis=1, out=out_idx)
    reducer.reduceat(data[order], starts, out=out_data)
    return out_idx, out_data


def shift_axis(
    idx: Index,
    data: Values,
    axis: int,
    n: int,
    axis_len: int,
    mode: str = "drop",
    out: Block | None = None,
) -> Block:
    """Each entry moved `n` positions along `axis`.

    Under `drop` an entry whose reference falls outside the axis is removed
    rather than replaced by a zero, so a position no entry reaches is absent.
    Under `wrap` the axis is cyclic and every entry survives.
    """
    if mode not in ("drop", "wrap"):
        raise ValueError(f"mode is 'drop' or 'wrap'; got {mode!r}")

    moved = idx[axis].astype(np.int64) + n
    if mode == "wrap":
        moved %= axis_len
        keep = np.arange(moved.size)
    else:
        keep = np.flatnonzero((moved >= 0) & (moved < axis_len))

    count = keep.size
    if out is None:
        out_idx = np.empty((idx.shape[0], count), dtype=np.int32)
        out_data = np.empty(count, dtype=np.float64)
    else:
        out_idx, out_data = out[0][:, :count], out[1][:count]
    np.take(idx, keep, axis=1, out=out_idx)
    np.take(data, keep, out=out_data)
    out_idx[axis] = moved[keep]
    return out_idx, out_data


def to_csr(
    idx: Index, data: Values, shape: Sequence[int]
) -> tuple[Index, Values, Index]:
    """A canonical two-dimensional block as CSR triplets.

    Canonical order is sorted by row and then column, which is CSR's own
    requirement, so the column indices and values are returned as views and
    only the row pointer is built.
    """
    rows = np.arange(shape[0] + 1, dtype=np.int32)
    indptr = np.searchsorted(idx[0], rows).astype(np.int32)
    return idx[1], data, indptr


def is_canonical(idx: Index, shape: Sequence[int]) -> bool:
    """Whether `idx` is sorted ascending by ravel key with no repeated key."""
    if idx.shape[1] < 2:
        return True
    keys = ravel(idx, shape)
    return bool(np.all(np.diff(keys) > 0))


def lookup(keys: Keys, probe: Positions) -> Positions:
    """Each entry of `probe` as its position in sorted-unique `keys`, or -1.

    `probe` may repeat and need not be sorted, which is what lets an operand
    carrying several entries per shared coordinate resolve against one that
    carries a single entry for it.
    """
    if keys.size == 0:
        return np.full(probe.size, -1, dtype=np.int64)
    at = np.searchsorted(keys, probe)
    np.minimum(at, keys.size - 1, out=at)
    at[keys[at] != probe] = -1
    return at
