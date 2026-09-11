"""Index and value operations over plain numpy buffers.

Every function takes and returns numpy arrays, without labels or dimension
names. A compiled module with the same signatures can replace this module.
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
    """Return one int64 key per entry of `idx`, in C order.

    Over no dimensions every key is 0. Raises OverflowError when the product
    of `shape` exceeds the int64 range.
    """
    if not len(shape):
        return np.zeros(np.shape(idx)[1], dtype=np.int64)
    total = 1
    for size in shape:
        total *= int(size)
        if total > _INT64_MAX:
            raise OverflowError(
                f"shape {tuple(shape)} exceeds the int64 range of a raveled "
                f"index key; reduce the number or the extent of the dimensions"
            )
    keys = idx[0].astype(np.int64)
    for axis in range(1, len(shape)):
        keys *= int(shape[axis])
        keys += idx[axis]
    return keys


def unravel(keys: Positions, shape: Sequence[int]) -> Index:
    """Return the `(ndim, n)` index matrix of a set of C-order keys."""
    out = np.empty((len(shape), keys.size), dtype=np.int32)
    rest = keys
    for axis in range(len(shape) - 1, -1, -1):
        size = int(shape[axis])
        out[axis] = rest % size
        rest = rest // size
    return out


def distinct(keys: Keys) -> Keys:
    """Return the sorted unique keys, in one pass when the keys do not descend.

    Keys that strictly ascend are returned as given, without a copy.
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
    """Return the position of a repeated key, or -1 when the keys are distinct.

    The position is that of the second occurrence of the smallest repeated
    key.
    """
    if keys.size < 2:
        return -1
    order = np.argsort(keys, kind="stable")
    repeat = keys[order][1:] == keys[order][:-1]
    hit = repeat.nonzero()[0]
    return -1 if not hit.size else int(order[1:][hit[0]])


def _emit(idx: Index, data: Values, out: Block | None) -> Block:
    """Return `idx` and `data` copied into the start of `out`, or as given."""
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
    """Return `idx` and `data` sorted by index, with no repeated entry.

    `on_duplicate` is "sum" or "raise". At a repeated entry, "sum" adds the
    values and "raise" raises ValueError. Raises ValueError for another
    `on_duplicate`. Entries with value zero are kept. With `out` the result is
    written into `out`. The result does not share memory with a non-empty input.
    """
    if on_duplicate not in ("sum", "raise"):
        raise ValueError(f"on_duplicate is 'sum' or 'raise'; got {on_duplicate!r}")
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
            f"index {tuple(int(v) for v in idx[:, at])} is repeated; pass each "
            f"coordinate once"
        )
    starts = np.flatnonzero(first)
    return _emit(idx[:, starts], np.add.reduceat(data, starts), out)


def align(keys_a: Keys, keys_b: Keys, how: str) -> tuple[Keys, Positions, Positions]:
    """Merge two sorted unique key sets and return each operand's take-vector.

    `how` is "union" or "intersect". `take_a[i]` is the position of merged
    key `i` in `keys_a`, or -1 where `keys_a` does not contain it. `take_b`
    is the same for `keys_b`. Raises ValueError for another `how`.
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
    """Return the entries whose position along `axis` is in `take`, renumbered.

    An entry at position `take[i]` moves to position `i`. Entries at other
    positions are dropped. `take` must contain each position at most once.
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
    """Return the entries without `axis`, combining those at the same coordinate.

    `op` is "sum", "min" or "max". Entries whose remaining coordinates strictly
    ascend are returned without a sort. Removing the only axis of a non-empty
    block returns one entry. With `out` the result is written into `out`.
    Raises ValueError for another `op`.
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
    """Return each entry moved `n` positions along `axis`.

    Under mode "drop" an entry moved outside the axis is removed. Under mode
    "wrap" the axis is cyclic and every entry is kept. Raises ValueError for
    another mode.
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
    """Return a canonical two-dimensional block as CSR triplets.

    The column indices and the values are views of `idx` and `data`. Only the
    row pointer is computed.
    """
    rows = np.arange(shape[0] + 1, dtype=np.int32)
    indptr = np.searchsorted(idx[0], rows).astype(np.int32)
    return idx[1], data, indptr


def is_canonical(idx: Index, shape: Sequence[int]) -> bool:
    """Return True when `idx` ascends by raveled key with no repeated key."""
    if idx.shape[1] < 2:
        return True
    keys = ravel(idx, shape)
    return bool(np.all(np.diff(keys) > 0))


def lookup(keys: Keys, probe: Positions) -> Positions:
    """Return the position of each entry of `probe` in `keys`, or -1 if absent.

    `keys` must be sorted and unique. `probe` may repeat and may be unsorted.
    """
    if keys.size == 0:
        return np.full(probe.size, -1, dtype=np.int64)
    at = np.searchsorted(keys, probe)
    np.minimum(at, keys.size - 1, out=at)
    at[keys[at] != probe] = -1
    return at
