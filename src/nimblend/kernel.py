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


def span(shape: Sequence[int]) -> int:
    """Return the number of cells of `shape`.

    Raises OverflowError when the number exceeds the int64 range.
    """
    total = 1
    for size in shape:
        total *= int(size)
    if total > _INT64_MAX:
        raise OverflowError(
            f"shape {tuple(shape)} exceeds the int64 range of a raveled "
            f"index key; reduce the number or the extent of the dimensions"
        )
    return total


def ravel(idx: Index | Sequence[Index], shape: Sequence[int]) -> Keys:
    """Return one int64 key per entry of `idx`, in C order.

    Over no dimensions every key is 0. Raises OverflowError when the product
    of `shape` exceeds the int64 range.
    """
    if not len(shape):
        return np.zeros(np.shape(idx)[1], dtype=np.int64)
    span(shape)
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


def _run_starts(keys: Keys) -> npt.NDArray[np.bool_]:
    """Return True at the first position of each run of equal keys.

    `keys` ascends. Raises IndexError for an empty array.
    """
    first = np.empty(keys.size, dtype=bool)
    first[0] = True
    np.not_equal(keys[1:], keys[:-1], out=first[1:])
    return first


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
        first = _run_starts(keys)
        return keys[first]
    return np.unique(keys)


def first_unsorted(keys: Keys) -> int:
    """Return the first position `i` with `keys[i + 1] <= keys[i]`, or -1.

    The result is -1 when the keys strictly ascend.
    """
    if keys.size < 2:
        return -1
    step = keys[1:] <= keys[:-1]
    at = int(np.argmax(step))
    return at if bool(step[at]) else -1


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

    first = _run_starts(keys)
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
    first = _run_starts(keys)
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


def _sum_runs(keys: Keys, values: Values) -> tuple[Keys, Values]:
    """Return the distinct keys of sorted `keys` and the sum of each run."""
    first = _run_starts(keys)
    starts = np.flatnonzero(first)
    return keys[starts], np.add.reduceat(values, starts)


def weighted_sum_axis(
    idx: Index,
    data: Values,
    axis: int,
    weights: Values,
    shape: Sequence[int],
    block: int = 1 << 22,
) -> Block:
    """Return the entries without `axis`, summing each value times its weight.

    `weights` has one value per position along `axis`. Entries at the same
    remaining coordinate are summed, and the result is canonical. The entries
    are read in slices of at most `block`. A temporary has at most `block`
    elements, or one element per cell over the remaining axes when the cells
    do not outnumber the entries. Raises ValueError for a `block` below 1.
    """
    block = int(block)
    if block < 1:
        raise ValueError(f"block {block} is below 1; pass a block of 1 or more")
    kept = [a for a in range(idx.shape[0]) if a != axis]
    sub_shape = [shape[a] for a in kept]
    if data.size == 0:
        return np.empty((len(kept), 0), dtype=np.int32), np.empty(0, np.float64)
    cells = span(sub_shape)
    slices = [slice(at, at + block) for at in range(0, data.size, block)]
    if cells <= data.size:
        sums = np.zeros(cells, dtype=np.float64)
        seen = np.zeros(cells, dtype=bool)
        for at in slices:
            part_keys = ravel(idx[kept, at], sub_shape)
            part = data[at] * weights[idx[axis, at]]
            sums += np.bincount(part_keys, weights=part, minlength=cells)
            seen[part_keys] = True
        keys = np.flatnonzero(seen)
        return unravel(keys, sub_shape), sums[keys]
    keys = np.empty(0, dtype=np.int64)
    sums = np.empty(0, dtype=np.float64)
    for at in slices:
        part_keys = ravel(idx[kept, at], sub_shape)
        part = data[at] * weights[idx[axis, at]]
        if not bool(np.all(part_keys[1:] >= part_keys[:-1])):
            order = np.argsort(part_keys, kind="stable")
            part_keys = part_keys[order]
            part = part[order]
        part_keys, part = _sum_runs(part_keys, part)
        keys, take_old, take_new = align(keys, part_keys, "union")
        merged = np.zeros(keys.size, dtype=np.float64)
        has_old = take_old >= 0
        merged[has_old] = sums[take_old[has_old]]
        has_new = take_new >= 0
        merged[has_new] += part[take_new[has_new]]
        sums = merged
    return unravel(keys, sub_shape), sums


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
    return first_unsorted(ravel(idx, shape)) < 0


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


def select_axis(
    idx: Index, data: Values, axis: int, position: int, axis_len: int
) -> Block:
    """Return the entries at `position` along `axis`, without that axis."""
    kept_idx, kept_data = gather(idx, data, axis, np.array([position]), axis_len)
    return np.delete(kept_idx, axis, axis=0), kept_data


def take_filled(data: Values, take: Positions, fill: float = 0.0) -> Values:
    """Return `data[take]`, with `fill` where `take` is -1."""
    out = np.full(take.size, fill, dtype=np.float64)
    has = take >= 0
    out[has] = data[take[has]]
    return out


def compress(idx: Index, data: Values, keep: npt.NDArray[np.bool_]) -> Block:
    """Return the entries where `keep` is True, in their order."""
    return idx[:, keep], data[keep]


def multiply_lookup(idx: Index, data: Values, other: Values, take: Positions) -> Block:
    """Return the entries where `take` is not -1, each value times `other[take]`."""
    hit = take >= 0
    return idx[:, hit], data[hit] * other[take[hit]]


def multiply_join(
    idx_a: Index,
    data_a: Values,
    keys_a: Keys,
    idx_b: Index,
    data_b: Values,
    keys_b: Keys,
    axes_b: Sequence[int],
) -> Block:
    """Return one entry per pair of entries with equal keys, valued by their product.

    The index of an entry is its index in `idx_a`, then the axes `axes_b` of
    its index in `idx_b`. The entries are ordered by their position in
    `idx_a`, then by their position in `idx_b`. The keys do not need to be
    sorted or unique.
    """
    order = np.argsort(keys_b, kind="stable")
    sorted_b = keys_b[order]
    low = np.searchsorted(sorted_b, keys_a, "left")
    counts = np.searchsorted(sorted_b, keys_a, "right") - low
    total = int(counts.sum())
    take_a = np.repeat(np.arange(keys_a.size), counts)
    offsets = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
    take_b = order[np.repeat(low, counts) + offsets]
    held = idx_a.shape[0]
    index = np.empty((held + len(axes_b), total), dtype=np.int32)
    np.take(idx_a, take_a, axis=1, out=index[:held])
    for at, axis in enumerate(axes_b, start=held):
        np.take(idx_b[axis], take_b, out=index[at])
    return index, data_a[take_a] * data_b[take_b]


def cross(idx: Index, data: Values, sizes: Sequence[int]) -> Block:
    """Return each entry repeated once per cell of `sizes`, those positions appended.

    The new rows follow the rows of `idx`, and the cells of `sizes` are in C
    order. A canonical block returns a canonical block. Raises OverflowError
    when the number of cells of `sizes` exceeds the int64 range.
    """
    total = span(sizes)
    held = idx.shape[0]
    if data.size == 0:
        return np.empty((held + len(sizes), 0), dtype=np.int32), data
    count = data.size
    index = np.empty((held + len(sizes), count * total), dtype=np.int32)
    for axis in range(held):
        index[axis] = np.repeat(idx[axis], total)
    grid = unravel(np.arange(total, dtype=np.int64), sizes)
    for at in range(len(sizes)):
        index[held + at] = np.tile(grid[at], count)
    return index, np.repeat(data, total)


def cross_keys(left: Keys, right: Keys, span: int) -> Keys:
    """Return each left key paired with each right key, as `left * span + right`.

    The pairs are in the order of `left`, then in the order of `right`. Keys
    that ascend on both sides, with each right key below `span`, give pairs
    that ascend.
    """
    return (left[:, None] * span + right[None, :]).reshape(-1)


def regroup(
    idx: Index,
    data: Values,
    at: Positions,
    lead: int,
    start: int,
    out: Block | None = None,
) -> Block:
    """Return the entries with their first `lead` rows replaced by `at + start`.

    An entry where `at` is -1 is dropped. With `out` the result is written
    into `out`.
    """
    keep = at >= 0
    count = int(keep.sum())
    rest = idx.shape[0] - lead
    if out is None:
        out_idx = np.empty((1 + rest, count), dtype=np.int32)
        out_data = np.empty(count, dtype=np.float64)
    else:
        out_idx, out_data = out[0][:, :count], out[1][:count]
    np.add(at[keep], np.int32(start), out=out_idx[0], casting="unsafe")
    for row in range(rest):
        np.compress(keep, idx[lead + row], out=out_idx[1 + row])
    np.compress(keep, data, out=out_data)
    return out_idx, out_data


def densify(
    idx: Index, data: Values, shape: Sequence[int], fill: float
) -> npt.NDArray[np.float64]:
    """Return an array of `shape` with each value at its index and `fill` elsewhere.

    Over no axes the result has one cell.
    """
    out = np.full(tuple(shape), fill, dtype=np.float64)
    if not data.size:
        return out
    if not len(shape):
        out[()] = data[0]
        return out
    out[tuple(idx)] = data
    return out
