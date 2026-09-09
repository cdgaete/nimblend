"""A slow, obviously-correct dense form of a COO block, for tests to check against."""

import numpy as np


def to_dense(idx, data, shape):
    """A COO block as a dense value array and a dense presence mask.

    Duplicate entries accumulate, which is what a canonical block must not
    contain and what `canonicalize` must produce the same totals for.
    """
    values = np.zeros(shape, dtype=np.float64)
    present = np.zeros(shape, dtype=bool)
    for k in range(data.size):
        at = tuple(int(idx[a, k]) for a in range(idx.shape[0]))
        values[at] += data[k]
        present[at] = True
    return values, present


def random_block(rng, shape, nnz, allow_duplicates=False):
    """A random COO block over `shape` with `nnz` entries."""
    ndim = len(shape)
    idx = np.empty((ndim, nnz), dtype=np.int32)
    for axis, size in enumerate(shape):
        idx[axis] = rng.integers(0, size, nnz)
    if not allow_duplicates:
        flat = np.zeros(nnz, dtype=np.int64)
        for axis, size in enumerate(shape):
            flat = flat * size + idx[axis]
        _, keep = np.unique(flat, return_index=True)
        idx = idx[:, np.sort(keep)]
    return idx, rng.standard_normal(idx.shape[1])
