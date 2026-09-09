import tracemalloc

import numpy as np

from nimblend import kernel


def block(n_rows, n_cols):
    """Entries over (row, w, col) that reduce over w into one row per row."""
    per = n_rows * n_cols
    idx = np.empty((3, per), dtype=np.int32)
    idx[0] = np.repeat(np.arange(n_rows, dtype=np.int32), n_cols)
    idx[1] = np.tile(np.arange(n_cols, dtype=np.int32), n_rows)
    idx[2] = np.arange(per, dtype=np.int32)
    return idx, np.ones(per)


def test_reduce_into_out_matches_reduce_without_out():
    idx, data = block(20, 5)
    want_idx, want_data = kernel.reduce_axis(idx, data, 1, (20, 5, 100))
    dest = (np.empty((2, 100), dtype=np.int32), np.empty(100))
    got_idx, got_data = kernel.reduce_axis(idx, data, 1, (20, 5, 100), out=dest)
    assert np.array_equal(got_idx, want_idx)
    assert np.array_equal(got_data, want_data)
    assert np.shares_memory(got_data, dest[1])


def test_reduce_into_out_does_not_allocate_a_second_copy_of_the_block():
    # the block is 500 000 entries: 4 MB of index and 4 MB of values. A path
    # that builds the block and copies it holds both at once.
    idx, data = block(2000, 250)
    n_out = 500_000
    dest = (np.empty((2, n_out), dtype=np.int32), np.empty(n_out))
    tracemalloc.start()
    before, _ = tracemalloc.get_traced_memory()
    kernel.reduce_axis(idx, data, 1, (2000, 250, n_out), out=dest)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    block_bytes = n_out * (2 * 4 + 8)
    assert peak - before < block_bytes, (peak - before, block_bytes)


def test_min_and_max_also_write_into_out():
    idx = np.array([[0, 0, 1], [0, 1, 0]], dtype=np.int32)
    data = np.array([4.0, -1.0, 9.0])
    dest = (np.empty((1, 4), dtype=np.int32), np.empty(4))
    _, mins = kernel.reduce_axis(idx, data, 1, (2, 2), op="min", out=dest)
    assert list(mins) == [-1.0, 9.0]
    _, maxs = kernel.reduce_axis(idx, data, 1, (2, 2), op="max", out=dest)
    assert list(maxs) == [4.0, 9.0]


def test_reduce_into_a_non_contiguous_destination_slice():
    idx = np.array([[0, 0, 1], [0, 1, 0]], dtype=np.int32)
    data = np.array([1.0, 2.0, 5.0])
    big_idx = np.full((1, 10), -7, dtype=np.int32)
    big_data = np.full(10, -7.0)
    dest = (big_idx[:, 4:8], big_data[4:8])
    out_idx, out_data = kernel.reduce_axis(idx, data, 1, (2, 2), out=dest)
    assert list(out_data) == [3.0, 5.0]
    assert list(big_data[3:7]) == [-7.0, 3.0, 5.0, -7.0]
