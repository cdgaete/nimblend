import numpy as np

from nimblend import kernel


def test_cross_repeats_each_entry_over_every_new_cell():
    idx = np.array([[0, 2]], dtype=np.int32)
    out_idx, out_data = kernel.cross(idx, np.array([5.0, 7.0]), (2, 2))
    assert out_idx.tolist() == [
        [0, 0, 0, 0, 2, 2, 2, 2],
        [0, 0, 1, 1, 0, 0, 1, 1],
        [0, 1, 0, 1, 0, 1, 0, 1],
    ]
    assert out_data.tolist() == [5.0] * 4 + [7.0] * 4


def test_crossing_a_canonical_block_leaves_it_canonical():
    idx = np.array([[0, 1, 1], [2, 0, 1]], dtype=np.int32)
    out_idx, _ = kernel.cross(idx, np.ones(3), (3,))
    assert kernel.is_canonical(out_idx, (2, 3, 3))


def test_crossing_with_no_sizes_returns_the_block():
    idx = np.array([[0, 1]], dtype=np.int32)
    out_idx, out_data = kernel.cross(idx, np.array([1.0, 2.0]), ())
    assert out_idx.tolist() == [[0, 1]]
    assert out_data.tolist() == [1.0, 2.0]


def test_crossing_an_empty_block_is_empty():
    out_idx, out_data = kernel.cross(
        np.empty((1, 0), dtype=np.int32), np.empty(0), (3,)
    )
    assert out_idx.shape == (2, 0)
    assert out_data.size == 0


def test_cross_keys_numbers_each_key_times_every_new_cell():
    got = kernel.cross_keys(np.array([1, 4], dtype=np.int64), 3)
    assert got.tolist() == [3, 4, 5, 12, 13, 14]


def test_cross_keys_agrees_with_cross_over_the_raveled_index():
    idx = np.array([[0, 1, 1], [2, 0, 1]], dtype=np.int32)
    out_idx, _ = kernel.cross(idx, np.ones(3), (2, 2))
    keys = kernel.ravel(idx, (2, 3))
    assert np.array_equal(
        kernel.cross_keys(keys, 4), kernel.ravel(out_idx, (2, 3, 2, 2))
    )
