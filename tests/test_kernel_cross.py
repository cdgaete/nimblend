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


def test_crossing_an_empty_block_over_a_large_size_does_not_build_the_grid(
    monkeypatch,
):
    def boom(*_args, **_kwargs):
        raise AssertionError("unravel is called")

    monkeypatch.setattr(kernel, "unravel", boom)
    out_idx, out_data = kernel.cross(
        np.empty((1, 0), dtype=np.int32), np.empty(0), (40_000_000,)
    )
    assert out_idx.shape == (2, 0)
    assert out_data.size == 0


def test_cross_keys_pairs_each_left_key_with_each_right_key():
    left = np.array([1, 4], dtype=np.int64)
    right = np.array([0, 2], dtype=np.int64)
    got = kernel.cross_keys(left, right, 3)
    assert got.tolist() == [3, 5, 12, 14]


def test_cross_keys_of_sorted_keys_ascend():
    rng = np.random.default_rng(6)
    left = np.unique(rng.integers(0, 50, 20)).astype(np.int64)
    right = np.unique(rng.integers(0, 7, 5)).astype(np.int64)
    assert kernel.first_unsorted(kernel.cross_keys(left, right, 7)) == -1


def test_cross_keys_with_no_key_on_a_side_is_empty():
    keys = np.array([1, 2], dtype=np.int64)
    empty = np.empty(0, dtype=np.int64)
    assert kernel.cross_keys(keys, empty, 3).size == 0
    assert kernel.cross_keys(empty, keys, 3).size == 0


def test_cross_keys_with_every_right_key_agrees_with_cross_over_the_index():
    idx = np.array([[0, 1, 1], [2, 0, 1]], dtype=np.int32)
    out_idx, _ = kernel.cross(idx, np.ones(3), (2, 2))
    keys = kernel.ravel(idx, (2, 3))
    every = np.arange(4, dtype=np.int64)
    assert np.array_equal(
        kernel.cross_keys(keys, every, 4), kernel.ravel(out_idx, (2, 3, 2, 2))
    )
