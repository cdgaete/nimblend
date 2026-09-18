import numpy as np

from nimblend import kernel


def test_each_pair_of_equal_keys_gives_one_product_entry():
    # a: (b0, l0) and (b1, l0); b: (l0, t0) and (l0, t1); keyed by l
    idx_a = np.array([[0, 1], [0, 0]], dtype=np.int32)
    idx_b = np.array([[0, 0], [0, 1]], dtype=np.int32)
    keys = np.array([0, 0], dtype=np.int64)
    out_idx, out_data = kernel.multiply_join(
        idx_a, np.array([-1.0, 0.9]), keys, idx_b, np.array([2.0, 3.0]), keys, [1]
    )
    assert out_idx.tolist() == [[0, 0, 1, 1], [0, 0, 0, 0], [0, 1, 0, 1]]
    assert np.allclose(out_data, [-2.0, -3.0, 1.8, 2.7])


def test_a_key_without_a_pair_gives_no_entry():
    idx_a = np.array([[0, 1]], dtype=np.int32)
    idx_b = np.array([[1, 2]], dtype=np.int32)
    out_idx, out_data = kernel.multiply_join(
        idx_a,
        np.array([1.0, 2.0]),
        np.array([0, 1], dtype=np.int64),
        idx_b,
        np.array([5.0, 6.0]),
        np.array([1, 2], dtype=np.int64),
        [],
    )
    assert out_idx.tolist() == [[1]]
    assert out_data.tolist() == [10.0]


def test_the_join_matches_a_nested_loop_over_unsorted_keys():
    rng = np.random.default_rng(4)
    keys_a = rng.integers(0, 5, 30).astype(np.int64)
    keys_b = rng.integers(0, 5, 20).astype(np.int64)
    idx_a = np.arange(30, dtype=np.int32)[None, :]
    idx_b = np.stack([keys_b, np.arange(20)]).astype(np.int32)
    data_a = rng.standard_normal(30)
    data_b = rng.standard_normal(20)
    out_idx, out_data = kernel.multiply_join(
        idx_a, data_a, keys_a, idx_b, data_b, keys_b, [1]
    )
    wanted = [
        (i, j, data_a[i] * data_b[j])
        for i in range(30)
        for j in range(20)
        if keys_a[i] == keys_b[j]
    ]
    assert out_idx.tolist() == [[i for i, _, _ in wanted], [j for _, j, _ in wanted]]
    assert np.allclose(out_data, [v for _, _, v in wanted])


def test_joining_no_entries_is_empty():
    empty = np.empty((1, 0), dtype=np.int32)
    out_idx, out_data = kernel.multiply_join(
        empty,
        np.empty(0),
        np.empty(0, dtype=np.int64),
        np.array([[0]], dtype=np.int32),
        np.array([1.0]),
        np.array([0], dtype=np.int64),
        [0],
    )
    assert out_idx.shape == (2, 0)
    assert out_data.size == 0
