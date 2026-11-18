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


def _nested_loop(keys_a, data_a, keys_b, data_b):
    """Return the pairs of positions with equal keys and their products."""
    return [
        (i, j, data_a[i] * data_b[j])
        for i in range(keys_a.size)
        for j in range(keys_b.size)
        if keys_a[i] == keys_b[j]
    ]


def _joined(keys_a, data_a, keys_b, data_b):
    idx_a = np.arange(keys_a.size, dtype=np.int32)[None, :]
    idx_b = np.stack([keys_b, np.arange(keys_b.size)]).astype(np.int32)
    return kernel.multiply_join(idx_a, data_a, keys_a, idx_b, data_b, keys_b, [1])


def _refuse(monkeypatch, name, said):
    def refuse(*args, **kwargs):
        raise AssertionError(said)

    monkeypatch.setattr(np, name, refuse)


def test_the_join_matches_a_nested_loop_over_ascending_keys_with_repeats():
    rng = np.random.default_rng(5)
    keys_a = rng.integers(0, 7, 40).astype(np.int64)
    keys_b = np.sort(rng.integers(0, 6, 25)).astype(np.int64)
    data_a, data_b = rng.standard_normal(40), rng.standard_normal(25)
    out_idx, out_data = _joined(keys_a, data_a, keys_b, data_b)
    wanted = _nested_loop(keys_a, data_a, keys_b, data_b)
    assert out_idx.tolist() == [[i for i, _, _ in wanted], [j for _, j, _ in wanted]]
    assert out_data.tolist() == [v for _, _, v in wanted]


def test_the_join_sorts_no_second_keys_that_ascend(monkeypatch):
    rng = np.random.default_rng(6)
    keys_a = rng.integers(0, 7, 40).astype(np.int64)
    keys_b = np.sort(rng.integers(0, 6, 25)).astype(np.int64)
    data_a, data_b = rng.standard_normal(40), rng.standard_normal(25)
    wanted = _nested_loop(keys_a, data_a, keys_b, data_b)
    _refuse(monkeypatch, "argsort", "the ascending keys were sorted")
    out_idx, out_data = _joined(keys_a, data_a, keys_b, data_b)
    assert out_idx.tolist() == [[i for i, _, _ in wanted], [j for _, j, _ in wanted]]
    assert out_data.tolist() == [v for _, _, v in wanted]


def test_the_join_matches_a_nested_loop_over_distinct_ascending_keys():
    rng = np.random.default_rng(7)
    # the first keys repeat, are unsorted, and include keys the second lacks
    keys_a = rng.integers(0, 30, 60).astype(np.int64)
    keys_b = np.array([2, 3, 5, 8, 13, 21], dtype=np.int64)
    data_a, data_b = rng.standard_normal(60), rng.standard_normal(6)
    out_idx, out_data = _joined(keys_a, data_a, keys_b, data_b)
    wanted = _nested_loop(keys_a, data_a, keys_b, data_b)
    assert out_idx.tolist() == [[i for i, _, _ in wanted], [j for _, j, _ in wanted]]
    assert out_data.tolist() == [v for _, _, v in wanted]


def test_the_join_reads_distinct_ascending_keys_from_a_table(monkeypatch):
    rng = np.random.default_rng(8)
    # 6 keys over a range of 20 and 60 probes: 20 + 6 <= 4 * 60
    keys_a = rng.integers(0, 30, 60).astype(np.int64)
    keys_b = np.array([2, 3, 5, 8, 13, 21], dtype=np.int64)
    data_a, data_b = rng.standard_normal(60), rng.standard_normal(6)
    wanted = _nested_loop(keys_a, data_a, keys_b, data_b)
    _refuse(monkeypatch, "argsort", "the distinct ascending keys were sorted")
    _refuse(monkeypatch, "searchsorted", "the distinct ascending keys were searched")
    out_idx, out_data = _joined(keys_a, data_a, keys_b, data_b)
    assert out_idx.tolist() == [[i for i, _, _ in wanted], [j for _, j, _ in wanted]]
    assert out_data.tolist() == [v for _, _, v in wanted]


def test_the_join_with_no_second_entries_is_empty():
    out_idx, out_data = kernel.multiply_join(
        np.array([[0, 1]], dtype=np.int32),
        np.array([1.0, 2.0]),
        np.array([0, 1], dtype=np.int64),
        np.empty((2, 0), dtype=np.int32),
        np.empty(0),
        np.empty(0, dtype=np.int64),
        [1],
    )
    assert out_idx.shape == (2, 0)
    assert out_idx.dtype == np.int32
    assert out_data.size == 0


def test_the_join_of_one_second_entry_pairs_every_equal_first_key():
    out_idx, out_data = kernel.multiply_join(
        np.array([[0, 1, 2]], dtype=np.int32),
        np.array([1.0, 2.0, 3.0]),
        np.array([4, 9, 4], dtype=np.int64),
        np.array([[4], [7]], dtype=np.int32),
        np.array([10.0]),
        np.array([4], dtype=np.int64),
        [1],
    )
    assert out_idx.tolist() == [[0, 2], [7, 7]]
    assert out_data.tolist() == [10.0, 30.0]
