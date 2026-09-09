import numpy as np

from reference import random_block, to_dense


def test_to_dense_places_each_entry_at_its_index():
    idx = np.array([[0, 2], [1, 3]], dtype=np.int32)
    data = np.array([5.0, 7.0])
    values, present = to_dense(idx, data, (3, 4))
    assert values[0, 1] == 5.0
    assert values[2, 3] == 7.0
    assert present.sum() == 2


def test_to_dense_accumulates_duplicates():
    idx = np.array([[1, 1], [1, 1]], dtype=np.int32)
    data = np.array([2.0, 3.0])
    values, _ = to_dense(idx, data, (2, 2))
    assert values[1, 1] == 5.0


def test_random_block_is_unique_by_default():
    rng = np.random.default_rng(0)
    idx, data = random_block(rng, (10, 10), 40)
    flat = idx[0].astype(np.int64) * 10 + idx[1]
    assert np.unique(flat).size == flat.size
    assert data.size == idx.shape[1]
