import numpy as np

from nimblend import kernel


def test_compress_keeps_the_entries_marked_true():
    idx = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    keep = np.array([True, False, True])
    out_idx, out_data = kernel.compress(idx, np.array([1.0, 2.0, 3.0]), keep)
    assert out_idx.tolist() == [[0, 2], [3, 5]]
    assert out_data.tolist() == [1.0, 3.0]


def test_compress_keeping_nothing_is_empty():
    idx = np.array([[0, 1]], dtype=np.int32)
    out_idx, out_data = kernel.compress(idx, np.ones(2), np.zeros(2, dtype=bool))
    assert out_idx.shape == (1, 0)
    assert out_data.size == 0
