import numpy as np

from nimblend import kernel


def test_multiply_lookup_scales_each_entry_by_the_value_it_takes():
    idx = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    data = np.array([1.0, 2.0, 3.0])
    out_idx, out_data = kernel.multiply_lookup(
        idx, data, np.array([10.0, 100.0]), np.array([1, -1, 0])
    )
    assert out_idx.tolist() == [[0, 2], [3, 5]]
    assert out_data.tolist() == [100.0, 30.0]


def test_multiply_lookup_without_a_hit_is_empty():
    idx = np.array([[0, 1]], dtype=np.int32)
    out_idx, out_data = kernel.multiply_lookup(
        idx, np.ones(2), np.empty(0), np.array([-1, -1])
    )
    assert out_idx.shape == (1, 0)
    assert out_data.size == 0
