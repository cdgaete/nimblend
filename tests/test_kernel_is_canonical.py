import numpy as np

from nimblend import kernel


def test_sorted_unique_index_is_canonical():
    idx = np.array([[0, 0, 1], [0, 1, 0]], dtype=np.int32)
    assert kernel.is_canonical(idx, (2, 2))


def test_unsorted_index_is_not_canonical():
    idx = np.array([[1, 0], [0, 0]], dtype=np.int32)
    assert not kernel.is_canonical(idx, (2, 2))


def test_repeated_index_is_not_canonical():
    idx = np.array([[0, 0], [1, 1]], dtype=np.int32)
    assert not kernel.is_canonical(idx, (2, 2))


def test_empty_index_is_canonical():
    assert kernel.is_canonical(np.empty((2, 0), dtype=np.int32), (2, 2))


def test_canonicalize_output_is_canonical():
    rng = np.random.default_rng(11)
    idx = rng.integers(0, 6, (2, 40)).astype(np.int32)
    data = rng.standard_normal(40)
    out_idx, _ = kernel.canonicalize(idx, data, (6, 6))
    assert kernel.is_canonical(out_idx, (6, 6))
