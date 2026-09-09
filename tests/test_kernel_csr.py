import numpy as np
import scipy.sparse as sp

from nimblend import kernel


def test_csr_matches_scipy():
    idx = np.array([[0, 0, 2], [1, 3, 0]], dtype=np.int32)
    data = np.array([5.0, 6.0, 7.0])
    indices, values, indptr = kernel.to_csr(idx, data, (3, 4))
    got = sp.csr_array((values, indices, indptr), shape=(3, 4)).toarray()
    want = np.zeros((3, 4))
    want[0, 1], want[0, 3], want[2, 0] = 5.0, 6.0, 7.0
    assert np.array_equal(got, want)


def test_indices_and_values_are_views_not_copies():
    idx = np.array([[0, 1], [0, 1]], dtype=np.int32)
    data = np.array([1.0, 2.0])
    indices, values, _ = kernel.to_csr(idx, data, (2, 2))
    assert values.base is data or values is data
    assert indices.base is idx or indices.base is idx.base


def test_empty_rows_get_equal_pointers():
    idx = np.array([[0, 2], [0, 1]], dtype=np.int32)
    data = np.array([1.0, 2.0])
    _, _, indptr = kernel.to_csr(idx, data, (3, 2))
    assert list(indptr) == [0, 1, 1, 2]


def test_empty_block():
    idx = np.empty((2, 0), dtype=np.int32)
    data = np.empty(0)
    indices, values, indptr = kernel.to_csr(idx, data, (2, 2))
    assert indices.size == 0 and values.size == 0
    assert list(indptr) == [0, 0, 0]
