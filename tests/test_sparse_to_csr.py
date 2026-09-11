import re

import numpy as np
import pytest

from nimblend.coords import StoredCoord
from nimblend.sparse import SparseArray


def two_d():
    index = np.array([[0, 0, 2], [1, 3, 0]], dtype=np.int32)
    coords = {"r": StoredCoord(np.arange(3)), "c": StoredCoord(np.arange(4))}
    return SparseArray(index, np.array([5.0, 6.0, 7.0]), coords, ("r", "c"))


def test_to_csr_returns_views_and_a_row_pointer():
    arr = two_d()
    indices, values, indptr = arr.to_csr()
    assert list(indices) == [1, 3, 0]
    assert list(values) == [5.0, 6.0, 7.0]
    assert list(indptr) == [0, 2, 2, 3]
    assert np.shares_memory(values, arr.data)
    assert np.shares_memory(indices, arr.index)


def test_to_csr_refuses_an_array_that_is_not_two_dimensional():
    coords = {"r": StoredCoord(np.arange(2))}
    arr = SparseArray(np.array([[0]], dtype=np.int32), np.array([1.0]), coords, ("r",))
    with pytest.raises(ValueError, match="two"):
        arr.to_csr()


def test_indices_are_int32_so_a_solver_takes_them_without_a_cast():
    indices, _, indptr = two_d().to_csr()
    assert indices.dtype == np.int32
    assert indptr.dtype == np.int32


def test_to_csr_refuses_rows_outside_the_row_extent():
    # a block numbered from an offset has row positions beyond the extent of
    # its own coordinate
    index = np.array([[10, 11], [1, 0]], dtype=np.int32)
    coords = {"r": StoredCoord(np.arange(3)), "c": StoredCoord(np.arange(4))}
    arr = SparseArray.from_canonical(index, np.array([5.0, 6.0]), coords, ("r", "c"))
    message = (
        "row positions range from 10 to 11 and dimension 'r' has extent 3; "
        "call to_csr on an array with row positions from 0 to 2"
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        arr.to_csr()
