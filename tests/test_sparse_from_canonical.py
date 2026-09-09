import numpy as np
import pytest

from nimblend.coords import StoredCoord
from nimblend.sparse import SparseArray


def coords_2d(n_rows, n_cols):
    return {
        "r": StoredCoord(np.arange(n_rows)),
        "c": StoredCoord(np.arange(n_cols)),
    }


def test_from_canonical_takes_no_copy():
    index = np.array([[0, 0, 1], [0, 1, 0]], dtype=np.int32)
    data = np.array([1.0, 2.0, 3.0])
    arr = SparseArray.from_canonical(index, data, coords_2d(2, 2), ("r", "c"))
    assert np.shares_memory(arr.index, index)
    assert np.shares_memory(arr.data, data)


def test_from_canonical_produces_the_same_array_as_the_constructor():
    index = np.array([[0, 0, 1], [0, 1, 0]], dtype=np.int32)
    data = np.array([1.0, 2.0, 3.0])
    wrapped = SparseArray.from_canonical(index, data, coords_2d(2, 2), ("r", "c"))
    built = SparseArray(index, data, coords_2d(2, 2), ("r", "c"))
    assert np.array_equal(wrapped.to_dense(), built.to_dense())
    assert wrapped.dims == built.dims
    assert wrapped.shape == built.shape


def test_from_canonical_refuses_an_unknown_absence():
    index = np.array([[0]], dtype=np.int32)
    with pytest.raises(ValueError, match="empty"):
        SparseArray.from_canonical(
            index,
            np.array([1.0]),
            {"r": StoredCoord(np.arange(1))},
            ("r",),
            absence="maybe",
        )


def test_from_canonical_refuses_a_dimension_without_a_coordinate():
    index = np.array([[0]], dtype=np.int32)
    with pytest.raises(ValueError, match="coordinate"):
        SparseArray.from_canonical(index, np.array([1.0]), {}, ("r",))


def test_a_wrapped_array_takes_part_in_arithmetic():
    index = np.array([[0, 1]], dtype=np.int32)
    coords = {"r": StoredCoord(np.arange(3))}
    a = SparseArray.from_canonical(index, np.array([1.0, 2.0]), coords, ("r",))
    b = SparseArray.from_canonical(index, np.array([10.0, 20.0]), coords, ("r",))
    assert list((a + b).data) == [11.0, 22.0]
