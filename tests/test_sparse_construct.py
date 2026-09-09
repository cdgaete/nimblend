import numpy as np
import pytest

from nimblend import kernel
from nimblend.coords import StoredCoord
from nimblend.sparse import SparseArray


def build(values, labels, absence="empty"):
    return SparseArray.from_dense(values, labels, absence=absence)


def test_from_dense_stores_every_cell_including_the_zeros():
    values = np.array([[0.0, 2.0], [3.0, 0.0]])
    arr = build(values, {"x": np.array(["a", "b"]), "y": np.array([1, 2])})
    # a dense array states that every cell is present, so every cell is stored;
    # dropping the zeros here would turn present-and-zero into absent
    assert arr.nnz == 4
    assert np.array_equal(arr.to_dense(), values)


def test_stored_zero_is_present_and_survives_canonicalisation():
    index = np.array([[0], [0]], dtype=np.int32)
    arr = SparseArray(
        index,
        np.array([0.0]),
        {"x": StoredCoord(np.array(["a"])), "y": StoredCoord(np.array([1]))},
        ("x", "y"),
    )
    assert arr.nnz == 1
    assert arr.data[0] == 0.0


def test_index_is_canonical_after_construction():
    index = np.array([[1, 0], [0, 1]], dtype=np.int32)
    arr = SparseArray(
        index,
        np.array([5.0, 6.0]),
        {"x": StoredCoord(np.array(["a", "b"])), "y": StoredCoord(np.array([1, 2]))},
        ("x", "y"),
    )
    keys = kernel.ravel(arr.index, arr.shape)
    assert list(keys) == sorted(keys)


def test_absence_must_be_one_of_the_two():
    with pytest.raises(ValueError, match="empty"):
        build(
            np.zeros((1, 1)),
            {"x": np.array(["a"]), "y": np.array([1])},
            absence="maybe",
        )


def test_shape_and_dims_follow_the_coordinates():
    arr = build(np.zeros((2, 3)), {"x": np.array(["a", "b"]), "y": np.array([1, 2, 3])})
    assert arr.dims == ("x", "y")
    assert arr.shape == (2, 3)


def test_rename_and_transpose():
    values = np.arange(6, dtype=np.float64).reshape(2, 3)
    arr = build(values, {"x": np.array(["a", "b"]), "y": np.array([1, 2, 3])})
    assert arr.rename({"x": "z"}).dims == ("z", "y")
    assert np.array_equal(arr.transpose("y", "x").to_dense(), values.T)


def test_sel_by_label_drops_the_dimension():
    values = np.arange(6, dtype=np.float64).reshape(2, 3)
    arr = build(values, {"x": np.array(["a", "b"]), "y": np.array([1, 2, 3])})
    assert np.array_equal(arr.sel({"x": "b"}).to_dense(), values[1])


def test_sparse_array_satisfies_the_contract():
    from conformance import check_array_contract

    check_array_contract(
        lambda values, labels, absence: SparseArray.from_dense(
            values, labels, absence=absence
        )
    )


def test_a_grid_over_no_dimensions_is_one_entry():
    # np.indices states one row per dimension and one column per cell; over
    # no dimensions that is no rows and the single cell the frame holds
    arr = SparseArray.from_dense(np.array(4.0), {})
    assert arr.dims == ()
    assert arr.index.shape == (0, 1)
    assert list(arr.values()) == [4.0]
    assert arr.to_dense() == 4.0
