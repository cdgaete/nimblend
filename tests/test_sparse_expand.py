import numpy as np
import pytest

from nimblend import kernel
from nimblend.coords import StoredCoord
from nimblend.sparse import SparseArray


def block(values, labels):
    """An array holding only the nonzero cells of `values`."""
    arr = SparseArray.from_dense(np.asarray(values, dtype=np.float64), labels)
    keep = arr.data != 0.0
    return SparseArray.from_canonical(
        arr.index[:, keep], arr.data[keep], arr.coords, arr.dims
    )


def test_expand_appends_the_dimension_and_replicates_every_entry():
    arr = block(
        [[1.0, 0.0], [3.0, 4.0]], {"x": np.array(["a", "b"]), "y": np.array([0, 1])}
    )
    got = arr.expand(("z",), {"z": StoredCoord(np.array([7, 8, 9]))})
    assert got.dims == ("x", "y", "z")
    assert got.shape == (2, 2, 3)
    assert got.nnz == arr.nnz * 3


def test_an_expanded_entry_carries_its_value_at_every_new_coordinate():
    arr = block([[5.0]], {"x": np.array(["a"]), "y": np.array([0])})
    got = arr.expand(("z",), {"z": StoredCoord(np.array([7, 8]))})
    assert got.to_dense().tolist() == [[[5.0, 5.0]]]


def test_expand_leaves_the_result_canonical():
    arr = block(
        [[1.0, 0.0], [3.0, 4.0]], {"x": np.array(["a", "b"]), "y": np.array([0, 1])}
    )
    got = arr.expand(("z",), {"z": StoredCoord(np.array([7, 8, 9]))})
    assert kernel.is_canonical(got.index, got.shape)


def test_expand_adds_several_dimensions_at_once():
    arr = block([[5.0]], {"x": np.array(["a"]), "y": np.array([0])})
    got = arr.expand(
        ("z", "w"),
        {"z": StoredCoord(np.array([1, 2])), "w": StoredCoord(np.array([3, 4, 5]))},
    )
    assert got.dims == ("x", "y", "z", "w")
    assert got.nnz == 6


def test_expand_refuses_a_dimension_the_array_already_carries():
    arr = block([[5.0]], {"x": np.array(["a"]), "y": np.array([0])})
    with pytest.raises(ValueError, match="already has"):
        arr.expand(("x",), {"x": StoredCoord(np.array(["a"]))})


def test_expand_refuses_a_dimension_with_no_coordinate():
    arr = block([[5.0]], {"x": np.array(["a"]), "y": np.array([0])})
    with pytest.raises(ValueError, match="no coordinate"):
        arr.expand(("z",), {})


def test_expand_keeps_the_arrays_absence():
    arr = block([[5.0]], {"x": np.array(["a"]), "y": np.array([0])}).as_unknown()
    got = arr.expand(("z",), {"z": StoredCoord(np.array([1, 2]))})
    assert got.absence == "unknown"


def test_expanding_an_empty_array_leaves_it_empty():
    arr = block([[0.0]], {"x": np.array(["a"]), "y": np.array([0])})
    assert arr.nnz == 0
    got = arr.expand(("z",), {"z": StoredCoord(np.array([1, 2]))})
    assert got.nnz == 0
    assert got.dims == ("x", "y", "z")
