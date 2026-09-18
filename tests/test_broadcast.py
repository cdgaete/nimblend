import numpy as np
import pytest

from nimblend import DenseArray, SparseArray, StoredCoord

LABELS = {"x": np.array(["a", "b"]), "y": np.array([10, 20, 30])}
Z = {"z": StoredCoord(np.array([1, 2]))}


def sparse():
    # entries at (a, 20) and (b, 10) only
    index = np.array([[0, 1], [1, 0]], dtype=np.int32)
    coords = {name: StoredCoord(labels) for name, labels in LABELS.items()}
    return SparseArray(index, np.array([5.0, 7.0]), coords, ("x", "y"))


def dense():
    values = np.zeros((2, 3))
    values[0, 1], values[1, 0] = 5.0, 7.0
    mask = values != 0.0
    return DenseArray.from_dense(values, LABELS, mask=mask)


@pytest.fixture(params=[sparse, dense], ids=["sparse", "dense"])
def arr(request):
    return request.param()


def test_broadcast_returns_the_array_over_exactly_the_given_dims(arr):
    got = arr.broadcast(("y", "z", "x"), Z)
    assert got.dims == ("y", "z", "x")
    assert got.shape == (3, 2, 2)
    assert got.coords["z"] is Z["z"]
    assert got.nnz == 4


def test_broadcast_replicates_each_value_across_a_missing_dimension(arr):
    got = arr.broadcast(("x", "y", "z"), Z).to_dense()
    wanted = np.broadcast_to(arr.to_dense()[:, :, None], (2, 3, 2))
    assert np.array_equal(got, wanted)


def test_broadcast_to_the_same_dims_in_another_order_transposes(arr):
    got = arr.broadcast(("y", "x"), {})
    assert got.dims == ("y", "x")
    assert np.array_equal(got.to_dense(), arr.to_dense().T)


def test_broadcast_to_the_dims_the_array_has_returns_it(arr):
    assert arr.broadcast(("x", "y"), {}) is arr


def test_broadcast_reads_coords_only_for_the_missing_dimensions(arr):
    # a coordinate for a dimension the array has is not read
    other = {"x": StoredCoord(np.array(["zz"])), **Z}
    got = arr.broadcast(("x", "y", "z"), other)
    assert got.coords["x"] == arr.coords["x"]


def test_broadcast_keeps_the_absence(arr):
    got = arr.as_unknown().broadcast(("x", "y", "z"), Z)
    assert got.absence == "unknown"


def test_broadcast_raises_for_a_dimension_of_the_array_not_in_dims(arr):
    with pytest.raises(ValueError, match=r"dimension\(s\) \['y'\] of the array"):
        arr.broadcast(("x", "z"), Z)


def test_broadcast_raises_for_a_missing_dimension_without_a_coordinate(arr):
    with pytest.raises(ValueError, match="no coordinate for dimension"):
        arr.broadcast(("x", "y", "w"), Z)


def test_broadcast_raises_for_a_repeated_dimension(arr):
    with pytest.raises(ValueError, match="more than once"):
        arr.broadcast(("x", "y", "x"), {})


def test_an_empty_sparse_array_broadcasts_to_an_empty_array():
    arr = sparse()
    empty = SparseArray.from_canonical(
        arr.index[:, :0], arr.data[:0], arr.coords, arr.dims
    )
    got = empty.broadcast(("z", "x", "y"), Z)
    assert got.dims == ("z", "x", "y")
    assert got.nnz == 0


def test_addition_over_nested_frames_matches_an_explicit_broadcast():
    wide = sparse().broadcast(("x", "y", "z"), Z)
    narrow = sparse()
    assert np.array_equal(
        (narrow + wide).to_dense(),
        wide.to_dense() + narrow.broadcast(wide.dims, Z).to_dense(),
    )
