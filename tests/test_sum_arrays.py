import functools
import operator

import numpy as np
import pytest

import nimblend as nb

COORDS = {
    "x": nb.StoredCoord(np.array(["a", "b", "c"])),
    "k": nb.ProductCoord((4,)),
}


def array(cells, values, absence="empty", dims=("x", "k")):
    index = np.array(cells, dtype=np.int32).T.reshape(len(dims), -1)
    return nb.SparseArray(index, np.array(values, float), COORDS, dims, absence)


def arrays(absence):
    return [
        array([(0, 0), (1, 1), (2, 3)], [1.0, 2.0, 3.0], absence),
        array([(0, 0), (1, 2)], [10.0, 20.0], absence),
        array([(0, 0), (2, 3), (1, 1)], [100.0, 200.0, 300.0], absence),
    ]


def same(left, right):
    return (
        left.dims == right.dims
        and left.absence == right.absence
        and left.coordinates().tolist() == right.coordinates().tolist()
        and left.values().tolist() == right.values().tolist()
    )


@pytest.mark.parametrize("absence", ["empty", "unknown"])
def test_the_sum_equals_the_pairwise_sum(absence):
    given = arrays(absence)
    assert same(nb.sum_arrays(given), functools.reduce(operator.add, given))


def test_an_empty_absence_keeps_every_coordinate():
    got = nb.sum_arrays(arrays("empty"))
    assert got.coordinates().tolist() == [[0, 1, 1, 2], [0, 1, 2, 3]]
    assert got.values().tolist() == [111.0, 302.0, 20.0, 203.0]


def test_an_unknown_absence_keeps_the_coordinates_every_array_has():
    got = nb.sum_arrays(arrays("unknown"))
    assert got.coordinates().tolist() == [[0], [0]]
    assert got.values().tolist() == [111.0]


def test_the_result_is_canonical_over_the_frame_of_the_arrays():
    got = nb.sum_arrays(arrays("empty"))
    assert nb.is_canonical(got.coordinates(), got.shape)
    assert got.coords == COORDS


def test_one_array_is_returned_as_its_sum():
    one = arrays("empty")[0]
    assert same(nb.sum_arrays([one]), one)


def test_no_arrays_raise():
    with pytest.raises(ValueError, match="no array is given; pass at least one"):
        nb.sum_arrays([])


def test_arrays_over_different_dimensions_raise():
    swapped = array([(0, 0)], [1.0], dims=("k", "x"))
    with pytest.raises(ValueError, match="differ; conform one to the other first"):
        nb.sum_arrays([arrays("empty")[0], swapped])


def test_arrays_with_different_absence_raise():
    with pytest.raises(ValueError, match="declares absence"):
        nb.sum_arrays([arrays("empty")[0], arrays("unknown")[1]])


def test_a_dense_array_raises():
    dense = nb.DenseArray(np.zeros((3, 4)), COORDS, ("x", "k"))
    with pytest.raises(TypeError, match="sum_arrays adds SparseArray"):
        nb.sum_arrays([arrays("empty")[0], dense])
