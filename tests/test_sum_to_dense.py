import numpy as np
import pytest

import nimblend as nb
from nimblend import kernel

COORDS = {
    "x": nb.StoredCoord(np.array(["a", "b", "c"])),
    "k": nb.ProductCoord((4,)),
}


def array(cells, values, absence="empty", dims=("x", "k"), coords=COORDS):
    index = np.array(cells, dtype=np.int32).T.reshape(len(dims), -1)
    return nb.SparseArray(index, np.array(values, float), coords, dims, absence)


def arrays():
    return [
        array([(0, 0), (1, 1), (2, 3)], [1.0, 2.0, 3.0]),
        array([(0, 0), (1, 2)], [10.0, 20.0]),
        array([(0, 0), (2, 3), (1, 1)], [100.0, 200.0, 300.0]),
    ]


def random_arrays(seed, count=4):
    rng = np.random.default_rng(seed)
    held = []
    for _ in range(count):
        keep = np.flatnonzero(rng.random(12) < 0.5)
        cells = np.stack(np.unravel_index(keep, (3, 4))).T
        held.append(array(cells, rng.standard_normal(keep.size)))
    return held


def test_the_dense_sum_has_the_shape_of_the_frame():
    got = nb.sum_to_dense(arrays())
    assert got.shape == (3, 4)
    assert got.dtype == np.float64


def test_the_dense_sum_adds_the_values_at_each_coordinate():
    got = nb.sum_to_dense(arrays())
    assert got.tolist() == [
        [111.0, 0.0, 0.0, 0.0],
        [0.0, 302.0, 20.0, 0.0],
        [0.0, 0.0, 0.0, 203.0],
    ]


@pytest.mark.parametrize("seed", range(6))
def test_the_dense_sum_equals_the_merged_sum_as_dense(seed):
    given = random_arrays(seed)
    assert np.array_equal(nb.sum_to_dense(given), nb.sum_arrays(given).to_dense())


def test_the_dense_sum_merges_no_array(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the arrays were merged")

    given = arrays()
    monkeypatch.setattr(kernel, "merge_sum", refuse)
    assert nb.sum_to_dense(given)[0, 0] == 111.0


def test_the_dense_sum_of_one_array_is_a_new_buffer():
    one = arrays()[0]
    got = nb.sum_to_dense([one])
    assert np.array_equal(got, one.to_dense())
    got[0, 0] = -1.0
    assert one.to_dense()[0, 0] == 1.0


def test_the_dense_sum_over_no_dimensions_is_one_cell():
    cell = nb.SparseArray(np.empty((0, 1), dtype=np.int32), np.array([2.0]), {}, ())
    assert nb.sum_to_dense([cell, cell]) == 4.0
    assert nb.sum_to_dense([cell, cell]).shape == ()


def test_the_dense_sum_reads_an_iterator():
    assert nb.sum_to_dense(iter(arrays()))[2, 3] == 203.0


def test_no_arrays_are_refused():
    with pytest.raises(ValueError, match="no array is given"):
        nb.sum_to_dense([])


def test_arrays_over_different_dimensions_are_refused():
    other = array([(0, 0)], [1.0], dims=("k", "x"), coords=COORDS)
    with pytest.raises(ValueError, match="differ"):
        nb.sum_to_dense([arrays()[0], other])


def test_arrays_with_different_labels_are_refused():
    relabeled = dict(COORDS, x=nb.StoredCoord(np.array(["a", "b", "z"])))
    other = array([(0, 0)], [1.0], coords=relabeled)
    with pytest.raises(ValueError, match="different labels"):
        nb.sum_to_dense([arrays()[0], other])


def test_an_array_of_absence_unknown_is_refused():
    unknown = array([(0, 0)], [1.0], absence="unknown")
    with pytest.raises(ValueError, match="unknown"):
        nb.sum_to_dense([unknown, unknown])


def test_an_object_that_is_not_a_sparse_array_is_refused():
    with pytest.raises(TypeError, match="SparseArray"):
        nb.sum_to_dense([arrays()[0], np.zeros((3, 4))])
