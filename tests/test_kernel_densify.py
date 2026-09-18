import numpy as np

from nimblend import kernel


def test_densify_places_each_value_and_fills_the_rest():
    idx = np.array([[0, 1], [2, 0]], dtype=np.int32)
    got = kernel.densify(idx, np.array([5.0, 7.0]), (2, 3), -1.0)
    assert got.tolist() == [[-1.0, -1.0, 5.0], [7.0, -1.0, -1.0]]


def test_densify_of_no_entries_is_all_fill():
    got = kernel.densify(np.empty((2, 0), dtype=np.int32), np.empty(0), (2, 2), 3.0)
    assert got.tolist() == [[3.0, 3.0], [3.0, 3.0]]


def test_densify_over_no_axes_is_one_cell():
    got = kernel.densify(np.empty((0, 1), dtype=np.int32), np.array([4.0]), (), 0.0)
    assert got.shape == ()
    assert float(got) == 4.0
    empty = kernel.densify(np.empty((0, 0), dtype=np.int32), np.empty(0), (), 9.0)
    assert float(empty) == 9.0
