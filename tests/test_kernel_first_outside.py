import numpy as np

from nimblend import kernel


def test_positions_inside_the_shape_have_no_outside_column():
    idx = np.array([[0, 1, 1], [0, 2, 1]], dtype=np.int32)
    assert kernel.first_outside(idx, (2, 3)) == -1


def test_the_column_is_the_first_outside_along_any_axis():
    # column 2 is past the end of axis 1, column 1 below the start of axis 0
    idx = np.array([[0, -1, 1], [0, 0, 3]], dtype=np.int32)
    assert kernel.first_outside(idx, (2, 3)) == 1


def test_a_position_equal_to_the_extent_is_outside():
    assert kernel.first_outside(np.array([[2]], dtype=np.int32), (2,)) == 0


def test_an_index_with_no_columns_has_no_outside_column():
    assert kernel.first_outside(np.empty((2, 0), dtype=np.int32), (2, 3)) == -1


def test_an_index_over_no_dimensions_has_no_outside_column():
    assert kernel.first_outside(np.empty((0, 4), dtype=np.int32), ()) == -1
