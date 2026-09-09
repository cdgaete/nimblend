"""Values moved along a dimension, dropping at the ends or wrapping."""

import numpy as np
import pytest

from nimblend import DenseArray

LABELS = {"x": np.array(["a", "b", "c"]), "y": np.array([10, 20])}
VALUES = np.arange(6, dtype=np.float64).reshape(3, 2)
MASK = np.array([[True, True], [False, True], [True, False]])


def full():
    """Every coordinate of a 3x2 frame present."""
    return DenseArray.from_dense(VALUES, LABELS)


def holed():
    """The same frame with (1, 0) and (2, 1) absent."""
    return DenseArray.from_dense(VALUES, LABELS, "empty", MASK)


def test_a_positive_shift_moves_values_forward_and_drops_the_far_end():
    # row a leaves the frame; rows a and b arrive at b and c
    got = full().shift({"x": 1})
    assert np.array_equal(got.to_dense(), [[0.0, 0.0], [0.0, 1.0], [2.0, 3.0]])
    assert got.nnz == 4


def test_a_negative_shift_moves_values_back_and_drops_the_near_end():
    got = full().shift({"x": -1})
    assert np.array_equal(got.to_dense(), [[2.0, 3.0], [4.0, 5.0], [0.0, 0.0]])
    assert got.nnz == 4


def test_a_roll_wraps_what_a_shift_would_drop():
    got = full().roll({"x": 1})
    assert np.array_equal(got.to_dense(), [[4.0, 5.0], [0.0, 1.0], [2.0, 3.0]])
    assert got.nnz == 6


def test_a_shift_carries_absence_with_the_values():
    # (1, 0) is absent and lands on (2, 0); row a arrives whole at b
    got = holed().shift({"x": 1})
    assert np.array_equal(got.to_dense(), [[0.0, 0.0], [0.0, 1.0], [0.0, 3.0]])
    assert got.nnz == 3


def test_a_roll_carries_absence_round_the_ends():
    got = holed().roll({"x": 1})
    assert np.array_equal(got.to_dense(), [[4.0, 0.0], [0.0, 1.0], [0.0, 3.0]])
    assert got.nnz == 4


def test_a_shift_past_the_extent_leaves_nothing():
    got = full().shift({"x": 5})
    assert got.nnz == 0
    assert np.array_equal(got.to_dense(), np.zeros((3, 2)))


def test_two_dimensions_shift_in_one_call():
    # only (0, 0), worth 0.0, survives both drops, and it survives as present
    got = holed().shift({"x": 1, "y": 1})
    assert got.nnz == 1
    assert got.present[1, 1]
    assert np.array_equal(got.to_dense(), np.zeros((3, 2)))


def test_a_shift_keeps_the_frame_it_was_given():
    got = full().shift({"x": 1})
    assert got.dims == ("x", "y")
    assert got.shape == (3, 2)
    assert np.array_equal(got.coords["x"].labels, LABELS["x"])


def test_an_unknown_array_shifts_and_keeps_its_tag():
    tagged = DenseArray.from_dense(np.where(MASK, VALUES, np.nan), LABELS, "unknown")
    got = tagged.shift({"x": 1})
    assert got.absence == "unknown"
    assert got.nnz == 3
    assert np.isnan(got.data[0, 0]) and np.isnan(got.data[0, 1])


def test_a_mode_that_is_neither_dropping_nor_wrapping_is_refused():
    with pytest.raises(ValueError, match="mode is 'drop' or 'wrap'"):
        full().shift({"x": 1}, mode="sideways")


def test_a_shift_does_not_disturb_the_array_it_was_applied_to():
    arr = holed()
    arr.shift({"x": 1})
    arr.roll({"x": 1})
    assert np.array_equal(arr.to_dense(), [[0.0, 1.0], [0.0, 3.0], [4.0, 0.0]])
    assert arr.nnz == 4


def test_a_shift_of_nothing_moves_and_drops_nothing():
    # a zero shift takes nothing out of the frame, so nothing leaves it
    for arr in (full(), holed()):
        got = arr.shift({"x": 0})
        assert got.nnz == arr.nnz
        assert np.array_equal(got.to_dense(), arr.to_dense())


def test_a_roll_of_nothing_moves_nothing():
    for arr in (full(), holed()):
        got = arr.roll({"x": 0})
        assert got.nnz == arr.nnz
        assert np.array_equal(got.to_dense(), arr.to_dense())


def test_a_zero_shift_beside_a_real_one_leaves_its_own_dimension_alone():
    got = full().shift({"x": 1, "y": 0})
    assert np.array_equal(got.to_dense(), [[0.0, 0.0], [0.0, 1.0], [2.0, 3.0]])
    assert got.nnz == 4
