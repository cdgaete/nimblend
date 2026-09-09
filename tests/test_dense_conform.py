"""A dense array read at exactly the labels asked for, in the order asked for."""

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


def test_labels_are_read_in_the_order_they_are_named():
    got = full().conform(
        ["x", "y"], {"x": np.array(["c", "a"]), "y": np.array([20, 10])}
    )
    assert np.array_equal(got.to_dense(), [[5.0, 4.0], [1.0, 0.0]])
    assert got.dims == ("x", "y")
    assert got.shape == (2, 2)


def test_a_subset_of_labels_narrows_the_frame():
    got = full().conform(["x", "y"], {"x": np.array(["b"]), "y": np.array([20])})
    assert np.array_equal(got.to_dense(), [[3.0]])
    assert got.shape == (1, 1)


def test_the_dimensions_come_back_in_the_order_conform_was_given():
    got = full().conform(
        ["y", "x"], {"x": np.array(["c", "a"]), "y": np.array([20, 10])}
    )
    assert got.dims == ("y", "x")
    assert np.array_equal(got.to_dense(), [[5.0, 1.0], [4.0, 0.0]])


def test_a_repeated_dimension_order_is_the_frame_itself():
    got = full().conform(["x", "y"], LABELS)
    assert got.dims == ("x", "y")
    assert np.array_equal(got.to_dense(), VALUES)


def test_absence_is_read_along_with_the_values():
    # x=c carries a value at y=10 alone, so reading it at y=20 reaches nothing
    got = holed().conform(["x", "y"], {"x": np.array(["c", "a"]), "y": np.array([20])})
    assert got.nnz == 1
    assert np.array_equal(got.to_dense(), [[0.0], [1.0]])
    assert not got.present[0, 0]


def test_the_labels_asked_for_are_the_labels_carried_back():
    wanted = np.array(["c", "a"])
    got = full().conform(["x", "y"], {"x": wanted, "y": np.array([20, 10])})
    assert np.array_equal(got.coords["x"].labels, wanted)
    assert np.array_equal(got.coords["y"].labels, np.array([20, 10]))


def test_an_unknown_array_conforms_and_keeps_its_tag():
    tagged = DenseArray.from_dense(np.where(MASK, VALUES, np.nan), LABELS, "unknown")
    got = tagged.conform(["x", "y"], {"x": np.array(["c", "a"]), "y": np.array([20])})
    assert got.absence == "unknown"
    assert got.nnz == 1
    assert np.isnan(got.data[0, 0])


def test_a_label_the_array_does_not_carry_is_refused():
    with pytest.raises(KeyError, match="is not carried"):
        full().conform(["x", "y"], {"x": np.array(["z"]), "y": np.array([10])})


def test_conform_does_not_disturb_the_array_it_was_applied_to():
    arr = holed()
    arr.conform(["x", "y"], {"x": np.array(["c", "a"]), "y": np.array([20])})
    assert np.array_equal(arr.to_dense(), [[0.0, 1.0], [0.0, 3.0], [4.0, 0.0]])
    assert arr.nnz == 4


def test_a_label_named_twice_is_refused():
    # a repeat would ask one position to occupy two
    with pytest.raises(ValueError, match="is named twice for dimension 'x'"):
        full().conform(["x", "y"], {"x": np.array(["a", "a"]), "y": LABELS["y"]})


def test_a_repeat_is_refused_on_whichever_dimension_carries_it():
    with pytest.raises(ValueError, match="is named twice for dimension 'y'"):
        full().conform(["x", "y"], {"x": LABELS["x"], "y": np.array([10, 10])})


def test_a_repeat_is_refused_before_any_value_is_read():
    with pytest.raises(ValueError, match="conform reads each position"):
        holed().conform(["x", "y"], {"x": np.array(["c", "c"]), "y": np.array([20])})
