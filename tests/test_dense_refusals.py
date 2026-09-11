"""What a dense array refuses, rather than guessing at."""

import numpy as np
import pytest

from nimblend import DenseArray
from nimblend.coords import StoredCoord

LABELS = {"x": np.array(["a", "b"]), "y": np.array([10, 20, 30])}
VALUES = np.arange(6, dtype=np.float64).reshape(2, 3)


def build(values=VALUES, labels=LABELS, absence="empty", mask=None):
    return DenseArray.from_dense(values, labels, absence, mask)


def test_a_declaration_that_is_neither_empty_nor_unknown_is_refused():
    with pytest.raises(ValueError, match="absence is 'empty' or 'unknown'"):
        build(absence="maybe")


def test_a_dimension_without_a_coordinate_is_refused():
    with pytest.raises(ValueError, match=r"no coordinate for dimension\(s\) \['y'\]"):
        DenseArray(VALUES, {"x": StoredCoord(LABELS["x"])}, ("x", "y"))


def test_values_that_do_not_fill_the_frame_are_refused():
    with pytest.raises(ValueError, match="the values have shape"):
        build(values=np.zeros((2, 2)))


def test_a_mask_that_does_not_fill_the_frame_is_refused():
    with pytest.raises(ValueError, match="the mask has shape"):
        build(mask=np.ones((2, 2), dtype=bool))


def test_expanding_onto_a_dimension_already_carried_is_refused():
    with pytest.raises(ValueError, match="already has"):
        build().expand(("x",), {"x": StoredCoord(np.array([1]))})


def test_expanding_without_a_coordinate_is_refused():
    with pytest.raises(ValueError, match=r"no coordinate for dimension\(s\) \['z'\]"):
        build().expand(("z",), {})


def test_transposing_to_a_frame_that_is_not_this_one_is_refused():
    with pytest.raises(
        ValueError, match=r"transpose requires each dimension of \('x', 'y'\) once"
    ):
        build().transpose("x")


def test_restricting_by_a_domain_over_a_dimension_not_carried_is_refused():
    foreign = build().rename({"x": "q"}).domain(("q",))
    with pytest.raises(ValueError, match="pass a domain over dimensions of the array"):
        build().restrict(foreign)


def test_operating_across_frames_that_share_no_dimension_is_refused():
    with pytest.raises(ValueError, match="share no dimension"):
        build() + build().rename({"x": "q", "y": "z"})


def test_operating_across_different_labels_is_refused():
    other = build(labels={"x": np.array(["a", "z"]), "y": LABELS["y"]})
    with pytest.raises(ValueError, match="have different labels"):
        build() + other


def test_a_skip_that_is_not_true_is_refused():
    with pytest.raises(ValueError, match="skip is True or None; got 1"):
        build().sum(skip=1)


def test_stating_both_skip_and_fill_is_refused():
    with pytest.raises(ValueError, match="skip= and fill= are given together"):
        build().sum(skip=True, fill=0.0)


def test_densifying_an_unknown_array_with_a_hole_needs_a_fill():
    holed = np.where([[True, False, True], [True, True, True]], VALUES, np.nan)
    with pytest.raises(ValueError, match="no value at 1 of 6 coordinates; pass fill="):
        build(values=holed, absence="unknown").to_dense()


def test_densifying_an_unknown_array_that_carries_its_whole_frame_needs_none():
    assert np.array_equal(build(absence="unknown").to_dense(), VALUES)
