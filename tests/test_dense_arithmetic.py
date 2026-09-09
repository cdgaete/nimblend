"""Scalar arithmetic over a dense array, and what it does to absence."""

import numpy as np

from nimblend import DenseArray

LABELS = {"x": np.array(["a", "b", "c"]), "y": np.array([10, 20])}
VALUES = np.arange(6, dtype=np.float64).reshape(3, 2)
MASK = np.array([[True, True], [False, True], [True, False]])


def empty():
    """Four of six coordinates present, absence contributing nothing."""
    return DenseArray.from_dense(VALUES, LABELS, "empty", MASK)


def unknown():
    """The same four coordinates, tagging absence with NaN."""
    return DenseArray.from_dense(np.where(MASK, VALUES, np.nan), LABELS, "unknown")


def test_a_product_by_a_scalar_scales_every_present_value():
    assert np.array_equal(
        (empty() * 3).to_dense(), [[0.0, 3.0], [0.0, 9.0], [12.0, 0.0]]
    )


def test_a_scalar_multiplies_from_either_side():
    assert np.array_equal((3 * empty()).to_dense(), (empty() * 3).to_dense())


def test_a_sum_with_a_scalar_reaches_the_present_coordinates_alone():
    # the absent (1, 0) and (2, 1) stay absent; 3 is added at the other four
    assert np.array_equal(
        (empty() + 3).to_dense(), [[3.0, 4.0], [0.0, 6.0], [7.0, 0.0]]
    )
    assert (empty() + 3).nnz == 4


def test_a_scalar_adds_from_either_side():
    assert np.array_equal((3 + empty()).to_dense(), (empty() + 3).to_dense())


def test_a_difference_with_a_scalar_subtracts_it():
    assert np.array_equal(
        (empty() - 3).to_dense(), [[-3.0, -2.0], [0.0, 0.0], [1.0, 0.0]]
    )


def test_a_scalar_on_the_left_subtracts_the_array_from_it():
    # 3 - arr, not arr - 3: the present 4 at (2, 0) reaches -1
    assert np.array_equal(
        (3 - empty()).to_dense(), [[3.0, 2.0], [0.0, 0.0], [-1.0, 0.0]]
    )


def test_a_quotient_by_a_scalar_divides_every_present_value():
    assert np.array_equal(
        (empty() / 2).to_dense(), [[0.0, 0.5], [0.0, 1.5], [2.0, 0.0]]
    )


def test_negation_flips_every_present_value():
    assert np.array_equal(
        (-empty()).to_dense(), [[-0.0, -1.0], [0.0, -3.0], [-4.0, 0.0]]
    )
    assert (-empty()).nnz == 4


def test_a_product_by_zero_leaves_the_coordinates_present_and_worth_nothing():
    # a stored 0.0 is a coordinate that is present, not one that is absent
    got = empty() * 0
    assert got.nnz == 4
    assert np.array_equal(got.to_dense(), np.zeros((3, 2)))


def test_a_scalar_leaves_the_absent_coordinates_absent():
    for got in (empty() * 3, empty() + 3, empty() - 3, 3 - empty(), -empty()):
        assert got.nnz == 4, got
        assert not got.present[1, 0] and not got.present[2, 1], got


def test_an_unknown_array_keeps_its_tag_through_a_scalar():
    for got in (unknown() * 2, unknown() + 2, 5 - unknown(), -unknown()):
        assert got.absence == "unknown"
        assert got.nnz == 4
        assert np.isnan(got.data[1, 0]) and np.isnan(got.data[2, 1])


def test_a_scalar_does_not_disturb_the_array_it_was_applied_to():
    arr = empty()
    arr * 3
    -arr
    assert np.array_equal(arr.to_dense(), [[0.0, 1.0], [0.0, 3.0], [4.0, 0.0]])
    assert arr.nnz == 4


def test_the_repr_states_the_frame_the_count_and_the_declaration():
    assert repr(empty()) == (
        "DenseArray(('x', 'y'), shape=(3, 2), 4 present, absence='empty')"
    )
    assert repr(unknown()).endswith("absence='unknown')")
