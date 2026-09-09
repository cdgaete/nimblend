import warnings

import numpy as np
import pytest

from nimblend import kernel
from nimblend.coords import StoredCoord
from nimblend.sparse import SparseArray


def one_d(positions, values, size=4, absence="empty"):
    index = np.array([positions], dtype=np.int32)
    coords = {"x": StoredCoord(np.arange(size))}
    return SparseArray(index, np.array(values, dtype=float), coords, ("x",), absence)


def test_empty_addition_is_a_union_with_absence_as_identity():
    a = one_d([0, 1], [1.0, 2.0])
    b = one_d([1, 3], [10.0, 30.0])
    got = a + b
    assert list(got.index[0]) == [0, 1, 3]
    assert list(got.data) == [1.0, 12.0, 30.0]


def test_unknown_addition_propagates_absence():
    a = one_d([0, 1], [1.0, 2.0], absence="unknown")
    b = one_d([1, 3], [10.0, 30.0], absence="unknown")
    got = a + b
    assert list(got.index[0]) == [1]
    assert list(got.data) == [12.0]


def test_multiplication_is_an_intersection_under_both_meanings():
    for absence in ("empty", "unknown"):
        a = one_d([0, 1], [2.0, 3.0], absence=absence)
        b = one_d([1, 2], [10.0, 20.0], absence=absence)
        got = a * b
        assert list(got.index[0]) == [1]
        assert list(got.data) == [30.0]


def test_mixing_the_two_meanings_raises():
    a = one_d([0], [1.0], absence="empty")
    b = one_d([0], [1.0], absence="unknown")
    with pytest.raises(ValueError, match="as_empty"):
        a + b


def test_conversion_lets_them_combine():
    a = one_d([0, 1], [1.0, 2.0], absence="empty")
    b = one_d([1], [5.0], absence="unknown")
    got = a + b.as_empty()
    assert list(got.data) == [1.0, 7.0]


def test_scalar_arithmetic_touches_only_stored_entries():
    a = one_d([0, 2], [1.0, 3.0])
    assert list((a * 2).data) == [2.0, 6.0]
    assert list((a + 1).data) == [2.0, 4.0]
    assert list((-a).data) == [-1.0, -3.0]


def test_division_by_an_absent_denominator_raises():
    a = one_d([0, 1], [1.0, 2.0])
    b = one_d([1], [4.0])
    with pytest.raises(ValueError, match="denominator is absent"):
        a / b


def test_subtraction_negates_the_right_hand_entries():
    a = one_d([0, 1], [1.0, 2.0])
    b = one_d([1, 3], [10.0, 30.0])
    got = a - b
    assert list(got.index[0]) == [0, 1, 3]
    assert list(got.data) == [1.0, -8.0, -30.0]


def two_d(positions, values, shape=(5, 4), absence="empty"):
    index = np.array(positions, dtype=np.int32).T.reshape(2, -1)
    coords = {
        "x": StoredCoord(np.arange(shape[0])),
        "y": StoredCoord(np.arange(shape[1])),
    }
    return SparseArray(
        index, np.array(values, dtype=float), coords, ("x", "y"), absence
    )


def test_a_combined_result_is_already_in_canonical_order():
    # from_canonical trusts this of every _combine result: partial overlap,
    # disjoint operands, and one operand a subset of the other.
    cases = [
        (one_d([0, 1], [1.0, 2.0]), one_d([1, 3], [10.0, 30.0])),
        (one_d([0, 2], [1.0, 3.0]), one_d([1, 3], [10.0, 30.0])),
        (one_d([0, 1, 2, 3], [1.0, 2.0, 3.0, 4.0]), one_d([2], [9.0])),
        (
            two_d([(0, 3), (2, 1), (4, 0)], [1.0, 2.0, 3.0]),
            two_d([(0, 0), (2, 1)], [5.0, 6.0]),
        ),
        (two_d([(1, 1)], [1.0]), two_d([(0, 0), (4, 3)], [5.0, 6.0])),
    ]
    for a, b in cases:
        for got in (a + b, a * b):
            assert kernel.is_canonical(got.index, got.shape)
            assert got.index.dtype == np.int32 and got.data.dtype == np.float64


def test_a_combined_result_owns_buffers_neither_operand_shares():
    a = one_d([0, 1], [1.0, 2.0])
    b = one_d([1, 3], [10.0, 30.0])
    got = a + b
    for operand in (a, b):
        assert not np.shares_memory(got.index, operand.index)
        assert not np.shares_memory(got.data, operand.data)


def labelled(labels, positions, values, absence="empty"):
    coords = {"x": StoredCoord(np.array(labels))}
    return SparseArray(
        np.array([positions], dtype=np.int32),
        np.array(values, dtype=float),
        coords,
        ("x",),
        absence,
    )


def test_addition_refuses_operands_whose_labels_differ():
    # the same dimension name and extent, naming different members
    a = labelled(["a", "b"], [0, 1], [1.0, 2.0])
    b = labelled(["p", "q"], [0, 1], [10.0, 20.0])
    with pytest.raises(ValueError, match="different labels"):
        a + b
    with pytest.raises(ValueError, match="different labels"):
        a - b


def test_addition_refuses_operands_whose_extents_differ():
    a = labelled(["a", "b"], [0, 1], [1.0, 2.0])
    b = labelled(["a", "b", "c"], [0, 1], [10.0, 20.0])
    with pytest.raises(ValueError, match="different labels"):
        a + b


def test_a_broadcast_product_refuses_a_shared_dimension_with_other_labels():
    wide = SparseArray(
        np.array([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=np.int32),
        np.array([1.0, 2.0, 3.0, 4.0]),
        {"x": StoredCoord(np.array(["a", "b"])), "y": StoredCoord(np.array([10, 20]))},
        ("x", "y"),
    )
    narrow = labelled(["p", "q"], [0, 1], [5.0, 6.0])
    with pytest.raises(ValueError, match="different labels"):
        wide * narrow


def test_arithmetic_accepts_equal_labels_held_in_separate_coordinates():
    a = labelled(["a", "b"], [0, 1], [1.0, 2.0])
    b = labelled(["a", "b"], [1], [10.0])
    assert list((a + b).data) == [1.0, 12.0]


def test_dividing_by_a_stored_zero_answers_the_arithmetic():
    # a stored zero is a value the array carries, so the quotient is the IEEE
    # result; refusing a non-finite value is a caller's rule, not nimblend's
    a = one_d([0, 1, 2], [1.0, 2.0, 0.0])
    b = one_d([0, 1, 2], [0.0, 2.0, 0.0])
    got = a / b
    assert got.data[0] == np.inf
    assert got.data[1] == 1.0
    assert np.isnan(got.data[2])


def test_dividing_by_zero_reports_no_warning():
    a = one_d([0], [1.0])
    b = one_d([0], [0.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert (a / b).data[0] == np.inf
        assert (a / 0.0).data[0] == np.inf


def test_arithmetic_beside_division_still_reports_a_numpy_warning():
    # the quiet is scoped to a zero denominator, not spread over arithmetic
    a = one_d([0], [1e308])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(RuntimeWarning, match="overflow"):
            a + a


def test_division_needs_a_denominator_at_every_numerator_entry():
    # the numerator's entries decide: a denominator carrying more is fine,
    # carrying fewer is not
    a = one_d([1, 2], [6.0, 8.0])
    assert list((a / one_d([1, 2], [2.0, 4.0])).data) == [3.0, 2.0]
    assert list((a / one_d([0, 1, 2, 3], [9.0, 2.0, 4.0, 9.0])).data) == [3.0, 2.0]
    for short in (one_d([1], [2.0]), one_d([0, 3], [1.0, 1.0]), one_d([], [])):
        with pytest.raises(ValueError, match="denominator is absent"):
            a / short


def test_dividing_an_empty_numerator_answers_an_empty_array():
    assert (one_d([], []) / one_d([0, 1], [2.0, 3.0])).nnz == 0
