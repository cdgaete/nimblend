import numpy as np
import pytest

from nimblend import DenseArray

LABELS = {"x": np.array(["a", "b"]), "y": np.array([10, 20, 30])}


def empty(values, mask=None):
    return DenseArray.from_dense(np.asarray(values, float), LABELS, "empty", mask)


def unknown(values):
    return DenseArray.from_dense(np.asarray(values, float), LABELS, "unknown")


def holes(*cells):
    mask = np.ones((2, 3), dtype=bool)
    for cell in cells:
        mask[cell] = False
    return mask


def test_an_empty_absence_is_the_additive_identity():
    a = empty(np.ones((2, 3)), holes((0, 0)))
    b = empty(np.full((2, 3), 2.0), holes((0, 1)))
    total = a + b
    # (0,0) takes b alone, (0,1) takes a alone, the rest take both
    assert total.to_dense().tolist() == [[2.0, 1.0, 3.0], [3.0, 3.0, 3.0]]
    assert total.nnz == 6


def test_an_unknown_absence_propagates_through_a_sum():
    a = unknown([[np.nan, 1.0, 1.0], [1.0, 1.0, 1.0]])
    b = unknown([[2.0, np.nan, 2.0], [2.0, 2.0, 2.0]])
    total = a + b
    assert total.nnz == 4
    assert np.isnan(total.data[0, 0]) and np.isnan(total.data[0, 1])
    assert total.data[1, 0] == 3.0


def test_a_product_is_an_intersection_under_either_declaration():
    a = empty(np.full((2, 3), 3.0), holes((0, 0)))
    b = empty(np.full((2, 3), 2.0), holes((0, 1)))
    assert (a * b).nnz == 4
    c = unknown([[np.nan, 3.0, 3.0], [3.0, 3.0, 3.0]])
    d = unknown([[2.0, np.nan, 2.0], [2.0, 2.0, 2.0]])
    assert (c * d).nnz == 4


def test_a_quotient_refuses_an_absent_denominator():
    a = empty(np.full((2, 3), 6.0))
    b = empty(np.full((2, 3), 2.0), holes((1, 2)))
    with pytest.raises(ValueError, match="denominator is absent"):
        a / b


def test_a_quotient_over_a_present_denominator_divides():
    a = empty(np.full((2, 3), 6.0), holes((0, 0)))
    b = empty(np.full((2, 3), 2.0), holes((0, 0)))
    assert (a / b).to_dense().tolist() == [[0.0, 3.0, 3.0], [3.0, 3.0, 3.0]]


def test_mixing_the_two_declarations_is_refused():
    with pytest.raises(ValueError, match="as_empty"):
        empty(np.ones((2, 3))) + unknown(np.ones((2, 3)))


def test_an_unknown_reduction_states_its_policy_or_raises():
    arr = unknown([[np.nan, 1.0, 2.0], [3.0, 4.0, 5.0]])
    with pytest.raises(ValueError, match="skip=True"):
        arr.sum()
    assert arr.sum(skip=True) == 15.0
    assert arr.sum(fill=0.0) == 15.0
    assert arr.sum(fill=10.0) == 25.0


def test_an_empty_reduction_runs_over_the_entries_present():
    arr = empty([[1.0, 1.0, 2.0], [3.0, 4.0, 5.0]], holes((0, 0)))
    assert arr.sum() == 15.0
    assert arr.sum("x").to_dense().tolist() == [3.0, 5.0, 7.0]
    assert arr.mean(skip=True) == 3.0
    assert arr.min() == 1.0
    assert arr.max() == 5.0


def test_a_reduction_leaving_no_present_entry_leaves_the_row_absent():
    arr = empty([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], holes((0, 0), (1, 0)))
    reduced = arr.sum("x")
    assert reduced.nnz == 2
    assert not bool(reduced.mask[0])


def test_a_stored_zero_is_not_an_absence_in_a_reduction():
    arr = empty([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    assert arr.nnz == 6
    assert arr.mean(skip=True) == 0.0


def test_a_mean_over_an_axis_counts_only_the_entries_present():
    arr = empty([[9.0, 1.0, 2.0], [3.0, 4.0, 5.0]], holes((0, 0)))
    # x=a is absent at y=10, so that column's mean is 3.0 over one entry
    assert arr.mean("x").to_dense().tolist() == [3.0, 2.5, 3.5]
    assert arr.mean("y").to_dense().tolist() == [1.5, 4.0]


def test_a_mean_over_an_axis_with_nothing_present_leaves_the_entry_absent():
    arr = empty(np.ones((2, 3)), holes((0, 0), (1, 0)))
    reduced = arr.mean("x")
    assert reduced.nnz == 2
    assert not bool(reduced.mask[0])


def test_a_quotient_of_two_unknown_arrays_keeps_the_declaration():
    a = unknown(np.full((2, 3), 6.0))
    b = unknown(np.full((2, 3), 2.0))
    assert (a / b).absence == "unknown"
    assert (a / b).to_dense().tolist() == [[3.0] * 3, [3.0] * 3]


def test_restating_the_declaration_an_array_already_holds_returns_it():
    e = empty(np.ones((2, 3)))
    u = unknown(np.ones((2, 3)))
    assert e.as_empty() is e
    assert u.as_unknown() is u
