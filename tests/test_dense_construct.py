import numpy as np
import pytest

from nimblend import DenseArray


def build(values, labels, absence="empty", mask=None):
    return DenseArray.from_dense(values, labels, absence=absence, mask=mask)


LABELS = {"x": np.array(["a", "b"]), "y": np.array([10, 20, 30])}
VALUES = np.arange(6, dtype=np.float64).reshape(2, 3)


def test_dense_array_satisfies_the_contract():
    from conformance import check_array_contract

    check_array_contract(
        lambda values, labels, absence: DenseArray.from_dense(
            values, labels, absence=absence
        )
    )


def test_an_empty_array_carries_a_mask_and_an_unknown_one_carries_none():
    empty = build(VALUES, LABELS, "empty")
    unknown = build(VALUES, LABELS, "unknown")
    assert empty.mask is not None
    assert unknown.mask is None
    assert empty.nnz == 6
    assert unknown.nnz == 6


def test_an_unknown_array_reads_its_absence_from_a_nan():
    values = VALUES.copy()
    values[0, 1] = np.nan
    arr = build(values, LABELS, "unknown")
    assert arr.nnz == 5
    assert arr.values().tolist() == [0.0, 2.0, 3.0, 4.0, 5.0]
    assert arr.coordinates().tolist() == [[0, 0, 1, 1, 1], [0, 2, 0, 1, 2]]


def test_an_empty_array_reads_its_absence_from_the_mask():
    mask = np.ones((2, 3), dtype=bool)
    mask[0, 1] = False
    arr = build(VALUES, LABELS, "empty", mask)
    assert arr.nnz == 5
    assert arr.values().tolist() == [0.0, 2.0, 3.0, 4.0, 5.0]


def test_an_unknown_array_refuses_a_mask():
    with pytest.raises(ValueError, match="carries no mask"):
        build(VALUES, LABELS, "unknown", np.ones((2, 3), dtype=bool))


def test_a_stored_zero_is_present_and_worth_nothing():
    values = np.zeros((2, 3))
    empty = build(values, LABELS, "empty")
    unknown = build(values, LABELS, "unknown")
    assert empty.nnz == 6
    assert unknown.nnz == 6


def test_converting_between_declarations_moves_where_absence_lives():
    mask = np.ones((2, 3), dtype=bool)
    mask[1, 2] = False
    empty = build(VALUES, LABELS, "empty", mask)
    unknown = empty.as_unknown()
    assert unknown.mask is None
    assert np.isnan(unknown.data[1, 2])
    assert unknown.nnz == 5
    back = unknown.as_empty()
    assert back.mask is not None
    assert not bool(back.mask[1, 2])
    assert back.nnz == 5
