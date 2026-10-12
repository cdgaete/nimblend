import numpy as np
import pytest

import nimblend as nb


def coords():
    labels = np.array(["a", "b"])
    return {"r": nb.StoredCoord(labels), "k": nb.StoredCoord(labels)}


def repeated():
    # (a, a) is given twice: a production and an own use of one product
    labels = {"r": np.array(["a", "a", "b"]), "k": np.array(["a", "a", "b"])}
    return labels, np.array([1.0, -0.25, 3.0])


def test_a_repeated_coordinate_raises_by_default():
    labels, values = repeated()
    with pytest.raises(
        ValueError, match=r"index \(0, 0\) is repeated; pass each coordinate once"
    ):
        nb.from_long(("r", "k"), coords(), labels, values)


def test_raise_passed_explicitly_raises_the_same_error():
    labels, values = repeated()
    with pytest.raises(
        ValueError, match=r"index \(0, 0\) is repeated; pass each coordinate once"
    ):
        nb.from_long(("r", "k"), coords(), labels, values, on_duplicate="raise")


def test_sum_adds_the_values_at_a_repeated_coordinate():
    labels, values = repeated()
    arr = nb.from_long(("r", "k"), coords(), labels, values, on_duplicate="sum")
    assert arr.nnz == 2
    assert arr.to_dense().tolist() == [[0.75, 0.0], [0.0, 3.0]]


def test_sum_adds_the_rows_of_a_coordinate_in_input_order_with_reduceat():
    # the rows of `a` are interleaved with those of `b`; a stable sort keeps
    # each coordinate's rows in input order, and np.add.reduceat adds them.
    # The three magnitudes make the result depend on the order of the rows.
    one = {"r": nb.StoredCoord(np.array(["a", "b"]))}
    at = {"r": np.array(["a", "b", "a", "b", "a"])}
    for a_values in ([1e16, 1.0, -1e16], [1e16, -1e16, 1.0], [1.0, 1e16, -1e16]):
        values = np.array([a_values[0], 5.0, a_values[1], 7.0, a_values[2]])
        arr = nb.from_long(("r",), one, at, values, on_duplicate="sum")
        expected = np.add.reduceat(np.array(a_values), [0])[0]
        assert arr.values().tolist() == [expected, 12.0]


def test_two_rows_sum_to_the_same_value_in_either_order():
    one = {"r": nb.StoredCoord(np.array(["a"]))}
    at = {"r": np.array(["a", "a"])}
    forward = nb.from_long(("r",), one, at, np.array([0.1, 0.7]), on_duplicate="sum")
    backward = nb.from_long(("r",), one, at, np.array([0.7, 0.1]), on_duplicate="sum")
    assert forward.values().tolist() == backward.values().tolist()


def test_a_sum_of_zero_is_an_entry():
    labels = {"r": np.array(["a", "a"]), "k": np.array(["b", "b"])}
    arr = nb.from_long(
        ("r", "k"), coords(), labels, np.array([2.0, -2.0]), on_duplicate="sum"
    )
    assert arr.nnz == 1
    assert arr.values().tolist() == [0.0]


def test_sum_keeps_the_declared_absence():
    labels, values = repeated()
    arr = nb.from_long(
        ("r", "k"), coords(), labels, values, absence="unknown", on_duplicate="sum"
    )
    assert arr.absence == "unknown"


def test_another_on_duplicate_raises():
    labels, values = repeated()
    with pytest.raises(
        ValueError, match=r"on_duplicate is 'sum' or 'raise'; got 'mean'"
    ):
        nb.from_long(("r", "k"), coords(), labels, values, on_duplicate="mean")


def test_sum_raises_for_a_label_the_coordinate_does_not_contain():
    labels = {"r": np.array(["a", "z"]), "k": np.array(["a", "a"])}
    with pytest.raises(KeyError):
        nb.from_long(("r", "k"), coords(), labels, np.ones(2), on_duplicate="sum")


def test_without_a_repeat_sum_and_raise_return_the_same_array():
    labels = {"r": np.array(["b", "a", "a"]), "k": np.array(["a", "b", "a"])}
    values = np.array([3.0, 2.0, 1.0])
    raised = nb.from_long(("r", "k"), coords(), labels, values)
    summed = nb.from_long(("r", "k"), coords(), labels, values, on_duplicate="sum")
    assert np.array_equal(raised.index, summed.index)
    assert np.array_equal(raised.data, summed.data)


def test_sum_over_no_rows_returns_an_empty_array():
    empty = {"r": np.array([], dtype=str), "k": np.array([], dtype=str)}
    arr = nb.from_long(("r", "k"), coords(), empty, np.array([]), on_duplicate="sum")
    assert arr.nnz == 0
    assert arr.shape == (2, 2)


def test_sum_reads_the_index_matrix_of_a_product_coordinate():
    # a ProductCoord label column is an index matrix with one column per label
    cells = {"p": nb.ProductCoord((2, 3))}
    labels = {"p": np.array([[1, 0, 1], [2, 0, 2]])}
    arr = nb.from_long(
        ("p",), cells, labels, np.array([1.0, 4.0, 2.0]), on_duplicate="sum"
    )
    assert arr.nnz == 2
    assert arr.to_dense().tolist() == [4.0, 0.0, 0.0, 0.0, 0.0, 3.0]
