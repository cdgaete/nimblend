import numpy as np

from nimblend.coords import StoredCoord
from nimblend.sparse import SparseArray


def xy(values):
    """A 2x3 array holding only the nonzero cells of `values`."""
    arr = SparseArray.from_dense(
        np.asarray(values, dtype=np.float64),
        {"x": np.array(["a", "b"]), "y": np.array([10, 20, 30])},
    )
    keep = arr.data != 0.0
    return SparseArray.from_canonical(
        arr.index[:, keep], arr.data[keep], arr.coords, arr.dims
    )


def test_values_answer_the_entries_in_canonical_order():
    arr = xy([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
    assert arr.values().tolist() == [1.0, 2.0, 3.0]


def test_values_and_coordinates_are_the_two_halves_of_one_entry_list():
    arr = xy([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
    assert arr.values().size == arr.nnz
    assert arr.coordinates().shape == (2, arr.nnz)
    assert arr.coordinates().tolist() == [[0, 0, 1], [0, 2, 1]]


def test_values_are_a_copy_that_does_not_write_through():
    arr = xy([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    got = arr.values()
    got[0] = 99.0
    assert arr.data[0] == 1.0


def test_values_keep_a_stored_zero():
    # a stored 0.0 is a value the array carries, not an absence
    arr = SparseArray.from_canonical(
        np.array([[0, 1]], dtype=np.int32),
        np.array([0.0, 5.0]),
        {"x": StoredCoord(np.array(["a", "b"]))},
        ("x",),
    )
    assert arr.values().tolist() == [0.0, 5.0]


def test_values_of_an_array_carrying_no_entry_is_empty():
    arr = xy([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    assert arr.nnz == 0
    assert arr.values().size == 0
    assert arr.values().dtype == np.float64


def test_values_read_a_domain_in_member_order_after_restrict():
    # the composition a consumer states: restrict to a domain, then read
    arr = xy([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
    first_row = SparseArray.from_canonical(
        arr.index[:, :2], arr.data[:2], arr.coords, arr.dims
    )
    assert arr.restrict(first_row.domain()).values().tolist() == [1.0, 2.0]
