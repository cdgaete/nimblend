import numpy as np
import pytest

from nimblend import kernel
from nimblend.buffer import EntryBuffer
from nimblend.sparse import SparseArray


def ijc():
    """A (2, 3, 4) array with four entries, holding only those."""
    values = np.zeros((2, 3, 4))
    values[0, 0, 1] = 5.0
    values[0, 2, 0] = 7.0
    values[1, 1, 2] = 4.0
    values[1, 1, 3] = 9.0
    arr = SparseArray.from_dense(
        values,
        {
            "i": np.array(["p", "q"]),
            "j": np.array([10, 20, 30]),
            "c": np.arange(4),
        },
    )
    keep = arr.data != 0.0
    return SparseArray.from_canonical(
        arr.index[:, keep], arr.data[keep], arr.coords, arr.dims
    )


def test_group_collapses_the_named_dimensions_into_one():
    got = ijc().group(("i", "j"), "g")
    assert got.dims == ("g", "c")
    # three distinct (i, j) coordinates carry the four entries
    assert got.shape == (3, 4)
    assert got.nnz == 4
    assert got.index[0].tolist() == [0, 1, 2, 2]
    assert got.index[1].tolist() == [1, 0, 2, 3]
    assert got.data.tolist() == [5.0, 7.0, 4.0, 9.0]


def test_group_numbers_the_new_dimension_from_the_offset():
    got = ijc().group(("i", "j"), "g", offset=10)
    assert got.index[0].tolist() == [10, 11, 12, 12]


def test_the_new_dimension_reads_back_the_coordinates_it_stands_for():
    got = ijc().group(("i", "j"), "g", offset=10)
    assert got.coords["g"].to_index(np.array([10, 12])).tolist() == [[0, 1], [0, 1]]


def test_group_leaves_the_result_canonical():
    got = ijc().group(("i", "j"), "g")
    assert kernel.is_canonical(got.index, got.shape)


def test_group_drops_entries_a_supplied_domain_does_not_carry():
    arr = ijc()
    first_two = SparseArray.from_canonical(
        arr.index[:, :2], arr.data[:2], arr.coords, arr.dims
    )
    got = arr.group(("i", "j"), "g", domain=first_two.domain(("i", "j")))
    assert got.nnz == 2
    assert got.data.tolist() == [5.0, 7.0]


def test_group_writes_into_a_supplied_buffer():
    arr = ijc()
    buffer = EntryBuffer(2, 4)
    got = arr.group(("i", "j"), "g", out=buffer.reserve(4))
    assert got.data.base is buffer.data or got.data.base is buffer.data.base
    assert buffer.data[:4].tolist() == [5.0, 7.0, 4.0, 9.0]


def test_group_refuses_dimensions_that_are_not_a_leading_prefix():
    with pytest.raises(ValueError, match="leading prefix"):
        ijc().group(("j", "c"), "g")


def test_group_refuses_a_name_the_remaining_dimensions_already_use():
    with pytest.raises(ValueError, match="is among the remaining dimensions"):
        ijc().group(("i", "j"), "c")


def test_group_keeps_the_arrays_absence():
    got = ijc().as_unknown().group(("i", "j"), "g")
    assert got.absence == "unknown"


def test_group_refuses_a_negative_offset():
    # a negative index wraps to the far end of the axis, scrambling the block
    with pytest.raises(ValueError, match="offset"):
        ijc().group(("i", "j"), "g", offset=-1)


def test_grouping_no_dimensions_collapses_every_entry_into_one_row():
    arr = ijc()
    got = arr.group((), "g", offset=5)
    assert got.dims == ("g", "i", "j", "c")
    assert got.shape[0] == 1
    assert got.nnz == arr.nnz
    assert got.coordinates()[0].tolist() == [5, 5, 5, 5]


def test_a_domain_over_no_dimensions_carries_one_member():
    assert ijc().domain(()).size == 1
    assert ijc().domain(()).dims == ()


def test_a_domain_over_no_dimensions_of_an_empty_array_carries_none():
    arr = ijc()
    empty = SparseArray.from_canonical(
        arr.index[:, :0], arr.data[:0], arr.coords, arr.dims
    )
    assert empty.domain(()).size == 0
