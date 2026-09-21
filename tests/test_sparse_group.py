import re

import numpy as np
import pytest

from nimblend import kernel
from nimblend.buffer import EntryBuffer
from nimblend.coords import ProductCoord
from nimblend.domain import Domain
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


def test_group_numbers_the_new_dimension_from_start_inside_the_given_coord():
    got = ijc().group(("i", "j"), "g", coord=ProductCoord((20,)), start=10)
    assert got.coords["g"] == ProductCoord((20,))
    assert got.shape == (20, 4)
    assert got.index[0].tolist() == [10, 11, 12, 12]


def test_the_new_dimension_reads_back_the_coordinates_it_stands_for():
    got = ijc().group(("i", "j"), "g")
    assert got.coords["g"].to_index(np.array([0, 2])).tolist() == [[0, 1], [0, 1]]


def test_without_a_coord_the_new_dimension_is_the_domain_numbered_from_zero():
    arr = ijc()
    got = arr.group(("i", "j"), "g")
    assert got.coords["g"] == arr.domain(("i", "j")).as_coord()


def test_every_position_of_a_numbered_group_is_inside_its_coordinate():
    got = ijc().group(("i", "j"), "g", coord=ProductCoord((20,)), start=10)
    dense = got.to_dense()
    assert dense.shape == (20, 4)
    assert dense[10, 1] == 5.0
    assert dense[12, 3] == 9.0
    assert got.domain().labels()["g"][:, 0].tolist() == [10]
    assert got.restrict(Domain.full(got.dims, got.coords)).nnz == got.nnz
    _, _, indptr = got.to_csr()
    assert indptr.tolist() == [0] * 11 + [1, 2, 4] + [4] * 7


def test_group_raises_for_a_start_without_a_coord_that_covers_it():
    # the default coord has one position per member and no room for a start
    with pytest.raises(ValueError, match="has extent 3; pass a smaller start"):
        ijc().group(("i", "j"), "g", start=5)


def test_group_raises_for_members_that_end_beyond_the_coord():
    message = (
        "3 member(s) numbered from 18 end at position 20, and dimension 'g' "
        "has extent 20; pass a smaller start or a larger coord"
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        ijc().group(("i", "j"), "g", coord=ProductCoord((20,)), start=18)


def test_group_accepts_members_that_end_at_the_last_position_of_the_coord():
    got = ijc().group(("i", "j"), "g", coord=ProductCoord((20,)), start=17)
    assert got.index[0].tolist() == [17, 18, 19, 19]


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


def test_group_refuses_a_negative_start():
    # a negative index wraps to the far end of the axis, scrambling the block
    with pytest.raises(ValueError, match="start -1 is negative"):
        ijc().group(("i", "j"), "g", coord=ProductCoord((20,)), start=-1)


def test_grouping_no_dimensions_collapses_every_entry_into_one_row():
    arr = ijc()
    got = arr.group((), "g", coord=ProductCoord((6,)), start=5)
    assert got.dims == ("g", "i", "j", "c")
    assert got.shape[0] == 6
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


def test_group_raises_for_a_destination_larger_than_its_entries():
    # a slot left unwritten would reach the matrix as whatever it held
    buffer = EntryBuffer(2, 5)
    with pytest.raises(ValueError, match=re.escape("destination has 5 entries and 4")):
        ijc().group(("i", "j"), "g", out=buffer.reserve(5))


def test_group_raises_for_a_destination_smaller_than_its_entries():
    buffer = EntryBuffer(2, 3)
    with pytest.raises(ValueError, match=re.escape("destination has 3 entries and 4")):
        ijc().group(("i", "j"), "g", out=buffer.reserve(3))


def test_group_writes_nothing_into_a_destination_of_another_size():
    buffer = EntryBuffer(2, 5)
    buffer.index[:] = -7
    buffer.data[:] = -7.0
    with pytest.raises(ValueError):
        ijc().group(("i", "j"), "g", out=buffer.reserve(5))
    assert (buffer.index == -7).all() and (buffer.data == -7.0).all()


def test_group_asks_reserve_for_exactly_the_entries_it_writes():
    buffer = EntryBuffer(2, 10)
    asked = []

    def reserve(n):
        asked.append(n)
        return buffer.reserve(n)

    got = ijc().group(("i", "j"), "g", reserve=reserve)
    expected = ijc().group(("i", "j"), "g")
    assert asked == [4]
    assert buffer.at == 4
    assert got.coordinates().tolist() == expected.coordinates().tolist()
    assert buffer.data[:4].tolist() == expected.values().tolist()


def test_group_raises_for_a_reservation_of_another_size():
    buffer = EntryBuffer(2, 10)
    with pytest.raises(ValueError, match=re.escape("destination has 6 entries and 4")):
        ijc().group(("i", "j"), "g", reserve=lambda n: buffer.reserve(n + 2))


def test_group_raises_for_out_and_reserve_together():
    buffer = EntryBuffer(2, 10)
    with pytest.raises(ValueError, match="pass out or reserve"):
        ijc().group(("i", "j"), "g", out=buffer.reserve(4), reserve=buffer.reserve)


def test_a_dense_array_groups_through_reserve():
    arr = ijc()
    dense = arr.to_dense()
    from nimblend.dense import DenseArray

    held = DenseArray(dense, arr.coords, arr.dims, mask=dense != 0.0)
    buffer = EntryBuffer(2, 10)
    got = held.group(("i", "j"), "g", reserve=buffer.reserve)
    assert got.nnz == 4 and buffer.at == 4
