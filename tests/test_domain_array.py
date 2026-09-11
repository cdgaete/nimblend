import numpy as np
import pytest

from nimblend import kernel
from nimblend.coords import ProductCoord, StoredCoord
from nimblend.domain import Domain


def coords_xy():
    return {
        "x": StoredCoord(np.array(["a", "b"])),
        "y": StoredCoord(np.array([10, 20, 30])),
    }


def domain(codes):
    return Domain(np.array(codes, dtype=np.int64), ("x", "y"), coords_xy(), (2, 3))


def test_a_domain_values_its_members_in_the_order_it_carries_them():
    # members (a, 10), (a, 30) and (b, 20) are codes 0, 2 and 4
    got = domain([0, 2, 4]).array(np.array([5.0, 6.0, 7.0]))
    assert got.dims == ("x", "y")
    assert got.nnz == 3
    dense = got.to_dense()
    assert dense[0, 0] == 5.0
    assert dense[0, 2] == 6.0
    assert dense[1, 1] == 7.0


def test_the_array_carries_the_domains_own_coordinates():
    held = coords_xy()
    got = Domain(np.array([0, 4], dtype=np.int64), ("x", "y"), held, (2, 3)).array(
        np.array([1.0, 2.0])
    )
    assert got.coords["x"] is held["x"]
    assert got.coords["y"] is held["y"]


def test_a_value_column_of_another_length_is_refused():
    with pytest.raises(ValueError, match="requires values of shape"):
        domain([0, 2, 4]).array(np.array([5.0, 6.0]))


def test_the_array_declares_the_absence_it_is_given():
    assert domain([0]).array(np.array([1.0]), absence="unknown").absence == "unknown"
    assert domain([0]).array(np.array([1.0])).absence == "empty"


def test_the_entries_are_canonical_as_written():
    got = domain([0, 2, 4]).array(np.array([5.0, 6.0, 7.0]))
    assert kernel.is_canonical(got.index, got.shape)


def test_a_full_domain_values_the_cells_of_the_product_in_row_major_order():
    # the property a caller relies on to state a whole grid: a full domain's
    # members ascend with the ravel key, which is what `values.ravel()` orders
    values = np.arange(6, dtype=np.float64).reshape(2, 3)
    got = Domain.full(("x", "y"), coords_xy()).array(values.ravel())
    assert np.array_equal(got.to_dense(), values)


def test_an_empty_domain_answers_an_array_carrying_nothing():
    got = domain([]).array(np.empty(0))
    assert got.nnz == 0
    assert got.dims == ("x", "y")


def test_a_member_is_paired_with_its_own_position_along_a_new_dimension():
    # three members numbered from 4 in an extent of ten
    got = domain([0, 2, 4]).identity("k", ProductCoord((10,)), start=4)
    assert got.dims == ("x", "y", "k")
    assert got.nnz == 3
    assert got.coordinates().tolist() == [[0, 0, 1], [0, 2, 1], [4, 5, 6]]
    assert got.values().tolist() == [1.0, 1.0, 1.0]


def test_the_positions_are_the_ones_as_coord_states():
    held = domain([0, 2, 4])
    got = held.identity("k", ProductCoord((10,)), start=4)
    stated = held.as_coord(4).to_position(held.coordinates())
    assert got.coordinates()[2].tolist() == stated.tolist()


def test_the_identity_numbers_from_zero_by_default():
    got = domain([0, 2, 4]).identity("k", ProductCoord((3,)))
    assert got.coordinates()[2].tolist() == [0, 1, 2]


def test_a_destination_too_short_for_the_members_is_refused():
    with pytest.raises(ValueError, match="has extent 6"):
        domain([0, 2, 4]).identity("k", ProductCoord((6,)), start=4)


def test_a_dimension_the_domain_already_carries_is_refused():
    with pytest.raises(ValueError, match="already has"):
        domain([0, 2]).identity("x", ProductCoord((10,)))


def test_the_identity_entries_are_canonical_as_written():
    got = domain([0, 2, 4]).identity("k", ProductCoord((10,)), start=4)
    assert kernel.is_canonical(got.index, got.shape)


def test_the_identity_declares_the_absence_it_is_given():
    got = domain([0]).identity("k", ProductCoord((2,)), absence="unknown")
    assert got.absence == "unknown"
