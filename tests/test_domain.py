import numpy as np
import pytest

from nimblend.coords import StoredCoord
from nimblend.domain import Domain


def coords_xy():
    return {
        "x": StoredCoord(np.array(["a", "b"])),
        "y": StoredCoord(np.array([10, 20, 30])),
    }


def test_a_domain_carries_its_members_dimensions_and_shape():
    domain = Domain(np.array([0, 4]), ("x", "y"), coords_xy(), (2, 3))
    assert domain.dims == ("x", "y")
    assert domain.shape == (2, 3)
    assert domain.size == 2
    assert len(domain) == 2


def test_coordinates_are_the_multi_index_of_each_member():
    # over shape (2, 3) code 0 is (0, 0) and code 4 is (1, 1)
    domain = Domain(np.array([0, 4]), ("x", "y"), coords_xy(), (2, 3))
    assert domain.coordinates().tolist() == [[0, 1], [0, 1]]


def test_labels_answer_each_member_in_its_dimensions_own_labels():
    domain = Domain(np.array([0, 4]), ("x", "y"), coords_xy(), (2, 3))
    got = domain.labels()
    assert list(got["x"]) == ["a", "b"]
    assert list(got["y"]) == [10, 20]


def test_a_domain_keeps_only_the_coordinates_of_its_own_dimensions():
    extra = coords_xy()
    extra["z"] = StoredCoord(np.array([1]))
    domain = Domain(np.array([0]), ("x", "y"), extra, (2, 3))
    assert set(domain.coords) == {"x", "y"}


def test_a_domain_refuses_codes_that_repeat():
    with pytest.raises(ValueError, match="ascend without repeating"):
        Domain(np.array([0, 4, 4]), ("x", "y"), coords_xy(), (2, 3))


def test_a_domain_refuses_codes_out_of_order():
    with pytest.raises(ValueError, match="ascend without repeating"):
        Domain(np.array([4, 0]), ("x", "y"), coords_xy(), (2, 3))


def test_a_domain_refuses_a_shape_naming_a_different_number_of_axes():
    with pytest.raises(ValueError, match="different number of axes"):
        Domain(np.array([0]), ("x", "y"), coords_xy(), (2,))


def test_a_domain_refuses_a_dimension_with_no_coordinate():
    with pytest.raises(ValueError, match="no coordinate"):
        Domain(np.array([0]), ("x", "z"), coords_xy(), (2, 3))


def test_an_empty_domain_carries_no_members():
    domain = Domain(np.array([], dtype=np.int64), ("x", "y"), coords_xy(), (2, 3))
    assert domain.size == 0
    assert domain.coordinates().shape == (2, 0)


def test_as_coord_numbers_the_members_the_domain_carries():
    # over shape (2, 3) code 0 is (0, 0) and code 4 is (1, 1)
    coord = Domain(np.array([0, 4]), ("x", "y"), coords_xy(), (2, 3)).as_coord()
    assert len(coord) == 2
    assert list(coord.to_position(np.array([[1, 0], [1, 0]], dtype=np.int32))) == [1, 0]
    assert coord.to_index(np.array([0, 1])).tolist() == [[0, 1], [0, 1]]


def test_as_coord_numbers_from_the_start_it_is_given():
    coord = Domain(np.array([0, 4]), ("x", "y"), coords_xy(), (2, 3)).as_coord(100)
    assert list(coord.to_position(np.array([[0, 1], [0, 1]], dtype=np.int32))) == [
        100,
        101,
    ]


def test_as_coord_refuses_a_member_the_domain_does_not_carry():
    coord = Domain(np.array([0, 4]), ("x", "y"), coords_xy(), (2, 3)).as_coord()
    with pytest.raises(KeyError, match="not carried by this subset"):
        coord.to_position(np.array([[0], [2]], dtype=np.int32))


def test_is_full_holds_where_every_coordinate_of_the_product_is_carried():
    assert Domain.full(("x", "y"), coords_xy()).is_full


def test_is_full_is_false_where_a_member_is_missing():
    # the product spans six members and this domain carries five
    codes = np.array([0, 1, 2, 3, 4])
    assert not Domain(codes, ("x", "y"), coords_xy(), (2, 3)).is_full


def test_is_full_holds_for_a_domain_carrying_one_member_of_one_cell():
    single = {"x": StoredCoord(np.array(["a"]))}
    assert Domain(np.array([0]), ("x",), single, (1,)).is_full


def test_expanding_a_full_domain_leaves_it_full():
    coords = coords_xy()
    coords["z"] = StoredCoord(np.array([7, 8]))
    assert Domain.full(("x", "y"), coords).expand(("z",), coords).is_full


def test_expanding_a_partial_domain_leaves_it_partial():
    coords = coords_xy()
    coords["z"] = StoredCoord(np.array([7, 8]))
    partial = Domain(np.array([0, 4]), ("x", "y"), coords_xy(), (2, 3))
    assert not partial.expand(("z",), coords).is_full


def test_coordinates_are_a_fresh_int32_matrix_of_one_row_per_dimension():
    held = Domain(np.array([0, 4]), ("x", "y"), coords_xy(), (2, 3))
    got = held.coordinates()
    assert got.dtype == np.int32
    assert got.shape == (2, held.size)
    got[:] = 0
    assert held.coordinates().tolist() == [[0, 1], [0, 1]]
