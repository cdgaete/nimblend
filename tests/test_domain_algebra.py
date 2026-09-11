import numpy as np
import pytest

from nimblend.coords import StoredCoord
from nimblend.domain import Domain
from nimblend.sparse import SparseArray


def coords_xy():
    return {
        "x": StoredCoord(np.array(["a", "b"])),
        "y": StoredCoord(np.array([10, 20, 30])),
    }


def domain(codes):
    return Domain(np.array(codes, dtype=np.int64), ("x", "y"), coords_xy(), (2, 3))


def block(values):
    """A 2x3 array holding only its nonzero cells."""
    arr = SparseArray.from_dense(
        np.asarray(values, dtype=np.float64),
        {"x": np.array(["a", "b"]), "y": np.array([10, 20, 30])},
    )
    keep = arr.data != 0.0
    return SparseArray.from_canonical(
        arr.index[:, keep], arr.data[keep], arr.coords, arr.dims
    )


def test_intersect_keeps_the_members_both_domains_carry():
    got = domain([0, 2, 4]).intersect(domain([2, 4, 5]))
    assert list(got.codes) == [2, 4]
    assert got.dims == ("x", "y")


def test_union_keeps_the_members_either_domain_carries():
    got = domain([0, 2]).union(domain([2, 5]))
    assert list(got.codes) == [0, 2, 5]


def test_intersecting_disjoint_domains_leaves_no_member():
    assert domain([0, 1]).intersect(domain([4, 5])).size == 0


def test_intersect_refuses_a_domain_over_different_dimensions():
    other = Domain(
        np.array([0]),
        ("x", "w"),
        {"x": StoredCoord(np.array(["a", "b"])), "w": StoredCoord(np.array([1, 2, 3]))},
        (2, 3),
    )
    with pytest.raises(ValueError, match="combine domains over the same frame"):
        domain([0]).intersect(other)


def test_positions_of_answers_each_entry_its_place_in_the_domain():
    # cells (0,0) and (1,1) are codes 0 and 4
    arr = block([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    assert list(domain([0, 4]).positions_of(arr)) == [0, 1]


def test_positions_of_answers_minus_one_for_an_entry_the_domain_omits():
    arr = block([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    assert list(domain([4]).positions_of(arr)) == [-1, 0]


def test_positions_of_reads_the_domains_dimensions_out_of_a_wider_array():
    values = np.zeros((2, 3, 2))
    values[0, 0, 1] = 5.0
    values[1, 1, 0] = 6.0
    arr = SparseArray.from_dense(
        values,
        {
            "x": np.array(["a", "b"]),
            "y": np.array([10, 20, 30]),
            "c": np.array([0, 1]),
        },
    )
    keep = arr.data != 0.0
    arr = SparseArray.from_canonical(
        arr.index[:, keep], arr.data[keep], arr.coords, arr.dims
    )
    assert list(domain([0, 4]).positions_of(arr)) == [0, 1]


def test_positions_of_coordinates_answers_each_column_its_place():
    # over shape (2, 3) code 0 is (0, 0), code 4 is (1, 1) and code 2 is (0, 2)
    index = np.array([[0, 1, 0], [0, 1, 2]], dtype=np.int32)
    assert list(domain([0, 4]).positions_of_coordinates(index)) == [0, 1, -1]


def test_positions_of_coordinates_answers_the_same_places_as_an_array_does():
    arr = block([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    held = domain([0, 4])
    assert list(held.positions_of_coordinates(arr.coordinates())) == list(
        held.positions_of(arr)
    )


def test_positions_of_coordinates_reads_a_column_repeated_and_out_of_order():
    index = np.array([[1, 0, 1], [1, 0, 1]], dtype=np.int32)
    assert list(domain([0, 4]).positions_of_coordinates(index)) == [1, 0, 1]


def test_positions_of_coordinates_answers_nothing_for_no_columns():
    index = np.empty((2, 0), dtype=np.int32)
    assert list(domain([0, 4]).positions_of_coordinates(index)) == []


def test_positions_of_coordinates_refuses_an_index_of_the_wrong_rank():
    with pytest.raises(ValueError, match="2 rows"):
        domain([0]).positions_of_coordinates(np.array([[0, 1]], dtype=np.int32))


def test_positions_of_refuses_an_array_whose_shape_differs():
    arr = SparseArray.from_dense(
        np.ones((2, 2)), {"x": np.array(["a", "b"]), "y": np.array([10, 20])}
    )
    with pytest.raises(ValueError, match="shape"):
        domain([0]).positions_of(arr)


def test_domain_algebra_refuses_domains_whose_labels_differ():
    other = Domain(
        np.array([0, 4], dtype=np.int64),
        ("x", "y"),
        {
            "x": StoredCoord(np.array(["p", "q"])),
            "y": StoredCoord(np.array([10, 20, 30])),
        },
        (2, 3),
    )
    for combine in (Domain.intersect, Domain.union):
        with pytest.raises(ValueError, match="different labels"):
            combine(domain([0, 4]), other)


def test_domain_algebra_accepts_equal_labels_held_in_separate_coordinates():
    # the helper builds a fresh coordinate per domain
    assert list(domain([0, 4]).intersect(domain([4, 5])).codes) == [4]


def test_difference_keeps_the_members_the_other_domain_does_not_carry():
    got = domain([0, 2, 4]).difference(domain([2, 5]))
    assert list(got.codes) == [0, 4]


def test_difference_from_a_domain_that_carries_everything_leaves_nothing():
    assert domain([0, 2]).difference(domain([0, 2, 4])).size == 0


def test_difference_from_an_empty_domain_keeps_every_member():
    empty = domain([])
    assert list(domain([1, 3]).difference(empty).codes) == [1, 3]
    assert empty.difference(domain([1, 3])).size == 0


def test_difference_refuses_a_domain_over_different_labels():
    other = Domain(
        np.array([0], dtype=np.int64),
        ("x", "y"),
        {
            "x": StoredCoord(np.array(["p", "q"])),
            "y": StoredCoord(np.array([10, 20, 30])),
        },
        (2, 3),
    )
    with pytest.raises(ValueError, match="different labels"):
        domain([0]).difference(other)


def test_full_carries_every_coordinate_of_the_product():
    got = Domain.full(("x", "y"), coords_xy())
    assert got.size == 6
    assert list(got.codes) == [0, 1, 2, 3, 4, 5]
    assert got.dims == ("x", "y") and got.shape == (2, 3)


def test_full_names_every_member_by_its_labels():
    got = Domain.full(("x", "y"), coords_xy()).labels()
    assert list(got["x"]) == ["a", "a", "a", "b", "b", "b"]
    assert list(got["y"]) == [10, 20, 30, 10, 20, 30]


def test_full_minus_a_subset_names_what_the_subset_leaves_out():
    got = Domain.full(("x", "y"), coords_xy()).difference(domain([0, 1, 2, 3, 4]))
    assert list(got.codes) == [5]
    assert (got.labels()["x"][0], got.labels()["y"][0]) == ("b", 30)


def test_full_refuses_a_dimension_it_has_no_coordinate_for():
    with pytest.raises(ValueError, match="no coordinate"):
        Domain.full(("x", "z"), coords_xy())


def test_expand_crosses_every_member_with_the_dimensions_full_extent():
    coords = coords_xy()
    coords["z"] = StoredCoord(np.array([7, 8]))
    # codes 0 and 4 over (2, 3) are (0, 0) and (1, 1)
    got = domain([0, 4]).expand(("z",), coords)
    assert got.dims == ("x", "y", "z")
    assert got.shape == (2, 3, 2)
    assert got.coordinates().tolist() == [[0, 0, 1, 1], [0, 0, 1, 1], [0, 1, 0, 1]]


def test_expand_over_no_dimension_leaves_the_members_alone():
    held = domain([0, 4])
    got = held.expand((), coords_xy())
    assert got.dims == held.dims
    assert got.coordinates().tolist() == held.coordinates().tolist()


def test_expand_multiplies_the_member_count_by_the_extent_added():
    coords = coords_xy()
    coords["z"] = StoredCoord(np.array([7, 8]))
    coords["w"] = StoredCoord(np.array(["p", "q", "r"]))
    assert domain([0, 4]).expand(("z", "w"), coords).size == 2 * 2 * 3


def test_expand_refuses_a_dimension_already_carried():
    with pytest.raises(ValueError, match="already has"):
        domain([0]).expand(("y",), coords_xy())


def test_expand_refuses_a_dimension_with_no_coordinate():
    with pytest.raises(ValueError, match="no coordinate"):
        domain([0]).expand(("z",), coords_xy())


def test_expand_names_the_same_members_a_full_cross_product_does():
    coords = coords_xy()
    coords["z"] = StoredCoord(np.array([7, 8]))
    got = Domain.full(("x", "y"), coords).expand(("z",), coords)
    assert list(got.codes) == list(Domain.full(("x", "y", "z"), coords).codes)


def test_transpose_keeps_the_members_and_reorders_the_dimensions():
    got = domain([0, 4]).transpose("y", "x")
    assert got.dims == ("y", "x")
    assert got.shape == (3, 2)
    # (0, 0) and (1, 1) read the other way round are (0, 0) and (1, 1)
    assert got.coordinates().tolist() == [[0, 1], [0, 1]]


def test_transpose_reorders_the_members_into_the_new_codes_order():
    # (0, 2) and (1, 0) over (2, 3) are codes 2 and 3; read as (y, x) they
    # are (2, 0) and (0, 1), which ascend the other way round
    got = domain([2, 3]).transpose("y", "x")
    assert got.coordinates().tolist() == [[0, 2], [1, 0]]


def test_transpose_back_again_is_the_domain_it_started_from():
    held = domain([0, 2, 4])
    assert list(held.transpose("y", "x").transpose("x", "y").codes) == list(held.codes)


def test_transpose_to_the_order_already_held_is_the_same_domain():
    held = domain([0, 4])
    assert held.transpose("x", "y") is held


def test_transpose_refuses_dimensions_that_are_not_the_domains_own():
    for named in (("x",), ("x", "z"), ("x", "y", "y")):
        with pytest.raises(ValueError, match="requires each dimension"):
            domain([0]).transpose(*named)


def test_expand_then_transpose_is_the_cross_product_in_the_order_asked():
    coords = coords_xy()
    coords["z"] = StoredCoord(np.array([7, 8]))
    got = domain([0, 4]).expand(("z",), coords).transpose("z", "x", "y")
    assert got.dims == ("z", "x", "y")
    members = sorted(zip(*got.coordinates().tolist()))
    assert members == [(0, 0, 0), (0, 1, 1), (1, 0, 0), (1, 1, 1)]
