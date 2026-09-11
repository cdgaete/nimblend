import numpy as np
import pytest

from nimblend.coords import ProductCoord, StoredCoord
from nimblend.domain import Domain


def coords_xy():
    return {
        "x": StoredCoord(np.array(["a", "b"])),
        "y": StoredCoord(np.array([10, 20, 30])),
    }


def test_a_domain_is_built_from_one_label_column_per_dimension():
    got = Domain.from_labels(
        ("x", "y"),
        coords_xy(),
        {"x": np.array(["a", "b"]), "y": np.array([10, 20])},
    )
    assert got.dims == ("x", "y")
    assert got.shape == (2, 3)
    assert got.size == 2
    # over shape (2, 3), ("a", 10) is code 0 and ("b", 20) is code 4
    assert list(got.codes) == [0, 4]


def test_the_members_read_back_in_the_labels_they_were_named_in():
    got = Domain.from_labels(
        ("x", "y"),
        coords_xy(),
        {"x": np.array(["b", "a"]), "y": np.array([30, 10])},
    )
    assert list(got.labels()["x"]) == ["a", "b"]
    assert list(got.labels()["y"]) == [10, 30]


def test_members_named_out_of_order_are_sorted():
    got = Domain.from_labels(
        ("x", "y"),
        coords_xy(),
        {"x": np.array(["b", "a"]), "y": np.array([10, 30])},
    )
    assert list(got.codes) == [2, 3]


def test_a_member_named_twice_raises_naming_it():
    with pytest.raises(ValueError, match="appears twice"):
        Domain.from_labels(
            ("x", "y"),
            coords_xy(),
            {"x": np.array(["a", "a"]), "y": np.array([10, 10])},
        )


def test_a_label_the_coordinate_does_not_carry_raises():
    with pytest.raises(KeyError):
        Domain.from_labels(
            ("x", "y"),
            coords_xy(),
            {"x": np.array(["z"]), "y": np.array([10])},
        )


def test_label_columns_of_differing_length_raise():
    with pytest.raises(ValueError, match="different lengths"):
        Domain.from_labels(
            ("x", "y"),
            coords_xy(),
            {"x": np.array(["a", "b"]), "y": np.array([10])},
        )


def test_a_dimension_with_no_label_column_raises():
    with pytest.raises(ValueError, match="no label column"):
        Domain.from_labels(("x", "y"), coords_xy(), {"x": np.array(["a"])})


def test_a_dimension_with_no_coordinate_raises():
    with pytest.raises(ValueError, match="no coordinate"):
        Domain.from_labels(
            ("x", "z"),
            coords_xy(),
            {"x": np.array(["a"]), "z": np.array([1])},
        )


def test_naming_no_dimension_at_all_raises():
    with pytest.raises(ValueError, match="at least one dimension"):
        Domain.from_labels((), {}, {})


def test_a_coordinate_that_cannot_read_a_label_column_raises():
    # a generated coordinate reads an index matrix, not a column of labels
    with pytest.raises(ValueError, match="one position per label"):
        Domain.from_labels(
            ("g",), {"g": ProductCoord((2, 3))}, {"g": np.array([0, 1, 2])}
        )


def test_a_domain_from_no_members_carries_none():
    got = Domain.from_labels(
        ("x", "y"),
        coords_xy(),
        {"x": np.array([], dtype="<U1"), "y": np.array([], dtype=np.int64)},
    )
    assert got.size == 0
    assert got.dims == ("x", "y")


def test_a_domain_is_built_from_an_index_matrix():
    # each column names one member: (0, 0) and (1, 1) over shape (2, 3)
    got = Domain.from_coordinates(
        ("x", "y"), coords_xy(), np.array([[0, 1], [0, 1]], dtype=np.int32)
    )
    assert list(got.codes) == [0, 4]
    assert got.shape == (2, 3)


def test_an_index_matrix_names_its_members_in_any_order():
    got = Domain.from_coordinates(
        ("x", "y"), coords_xy(), np.array([[1, 0], [1, 0]], dtype=np.int32)
    )
    assert list(got.codes) == [0, 4]


def test_an_index_matrix_naming_a_member_twice_raises():
    with pytest.raises(ValueError, match="appears twice"):
        Domain.from_coordinates(
            ("x", "y"), coords_xy(), np.array([[0, 0], [1, 1]], dtype=np.int32)
        )


def test_an_index_matrix_with_the_wrong_number_of_rows_raises():
    with pytest.raises(ValueError, match="rows"):
        Domain.from_coordinates(
            ("x", "y"), coords_xy(), np.array([[0, 1]], dtype=np.int32)
        )


def test_the_two_constructors_agree_on_the_same_members():
    by_label = Domain.from_labels(
        ("x", "y"),
        coords_xy(),
        {"x": np.array(["a", "b"]), "y": np.array([10, 20])},
    )
    by_index = Domain.from_coordinates(
        ("x", "y"), coords_xy(), np.array([[0, 1], [0, 1]], dtype=np.int32)
    )
    assert list(by_label.codes) == list(by_index.codes)
