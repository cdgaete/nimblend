import numpy as np
import pytest

from nimblend import coords


def test_stored_coord_resolves_by_label():
    c = coords.StoredCoord(np.array(["a", "b", "c"]))
    assert len(c) == 3
    assert list(c.to_position(np.array(["c", "a"]))) == [2, 0]
    assert list(c.to_index(np.array([1, 2]))) == ["b", "c"]


def test_stored_coord_refuses_an_absent_label():
    c = coords.StoredCoord(np.array(["a", "b"]))
    with pytest.raises(KeyError, match="z"):
        c.to_position(np.array(["z"]))


def test_product_coord_is_stride_arithmetic():
    c = coords.ProductCoord((4, 5))
    assert len(c) == 20
    idx = np.array([[1, 3], [2, 4]], dtype=np.int32)
    assert list(c.to_position(idx)) == [1 * 5 + 2, 3 * 5 + 4]
    assert np.array_equal(c.to_index(np.array([7, 19])), np.array([[1, 3], [2, 4]]))


def test_product_coord_offsets_by_start():
    c = coords.ProductCoord((2, 3), start=100)
    idx = np.array([[0], [0]], dtype=np.int32)
    assert list(c.to_position(idx)) == [100]
    assert np.array_equal(c.to_index(np.array([100])), np.array([[0], [0]]))


def test_subset_coord_position_is_the_entry_rank():
    # a subset of a 4x4 grid holding three cells
    sizes = (4, 4)
    codes = np.array([0 * 4 + 1, 2 * 4 + 2, 3 * 4 + 0], dtype=np.int64)
    c = coords.SubsetCoord(codes, sizes)
    assert len(c) == 3
    idx = np.array([[2, 0], [2, 1]], dtype=np.int32)
    assert list(c.to_position(idx)) == [1, 0]
    assert np.array_equal(c.to_index(np.array([2])), np.array([[3], [0]]))


def test_subset_coord_refuses_a_cell_it_does_not_hold():
    c = coords.SubsetCoord(np.array([1, 5], dtype=np.int64), (3, 3))
    with pytest.raises(KeyError, match="not carried"):
        c.to_position(np.array([[2], [2]], dtype=np.int32))


def test_stored_coords_are_equal_when_they_name_the_same_labels():
    a = coords.StoredCoord(np.array(["a", "b"]))
    assert a == coords.StoredCoord(np.array(["a", "b"]))
    assert a != coords.StoredCoord(np.array(["a", "c"]))
    assert a != coords.StoredCoord(np.array(["a", "b", "c"]))


def test_generated_coords_are_equal_on_their_rule_not_their_extent():
    assert coords.ProductCoord((2, 3)) == coords.ProductCoord((2, 3))
    assert coords.ProductCoord((2, 3)) != coords.ProductCoord((2, 3), start=6)
    assert coords.ProductCoord((2, 3)) != coords.ProductCoord((3, 2))
    codes = np.array([1, 5], dtype=np.int64)
    assert coords.SubsetCoord(codes, (3, 3)) == coords.SubsetCoord(codes, (3, 3))
    assert coords.SubsetCoord(codes, (3, 3)) != coords.SubsetCoord(
        np.array([1, 6], dtype=np.int64), (3, 3)
    )


def test_coords_of_different_kinds_are_never_equal():
    # a product of two labels and two stored labels number the same positions
    assert coords.StoredCoord(np.array([0, 1])) != coords.ProductCoord((2,))
