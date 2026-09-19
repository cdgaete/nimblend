import numpy as np
import pytest

from nimblend import frame
from nimblend.coords import StoredCoord
from nimblend.sparse import SparseArray


def narrow():
    index = np.array([[0, 1]], dtype=np.int32)
    coords = {"P": StoredCoord(np.arange(2))}
    return SparseArray(index, np.array([1.0, 2.0]), coords, ("P",))


def wide():
    index = np.array([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=np.int32)
    coords = {"P": StoredCoord(np.arange(2)), "Q": StoredCoord(np.arange(2))}
    return SparseArray(index, np.ones(4), coords, ("P", "Q"))


def test_nested_operands_puts_the_narrower_frame_first_in_either_argument_order():
    a, b = frame.nested_operands(narrow(), wide())
    assert a.dims == ("P",)
    assert b.dims == ("P", "Q")
    a, b = frame.nested_operands(wide(), narrow())
    assert a.dims == ("P",)
    assert b.dims == ("P", "Q")


def test_nested_operands_raises_for_dimensions_that_do_not_nest():
    other = SparseArray(
        np.array([[0]], dtype=np.int32),
        np.ones(1),
        {"Z": StoredCoord(np.arange(1))},
        ("Z",),
    )
    with pytest.raises(ValueError, match="are not a subset of"):
        frame.nested_operands(other, wide())


def test_nested_operands_raises_for_a_shared_dimension_of_differing_size():
    three = SparseArray(
        np.array([[0, 1, 2]], dtype=np.int32),
        np.ones(3),
        {"P": StoredCoord(np.arange(3))},
        ("P",),
    )
    with pytest.raises(ValueError, match="size"):
        frame.nested_operands(three, wide())


def test_nested_operands_raises_for_different_labels_on_the_shared_dimension():
    narrow_labels = SparseArray(
        np.array([[0, 1]], dtype=np.int32),
        np.ones(2),
        {"P": StoredCoord(np.array(["x", "y"]))},
        ("P",),
    )
    wide_labels = SparseArray(
        np.array([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=np.int32),
        np.ones(4),
        {"P": StoredCoord(np.array(["x", "z"])), "Q": StoredCoord(np.arange(2))},
        ("P", "Q"),
    )
    with pytest.raises(ValueError, match="have different labels"):
        frame.nested_operands(narrow_labels, wide_labels)


def test_nested_operands_raises_for_different_absence():
    with pytest.raises(ValueError, match="declares absence"):
        frame.nested_operands(narrow().as_unknown(), wide())
