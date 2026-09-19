import numpy as np
import pytest

from nimblend import frame
from nimblend.coords import StoredCoord
from nimblend.sparse import SparseArray


def incidence():
    # two buses and one link: l0 leaves b0 and arrives at b1
    index = np.array([[0, 1], [0, 0]], dtype=np.int32)
    coords = {"B": StoredCoord(np.arange(2)), "L": StoredCoord(np.arange(1))}
    return SparseArray(index, np.array([-1.0, 0.9]), coords, ("B", "L"))


def link_columns():
    # one column per (l, t), value 1.0, over a single link and two hours
    index = np.array([[0, 0], [0, 1], [0, 1]], dtype=np.int32)
    coords = {
        "L": StoredCoord(np.arange(1)),
        "T": StoredCoord(np.arange(2)),
        "col": StoredCoord(np.arange(2)),
    }
    return SparseArray(index, np.ones(2), coords, ("L", "T", "col"))


def test_overlap_dims_returns_the_shared_dims_the_extra_dims_and_the_coordinates():
    left = incidence()
    right = link_columns()
    shared, extra, coords = frame.overlap_dims(left, right)
    assert shared == ("L",)
    assert extra == ("T", "col")
    assert coords["B"] is left.coords["B"]
    assert coords["L"] is left.coords["L"]
    assert coords["T"] is right.coords["T"]
    assert coords["col"] is right.coords["col"]


def test_overlap_dims_raises_for_different_absence():
    with pytest.raises(ValueError, match="declares absence"):
        frame.overlap_dims(incidence().as_unknown(), link_columns())


def test_overlap_dims_raises_for_different_labels_on_a_shared_dimension():
    left = SparseArray.from_dense(
        np.ones((2, 2)), {"B": np.arange(2), "L": np.array(["x", "y"])}
    )
    right = SparseArray.from_dense(
        np.ones((2, 2)), {"L": np.array(["x", "z"]), "T": np.arange(2)}
    )
    with pytest.raises(ValueError, match="have different labels"):
        frame.overlap_dims(left, right)
