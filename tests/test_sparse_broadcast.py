import numpy as np
import pytest

from nimblend.coords import StoredCoord
from nimblend.sparse import SparseArray


def narrow():
    # cost over (P, W), one entry per cell of a 2x2 grid
    index = np.array([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=np.int32)
    coords = {"P": StoredCoord(np.arange(2)), "W": StoredCoord(np.arange(2))}
    return SparseArray(index, np.array([1.0, 2.0, 3.0, 4.0]), coords, ("P", "W"))


def wide():
    # a column indicator over (P, W, col): one column per (p, w), value 1.0
    index = np.array([[0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 2, 3]], dtype=np.int32)
    coords = {
        "P": StoredCoord(np.arange(2)),
        "W": StoredCoord(np.arange(2)),
        "col": StoredCoord(np.arange(4)),
    }
    return SparseArray(index, np.ones(4), coords, ("P", "W", "col"))


def test_broadcast_multiply_carries_the_wide_frame():
    got = narrow() * wide()
    assert got.dims == ("P", "W", "col")
    assert list(got.data) == [1.0, 2.0, 3.0, 4.0]
    assert list(got.index[2]) == [0, 1, 2, 3]


def test_broadcast_multiply_is_commutative():
    assert np.array_equal((narrow() * wide()).data, (wide() * narrow()).data)


def test_a_coordinate_absent_from_the_narrow_operand_drops_the_entry():
    # cost carries only (0,0) and (1,1); the other two columns have no term
    index = np.array([[0, 1], [0, 1]], dtype=np.int32)
    coords = {"P": StoredCoord(np.arange(2)), "W": StoredCoord(np.arange(2))}
    sparse_cost = SparseArray(index, np.array([5.0, 7.0]), coords, ("P", "W"))
    got = sparse_cost * wide()
    assert list(got.index[2]) == [0, 3]
    assert list(got.data) == [5.0, 7.0]


def test_broadcast_over_a_dimension_carrying_several_entries():
    # the narrow operand is over P only; every (p, col) entry takes p's value
    index = np.array([[0]], dtype=np.int32)
    coords = {"P": StoredCoord(np.arange(2))}
    per_p = SparseArray(index, np.array([10.0]), coords, ("P",))
    got = per_p * wide()
    assert list(got.data) == [10.0, 10.0]
    assert list(got.index[0]) == [0, 0]
    assert list(got.index[2]) == [0, 1]


def test_broadcast_result_is_canonical():
    from nimblend import kernel

    got = narrow() * wide()
    assert kernel.is_canonical(got.index, got.shape)


def test_addition_over_a_nesting_frame_carries_the_wider_one():
    got = narrow() + wide()
    assert got.dims == ("P", "W", "col")
    # the narrow operand supplies a term at every column of the wider frame,
    # and the wide one adds its 1.0 at the single column each (p, w) states
    assert got.nnz == 16
    assert got.to_dense()[0, 0, 0] == 2.0
    assert got.to_dense()[0, 0, 1] == 1.0


def test_a_shared_dimension_of_differing_size_raises():
    index = np.array([[0, 1, 2]], dtype=np.int32)
    coords = {"P": StoredCoord(np.arange(3))}
    three = SparseArray(index, np.ones(3), coords, ("P",))
    with pytest.raises(ValueError, match="size"):
        three * wide()


def test_unrelated_dimension_sets_raise():
    index = np.array([[0]], dtype=np.int32)
    other = SparseArray(index, np.ones(1), {"Z": StoredCoord(np.arange(1))}, ("Z",))
    with pytest.raises(ValueError, match="share no dimension"):
        other * wide()


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


def test_overlapping_frames_multiply_out_to_the_union():
    got = incidence() * link_columns()
    assert got.dims == ("B", "L", "T", "col")
    # each of the two link columns reaches both bus rows
    assert got.nnz == 4
    assert list(got.index[0]) == [0, 0, 1, 1]
    assert list(got.index[3]) == [0, 1, 0, 1]
    assert list(got.data) == [-1.0, -1.0, 0.9, 0.9]


def test_an_overlapping_product_matches_a_dense_reference():
    a = SparseArray.from_dense(
        np.arange(6.0).reshape(3, 2), {"B": np.arange(3), "L": np.arange(2)}
    )
    b = SparseArray.from_dense(
        np.arange(4.0).reshape(2, 2) + 1.0, {"L": np.arange(2), "T": np.arange(2)}
    )
    got = a * b
    assert got.dims == ("B", "L", "T")
    expect = a.to_dense()[:, :, None] * b.to_dense()[None, :, :]
    assert np.allclose(got.to_dense(), expect)


def test_a_shared_coordinate_one_operand_lacks_drops_the_pair():
    # the incidence carries link 0 only, so link 1's columns have no factor
    index = np.array([[0, 1], [0, 0], [0, 1]], dtype=np.int32)
    coords = {
        "L": StoredCoord(np.arange(2)),
        "T": StoredCoord(np.arange(1)),
        "col": StoredCoord(np.arange(2)),
    }
    columns = SparseArray(index, np.ones(2), coords, ("L", "T", "col"))
    inc = SparseArray(
        np.array([[0], [0]], dtype=np.int32),
        np.array([-1.0]),
        {"B": StoredCoord(np.arange(1)), "L": StoredCoord(np.arange(2))},
        ("B", "L"),
    )
    got = inc * columns
    assert got.nnz == 1
    assert list(got.index[3]) == [0]


def test_overlapping_operands_disagreeing_on_labels_are_refused():
    a = SparseArray.from_dense(
        np.ones((2, 2)), {"B": np.arange(2), "L": np.array(["x", "y"])}
    )
    b = SparseArray.from_dense(
        np.ones((2, 2)), {"L": np.array(["x", "z"]), "T": np.arange(2)}
    )
    with pytest.raises(ValueError, match="have different labels"):
        a * b


def test_overlapping_operands_disagreeing_on_absence_are_refused():
    a = SparseArray.from_dense(np.ones((2, 2)), {"B": np.arange(2), "L": np.arange(2)})
    b = SparseArray.from_dense(
        np.ones((2, 2)), {"L": np.arange(2), "T": np.arange(2)}, absence="unknown"
    )
    with pytest.raises(ValueError, match="declares absence"):
        a * b


def test_nesting_frames_take_the_broadcast_path():
    # neither operand introduces a dimension, so the wide frame is the result
    got = narrow() * wide()
    assert got.dims == ("P", "W", "col")
