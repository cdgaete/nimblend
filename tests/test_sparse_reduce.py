import numpy as np
import pytest

from nimblend.coords import StoredCoord
from nimblend.sparse import SparseArray


def two_d(entries, shape=(2, 3), absence="empty"):
    idx = np.array(
        [[p for p, _, _ in entries], [q for _, q, _ in entries]], dtype=np.int32
    )
    data = np.array([v for _, _, v in entries], dtype=float)
    coords = {
        "x": StoredCoord(np.arange(shape[0])),
        "y": StoredCoord(np.arange(shape[1])),
    }
    return SparseArray(idx, data, coords, ("x", "y"), absence)


def test_sum_over_a_dimension_drops_it():
    arr = two_d([(0, 0, 1.0), (0, 1, 2.0), (1, 2, 5.0)])
    got = arr.sum("y")
    assert got.dims == ("x",)
    assert list(got.data) == [3.0, 5.0]


def test_reducing_the_only_dimension_leaves_an_array_over_none():
    # the frame a reduction leaves is the dimensions it did not reduce; over
    # the last one that is no dimensions, which is one cell holding the total
    coords = {"x": StoredCoord(np.arange(3))}
    arr = SparseArray(
        np.array([[0, 1, 2]], dtype=np.int32), np.array([1.0, 2.0, 3.0]), coords, ("x",)
    )
    got = arr.sum("x")
    assert got.dims == ()
    assert got.shape == ()
    assert got.nnz == 1
    assert list(got.values()) == [6.0]
    assert got.to_dense() == 6.0


def test_reducing_the_only_dimension_carries_the_op_and_the_fill_path():
    coords = {"x": StoredCoord(np.arange(3))}
    arr = SparseArray(
        np.array([[0, 1, 2]], dtype=np.int32), np.array([1.0, 2.0, 3.0]), coords, ("x",)
    )
    assert arr.min("x", skip=True).to_dense() == 1.0
    assert arr.max("x", skip=True).to_dense() == 3.0
    assert arr.mean("x", skip=True).to_dense() == 2.0
    # the fill path densifies the frame instead of reducing the buffer, and
    # arrives at the same array over no dimensions
    assert arr.sum("x", fill=0.0).dims == ()
    assert arr.sum("x", fill=0.0).to_dense() == 6.0


def test_sum_over_everything_is_a_number():
    arr = two_d([(0, 0, 1.0), (1, 2, 5.0)])
    assert arr.sum() == 6.0


def test_empty_reduction_needs_no_policy():
    arr = two_d([(0, 0, 4.0)], absence="empty")
    assert arr.mean("y").dims == ("x",)


def test_unknown_reduction_without_a_policy_raises():
    arr = two_d([(0, 0, 4.0)], absence="unknown")
    with pytest.raises(ValueError, match="skip"):
        arr.mean("y")


def test_unknown_reduction_with_skip_uses_present_entries_only():
    arr = two_d([(0, 0, 2.0), (0, 1, 4.0)], absence="unknown")
    assert arr.mean("y", skip=True).data[0] == 3.0


def test_unknown_reduction_with_fill_counts_the_absences():
    arr = two_d([(0, 0, 2.0), (0, 1, 4.0)], absence="unknown")
    assert arr.mean("y", fill=0.0).data[0] == 2.0


def test_whole_array_reduction_with_fill_counts_the_absences():
    # two entries in a 2x3 grid: four cells are absent
    arr = two_d([(0, 0, 1.0), (1, 2, 4.0)], absence="unknown")
    assert arr.sum(fill=100.0) == 1.0 + 4.0 + 4 * 100.0
    assert arr.mean(fill=0.0) == 5.0 / 6.0


def test_whole_array_extremes_reach_the_fill_value():
    # the fill sits outside the range of the stored entries, so a reduction
    # that honours it must return the fill itself
    arr = two_d([(0, 0, 1.0), (1, 2, 4.0)], absence="unknown")
    assert arr.min(fill=-5.0) == -5.0
    assert arr.max(fill=99.0) == 99.0


def test_whole_array_reduction_with_skip_uses_present_entries_only():
    arr = two_d([(0, 0, 1.0), (1, 2, 4.0)], absence="unknown")
    assert arr.sum(skip=True) == 5.0
    assert arr.min(skip=True) == 1.0
    assert arr.max(skip=True) == 4.0


def test_a_fill_that_matches_the_absence_reading_changes_nothing():
    arr = two_d([(0, 0, 1.0), (1, 2, 4.0)], absence="unknown")
    assert arr.sum(fill=0.0) == arr.sum(skip=True)


def test_shift_drops_entries_leaving_the_axis():
    arr = two_d([(0, 0, 1.0), (1, 0, 2.0)])
    got = arr.shift({"x": 1})
    assert list(got.index[0]) == [1]
    assert list(got.data) == [1.0]


def test_roll_wraps_every_entry():
    arr = two_d([(0, 0, 1.0), (1, 0, 2.0)])
    got = arr.roll({"x": 1})
    assert got.nnz == 2
    assert sorted(got.index[0]) == [0, 1]


def test_conform_reads_at_the_given_labels():
    arr = two_d([(0, 0, 1.0), (1, 1, 2.0)], shape=(2, 3))
    got = arr.conform(["x", "y"], {"x": np.array([1, 0]), "y": np.array([0, 1, 2])})
    assert got.to_dense()[0, 1] == 2.0


def test_conform_refuses_a_label_the_array_lacks():
    arr = two_d([(0, 0, 1.0)])
    with pytest.raises(KeyError):
        arr.conform(["x", "y"], {"x": np.array([9]), "y": np.array([0, 1, 2])})


def test_a_reduction_refuses_a_skip_that_is_not_true():
    arr = two_d([(0, 0, 1.0)], absence="unknown")
    with pytest.raises(ValueError, match="skip states"):
        arr.sum(skip=False)


def test_a_reduction_refuses_both_policies_under_either_absence():
    for absence in ("empty", "unknown"):
        arr = two_d([(0, 0, 1.0)], absence=absence)
        with pytest.raises(ValueError, match="not both"):
            arr.sum(skip=True, fill=0.0)


def test_densifying_an_unknown_array_states_what_absence_carries():
    arr = two_d([(0, 0, 1.0)], absence="unknown")
    with pytest.raises(ValueError, match="fill="):
        arr.to_dense()
    assert arr.to_dense(fill=0.0)[1, 2] == 0.0
    assert np.isnan(arr.to_dense(fill=np.nan)[1, 2])
    assert arr.to_dense(fill=np.nan)[0, 0] == 1.0


def test_densifying_an_empty_array_carries_zero_where_it_holds_nothing():
    arr = two_d([(0, 0, 1.0)])
    assert arr.to_dense()[1, 2] == 0.0
    assert arr.to_dense(fill=7.0)[1, 2] == 7.0


def test_conform_refuses_a_label_named_twice():
    arr = two_d([(0, 0, 1.0), (1, 1, 2.0)], shape=(2, 3))
    with pytest.raises(ValueError, match="named twice"):
        arr.conform(["x", "y"], {"x": np.array([0, 0, 1]), "y": np.array([0, 1, 2])})


def test_conform_refuses_a_repeat_on_any_dimension():
    arr = two_d([(0, 0, 1.0)], shape=(2, 3))
    with pytest.raises(ValueError, match="'y'"):
        arr.conform(["x", "y"], {"x": np.array([0, 1]), "y": np.array([2, 2])})


def test_conform_reads_no_label_at_all():
    arr = two_d([(0, 0, 1.0), (1, 1, 2.0)], shape=(2, 3))
    got = arr.conform(["x", "y"], {"x": np.array([], dtype=int), "y": np.array([0, 1])})
    assert got.shape == (0, 2) and got.nnz == 0
