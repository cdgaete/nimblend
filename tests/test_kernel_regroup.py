import numpy as np

from nimblend import kernel


def test_regroup_replaces_the_leading_rows_by_one_row():
    idx = np.array([[0, 0, 1], [1, 2, 0], [5, 6, 7]], dtype=np.int32)
    data = np.array([1.0, 2.0, 3.0])
    at = np.array([0, 1, 2])
    out_idx, out_data = kernel.regroup(idx, data, at, 2, 10)
    assert out_idx.tolist() == [[10, 11, 12], [5, 6, 7]]
    assert out_data.tolist() == [1.0, 2.0, 3.0]


def test_regroup_drops_an_entry_at_minus_one():
    idx = np.array([[0, 1, 2], [4, 5, 6]], dtype=np.int32)
    out_idx, out_data = kernel.regroup(
        idx, np.array([1.0, 2.0, 3.0]), np.array([0, -1, 1]), 1, 0
    )
    assert out_idx.tolist() == [[0, 1], [4, 6]]
    assert out_data.tolist() == [1.0, 3.0]


def test_regroup_writes_into_a_supplied_destination():
    idx = np.array([[0, 1], [4, 5]], dtype=np.int32)
    dest_idx = np.full((2, 5), -7, dtype=np.int32)
    dest_data = np.full(5, -7.0)
    out_idx, out_data = kernel.regroup(
        idx, np.array([1.0, 2.0]), np.array([0, 1]), 1, 3, out=(dest_idx, dest_data)
    )
    assert out_data.base is dest_data or out_data.base is dest_data.base
    assert dest_idx[:, :2].tolist() == [[3, 4], [4, 5]]
    assert dest_data[2] == -7.0


def test_regroup_of_no_leading_rows_adds_one_row():
    idx = np.array([[4, 5]], dtype=np.int32)
    out_idx, _ = kernel.regroup(idx, np.array([1.0, 2.0]), np.array([0, 0]), 0, 2)
    assert out_idx.tolist() == [[2, 2], [4, 5]]
