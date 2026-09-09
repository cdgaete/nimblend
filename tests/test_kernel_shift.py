import numpy as np
import pytest

from nimblend import kernel


def test_drop_removes_entries_that_leave_the_axis():
    idx = np.array([[0, 1, 2]], dtype=np.int32)
    data = np.array([1.0, 2.0, 3.0])
    out_idx, out_data = kernel.shift_axis(idx, data, 0, 1, 3, mode="drop")
    # the entry at position 2 would land on 3, which the axis does not hold
    assert list(out_idx[0]) == [1, 2]
    assert list(out_data) == [1.0, 2.0]


def test_drop_on_a_negative_shift():
    idx = np.array([[0, 1, 2]], dtype=np.int32)
    data = np.array([1.0, 2.0, 3.0])
    out_idx, out_data = kernel.shift_axis(idx, data, 0, -1, 3, mode="drop")
    assert list(out_idx[0]) == [0, 1]
    assert list(out_data) == [2.0, 3.0]


def test_wrap_keeps_every_entry():
    idx = np.array([[0, 1, 2]], dtype=np.int32)
    data = np.array([1.0, 2.0, 3.0])
    out_idx, out_data = kernel.shift_axis(idx, data, 0, 1, 3, mode="wrap")
    assert sorted(out_idx[0]) == [0, 1, 2]
    assert out_data.size == 3
    at = {int(p): v for p, v in zip(out_idx[0], out_data, strict=True)}
    assert at[0] == 3.0


def test_zero_shift_is_the_identity():
    idx = np.array([[0, 2]], dtype=np.int32)
    data = np.array([1.0, 5.0])
    out_idx, out_data = kernel.shift_axis(idx, data, 0, 0, 3)
    assert np.array_equal(out_idx, idx)
    assert np.array_equal(out_data, data)


def test_unknown_mode_raises():
    idx = np.array([[0]], dtype=np.int32)
    with pytest.raises(ValueError, match="drop"):
        kernel.shift_axis(idx, np.array([1.0]), 0, 1, 2, mode="fill")
