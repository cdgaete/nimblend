import numpy as np

from nimblend import kernel


def test_accumulate_adds_each_value_at_its_index():
    out = np.zeros((2, 3))
    idx = np.array([[0, 1, 1], [2, 0, 1]], dtype=np.int32)
    kernel.accumulate(out, idx, np.array([1.0, 2.0, 3.0]))
    assert out.tolist() == [[0.0, 0.0, 1.0], [2.0, 3.0, 0.0]]


def test_accumulate_adds_to_the_values_already_in_out():
    out = np.full((2, 2), 10.0)
    idx = np.array([[0, 1], [1, 1]], dtype=np.int32)
    kernel.accumulate(out, idx, np.array([1.0, -4.0]))
    assert out.tolist() == [[10.0, 11.0], [10.0, 6.0]]


def test_accumulate_of_no_entries_changes_nothing():
    out = np.full((2, 2), 3.0)
    kernel.accumulate(out, np.empty((2, 0), dtype=np.int32), np.empty(0))
    assert out.tolist() == [[3.0, 3.0], [3.0, 3.0]]


def test_accumulate_over_no_axes_adds_to_the_one_cell():
    out = np.zeros(())
    kernel.accumulate(out, np.empty((0, 1), dtype=np.int32), np.array([2.5]))
    kernel.accumulate(out, np.empty((0, 1), dtype=np.int32), np.array([1.0]))
    assert out == 3.5
