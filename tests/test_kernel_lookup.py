import numpy as np

from nimblend import kernel


def test_lookup_returns_the_position_of_each_probe():
    keys = np.array([10, 20, 30], dtype=np.int64)
    probe = np.array([30, 10, 20], dtype=np.int64)
    assert list(kernel.lookup(keys, probe)) == [2, 0, 1]


def test_lookup_marks_an_absent_probe_with_minus_one():
    keys = np.array([10, 30], dtype=np.int64)
    probe = np.array([10, 20, 30, 40], dtype=np.int64)
    assert list(kernel.lookup(keys, probe)) == [0, -1, 1, -1]


def test_lookup_tolerates_a_repeated_probe():
    keys = np.array([5, 7], dtype=np.int64)
    probe = np.array([7, 7, 5, 7], dtype=np.int64)
    assert list(kernel.lookup(keys, probe)) == [1, 1, 0, 1]


def test_lookup_against_empty_keys_is_all_absent():
    keys = np.empty(0, dtype=np.int64)
    probe = np.array([1, 2], dtype=np.int64)
    assert list(kernel.lookup(keys, probe)) == [-1, -1]


def test_lookup_of_an_empty_probe_is_empty():
    keys = np.array([1], dtype=np.int64)
    assert kernel.lookup(keys, np.empty(0, dtype=np.int64)).size == 0
