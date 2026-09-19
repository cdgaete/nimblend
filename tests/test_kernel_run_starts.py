import numpy as np
import pytest

from nimblend import kernel


def test_run_starts_is_true_only_at_the_first_position_of_each_run():
    keys = np.array([1, 1, 2, 2, 2, 5], dtype=np.int64)
    got = kernel._run_starts(keys)
    assert got.tolist() == [True, False, True, False, False, True]


def test_run_starts_is_true_at_every_position_of_distinct_ascending_keys():
    keys = np.array([1, 2, 3], dtype=np.int64)
    got = kernel._run_starts(keys)
    assert got.tolist() == [True, True, True]


def test_run_starts_of_one_key_is_true():
    keys = np.array([7], dtype=np.int64)
    got = kernel._run_starts(keys)
    assert got.tolist() == [True]


def test_run_starts_of_no_keys_raises_index_error():
    with pytest.raises(IndexError):
        kernel._run_starts(np.empty(0, dtype=np.int64))
