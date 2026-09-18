import numpy as np

from nimblend import kernel


def test_ascending_keys_have_no_unsorted_position():
    assert kernel.first_unsorted(np.array([1, 4, 9], dtype=np.int64)) == -1


def test_the_position_is_the_key_followed_by_one_not_above_it():
    assert kernel.first_unsorted(np.array([1, 4, 4, 2], dtype=np.int64)) == 1
    assert kernel.first_unsorted(np.array([5, 3], dtype=np.int64)) == 0


def test_fewer_than_two_keys_ascend():
    assert kernel.first_unsorted(np.empty(0, dtype=np.int64)) == -1
    assert kernel.first_unsorted(np.array([7], dtype=np.int64)) == -1
