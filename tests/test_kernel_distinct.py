import numpy as np

from nimblend import kernel


def test_distinct_hands_back_strictly_ascending_keys_unchanged():
    keys = np.array([1, 4, 9], dtype=np.int64)
    got = kernel.distinct(keys)
    assert list(got) == [1, 4, 9]
    # nothing to remove, so no copy is taken
    assert got is keys


def test_distinct_drops_adjacent_repeats():
    keys = np.array([1, 1, 4, 4, 4, 9], dtype=np.int64)
    assert list(kernel.distinct(keys)) == [1, 4, 9]


def test_distinct_sorts_keys_that_arrive_out_of_order():
    keys = np.array([9, 1, 4, 1], dtype=np.int64)
    assert list(kernel.distinct(keys)) == [1, 4, 9]


def test_distinct_of_fewer_than_two_keys_is_the_input():
    empty = np.array([], dtype=np.int64)
    assert kernel.distinct(empty).size == 0
    one = np.array([7], dtype=np.int64)
    assert list(kernel.distinct(one)) == [7]
