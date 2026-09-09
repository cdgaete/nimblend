import numpy as np

from nimblend import kernel


def test_distinct_keys_report_no_repeat():
    assert kernel.first_repeat(np.array([3, 1, 4, 9], dtype=np.int64)) == -1


def test_the_position_reported_is_the_second_key_to_carry_the_value():
    # the 1 at position 3 repeats the 1 at position 0
    assert kernel.first_repeat(np.array([1, 4, 9, 1], dtype=np.int64)) == 3


def test_adjacent_repeats_report_the_second_of_the_pair():
    assert kernel.first_repeat(np.array([4, 4, 9], dtype=np.int64)) == 1


def test_the_earliest_repeat_is_the_one_reported():
    # 9 repeats at position 4 and 1 repeats at position 3
    assert kernel.first_repeat(np.array([1, 9, 4, 1, 9], dtype=np.int64)) == 3


def test_a_value_carried_three_times_reports_its_second_position():
    assert kernel.first_repeat(np.array([7, 7, 7], dtype=np.int64)) == 1


def test_fewer_than_two_keys_can_carry_no_repeat():
    assert kernel.first_repeat(np.array([], dtype=np.int64)) == -1
    assert kernel.first_repeat(np.array([7], dtype=np.int64)) == -1


def test_the_result_is_a_python_int():
    assert type(kernel.first_repeat(np.array([1, 1], dtype=np.int64))) is int
    assert type(kernel.first_repeat(np.array([1, 2], dtype=np.int64))) is int
