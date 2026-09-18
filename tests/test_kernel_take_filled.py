import numpy as np

from nimblend import kernel


def test_take_filled_reads_each_position_and_fills_the_absent_ones():
    data = np.array([10.0, 20.0, 30.0])
    take = np.array([2, -1, 0, -1])
    assert kernel.take_filled(data, take).tolist() == [30.0, 0.0, 10.0, 0.0]


def test_take_filled_uses_the_fill_it_is_given():
    data = np.array([10.0])
    assert kernel.take_filled(data, np.array([-1, 0]), 7.0).tolist() == [7.0, 10.0]


def test_take_filled_from_no_values_is_all_fill():
    got = kernel.take_filled(np.empty(0), np.array([-1, -1]))
    assert got.tolist() == [0.0, 0.0]
