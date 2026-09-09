import numpy as np
import pytest

from nimblend import kernel
from reference import random_block, to_dense


def test_reduce_drops_the_axis_and_sums_collisions():
    idx = np.array([[0, 0, 1], [0, 1, 0]], dtype=np.int32)
    data = np.array([1.0, 2.0, 5.0])
    out_idx, out_data = kernel.reduce_axis(idx, data, 1, (2, 2))
    assert out_idx.shape[0] == 1
    assert list(out_idx[0]) == [0, 1]
    assert list(out_data) == [3.0, 5.0]


def test_reduce_matches_the_dense_reference():
    rng = np.random.default_rng(5)
    shape = (6, 4, 5)
    idx, data = random_block(rng, shape, 90)
    dense, _ = to_dense(idx, data, shape)
    out_idx, out_data = kernel.reduce_axis(idx, data, 1, shape)
    got, _ = to_dense(out_idx, out_data, (6, 5))
    assert np.allclose(got, dense.sum(axis=1))


def test_reduce_keeps_distinct_columns_apart():
    # the model-building case: each summed position carries its own column,
    # so nothing collides and the entry count is unchanged
    idx = np.array([[0, 0, 0], [0, 1, 2], [7, 8, 9]], dtype=np.int32)
    data = np.ones(3)
    out_idx, out_data = kernel.reduce_axis(idx, data, 1, (1, 3, 10))
    assert out_data.size == 3
    assert list(out_idx[1]) == [7, 8, 9]


def test_min_and_max_reductions():
    idx = np.array([[0, 0, 1], [0, 1, 0]], dtype=np.int32)
    data = np.array([4.0, -1.0, 9.0])
    _, mins = kernel.reduce_axis(idx, data, 1, (2, 2), op="min")
    _, maxs = kernel.reduce_axis(idx, data, 1, (2, 2), op="max")
    assert list(mins) == [-1.0, 9.0]
    assert list(maxs) == [4.0, 9.0]


def test_reducing_the_last_axis_leaves_one_entry_over_no_axes():
    # over no dimensions the product of the sizes is one, so every entry
    # lands on the single coordinate that product holds
    idx = np.array([[0, 1, 2]], dtype=np.int32)
    data = np.array([1.0, 2.0, 5.0])
    out_idx, out_data = kernel.reduce_axis(idx, data, 0, (3,))
    assert out_idx.shape == (0, 1)
    assert list(out_data) == [8.0]


def test_reducing_the_last_axis_carries_the_op_it_is_given():
    idx = np.array([[0, 1, 2]], dtype=np.int32)
    data = np.array([4.0, -1.0, 9.0])
    assert list(kernel.reduce_axis(idx, data, 0, (3,), op="min")[1]) == [-1.0]
    assert list(kernel.reduce_axis(idx, data, 0, (3,), op="max")[1]) == [9.0]


def test_unknown_op_raises():
    idx = np.array([[0]], dtype=np.int32)
    with pytest.raises(ValueError, match="sum"):
        kernel.reduce_axis(idx, np.array([1.0]), 0, (1,), op="median")
