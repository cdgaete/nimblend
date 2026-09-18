import numpy as np
import pytest

from nimblend import kernel
from reference import random_block, to_dense


def canonical(idx, data, shape):
    return kernel.canonicalize(idx, data, shape, on_duplicate="raise")


def test_weighted_sum_scales_each_value_by_its_weight_and_drops_the_axis():
    idx = np.array([[0, 0, 1], [0, 1, 0]], dtype=np.int32)
    data = np.array([1.0, 2.0, 5.0])
    weights = np.array([10.0, 100.0])
    out_idx, out_data = kernel.weighted_sum_axis(idx, data, 0, weights, (2, 2))
    # column 0: 1 * 10 + 5 * 100; column 1: 2 * 10
    assert out_idx.tolist() == [[0, 1]]
    assert out_data.tolist() == [510.0, 20.0]


# 90 entries over (6, 4, 5) outnumber the remaining cells; 40 entries over
# (12, 10, 9) do not, and the two cases take the two accumulations
@pytest.mark.parametrize("shape,nnz", [((6, 4, 5), 90), ((12, 10, 9), 40)])
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("block", [1, 7, 1 << 22])
def test_weighted_sum_matches_the_dense_reference(shape, nnz, axis, block):
    rng = np.random.default_rng(axis)
    idx, data = canonical(*random_block(rng, shape, nnz), shape)
    weights = rng.standard_normal(shape[axis])
    dense, present = to_dense(idx, data, shape)
    view = [None] * len(shape)
    view[axis] = slice(None)
    wanted = (dense * weights[tuple(view)]).sum(axis=axis)
    out_idx, out_data = kernel.weighted_sum_axis(
        idx, data, axis, weights, shape, block=block
    )
    kept = tuple(size for a, size in enumerate(shape) if a != axis)
    got, got_present = to_dense(out_idx, out_data, kept)
    assert np.allclose(got, wanted)
    assert np.array_equal(got_present, present.any(axis=axis))
    assert kernel.is_canonical(out_idx, kept)


@pytest.mark.parametrize("block", [1, 3, 1 << 22])
def test_weighted_sum_matches_reduce_axis_over_the_weighted_values(block):
    rng = np.random.default_rng(11)
    shape = (40, 30)
    idx, data = canonical(*random_block(rng, shape, 500), shape)
    weights = rng.standard_normal(shape[0])
    ref_idx, ref_data = kernel.reduce_axis(idx, data * weights[idx[0]], 0, shape)
    out_idx, out_data = kernel.weighted_sum_axis(
        idx, data, 0, weights, shape, block=block
    )
    assert np.array_equal(out_idx, ref_idx)
    assert np.allclose(out_data, ref_data)


def test_weighted_sum_keeps_a_coordinate_whose_sum_is_zero():
    idx = np.array([[0, 1], [0, 0]], dtype=np.int32)
    data = np.array([1.0, -1.0])
    out_idx, out_data = kernel.weighted_sum_axis(idx, data, 0, np.ones(2), (2, 1))
    assert out_idx.tolist() == [[0]]
    assert out_data.tolist() == [0.0]


def test_weighted_sum_of_an_empty_block_is_empty():
    idx = np.empty((2, 0), dtype=np.int32)
    out_idx, out_data = kernel.weighted_sum_axis(
        idx, np.empty(0), 1, np.ones(3), (2, 3)
    )
    assert out_idx.shape == (1, 0)
    assert out_data.size == 0


def test_weighted_sum_over_the_only_axis_leaves_one_entry_over_no_axes():
    idx = np.array([[0, 1, 2]], dtype=np.int32)
    data = np.array([1.0, 2.0, 5.0])
    out_idx, out_data = kernel.weighted_sum_axis(
        idx, data, 0, np.array([1.0, 2.0, 3.0]), (3,), block=2
    )
    assert out_idx.shape == (0, 1)
    assert out_data.tolist() == [20.0]


def test_weighted_sum_raises_for_a_block_below_one():
    idx = np.array([[0]], dtype=np.int32)
    with pytest.raises(ValueError, match="block 0 is below 1"):
        kernel.weighted_sum_axis(idx, np.ones(1), 0, np.ones(1), (1,), block=0)


def test_weighted_sum_allocates_no_temporary_of_the_size_of_the_input():
    import tracemalloc

    # 4000 positions along the summed axis with 100 entries each, over 500
    # remaining positions: the input holds 400 000 entries, the result 500
    rows, per_row, cols = 4000, 100, 500
    n = rows * per_row
    idx = np.stack([np.repeat(np.arange(rows), per_row), np.arange(n) % cols]).astype(
        np.int32
    )
    idx, data = canonical(
        idx, np.random.default_rng(3).standard_normal(n), (rows, cols)
    )
    weights = np.random.default_rng(4).standard_normal(rows)
    tracemalloc.start()
    kernel.weighted_sum_axis(idx, data, 0, weights, (rows, cols), block=1 << 12)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # a product array would hold data.nbytes of values and twice that of index
    assert peak < data.nbytes // 4


def test_weighted_sum_over_a_wide_frame_allocates_no_temporary_of_the_input_size():
    import tracemalloc

    # the remaining cells outnumber the entries, and the entries fall on 500
    # distinct remaining coordinates
    rows, per_row, cols = 4000, 100, 500
    n = rows * per_row
    idx = np.stack(
        [
            np.repeat(np.arange(rows), per_row),
            np.arange(n) % cols,
            np.zeros(n, dtype=np.int64),
        ]
    ).astype(np.int32)
    shape = (rows, cols, 10_000)
    idx, data = canonical(idx, np.random.default_rng(3).standard_normal(n), shape)
    weights = np.random.default_rng(4).standard_normal(rows)
    tracemalloc.start()
    out_idx, _ = kernel.weighted_sum_axis(idx, data, 0, weights, shape, block=1 << 12)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert out_idx.shape == (2, cols)
    assert peak < data.nbytes // 4
