import numpy as np
import pytest

from nimblend import kernel


def block(cells, values):
    return np.array(cells, dtype=np.int32).T.reshape(2, -1), np.array(values, float)


def test_the_union_keeps_every_cell_and_sums_the_repeats():
    a = block([(0, 0), (1, 2)], [1.0, 2.0])
    b = block([(0, 0), (0, 1)], [10.0, 20.0])
    c = block([(1, 2)], [100.0])
    idx, data = kernel.merge_sum(
        [a[0], b[0], c[0]], [a[1], b[1], c[1]], (2, 3), "union"
    )
    assert idx.tolist() == [[0, 0, 1], [0, 1, 2]]
    assert data.tolist() == [11.0, 20.0, 102.0]


def test_the_intersection_keeps_the_cells_every_block_has():
    a = block([(0, 0), (1, 2)], [1.0, 2.0])
    b = block([(0, 0), (1, 2), (0, 1)], [10.0, 20.0, 30.0])
    c = block([(1, 2)], [100.0])
    idx, data = kernel.merge_sum(
        [a[0], b[0], c[0]], [a[1], b[1], c[1]], (2, 3), "intersect"
    )
    assert idx.tolist() == [[1], [2]]
    assert data.tolist() == [122.0]


def test_the_result_equals_the_pairwise_sum_in_block_order():
    # values added in block order give the same floating point result as a
    # left fold of pairwise additions
    rng = np.random.default_rng(0)
    blocks = []
    for _ in range(5):
        keys = np.unique(rng.integers(0, 60, 25))
        idx = np.stack(np.unravel_index(keys, (6, 10))).astype(np.int32)
        blocks.append((idx, rng.normal(size=keys.size)))
    idx, data = kernel.merge_sum(
        [b[0] for b in blocks], [b[1] for b in blocks], (6, 10), "union"
    )
    total = {}
    for b_idx, b_data in blocks:
        for key, value in zip(kernel.ravel(b_idx, (6, 10)).tolist(), b_data.tolist()):
            total[key] = total[key] + value if key in total else value
    assert kernel.ravel(idx, (6, 10)).tolist() == sorted(total)
    assert data.tolist() == [total[k] for k in sorted(total)]


def test_a_stored_zero_and_a_sum_of_zero_are_kept():
    a = block([(0, 0), (0, 1)], [0.0, 1.0])
    b = block([(0, 1)], [-1.0])
    idx, data = kernel.merge_sum([a[0], b[0]], [a[1], b[1]], (1, 2), "union")
    assert idx.tolist() == [[0, 0], [0, 1]]
    assert data.tolist() == [0.0, 0.0]


def test_blocks_with_no_entries_merge_to_no_entries():
    empty = np.empty((2, 0), dtype=np.int32), np.empty(0)
    idx, data = kernel.merge_sum(
        [empty[0], empty[0]], [empty[1], empty[1]], (2, 3), "union"
    )
    assert idx.shape == (2, 0) and data.size == 0


def test_a_mode_other_than_union_or_intersect_raises():
    a = block([(0, 0)], [1.0])
    with pytest.raises(ValueError, match="how is 'union' or 'intersect'"):
        kernel.merge_sum([a[0]], [a[1]], (1, 1), "outer")
