import numpy as np

from nimblend import kernel


def test_select_keeps_the_entries_at_one_position_without_the_axis():
    idx = np.array([[0, 0, 1, 1], [0, 2, 1, 2]], dtype=np.int32)
    data = np.array([1.0, 2.0, 3.0, 4.0])
    out_idx, out_data = kernel.select_axis(idx, data, 0, 1, 2)
    assert out_idx.tolist() == [[1, 2]]
    assert out_data.tolist() == [3.0, 4.0]


def test_select_along_an_inner_axis_keeps_the_other_rows_in_order():
    idx = np.array([[0, 0, 1, 1], [0, 2, 1, 2]], dtype=np.int32)
    data = np.array([1.0, 2.0, 3.0, 4.0])
    out_idx, out_data = kernel.select_axis(idx, data, 1, 2, 3)
    assert out_idx.tolist() == [[0, 1]]
    assert out_data.tolist() == [2.0, 4.0]


def test_select_at_a_position_without_entries_is_empty():
    idx = np.array([[0, 1]], dtype=np.int32)
    out_idx, out_data = kernel.select_axis(idx, np.array([1.0, 2.0]), 0, 2, 3)
    assert out_idx.shape == (0, 0)
    assert out_data.size == 0


def test_select_matches_gather_followed_by_dropping_the_axis():
    rng = np.random.default_rng(2)
    idx = np.stack([rng.integers(0, n, 50) for n in (4, 5, 6)]).astype(np.int32)
    idx, data = kernel.canonicalize(idx, rng.standard_normal(50), (4, 5, 6))
    for axis, extent in enumerate((4, 5, 6)):
        for position in range(extent):
            got_idx, got_data = kernel.select_axis(idx, data, axis, position, extent)
            ref_idx, ref_data = kernel.gather(
                idx, data, axis, np.array([position]), extent
            )
            assert np.array_equal(got_idx, np.delete(ref_idx, axis, axis=0))
            assert np.array_equal(got_data, ref_data)
