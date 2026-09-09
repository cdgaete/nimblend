import numpy as np

from nimblend import kernel


def test_gather_keeps_selected_positions_and_renumbers():
    idx = np.array([[0, 1, 2, 3], [0, 0, 1, 1]], dtype=np.int32)
    data = np.array([10.0, 11.0, 12.0, 13.0])
    out_idx, out_data = kernel.gather(idx, data, 0, np.array([1, 3]), 4)
    assert list(out_data) == [11.0, 13.0]
    assert list(out_idx[0]) == [0, 1]
    assert list(out_idx[1]) == [0, 1]


def test_gather_dropping_everything_gives_an_empty_block():
    idx = np.array([[0, 1]], dtype=np.int32)
    data = np.array([1.0, 2.0])
    out_idx, out_data = kernel.gather(idx, data, 0, np.array([], dtype=np.int64), 2)
    assert out_idx.shape == (1, 0)
    assert out_data.size == 0


def test_gather_writes_into_a_supplied_destination():
    idx = np.array([[0, 1, 2]], dtype=np.int32)
    data = np.array([1.0, 2.0, 3.0])
    dest_idx = np.full((1, 8), -7, dtype=np.int32)
    dest_data = np.full(8, -7.0)
    out_idx, out_data = kernel.gather(
        idx, data, 0, np.array([0, 2]), 3, out=(dest_idx, dest_data)
    )
    assert list(out_data) == [1.0, 3.0]
    # the destination was written in place, not replaced
    assert out_data.base is dest_data or out_data.base is dest_data.base
    assert dest_data[2] == -7.0


def test_gather_reorders_to_take_order():
    idx = np.array([[0, 1, 2]], dtype=np.int32)
    data = np.array([1.0, 2.0, 3.0])
    out_idx, out_data = kernel.gather(idx, data, 0, np.array([2, 0]), 3)
    assert sorted(out_data) == [1.0, 3.0]
    order = {int(p): v for p, v in zip(out_idx[0], out_data, strict=True)}
    assert order[0] == 3.0  # position 2 became position 0
    assert order[1] == 1.0  # position 0 became position 1
