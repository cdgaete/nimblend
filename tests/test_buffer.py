import numpy as np
import pytest

from nimblend import kernel
from nimblend.buffer import EntryBuffer
from nimblend.coords import ProductCoord, StoredCoord


def test_reserve_hands_out_successive_slices():
    buf = EntryBuffer(2, 10)
    first_idx, first_data = buf.reserve(3)
    second_idx, second_data = buf.reserve(4)
    assert first_idx.shape == (2, 3)
    assert second_idx.shape == (2, 4)
    assert second_data.shape == (4,)
    assert buf.at == 7
    first_data[:] = 1.0
    second_data[:] = 2.0
    assert list(buf.data[:7]) == [1.0] * 3 + [2.0] * 4


def test_a_reserved_slice_writes_into_the_buffer():
    buf = EntryBuffer(1, 8)
    dest = buf.reserve(3)
    idx = np.array([[2, 0, 1]], dtype=np.int32)
    kernel.canonicalize(idx, np.array([3.0, 1.0, 2.0]), (3,), out=dest)
    assert list(buf.data[:3]) == [1.0, 2.0, 3.0]
    assert np.shares_memory(dest[1], buf.data)


def test_reserving_past_the_capacity_raises():
    buf = EntryBuffer(2, 4)
    buf.reserve(3)
    with pytest.raises(ValueError, match="capacity"):
        buf.reserve(2)


def test_written_returns_only_the_prefix():
    buf = EntryBuffer(2, 10)
    buf.reserve(3)
    idx, data = buf.written()
    assert idx.shape == (2, 3)
    assert data.shape == (3,)


def test_array_wraps_the_prefix_without_copying():
    buf = EntryBuffer(2, 10)
    idx, data = buf.reserve(2)
    idx[:] = np.array([[0, 1], [0, 1]], dtype=np.int32)
    data[:] = [5.0, 6.0]
    coords = {"r": StoredCoord(np.arange(2)), "c": StoredCoord(np.arange(2))}
    arr = buf.array(coords, ("r", "c"))
    assert np.shares_memory(arr.data, buf.data)
    assert arr.nnz == 2
    assert arr.to_dense()[1, 1] == 6.0


def test_an_empty_buffer_yields_an_empty_array():
    buf = EntryBuffer(1, 4)
    coords = {"r": StoredCoord(np.arange(2))}
    assert buf.array(coords, ("r",)).nnz == 0


def test_a_growing_buffer_reserves_beyond_its_capacity():
    buffer = EntryBuffer(2, 2, resizable=True)
    index, data = buffer.reserve(2)
    index[:] = [[0, 0], [0, 1]]
    data[:] = [1.0, 2.0]
    index, data = buffer.reserve(3)
    index[:] = [[1, 1, 2], [0, 1, 0]]
    data[:] = [3.0, 4.0, 5.0]
    assert buffer.capacity >= 5
    assert buffer.written()[0].tolist() == [[0, 0, 1, 1, 2], [0, 1, 0, 1, 0]]
    assert buffer.written()[1].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_a_growth_at_least_doubles_the_capacity():
    buffer = EntryBuffer(1, 4, resizable=True)
    buffer.reserve(4)
    buffer.reserve(1)
    assert buffer.capacity >= 8


def test_a_growing_buffer_of_no_capacity_reserves():
    buffer = EntryBuffer(2, 0, resizable=True)
    assert buffer.reserve(3)[1].size == 3


def test_a_view_reserved_before_a_growth_is_not_a_view_of_the_grown_buffer():
    buffer = EntryBuffer(1, 1, resizable=True)
    _, before = buffer.reserve(1)
    buffer.reserve(1)
    assert not np.shares_memory(before, buffer.data)


def test_the_array_of_a_grown_buffer_is_canonical_with_every_entry():
    buffer = EntryBuffer(1, 1, resizable=True)
    for at in range(5):
        index, data = buffer.reserve(1)
        index[0, 0] = at
        data[0] = float(at)
    array = buffer.array({"k": ProductCoord((5,))}, ("k",))
    assert array.to_dense().tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_a_buffer_that_does_not_grow_raises_beyond_its_capacity():
    with pytest.raises(ValueError, match="exceeds the capacity"):
        EntryBuffer(1, 1).reserve(2)
