import numpy as np
import pytest

from nimblend import kernel
from nimblend.buffer import EntryBuffer
from nimblend.coords import StoredCoord


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
