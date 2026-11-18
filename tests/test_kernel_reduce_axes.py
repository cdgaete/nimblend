import numpy as np
import pytest

from nimblend import kernel
from reference import random_block, to_dense


def canonical(rng, shape, nnz):
    """Return a random block over `shape` in canonical order."""
    idx, data = random_block(rng, shape, nnz)
    return kernel.canonicalize(idx, data, shape)


def _count_sorts(monkeypatch):
    calls = []
    sort = np.argsort

    def counted(*args, **kwargs):
        calls.append(1)
        return sort(*args, **kwargs)

    monkeypatch.setattr(np, "argsort", counted)
    return calls


def test_reducing_several_axes_matches_the_dense_reference():
    rng = np.random.default_rng(11)
    shape = (5, 4, 6, 3)
    idx, data = canonical(rng, shape, 160)
    dense, _ = to_dense(idx, data, shape)
    out_idx, out_data = kernel.reduce_axes(idx, data, [1, 3], shape)
    got, _ = to_dense(out_idx, out_data, (5, 6))
    assert np.allclose(got, dense.sum(axis=(1, 3)))


@pytest.mark.parametrize("op", ["sum", "min", "max"])
def test_reducing_two_axes_equals_reducing_each_in_turn(op):
    rng = np.random.default_rng(12)
    shape = (5, 4, 6, 3)
    idx, data = canonical(rng, shape, 160)
    once_idx, once_data = kernel.reduce_axes(idx, data, [1, 3], shape, op=op)
    step_idx, step_data = kernel.reduce_axes(idx, data, [3], shape, op=op)
    step_idx, step_data = kernel.reduce_axes(step_idx, step_data, [1], shape[:3], op=op)
    assert np.array_equal(once_idx, step_idx)
    assert np.allclose(once_data, step_data)


def test_the_axes_are_reduced_in_any_order_given():
    rng = np.random.default_rng(13)
    shape = (5, 4, 6, 3)
    idx, data = canonical(rng, shape, 160)
    ahead = kernel.reduce_axes(idx, data, [1, 3], shape)
    back = kernel.reduce_axes(idx, data, [3, 1], shape)
    assert np.array_equal(ahead[0], back[0])
    assert np.array_equal(ahead[1], back[1])


def test_the_result_of_several_axes_is_canonical():
    rng = np.random.default_rng(14)
    shape = (5, 4, 6, 3)
    idx, data = canonical(rng, shape, 160)
    out_idx, _ = kernel.reduce_axes(idx, data, [0, 2], shape)
    assert kernel.is_canonical(out_idx, (4, 3))


def test_reducing_several_axes_sorts_once(monkeypatch):
    rng = np.random.default_rng(15)
    shape = (5, 4, 6, 3)
    idx, data = canonical(rng, shape, 160)
    calls = _count_sorts(monkeypatch)
    kernel.reduce_axes(idx, data, [0, 2], shape)
    assert len(calls) == 1


def test_kept_keys_that_strictly_ascend_are_not_sorted(monkeypatch):
    # each entry has its own position on the last axis, in ascending order
    idx = np.array([[0, 0, 1, 1], [0, 1, 0, 1], [4, 5, 6, 7]], dtype=np.int32)
    data = np.array([1.0, 2.0, 3.0, 4.0])
    calls = _count_sorts(monkeypatch)
    out_idx, out_data = kernel.reduce_axes(idx, data, [0, 1], (2, 2, 9))
    assert not calls
    assert out_idx.tolist() == [[4, 5, 6, 7]]
    assert out_data.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_reducing_every_axis_returns_one_entry():
    idx = np.array([[0, 0, 1], [0, 1, 0]], dtype=np.int32)
    data = np.array([4.0, -1.0, 9.0])
    out_idx, out_data = kernel.reduce_axes(idx, data, [0, 1], (2, 2))
    assert out_idx.shape == (0, 1)
    assert out_data.tolist() == [12.0]
    assert kernel.reduce_axes(idx, data, [0, 1], (2, 2), op="min")[1].tolist() == [-1.0]


def test_reducing_axes_of_no_entries_returns_no_entries():
    idx = np.empty((3, 0), dtype=np.int32)
    out_idx, out_data = kernel.reduce_axes(idx, np.empty(0), [0, 2], (2, 3, 4))
    assert out_idx.shape == (1, 0)
    assert out_data.size == 0


def test_reducing_several_axes_writes_into_out():
    rng = np.random.default_rng(16)
    shape = (5, 4, 6, 3)
    idx, data = canonical(rng, shape, 160)
    want_idx, want_data = kernel.reduce_axes(idx, data, [1, 3], shape)
    dest = (np.empty((2, 160), dtype=np.int32), np.empty(160))
    got_idx, got_data = kernel.reduce_axes(idx, data, [1, 3], shape, out=dest)
    assert np.array_equal(got_idx, want_idx)
    assert np.array_equal(got_data, want_data)
    assert np.shares_memory(got_data, dest[1])


def test_reduce_axes_refuses_an_unknown_op():
    idx = np.array([[0]], dtype=np.int32)
    with pytest.raises(ValueError, match="sum, min, max"):
        kernel.reduce_axes(idx, np.array([1.0]), [0], (1,), op="median")
