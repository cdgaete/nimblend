import numpy as np
import pytest

from nimblend import kernel
from reference import random_block, to_dense


def test_canonical_output_is_sorted_and_unique():
    rng = np.random.default_rng(2)
    shape = (6, 5)
    idx, data = random_block(rng, shape, 20, allow_duplicates=True)
    out_idx, out_data = kernel.canonicalize(idx, data, shape)
    keys = kernel.ravel(out_idx, shape)
    assert np.all(np.diff(keys) > 0)
    assert out_data.size == out_idx.shape[1]


def test_canonicalize_preserves_totals():
    rng = np.random.default_rng(3)
    shape = (8, 4, 3)
    idx, data = random_block(rng, shape, 200, allow_duplicates=True)
    before, _ = to_dense(idx, data, shape)
    out_idx, out_data = kernel.canonicalize(idx, data, shape)
    after, _ = to_dense(out_idx, out_data, shape)
    assert np.allclose(before, after)


def test_canonicalize_keeps_an_explicit_zero():
    idx = np.array([[0, 1]], dtype=np.int32)
    data = np.array([0.0, 3.0])
    out_idx, out_data = kernel.canonicalize(idx, data, (2,))
    assert out_idx.shape[1] == 2
    assert out_data[0] == 0.0


def test_raise_policy_names_the_repeated_index():
    idx = np.array([[1, 1], [2, 2]], dtype=np.int32)
    data = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match=r"repeat"):
        kernel.canonicalize(idx, data, (3, 3), on_duplicate="raise")


def test_empty_block_canonicalizes_to_empty():
    idx = np.empty((2, 0), dtype=np.int32)
    data = np.empty(0)
    out_idx, out_data = kernel.canonicalize(idx, data, (3, 3))
    assert out_idx.shape == (2, 0)
    assert out_data.size == 0


def test_canonicalize_writes_into_a_supplied_destination():
    idx = np.array([[2, 0, 1]], dtype=np.int32)
    data = np.array([3.0, 1.0, 2.0])
    dest_idx = np.full((1, 8), -7, dtype=np.int32)
    dest_data = np.full(8, -7.0)
    out_idx, out_data = kernel.canonicalize(idx, data, (3,), out=(dest_idx, dest_data))
    assert list(out_data) == [1.0, 2.0, 3.0]
    assert dest_data[3] == -7.0  # beyond the written prefix
    assert out_data.base is dest_data or out_data.base is dest_data.base


def test_ordered_input_canonicalizes_to_the_same_thing_as_shuffled_input():
    rng = np.random.default_rng(11)
    shape = (7, 5, 3)
    idx, data = random_block(rng, shape, 60, allow_duplicates=True)
    ordered_idx, ordered_data = kernel.canonicalize(idx, data, shape)

    scramble = rng.permutation(ordered_idx.shape[1])
    again_idx, again_data = kernel.canonicalize(
        ordered_idx[:, scramble], ordered_data[scramble], shape
    )
    assert np.array_equal(again_idx, ordered_idx)
    assert np.allclose(again_data, ordered_data)

    # a second pass over an already canonical block changes nothing
    twice_idx, twice_data = kernel.canonicalize(ordered_idx, ordered_data, shape)
    assert np.array_equal(twice_idx, ordered_idx)
    assert np.allclose(twice_data, ordered_data)


def test_sorted_input_carrying_repeats_still_sums_them():
    # non-decreasing rather than increasing: no sort is needed, but the
    # repeated coordinate must still collapse to one entry
    idx = np.array([[0, 1, 1, 2]], dtype=np.int32)
    data = np.array([1.0, 2.0, 3.0, 4.0])
    out_idx, out_data = kernel.canonicalize(idx, data, (3,))
    assert list(out_idx[0]) == [0, 1, 2]
    assert list(out_data) == [1.0, 5.0, 4.0]


def test_a_canonical_result_owns_its_buffers():
    # the plain door copies whatever it is handed; from_canonical is the one
    # that shares memory with its caller
    idx = np.array([[0, 1, 2]], dtype=np.int32)
    data = np.array([1.0, 2.0, 3.0])
    for source_idx, source_data in (
        (idx, data),
        (idx[:, ::-1].copy(), data[::-1].copy()),
    ):
        out_idx, out_data = kernel.canonicalize(source_idx, source_data, (3,))
        assert not np.shares_memory(out_idx, source_idx)
        assert not np.shares_memory(out_data, source_data)


def test_an_ordered_block_is_not_permuted(monkeypatch):
    # the permutation an ordered block would produce is the identity, and
    # applying it gathers the index and the data at random; neither the sort
    # nor the gathers are performed
    def refuse(*args, **kwargs):
        raise AssertionError("an ordered block was sorted")

    monkeypatch.setattr(np, "argsort", refuse)

    idx = np.array([[0, 0, 1], [0, 2, 1]], dtype=np.int32)
    data = np.array([1.0, 2.0, 3.0])
    out_idx, out_data = kernel.canonicalize(idx, data, (2, 3))
    assert np.array_equal(out_idx, idx)
    assert list(out_data) == [1.0, 2.0, 3.0]

    # keys that repeat are adjacent in an ordered block, so they collapse
    # without a sort either
    repeated = np.array([[0, 1, 1, 2]], dtype=np.int32)
    out_idx, out_data = kernel.canonicalize(
        repeated, np.array([1.0, 2.0, 3.0, 4.0]), (3,)
    )
    assert list(out_idx[0]) == [0, 1, 2]
    assert list(out_data) == [1.0, 5.0, 4.0]
