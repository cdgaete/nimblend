import time

import numpy as np
import pytest

from nimblend import kernel


def test_intersect_keeps_only_shared_keys():
    a = np.array([1, 3, 5, 7], dtype=np.int64)
    b = np.array([3, 4, 5], dtype=np.int64)
    merged, ta, tb = kernel.align(a, b, "intersect")
    assert list(merged) == [3, 5]
    assert list(a[ta]) == [3, 5]
    assert list(b[tb]) == [3, 5]
    assert (ta >= 0).all() and (tb >= 0).all()


def test_union_marks_absent_entries_with_minus_one():
    a = np.array([1, 3], dtype=np.int64)
    b = np.array([3, 9], dtype=np.int64)
    merged, ta, tb = kernel.align(a, b, "union")
    assert list(merged) == [1, 3, 9]
    assert list(ta) == [0, 1, -1]
    assert list(tb) == [-1, 0, 1]


def test_disjoint_intersect_is_empty():
    a = np.array([1, 2], dtype=np.int64)
    b = np.array([8, 9], dtype=np.int64)
    merged, ta, tb = kernel.align(a, b, "intersect")
    assert merged.size == 0 and ta.size == 0 and tb.size == 0


def test_empty_operand_is_handled():
    a = np.array([1, 2], dtype=np.int64)
    b = np.empty(0, dtype=np.int64)
    merged, ta, tb = kernel.align(a, b, "intersect")
    assert merged.size == 0
    merged, ta, tb = kernel.align(a, b, "union")
    assert list(merged) == [1, 2]
    assert list(tb) == [-1, -1]


def test_take_vectors_reconstruct_both_operands():
    rng = np.random.default_rng(4)
    a = np.unique(rng.integers(0, 500, 200)).astype(np.int64)
    b = np.unique(rng.integers(0, 500, 200)).astype(np.int64)
    merged, ta, tb = kernel.align(a, b, "union")
    assert np.array_equal(merged[ta >= 0], a)
    assert np.array_equal(merged[tb >= 0], b)


def test_unknown_how_raises():
    a = np.array([1], dtype=np.int64)
    with pytest.raises(ValueError, match="union"):
        kernel.align(a, a, "outer")


def test_union_matches_a_hash_union_on_randomised_operands():
    rng = np.random.default_rng(11)
    for _ in range(300):
        hi = int(rng.integers(2, 400))
        a = np.unique(rng.integers(0, hi, int(rng.integers(0, 60)))).astype(np.int64)
        b = np.unique(rng.integers(0, hi, int(rng.integers(0, 60)))).astype(np.int64)
        merged, ta, tb = kernel.align(a, b, "union")
        assert np.array_equal(merged, np.union1d(a, b))
        assert np.array_equal(merged[ta >= 0], a)
        assert np.array_equal(merged[tb >= 0], b)
        assert np.array_equal(a[ta[ta >= 0]], a)
        assert np.array_equal(b[tb[tb >= 0]], b)
        assert np.array_equal(ta < 0, ~np.isin(merged, a))
        assert np.array_equal(tb < 0, ~np.isin(merged, b))


def test_union_of_sorted_keys_costs_about_what_a_search_costs():
    # align receives sorted unique keys, so it must merge them; recovering
    # that order by hashing costs an order of magnitude more than the
    # searchsorted the merge is built on.
    rng = np.random.default_rng(12)
    a = np.unique(rng.integers(0, 4_000_000, 800_000)).astype(np.int64)
    b = np.unique(rng.integers(0, 4_000_000, 800_000)).astype(np.int64)

    start = time.perf_counter()
    np.searchsorted(a, b)
    search = time.perf_counter() - start

    kernel.align(a, b, "union")
    start = time.perf_counter()
    kernel.align(a, b, "union")
    union = time.perf_counter() - start

    assert union < 8 * search, (
        f"align {union * 1e3:.1f} ms, search {search * 1e3:.1f} ms"
    )
