"""Tests for comparison operators."""

import numpy as np
from numpy.testing import assert_array_equal

from nimblend import Array


class TestComparisonWithScalar:
    """Comparison with scalar values."""

    def test_eq_scalar(self):
        arr = Array(np.array([1, 2, 3]), {"x": ["a", "b", "c"]})
        result = arr == 2
        assert_array_equal(result.data, [False, True, False])

    def test_ne_scalar(self):
        arr = Array(np.array([1, 2, 3]), {"x": ["a", "b", "c"]})
        result = arr != 2
        assert_array_equal(result.data, [True, False, True])

    def test_lt_scalar(self):
        arr = Array(np.array([1, 2, 3]), {"x": ["a", "b", "c"]})
        result = arr < 2
        assert_array_equal(result.data, [True, False, False])

    def test_le_scalar(self):
        arr = Array(np.array([1, 2, 3]), {"x": ["a", "b", "c"]})
        result = arr <= 2
        assert_array_equal(result.data, [True, True, False])

    def test_gt_scalar(self):
        arr = Array(np.array([1, 2, 3]), {"x": ["a", "b", "c"]})
        result = arr > 2
        assert_array_equal(result.data, [False, False, True])

    def test_ge_scalar(self):
        arr = Array(np.array([1, 2, 3]), {"x": ["a", "b", "c"]})
        result = arr >= 2
        assert_array_equal(result.data, [False, True, True])


class TestComparisonWithArray:
    """Comparison between arrays with alignment."""

    def test_eq_aligned_arrays(self):
        arr1 = Array(np.array([1, 2, 3]), {"x": ["a", "b", "c"]})
        arr2 = Array(np.array([1, 0, 3]), {"x": ["a", "b", "c"]})
        result = arr1 == arr2
        assert_array_equal(result.data, [True, False, True])

    def test_lt_partial_overlap(self):
        arr1 = Array(np.array([1, 2]), {"x": ["a", "b"]})
        arr2 = Array(np.array([2, 3]), {"x": ["b", "c"]})
        result = arr1 < arr2
        # a: 1 < 0 = False, b: 2 < 2 = False, c: 0 < 3 = True
        assert_array_equal(result.data, [False, False, True])

    def test_comparison_preserves_coords(self):
        arr = Array(np.array([1, 2, 3]), {"x": ["a", "b", "c"]})
        result = arr > 1
        assert result.dims == ["x"]
        assert_array_equal(result.coords["x"], ["a", "b", "c"])

    def test_comparison_returns_bool_dtype(self):
        arr = Array(np.array([1.0, 2.0, 3.0]), {"x": ["a", "b", "c"]})
        result = arr > 1.5
        assert result.data.dtype == bool
