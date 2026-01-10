"""Tests for coarsen operation."""

import numpy as np
import pytest

from nimblend import Array


class TestCoarsen:
    """Tests for Array.coarsen()."""

    def test_coarsen_mean(self):
        """Coarsen with mean aggregation."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        arr = Array(data, {"x": [0, 1, 2, 3, 4, 5]})

        result = arr.coarsen({"x": 2}, func="mean")
        expected = np.array([1.5, 3.5, 5.5])
        np.testing.assert_array_equal(result.data, expected)
        np.testing.assert_array_equal(result.coords["x"], [0, 2, 4])

    def test_coarsen_sum(self):
        """Coarsen with sum aggregation."""
        data = np.array([1.0, 2.0, 3.0, 4.0])
        arr = Array(data, {"x": [0, 1, 2, 3]})

        result = arr.coarsen({"x": 2}, func="sum")
        expected = np.array([3.0, 7.0])
        np.testing.assert_array_equal(result.data, expected)

    def test_coarsen_min_max(self):
        """Coarsen with min/max aggregation."""
        data = np.array([1.0, 5.0, 2.0, 8.0])
        arr = Array(data, {"x": [0, 1, 2, 3]})

        result_min = arr.coarsen({"x": 2}, func="min")
        result_max = arr.coarsen({"x": 2}, func="max")

        np.testing.assert_array_equal(result_min.data, [1.0, 2.0])
        np.testing.assert_array_equal(result_max.data, [5.0, 8.0])

    def test_coarsen_trim(self):
        """Coarsen trims incomplete windows."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # 5 elements, window=2
        arr = Array(data, {"x": [0, 1, 2, 3, 4]})

        result = arr.coarsen({"x": 2}, func="mean", boundary="trim")
        expected = np.array([1.5, 3.5])  # Last element dropped
        np.testing.assert_array_equal(result.data, expected)

    def test_coarsen_pad(self):
        """Coarsen pads incomplete windows with zeros."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        arr = Array(data, {"x": [0, 1, 2, 3, 4]})

        result = arr.coarsen({"x": 2}, func="sum", boundary="pad")
        expected = np.array([3.0, 7.0, 5.0])  # Last window: 5+0
        np.testing.assert_array_equal(result.data, expected)

    def test_coarsen_2d(self):
        """Coarsen works on 2D arrays."""
        data = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        arr = Array(data, {"x": ["a", "b"], "y": [0, 1, 2, 3]})

        result = arr.coarsen({"y": 2}, func="mean")
        expected = np.array([[1.5, 3.5], [5.5, 7.5]])
        np.testing.assert_array_equal(result.data, expected)
        np.testing.assert_array_equal(result.coords["y"], [0, 2])

    def test_coarsen_multi_dim(self):
        """Coarsen works on multiple dimensions."""
        data = np.arange(16).reshape(4, 4).astype(float)
        arr = Array(data, {"x": [0, 1, 2, 3], "y": [0, 1, 2, 3]})

        result = arr.coarsen({"x": 2, "y": 2}, func="mean")
        expected = np.array([[2.5, 4.5], [10.5, 12.5]])
        np.testing.assert_array_equal(result.data, expected)

    def test_coarsen_invalid_dim(self):
        """Coarsen raises on invalid dimension."""
        arr = Array(np.array([1, 2, 3]), {"x": [0, 1, 2]})

        with pytest.raises(KeyError, match="not found"):
            arr.coarsen({"z": 2})

    def test_coarsen_invalid_func(self):
        """Coarsen raises on invalid function."""
        arr = Array(np.array([1, 2, 3, 4]), {"x": [0, 1, 2, 3]})

        with pytest.raises(ValueError, match="Unknown function"):
            arr.coarsen({"x": 2}, func="invalid")

    def test_coarsen_invalid_window(self):
        """Coarsen raises on invalid window size."""
        arr = Array(np.array([1, 2, 3]), {"x": [0, 1, 2]})

        with pytest.raises(ValueError, match="Window size"):
            arr.coarsen({"x": 0})
