"""Tests for diff, cumsum, cumprod, argmax, argmin methods."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from nimblend import Array


class TestDiff:
    """Tests for diff method."""

    def test_diff_1d(self):
        """First differences on 1D array."""
        arr = Array(np.array([1, 3, 6, 10]), {"x": ["a", "b", "c", "d"]})
        result = arr.diff("x")

        assert result.shape == (3,)
        assert_array_equal(result.data, [2, 3, 4])
        assert_array_equal(result.coords["x"], ["b", "c", "d"])

    def test_diff_2d_first_dim(self):
        """Differences along first dimension."""
        data = np.array([[1, 2], [4, 6], [9, 12]])
        arr = Array(data, {"x": ["a", "b", "c"], "y": [0, 1]})
        result = arr.diff("x")

        assert result.shape == (2, 2)
        assert_array_equal(result.data, [[3, 4], [5, 6]])
        assert_array_equal(result.coords["x"], ["b", "c"])

    def test_diff_2d_second_dim(self):
        """Differences along second dimension."""
        data = np.array([[1, 3, 6], [10, 14, 19]])
        arr = Array(data, {"x": ["a", "b"], "y": [0, 1, 2]})
        result = arr.diff("y")

        assert result.shape == (2, 2)
        assert_array_equal(result.data, [[2, 3], [4, 5]])

    def test_diff_n2(self):
        """Second differences."""
        arr = Array(np.array([1, 3, 6, 10, 15]), {"x": list("abcde")})
        result = arr.diff("x", n=2)

        assert result.shape == (3,)
        assert_array_equal(result.data, [1, 1, 1])

    def test_diff_invalid_dim(self):
        """Error on invalid dimension."""
        arr = Array(np.array([1, 2, 3]), {"x": ["a", "b", "c"]})
        with pytest.raises(KeyError, match="not found"):
            arr.diff("y")


class TestCumsum:
    """Tests for cumsum method."""

    def test_cumsum_1d(self):
        """Cumulative sum on 1D array."""
        arr = Array(np.array([1, 2, 3, 4]), {"x": ["a", "b", "c", "d"]})
        result = arr.cumsum("x")

        assert result.shape == (4,)
        assert_array_equal(result.data, [1, 3, 6, 10])
        assert_array_equal(result.coords["x"], arr.coords["x"])

    def test_cumsum_2d(self):
        """Cumulative sum along dimension."""
        data = np.array([[1, 2], [3, 4], [5, 6]])
        arr = Array(data, {"x": ["a", "b", "c"], "y": [0, 1]})
        result = arr.cumsum("x")

        assert result.shape == (3, 2)
        assert_array_equal(result.data, [[1, 2], [4, 6], [9, 12]])

    def test_cumsum_invalid_dim(self):
        """Error on invalid dimension."""
        arr = Array(np.array([1, 2, 3]), {"x": ["a", "b", "c"]})
        with pytest.raises(KeyError, match="not found"):
            arr.cumsum("y")


class TestCumprod:
    """Tests for cumprod method."""

    def test_cumprod_1d(self):
        """Cumulative product on 1D array."""
        arr = Array(np.array([1, 2, 3, 4]), {"x": ["a", "b", "c", "d"]})
        result = arr.cumprod("x")

        assert result.shape == (4,)
        assert_array_equal(result.data, [1, 2, 6, 24])

    def test_cumprod_2d(self):
        """Cumulative product along dimension."""
        data = np.array([[1, 2], [3, 4], [5, 6]])
        arr = Array(data, {"x": ["a", "b", "c"], "y": [0, 1]})
        result = arr.cumprod("x")

        assert_array_equal(result.data, [[1, 2], [3, 8], [15, 48]])


class TestArgmax:
    """Tests for argmax method."""

    def test_argmax_1d(self):
        """Argmax on 1D returns scalar."""
        arr = Array(np.array([1, 5, 3, 2]), {"x": ["a", "b", "c", "d"]})
        result = arr.argmax("x")
        assert result == 1

    def test_argmax_2d(self):
        """Argmax along dimension."""
        data = np.array([[1, 5], [3, 2], [4, 6]])
        arr = Array(data, {"x": ["a", "b", "c"], "y": [0, 1]})
        result = arr.argmax("x")

        assert result.shape == (2,)
        assert_array_equal(result.data, [2, 2])  # max at index 2 for both cols
        assert "x" not in result.dims

    def test_argmax_invalid_dim(self):
        """Error on invalid dimension."""
        arr = Array(np.array([1, 2, 3]), {"x": ["a", "b", "c"]})
        with pytest.raises(KeyError, match="not found"):
            arr.argmax("y")


class TestArgmin:
    """Tests for argmin method."""

    def test_argmin_1d(self):
        """Argmin on 1D returns scalar."""
        arr = Array(np.array([3, 1, 5, 2]), {"x": ["a", "b", "c", "d"]})
        result = arr.argmin("x")
        assert result == 1

    def test_argmin_2d(self):
        """Argmin along dimension."""
        data = np.array([[3, 5], [1, 2], [4, 6]])
        arr = Array(data, {"x": ["a", "b", "c"], "y": [0, 1]})
        result = arr.argmin("x")

        assert result.shape == (2,)
        assert_array_equal(result.data, [1, 1])  # min at index 1 for both cols
