"""Tests for fillna() and dropna() methods - NaN handling."""

import numpy as np
from numpy.testing import assert_array_equal

from nimblend import Array


class TestFillna:
    """Tests for fillna()."""

    def test_fillna_replaces_nan(self):
        """fillna replaces NaN values."""
        data = np.array([1.0, np.nan, 3.0, np.nan])
        arr = Array(data, {"x": ["a", "b", "c", "d"]})

        result = arr.fillna(0)

        assert_array_equal(result.data, [1.0, 0.0, 3.0, 0.0])

    def test_fillna_2d(self):
        """fillna works on 2D arrays."""
        data = np.array([[1.0, np.nan], [np.nan, 4.0]])
        arr = Array(data, {"x": ["a", "b"], "y": [0, 1]})

        result = arr.fillna(-1)

        assert_array_equal(result.data, [[1.0, -1.0], [-1.0, 4.0]])

    def test_fillna_no_nan(self):
        """fillna with no NaN returns equivalent array."""
        data = np.array([1.0, 2.0, 3.0])
        arr = Array(data, {"x": ["a", "b", "c"]})

        result = arr.fillna(0)

        assert_array_equal(result.data, data)

    def test_fillna_preserves_coords(self):
        """fillna preserves coordinates."""
        data = np.array([1.0, np.nan, 3.0])
        arr = Array(data, {"x": ["a", "b", "c"]})

        result = arr.fillna(0)

        assert_array_equal(result.coords["x"], ["a", "b", "c"])

    def test_fillna_preserves_name(self):
        """fillna preserves name."""
        arr = Array(np.array([1.0, np.nan]), {"x": ["a", "b"]}, name="test")

        result = arr.fillna(0)

        assert result.name == "test"


class TestDropna:
    """Tests for dropna()."""

    def test_dropna_removes_nan_coords(self):
        """dropna removes coordinates with NaN."""
        data = np.array([1.0, np.nan, 3.0, np.nan])
        arr = Array(data, {"x": ["a", "b", "c", "d"]})

        result = arr.dropna("x")

        assert result.dims == ["x"]
        assert_array_equal(result.coords["x"], ["a", "c"])
        assert_array_equal(result.data, [1.0, 3.0])

    def test_dropna_2d_along_dim(self):
        """dropna on 2D array removes rows/cols with any NaN."""
        data = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]])
        arr = Array(data, {"x": ["a", "b", "c"], "y": [0, 1]})

        result = arr.dropna("x")

        assert_array_equal(result.coords["x"], ["a", "c"])
        assert_array_equal(result.data, [[1.0, 2.0], [5.0, 6.0]])

    def test_dropna_no_nan(self):
        """dropna with no NaN returns equivalent array."""
        data = np.array([1.0, 2.0, 3.0])
        arr = Array(data, {"x": ["a", "b", "c"]})

        result = arr.dropna("x")

        assert_array_equal(result.data, data)
        assert_array_equal(result.coords["x"], ["a", "b", "c"])

    def test_dropna_preserves_name(self):
        """dropna preserves name."""
        arr = Array(np.array([1.0, np.nan, 3.0]), {"x": ["a", "b", "c"]}, name="test")

        result = arr.dropna("x")

        assert result.name == "test"
