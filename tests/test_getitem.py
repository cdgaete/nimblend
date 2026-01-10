"""Tests for __getitem__ - indexing and slicing."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from nimblend import Array


class TestGetitemDictSelection:
    """Dict-based selection (alias for sel)."""

    def test_getitem_dict_single_value(self):
        """Dict indexing works like sel."""
        data = np.array([[1, 2, 3], [4, 5, 6]])
        arr = Array(data, {"x": ["a", "b"], "y": [10, 20, 30]})

        result = arr[{"x": "a"}]

        assert result.dims == ["y"]
        assert_array_equal(result.data, [1, 2, 3])

    def test_getitem_dict_multiple_values(self):
        """Dict indexing with list of values."""
        data = np.array([[1, 2, 3], [4, 5, 6]])
        arr = Array(data, {"x": ["a", "b"], "y": [10, 20, 30]})

        result = arr[{"y": [10, 30]}]

        assert result.dims == ["x", "y"]
        assert_array_equal(result.data, [[1, 3], [4, 6]])

    def test_getitem_dict_returns_scalar(self):
        """Dict indexing all dims returns scalar."""
        data = np.array([[1, 2], [3, 4]])
        arr = Array(data, {"x": ["a", "b"], "y": [0, 1]})

        result = arr[{"x": "b", "y": 1}]

        assert result == 4


class TestGetitemSlicing:
    """Integer and slice indexing on first dimension."""

    def test_getitem_integer(self):
        """Single integer index."""
        data = np.array([[1, 2, 3], [4, 5, 6]])
        arr = Array(data, {"x": ["a", "b"], "y": [10, 20, 30]})

        result = arr[0]

        assert result.dims == ["y"]
        assert_array_equal(result.data, [1, 2, 3])

    def test_getitem_negative_integer(self):
        """Negative integer index."""
        data = np.array([[1, 2, 3], [4, 5, 6]])
        arr = Array(data, {"x": ["a", "b"], "y": [10, 20, 30]})

        result = arr[-1]

        assert result.dims == ["y"]
        assert_array_equal(result.data, [4, 5, 6])

    def test_getitem_slice(self):
        """Slice indexing."""
        data = np.array([[1, 2], [3, 4], [5, 6]])
        arr = Array(data, {"x": ["a", "b", "c"], "y": [0, 1]})

        result = arr[0:2]

        assert result.dims == ["x", "y"]
        assert_array_equal(result.coords["x"], ["a", "b"])
        assert_array_equal(result.data, [[1, 2], [3, 4]])

    def test_getitem_slice_step(self):
        """Slice with step."""
        data = np.array([1, 2, 3, 4, 5])
        arr = Array(data, {"x": ["a", "b", "c", "d", "e"]})

        result = arr[::2]

        assert_array_equal(result.coords["x"], ["a", "c", "e"])
        assert_array_equal(result.data, [1, 3, 5])

    def test_getitem_1d_integer_returns_scalar(self):
        """Integer on 1D array returns scalar."""
        data = np.array([10, 20, 30])
        arr = Array(data, {"x": ["a", "b", "c"]})

        result = arr[1]

        assert result == 20


class TestGetitemDimName:
    """Selection by dimension name (returns coord array)."""

    def test_getitem_dim_name(self):
        """String index returns coordinate array."""
        data = np.array([[1, 2], [3, 4]])
        arr = Array(data, {"x": ["a", "b"], "y": [10, 20]})

        result = arr["x"]

        assert_array_equal(result, ["a", "b"])

    def test_getitem_invalid_dim_name_raises(self):
        """Invalid dimension name raises KeyError."""
        arr = Array(np.array([1, 2]), {"x": ["a", "b"]})

        with pytest.raises(KeyError):
            arr["z"]


class TestGetitemEdgeCases:
    """Edge cases for __getitem__."""

    def test_getitem_preserves_name(self):
        """Indexing preserves array name."""
        arr = Array(
            np.array([[1, 2], [3, 4]]), {"x": ["a", "b"], "y": [0, 1]}, name="test"
        )

        result = arr[0]
        assert result.name == "test"

        result2 = arr[{"x": "a"}]
        assert result2.name == "test"
