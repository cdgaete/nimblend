"""Tests for sel() method - selection by coordinate labels."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from nimblend import Array


class TestSelSingleValue:
    """Select single values along dimensions."""

    def test_sel_single_value_2d(self):
        """Select single value reduces dimension."""
        data = np.array([[1, 2, 3], [4, 5, 6]])
        arr = Array(data, {"x": ["a", "b"], "y": [10, 20, 30]})

        result = arr.sel({"x": "a"})

        assert result.dims == ["y"]
        assert_array_equal(result.coords["y"], [10, 20, 30])
        assert_array_equal(result.data, [1, 2, 3])

    def test_sel_single_value_other_dim(self):
        """Select along second dimension."""
        data = np.array([[1, 2, 3], [4, 5, 6]])
        arr = Array(data, {"x": ["a", "b"], "y": [10, 20, 30]})

        result = arr.sel({"y": 20})

        assert result.dims == ["x"]
        assert_array_equal(result.coords["x"], ["a", "b"])
        assert_array_equal(result.data, [2, 5])

    def test_sel_returns_scalar(self):
        """Selecting all dimensions returns scalar."""
        data = np.array([[1, 2], [3, 4]])
        arr = Array(data, {"x": ["a", "b"], "y": [0, 1]})

        result = arr.sel({"x": "b", "y": 1})

        assert result == 4
        assert isinstance(result, (int, np.integer))


class TestSelMultipleValues:
    """Select multiple values along dimensions."""

    def test_sel_list_of_values(self):
        """Select multiple values preserves dimension."""
        data = np.array([[1, 2, 3], [4, 5, 6]])
        arr = Array(data, {"x": ["a", "b"], "y": [10, 20, 30]})

        result = arr.sel({"y": [10, 30]})

        assert result.dims == ["x", "y"]
        assert_array_equal(result.coords["y"], [10, 30])
        assert_array_equal(result.data, [[1, 3], [4, 6]])

    def test_sel_reorders_values(self):
        """Selection order is preserved."""
        data = np.array([1, 2, 3, 4])
        arr = Array(data, {"x": ["a", "b", "c", "d"]})

        result = arr.sel({"x": ["d", "b"]})

        assert_array_equal(result.coords["x"], ["d", "b"])
        assert_array_equal(result.data, [4, 2])

    def test_sel_multiple_dims_mixed(self):
        """Mix of single and multiple value selection."""
        data = np.arange(12).reshape(2, 3, 2)
        arr = Array(data, {"x": ["a", "b"], "y": [0, 1, 2], "z": ["p", "q"]})

        result = arr.sel({"x": "a", "y": [0, 2]})

        assert result.dims == ["y", "z"]
        assert_array_equal(result.coords["y"], [0, 2])
        assert_array_equal(result.data, [[0, 1], [4, 5]])


class TestSelEdgeCases:
    """Edge cases for sel()."""

    def test_sel_1d_array(self):
        """Selection on 1D array."""
        data = np.array([10, 20, 30])
        arr = Array(data, {"x": ["a", "b", "c"]})

        result = arr.sel({"x": "b"})
        assert result == 20

        result2 = arr.sel({"x": ["a", "c"]})
        assert_array_equal(result2.data, [10, 30])

    def test_sel_invalid_dim_raises(self):
        """Selecting non-existent dimension raises KeyError."""
        arr = Array(np.array([1, 2]), {"x": ["a", "b"]})

        with pytest.raises(KeyError):
            arr.sel({"z": "a"})

    def test_sel_invalid_label_raises(self):
        """Selecting non-existent label raises ValueError."""
        arr = Array(np.array([1, 2]), {"x": ["a", "b"]})

        with pytest.raises(ValueError):
            arr.sel({"x": "z"})

    def test_sel_preserves_name(self):
        """Selection preserves array name."""
        arr = Array(np.array([[1, 2], [3, 4]]), {"x": ["a", "b"], "y": [0, 1]},
                    name="test")

        result = arr.sel({"x": "a"})
        assert result.name == "test"
