"""Tests for isel() method - selection by integer index."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from nimblend import Array


class TestIselSingleIndex:
    """Select single indices along dimensions."""

    def test_isel_single_index_2d(self):
        """Select single index reduces dimension."""
        data = np.array([[1, 2, 3], [4, 5, 6]])
        arr = Array(data, {"x": ["a", "b"], "y": [10, 20, 30]})

        result = arr.isel({"x": 0})

        assert result.dims == ["y"]
        assert_array_equal(result.coords["y"], [10, 20, 30])
        assert_array_equal(result.data, [1, 2, 3])

    def test_isel_negative_index(self):
        """Negative indexing works."""
        data = np.array([[1, 2, 3], [4, 5, 6]])
        arr = Array(data, {"x": ["a", "b"], "y": [10, 20, 30]})

        result = arr.isel({"x": -1})

        assert result.dims == ["y"]
        assert_array_equal(result.data, [4, 5, 6])

    def test_isel_returns_scalar(self):
        """Selecting all dimensions returns scalar."""
        data = np.array([[1, 2], [3, 4]])
        arr = Array(data, {"x": ["a", "b"], "y": [0, 1]})

        result = arr.isel({"x": 1, "y": 1})

        assert result == 4


class TestIselMultipleIndices:
    """Select multiple indices along dimensions."""

    def test_isel_list_of_indices(self):
        """Select multiple indices preserves dimension."""
        data = np.array([[1, 2, 3], [4, 5, 6]])
        arr = Array(data, {"x": ["a", "b"], "y": [10, 20, 30]})

        result = arr.isel({"y": [0, 2]})

        assert result.dims == ["x", "y"]
        assert_array_equal(result.coords["y"], [10, 30])
        assert_array_equal(result.data, [[1, 3], [4, 6]])

    def test_isel_reorders_indices(self):
        """Index order is preserved."""
        data = np.array([1, 2, 3, 4])
        arr = Array(data, {"x": ["a", "b", "c", "d"]})

        result = arr.isel({"x": [3, 1]})

        assert_array_equal(result.coords["x"], ["d", "b"])
        assert_array_equal(result.data, [4, 2])


class TestIselEdgeCases:
    """Edge cases for isel()."""

    def test_isel_1d_array(self):
        """Selection on 1D array."""
        data = np.array([10, 20, 30])
        arr = Array(data, {"x": ["a", "b", "c"]})

        result = arr.isel({"x": 1})
        assert result == 20

    def test_isel_invalid_dim_raises(self):
        """Selecting non-existent dimension raises KeyError."""
        arr = Array(np.array([1, 2]), {"x": ["a", "b"]})

        with pytest.raises(KeyError):
            arr.isel({"z": 0})

    def test_isel_out_of_bounds_raises(self):
        """Out of bounds index raises IndexError."""
        arr = Array(np.array([1, 2]), {"x": ["a", "b"]})

        with pytest.raises(IndexError):
            arr.isel({"x": 5})

    def test_isel_preserves_name(self):
        """Selection preserves array name."""
        arr = Array(np.array([[1, 2], [3, 4]]), {"x": ["a", "b"], "y": [0, 1]},
                    name="test")

        result = arr.isel({"x": 0})
        assert result.name == "test"
