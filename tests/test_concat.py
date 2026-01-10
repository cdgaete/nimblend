"""Tests for concat function."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import nimblend as nb
from nimblend import Array


class TestConcat:
    """Tests for concat function."""

    def test_concat_1d(self):
        """Concatenate 1D arrays."""
        arr1 = Array(np.array([1, 2]), {"x": ["a", "b"]})
        arr2 = Array(np.array([3, 4]), {"x": ["c", "d"]})

        result = nb.concat([arr1, arr2], dim="x")

        assert result.shape == (4,)
        assert_array_equal(result.data, [1, 2, 3, 4])
        assert_array_equal(result.coords["x"], ["a", "b", "c", "d"])

    def test_concat_2d_first_dim(self):
        """Concatenate 2D arrays along first dimension."""
        arr1 = Array(np.array([[1, 2], [3, 4]]), {"x": ["a", "b"], "y": [0, 1]})
        arr2 = Array(np.array([[5, 6]]), {"x": ["c"], "y": [0, 1]})

        result = nb.concat([arr1, arr2], dim="x")

        assert result.shape == (3, 2)
        assert_array_equal(result.data, [[1, 2], [3, 4], [5, 6]])
        assert_array_equal(result.coords["x"], ["a", "b", "c"])
        assert_array_equal(result.coords["y"], [0, 1])

    def test_concat_2d_second_dim(self):
        """Concatenate 2D arrays along second dimension."""
        arr1 = Array(np.array([[1, 2], [3, 4]]), {"x": ["a", "b"], "y": [0, 1]})
        arr2 = Array(np.array([[5], [6]]), {"x": ["a", "b"], "y": [2]})

        result = nb.concat([arr1, arr2], dim="y")

        assert result.shape == (2, 3)
        assert_array_equal(result.data, [[1, 2, 5], [3, 4, 6]])
        assert_array_equal(result.coords["y"], [0, 1, 2])

    def test_concat_multiple(self):
        """Concatenate more than 2 arrays."""
        arr1 = Array(np.array([1]), {"x": ["a"]})
        arr2 = Array(np.array([2]), {"x": ["b"]})
        arr3 = Array(np.array([3]), {"x": ["c"]})

        result = nb.concat([arr1, arr2, arr3], dim="x")

        assert result.shape == (3,)
        assert_array_equal(result.data, [1, 2, 3])

    def test_concat_single_array(self):
        """Concatenating single array returns copy."""
        arr = Array(np.array([1, 2]), {"x": ["a", "b"]})
        result = nb.concat([arr], dim="x")

        assert_array_equal(result.data, arr.data)
        assert result is not arr

    def test_concat_empty_list_error(self):
        """Error on empty list."""
        with pytest.raises(ValueError, match="empty list"):
            nb.concat([], dim="x")

    def test_concat_mismatched_dims_error(self):
        """Error when dimensions don't match."""
        arr1 = Array(np.array([1, 2]), {"x": ["a", "b"]})
        arr2 = Array(np.array([[1, 2]]), {"x": ["a"], "y": [0, 1]})

        with pytest.raises(ValueError, match="different dimensions"):
            nb.concat([arr1, arr2], dim="x")

    def test_concat_mismatched_coords_error(self):
        """Error when non-concat coords don't match."""
        arr1 = Array(np.array([[1, 2], [3, 4]]), {"x": ["a", "b"], "y": [0, 1]})
        arr2 = Array(np.array([[5, 6]]), {"x": ["c"], "y": [0, 2]})  # y mismatch

        with pytest.raises(ValueError, match="do not match"):
            nb.concat([arr1, arr2], dim="x")

    def test_concat_invalid_dim_error(self):
        """Error on invalid dimension."""
        arr1 = Array(np.array([1, 2]), {"x": ["a", "b"]})
        arr2 = Array(np.array([3, 4]), {"x": ["c", "d"]})

        with pytest.raises(KeyError, match="not found"):
            nb.concat([arr1, arr2], dim="y")
