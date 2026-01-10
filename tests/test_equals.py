"""Tests for equals() method - full array comparison."""

import numpy as np

from nimblend import Array


class TestEqualsIdentical:
    """Test equals() on identical arrays."""

    def test_equals_same_array(self):
        """Array equals itself."""
        arr = Array(np.array([1, 2, 3]), {"x": ["a", "b", "c"]})

        assert arr.equals(arr)

    def test_equals_copy(self):
        """Array equals its copy."""
        arr = Array(np.array([[1, 2], [3, 4]]), {"x": ["a", "b"], "y": [0, 1]})

        assert arr.equals(arr.copy())

    def test_equals_identical_construction(self):
        """Separately constructed identical arrays are equal."""
        arr1 = Array(np.array([1, 2, 3]), {"x": ["a", "b", "c"]})
        arr2 = Array(np.array([1, 2, 3]), {"x": ["a", "b", "c"]})

        assert arr1.equals(arr2)


class TestEqualsDifferent:
    """Test equals() on different arrays."""

    def test_equals_different_data(self):
        """Different data values are not equal."""
        arr1 = Array(np.array([1, 2, 3]), {"x": ["a", "b", "c"]})
        arr2 = Array(np.array([1, 2, 4]), {"x": ["a", "b", "c"]})

        assert not arr1.equals(arr2)

    def test_equals_different_coords(self):
        """Different coordinates are not equal."""
        arr1 = Array(np.array([1, 2, 3]), {"x": ["a", "b", "c"]})
        arr2 = Array(np.array([1, 2, 3]), {"x": ["a", "b", "d"]})

        assert not arr1.equals(arr2)

    def test_equals_different_dims(self):
        """Different dimension names are not equal."""
        arr1 = Array(np.array([1, 2, 3]), {"x": ["a", "b", "c"]})
        arr2 = Array(np.array([1, 2, 3]), {"y": ["a", "b", "c"]})

        assert not arr1.equals(arr2)

    def test_equals_different_shape(self):
        """Different shapes are not equal."""
        arr1 = Array(np.array([1, 2, 3]), {"x": ["a", "b", "c"]})
        arr2 = Array(np.array([1, 2]), {"x": ["a", "b"]})

        assert not arr1.equals(arr2)

    def test_equals_different_dim_order(self):
        """Different dimension order are not equal."""
        data = np.array([[1, 2], [3, 4]])
        arr1 = Array(data, {"x": ["a", "b"], "y": [0, 1]})
        arr2 = Array(data.T, {"y": [0, 1], "x": ["a", "b"]})

        assert not arr1.equals(arr2)


class TestEqualsEdgeCases:
    """Edge cases for equals()."""

    def test_equals_float_arrays(self):
        """Float arrays equality."""
        arr1 = Array(np.array([1.0, 2.0, 3.0]), {"x": ["a", "b", "c"]})
        arr2 = Array(np.array([1.0, 2.0, 3.0]), {"x": ["a", "b", "c"]})

        assert arr1.equals(arr2)

    def test_equals_name_ignored(self):
        """Name differences don't affect equality."""
        arr1 = Array(np.array([1, 2]), {"x": ["a", "b"]}, name="foo")
        arr2 = Array(np.array([1, 2]), {"x": ["a", "b"]}, name="bar")

        assert arr1.equals(arr2)

    def test_equals_empty_arrays(self):
        """Empty arrays are equal if coords match."""
        arr1 = Array(np.array([]), {"x": []})
        arr2 = Array(np.array([]), {"x": []})

        assert arr1.equals(arr2)
