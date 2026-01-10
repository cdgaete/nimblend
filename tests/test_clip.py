"""Tests for clip() method - bound values to a range."""

import numpy as np
from numpy.testing import assert_array_equal

from nimblend import Array


class TestClipBasic:
    """Basic clip() functionality."""

    def test_clip_both_bounds(self):
        """Clip values to min and max."""
        data = np.array([1, 5, 10, 15, 20])
        arr = Array(data, {"x": ["a", "b", "c", "d", "e"]})

        result = arr.clip(5, 15)

        assert_array_equal(result.data, [5, 5, 10, 15, 15])

    def test_clip_only_min(self):
        """Clip with only min bound."""
        data = np.array([-5, 0, 5, 10])
        arr = Array(data, {"x": ["a", "b", "c", "d"]})

        result = arr.clip(min=0)

        assert_array_equal(result.data, [0, 0, 5, 10])

    def test_clip_only_max(self):
        """Clip with only max bound."""
        data = np.array([1, 5, 10, 15])
        arr = Array(data, {"x": ["a", "b", "c", "d"]})

        result = arr.clip(max=10)

        assert_array_equal(result.data, [1, 5, 10, 10])


class TestClipFloats:
    """clip() with float values."""

    def test_clip_float_data(self):
        """Clip float array."""
        data = np.array([0.1, 0.5, 0.9, 1.5])
        arr = Array(data, {"x": ["a", "b", "c", "d"]})

        result = arr.clip(0.2, 1.0)

        assert_array_equal(result.data, [0.2, 0.5, 0.9, 1.0])

    def test_clip_float_bounds(self):
        """Clip int array with float bounds."""
        data = np.array([1, 2, 3, 4, 5])
        arr = Array(data, {"x": ["a", "b", "c", "d", "e"]})

        result = arr.clip(1.5, 4.5)

        assert_array_equal(result.data, [1.5, 2, 3, 4, 4.5])


class TestClipMultiDimensional:
    """clip() on multi-dimensional arrays."""

    def test_clip_2d(self):
        """Clip 2D array."""
        data = np.array([[1, 2, 3], [4, 5, 6]])
        arr = Array(data, {"x": ["a", "b"], "y": [0, 1, 2]})

        result = arr.clip(2, 5)

        assert_array_equal(result.data, [[2, 2, 3], [4, 5, 5]])


class TestClipEdgeCases:
    """Edge cases for clip()."""

    def test_clip_preserves_coords(self):
        """clip() preserves coordinates."""
        data = np.array([[1, 10], [5, 15]])
        arr = Array(data, {"x": ["a", "b"], "y": [10, 20]})

        result = arr.clip(3, 12)

        assert result.dims == ["x", "y"]
        assert_array_equal(result.coords["x"], ["a", "b"])
        assert_array_equal(result.coords["y"], [10, 20])

    def test_clip_preserves_name(self):
        """clip() preserves array name."""
        arr = Array(np.array([1, 2, 3]), {"x": ["a", "b", "c"]}, name="test")

        result = arr.clip(0, 2)

        assert result.name == "test"

    def test_clip_no_change_when_in_range(self):
        """Values already in range are unchanged."""
        data = np.array([5, 6, 7, 8])
        arr = Array(data, {"x": ["a", "b", "c", "d"]})

        result = arr.clip(0, 100)

        assert_array_equal(result.data, data)

    def test_clip_negative_bounds(self):
        """Clip with negative bounds."""
        data = np.array([-10, -5, 0, 5, 10])
        arr = Array(data, {"x": ["a", "b", "c", "d", "e"]})

        result = arr.clip(-3, 3)

        assert_array_equal(result.data, [-3, -3, 0, 3, 3])
