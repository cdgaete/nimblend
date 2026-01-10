"""Tests for where() method - conditional replacement."""

import numpy as np
from numpy.testing import assert_array_equal

from nimblend import Array


class TestWhereBasic:
    """Basic where() functionality."""

    def test_where_replace_negatives(self):
        """Replace negative values with 0."""
        data = np.array([-1, 2, -3, 4])
        arr = Array(data, {"x": ["a", "b", "c", "d"]})

        result = arr.where(arr >= 0, 0)

        assert_array_equal(result.data, [0, 2, 0, 4])

    def test_where_with_scalar_other(self):
        """Replace based on condition with scalar."""
        data = np.array([[1, 5], [3, 8]])
        arr = Array(data, {"x": ["a", "b"], "y": [0, 1]})

        result = arr.where(arr > 3, -1)

        assert_array_equal(result.data, [[-1, 5], [-1, 8]])

    def test_where_preserves_when_true(self):
        """Values where condition is True are preserved."""
        data = np.array([1, 2, 3, 4, 5])
        arr = Array(data, {"x": ["a", "b", "c", "d", "e"]})

        result = arr.where(arr > 0, 99)

        assert_array_equal(result.data, data)  # All preserved


class TestWhereWithArrayOther:
    """where() with Array as replacement value."""

    def test_where_array_other(self):
        """Replace with values from another array."""
        data = np.array([1, 2, 3, 4])
        arr = Array(data, {"x": ["a", "b", "c", "d"]})
        other = Array(np.array([10, 20, 30, 40]), {"x": ["a", "b", "c", "d"]})

        result = arr.where(arr > 2, other)

        assert_array_equal(result.data, [10, 20, 3, 4])


class TestWhereBooleanCondition:
    """where() with boolean Array conditions."""

    def test_where_boolean_array_condition(self):
        """Condition is a boolean Array."""
        data = np.array([1, 2, 3, 4])
        arr = Array(data, {"x": ["a", "b", "c", "d"]})
        cond = Array(np.array([True, False, True, False]),
                     {"x": ["a", "b", "c", "d"]})

        result = arr.where(cond, 0)

        assert_array_equal(result.data, [1, 0, 3, 0])


class TestWhereEdgeCases:
    """Edge cases for where()."""

    def test_where_preserves_coords(self):
        """where() preserves coordinates."""
        data = np.array([[1, 2], [3, 4]])
        arr = Array(data, {"x": ["a", "b"], "y": [10, 20]})

        result = arr.where(arr > 2, 0)

        assert result.dims == ["x", "y"]
        assert_array_equal(result.coords["x"], ["a", "b"])
        assert_array_equal(result.coords["y"], [10, 20])

    def test_where_preserves_name(self):
        """where() preserves array name."""
        arr = Array(np.array([1, 2, 3]), {"x": ["a", "b", "c"]}, name="test")

        result = arr.where(arr > 1, 0)

        assert result.name == "test"

    def test_where_float_dtype(self):
        """where() works with float arrays."""
        data = np.array([1.5, -0.5, 2.5])
        arr = Array(data, {"x": ["a", "b", "c"]})

        result = arr.where(arr > 0, 0.0)

        assert_array_equal(result.data, [1.5, 0.0, 2.5])

    def test_where_2d(self):
        """where() works on 2D arrays."""
        data = np.array([[1, 2, 3], [4, 5, 6]])
        arr = Array(data, {"x": ["a", "b"], "y": [0, 1, 2]})

        result = arr.where(arr > 3, -1)

        assert_array_equal(result.data, [[-1, -1, -1], [4, 5, 6]])
