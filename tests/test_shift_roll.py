"""Tests for shift and roll operations."""

import numpy as np
import pytest

from nimblend import Array


class TestShift:
    """Tests for Array.shift()."""

    def test_shift_forward(self):
        """Shift forward pads start with fill_value."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        arr = Array(data, {"x": ["a", "b", "c", "d", "e"]})

        result = arr.shift({"x": 2})
        expected = np.array([0.0, 0.0, 1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result.data, expected)

    def test_shift_backward(self):
        """Shift backward pads end with fill_value."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        arr = Array(data, {"x": ["a", "b", "c", "d", "e"]})

        result = arr.shift({"x": -2})
        expected = np.array([3.0, 4.0, 5.0, 0.0, 0.0])
        np.testing.assert_array_equal(result.data, expected)

    def test_shift_custom_fill(self):
        """Custom fill_value is used."""
        data = np.array([1.0, 2.0, 3.0])
        arr = Array(data, {"x": ["a", "b", "c"]})

        result = arr.shift({"x": 1}, fill_value=np.nan)
        assert np.isnan(result.data[0])
        np.testing.assert_array_equal(result.data[1:], [1.0, 2.0])

    def test_shift_2d(self):
        """Shift works on 2D arrays."""
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        arr = Array(data, {"x": ["a", "b"], "y": [0, 1, 2]})

        result = arr.shift({"x": 1})
        expected = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        np.testing.assert_array_equal(result.data, expected)

    def test_shift_multiple_dims(self):
        """Shift works on multiple dimensions at once."""
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        arr = Array(data, {"x": ["a", "b"], "y": [0, 1, 2]})

        result = arr.shift({"x": 1, "y": 1})
        expected = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 2.0]])
        np.testing.assert_array_equal(result.data, expected)

    def test_shift_zero_is_noop(self):
        """Shift by 0 returns unchanged data."""
        data = np.array([1.0, 2.0, 3.0])
        arr = Array(data, {"x": ["a", "b", "c"]})

        result = arr.shift({"x": 0})
        np.testing.assert_array_equal(result.data, data)

    def test_shift_invalid_dim(self):
        """Shift raises on invalid dimension."""
        arr = Array(np.array([1, 2, 3]), {"x": [0, 1, 2]})

        with pytest.raises(KeyError, match="not found"):
            arr.shift({"z": 1})

    def test_shift_preserves_coords(self):
        """Shift preserves coordinates."""
        arr = Array(np.array([1, 2, 3]), {"x": ["a", "b", "c"]})
        result = arr.shift({"x": 1})
        np.testing.assert_array_equal(result.coords["x"], ["a", "b", "c"])


class TestRoll:
    """Tests for Array.roll()."""

    def test_roll_forward(self):
        """Roll forward wraps last values to start."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        arr = Array(data, {"x": ["a", "b", "c", "d", "e"]})

        result = arr.roll({"x": 2})
        expected = np.array([4.0, 5.0, 1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result.data, expected)

    def test_roll_backward(self):
        """Roll backward wraps first values to end."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        arr = Array(data, {"x": ["a", "b", "c", "d", "e"]})

        result = arr.roll({"x": -2})
        expected = np.array([3.0, 4.0, 5.0, 1.0, 2.0])
        np.testing.assert_array_equal(result.data, expected)

    def test_roll_2d(self):
        """Roll works on 2D arrays."""
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        arr = Array(data, {"x": ["a", "b"], "y": [0, 1, 2]})

        result = arr.roll({"y": 1})
        expected = np.array([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]])
        np.testing.assert_array_equal(result.data, expected)

    def test_roll_multiple_dims(self):
        """Roll works on multiple dimensions."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        arr = Array(data, {"x": ["a", "b"], "y": [0, 1]})

        result = arr.roll({"x": 1, "y": 1})
        expected = np.array([[4.0, 3.0], [2.0, 1.0]])
        np.testing.assert_array_equal(result.data, expected)

    def test_roll_invalid_dim(self):
        """Roll raises on invalid dimension."""
        arr = Array(np.array([1, 2, 3]), {"x": [0, 1, 2]})

        with pytest.raises(KeyError, match="not found"):
            arr.roll({"z": 1})

    def test_roll_preserves_coords(self):
        """Roll preserves coordinates."""
        arr = Array(np.array([1, 2, 3]), {"x": ["a", "b", "c"]})
        result = arr.roll({"x": 1})
        np.testing.assert_array_equal(result.coords["x"], ["a", "b", "c"])
