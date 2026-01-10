"""Tests for transpose() method."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from nimblend import Array


class TestTranspose:
    """Tests for transpose()."""

    def test_transpose_2d(self):
        """Transpose 2D array."""
        data = np.array([[1, 2, 3], [4, 5, 6]])
        arr = Array(data, {"x": ["a", "b"], "y": [0, 1, 2]})

        result = arr.transpose("y", "x")

        assert result.dims == ["y", "x"]
        assert result.shape == (3, 2)
        assert_array_equal(result.data, data.T)

    def test_transpose_3d(self):
        """Transpose 3D array."""
        data = np.arange(24).reshape(2, 3, 4)
        arr = Array(data, {"x": ["a", "b"], "y": [0, 1, 2], "z": [0, 1, 2, 3]})

        result = arr.transpose("z", "x", "y")

        assert result.dims == ["z", "x", "y"]
        assert result.shape == (4, 2, 3)

    def test_transpose_preserves_coords(self):
        """Transpose preserves coordinate values."""
        arr = Array(np.arange(6).reshape(2, 3),
                    {"x": ["a", "b"], "y": [10, 20, 30]})

        result = arr.transpose("y", "x")

        assert_array_equal(result.coords["x"], ["a", "b"])
        assert_array_equal(result.coords["y"], [10, 20, 30])

    def test_transpose_no_args_reverses(self):
        """Transpose with no args reverses dimension order."""
        arr = Array(np.arange(6).reshape(2, 3), {"x": ["a", "b"], "y": [0, 1, 2]})

        result = arr.transpose()

        assert result.dims == ["y", "x"]

    def test_T_property(self):
        """T property reverses dimensions."""
        arr = Array(np.arange(6).reshape(2, 3), {"x": ["a", "b"], "y": [0, 1, 2]})

        result = arr.T

        assert result.dims == ["y", "x"]
        assert_array_equal(result.data, arr.data.T)

    def test_transpose_missing_dim_raises(self):
        """Transpose with missing dimension raises."""
        arr = Array(np.arange(6).reshape(2, 3), {"x": ["a", "b"], "y": [0, 1, 2]})

        with pytest.raises(ValueError):
            arr.transpose("x")  # Missing y

    def test_transpose_preserves_name(self):
        """Transpose preserves array name."""
        arr = Array(np.arange(4).reshape(2, 2),
                    {"x": ["a", "b"], "y": [0, 1]}, name="test")

        assert arr.transpose("y", "x").name == "test"
