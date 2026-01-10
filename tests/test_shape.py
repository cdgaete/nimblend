"""Tests for squeeze() and expand_dims() methods."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from nimblend import Array


class TestSqueeze:
    """Tests for squeeze()."""

    def test_squeeze_all(self):
        """Squeeze all size-1 dimensions."""
        data = np.arange(6).reshape(1, 2, 1, 3)
        arr = Array(data, {"w": ["a"], "x": ["a", "b"], "y": ["c"], "z": [0, 1, 2]})

        result = arr.squeeze()

        assert result.dims == ["x", "z"]
        assert result.shape == (2, 3)

    def test_squeeze_specific_dim(self):
        """Squeeze a specific dimension."""
        data = np.arange(6).reshape(1, 2, 3)
        arr = Array(data, {"x": ["a"], "y": ["a", "b"], "z": [0, 1, 2]})

        result = arr.squeeze("x")

        assert result.dims == ["y", "z"]
        assert result.shape == (2, 3)

    def test_squeeze_preserves_data(self):
        """Squeeze preserves data values."""
        data = np.arange(6).reshape(1, 6)
        arr = Array(data, {"x": ["a"], "y": [0, 1, 2, 3, 4, 5]})

        result = arr.squeeze()

        assert_array_equal(result.data, np.arange(6))

    def test_squeeze_invalid_dim_raises(self):
        """Squeeze non-existent dim raises KeyError."""
        arr = Array(np.arange(6).reshape(2, 3), {"x": ["a", "b"], "y": [0, 1, 2]})

        with pytest.raises(KeyError):
            arr.squeeze("z")

    def test_squeeze_non_singleton_raises(self):
        """Squeeze on non-size-1 dimension raises ValueError."""
        arr = Array(np.arange(6).reshape(2, 3), {"x": ["a", "b"], "y": [0, 1, 2]})

        with pytest.raises(ValueError):
            arr.squeeze("x")

    def test_squeeze_preserves_name(self):
        """Squeeze preserves array name."""
        arr = Array(np.arange(3).reshape(1, 3),
                    {"x": ["a"], "y": [0, 1, 2]}, name="test")

        assert arr.squeeze().name == "test"


class TestExpandDims:
    """Tests for expand_dims()."""

    def test_expand_dims_basic(self):
        """Add new dimension."""
        arr = Array(np.arange(6).reshape(2, 3), {"x": ["a", "b"], "y": [0, 1, 2]})

        result = arr.expand_dims("z", "new")

        assert result.dims == ["z", "x", "y"]
        assert result.shape == (1, 2, 3)
        assert_array_equal(result.coords["z"], ["new"])

    def test_expand_dims_default_coord(self):
        """Default coordinate is 0."""
        arr = Array(np.arange(3), {"x": ["a", "b", "c"]})

        result = arr.expand_dims("y")

        assert_array_equal(result.coords["y"], [0])

    def test_expand_dims_existing_raises(self):
        """Adding existing dimension raises ValueError."""
        arr = Array(np.arange(3), {"x": ["a", "b", "c"]})

        with pytest.raises(ValueError):
            arr.expand_dims("x")

    def test_expand_dims_preserves_name(self):
        """expand_dims preserves array name."""
        arr = Array(np.arange(3), {"x": ["a", "b", "c"]}, name="test")

        assert arr.expand_dims("y").name == "test"


class TestSqueezeExpandRoundtrip:
    """Test squeeze and expand_dims together."""

    def test_expand_then_squeeze(self):
        """Expand then squeeze returns original shape."""
        arr = Array(np.arange(6).reshape(2, 3), {"x": ["a", "b"], "y": [0, 1, 2]})

        expanded = arr.expand_dims("z", "new")
        squeezed = expanded.squeeze("z")

        assert squeezed.dims == arr.dims
        assert squeezed.shape == arr.shape
        assert_array_equal(squeezed.data, arr.data)
