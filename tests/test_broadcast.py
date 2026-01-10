"""Tests for broadcast_like() method - expand to match shape."""

import numpy as np
from numpy.testing import assert_array_equal

from nimblend import Array


class TestBroadcastLikeBasic:
    """Basic broadcast_like() functionality."""

    def test_broadcast_add_dim(self):
        """Broadcast 1D to 2D by adding dimension."""
        arr = Array(np.array([1, 2, 3]), {"x": ["a", "b", "c"]})
        target = Array(np.zeros((3, 2)), {"x": ["a", "b", "c"], "y": [0, 1]})

        result = arr.broadcast_like(target)

        assert result.dims == ["x", "y"]
        assert result.shape == (3, 2)
        assert_array_equal(result.data, [[1, 1], [2, 2], [3, 3]])

    def test_broadcast_matching_dims(self):
        """Broadcast with same dims but different coords."""
        arr = Array(np.array([1, 2]), {"x": ["a", "b"]})
        target = Array(np.zeros((2, 3)), {"x": ["a", "b"], "y": [0, 1, 2]})

        result = arr.broadcast_like(target)

        assert result.dims == ["x", "y"]
        assert result.shape == (2, 3)
        assert_array_equal(result.data, [[1, 1, 1], [2, 2, 2]])


class TestBroadcastLikeMultiDim:
    """broadcast_like() on multi-dimensional arrays."""

    def test_broadcast_2d_to_3d(self):
        """Broadcast 2D to 3D."""
        arr = Array(np.array([[1, 2], [3, 4]]), {"x": ["a", "b"], "y": [0, 1]})
        target = Array(
            np.zeros((2, 2, 3)), {"x": ["a", "b"], "y": [0, 1], "z": ["p", "q", "r"]}
        )

        result = arr.broadcast_like(target)

        assert result.dims == ["x", "y", "z"]
        assert result.shape == (2, 2, 3)
        # Original data broadcast along z
        assert_array_equal(result.data[0, 0, :], [1, 1, 1])
        assert_array_equal(result.data[1, 1, :], [4, 4, 4])


class TestBroadcastLikeEdgeCases:
    """Edge cases for broadcast_like()."""

    def test_broadcast_preserves_coords(self):
        """Coordinates from target are used."""
        arr = Array(np.array([10, 20]), {"x": ["a", "b"]})
        target = Array(np.zeros((2, 2)), {"x": ["a", "b"], "y": ["p", "q"]})

        result = arr.broadcast_like(target)

        assert_array_equal(result.coords["x"], ["a", "b"])
        assert_array_equal(result.coords["y"], ["p", "q"])

    def test_broadcast_preserves_name(self):
        """broadcast_like() preserves array name."""
        arr = Array(np.array([1, 2]), {"x": ["a", "b"]}, name="test")
        target = Array(np.zeros((2, 3)), {"x": ["a", "b"], "y": [0, 1, 2]})

        result = arr.broadcast_like(target)

        assert result.name == "test"

    def test_broadcast_same_shape(self):
        """Broadcast to same shape returns equivalent array."""
        arr = Array(np.array([1, 2, 3]), {"x": ["a", "b", "c"]})
        target = Array(np.array([0, 0, 0]), {"x": ["a", "b", "c"]})

        result = arr.broadcast_like(target)

        assert_array_equal(result.data, arr.data)
        assert result.dims == arr.dims

    def test_broadcast_scalar_like(self):
        """Scalar-like array broadcasts to any shape."""
        arr = Array(np.array([[5]]), {"x": ["a"], "y": [0]})
        target = Array(np.zeros((3, 2)), {"x": ["a", "b", "c"], "y": [0, 1]})

        result = arr.broadcast_like(target)

        assert result.shape == (3, 2)
        # Only [0,0] has value 5, rest is 0 (outer join fills missing)
        assert result.data[0, 0] == 5
