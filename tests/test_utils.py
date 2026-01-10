"""Tests for rename(), copy(), and astype() methods."""

import numpy as np
from numpy.testing import assert_array_equal

from nimblend import Array


class TestRename:
    """Tests for rename()."""

    def test_rename_single_dim(self):
        """Rename one dimension."""
        arr = Array(np.arange(6).reshape(2, 3), {"x": ["a", "b"], "y": [0, 1, 2]})

        result = arr.rename({"x": "rows"})

        assert result.dims == ["rows", "y"]
        assert "rows" in result.coords
        assert "x" not in result.coords

    def test_rename_multiple_dims(self):
        """Rename multiple dimensions."""
        arr = Array(np.arange(6).reshape(2, 3), {"x": ["a", "b"], "y": [0, 1, 2]})

        result = arr.rename({"x": "rows", "y": "cols"})

        assert result.dims == ["rows", "cols"]

    def test_rename_preserves_coords(self):
        """Rename preserves coordinate values."""
        arr = Array(np.arange(6).reshape(2, 3), {"x": ["a", "b"], "y": [0, 1, 2]})

        result = arr.rename({"x": "rows"})

        assert_array_equal(result.coords["rows"], ["a", "b"])
        assert_array_equal(result.coords["y"], [0, 1, 2])

    def test_rename_preserves_data(self):
        """Rename preserves data."""
        arr = Array(np.arange(6).reshape(2, 3), {"x": ["a", "b"], "y": [0, 1, 2]})

        result = arr.rename({"x": "rows"})

        assert_array_equal(result.data, arr.data)


class TestCopy:
    """Tests for copy()."""

    def test_copy_creates_new_data(self):
        """Copy creates independent data array."""
        arr = Array(np.arange(6).reshape(2, 3), {"x": ["a", "b"], "y": [0, 1, 2]})

        copied = arr.copy()
        copied.data[0, 0] = 999

        assert arr.data[0, 0] == 0  # Original unchanged

    def test_copy_creates_new_coords(self):
        """Copy creates independent coords."""
        arr = Array(np.arange(3), {"x": np.array([1, 2, 3])})

        copied = arr.copy()
        copied.coords["x"][0] = 999

        assert arr.coords["x"][0] == 1  # Original unchanged

    def test_copy_preserves_name(self):
        """Copy preserves name."""
        arr = Array(np.arange(3), {"x": ["a", "b", "c"]}, name="test")

        assert arr.copy().name == "test"


class TestAstype:
    """Tests for astype()."""

    def test_astype_int_to_float(self):
        """Convert int to float."""
        arr = Array(np.array([1, 2, 3]), {"x": ["a", "b", "c"]})

        result = arr.astype(float)

        assert result.data.dtype == float
        assert_array_equal(result.data, [1.0, 2.0, 3.0])

    def test_astype_float_to_int(self):
        """Convert float to int."""
        arr = Array(np.array([1.5, 2.7, 3.2]), {"x": ["a", "b", "c"]})

        result = arr.astype(int)

        assert result.data.dtype == int
        assert_array_equal(result.data, [1, 2, 3])

    def test_astype_preserves_coords(self):
        """astype preserves coordinates."""
        arr = Array(np.array([1, 2, 3]), {"x": ["a", "b", "c"]})

        result = arr.astype(float)

        assert_array_equal(result.coords["x"], ["a", "b", "c"])

    def test_astype_preserves_name(self):
        """astype preserves name."""
        arr = Array(np.array([1, 2, 3]), {"x": ["a", "b", "c"]}, name="test")

        assert arr.astype(float).name == "test"


class TestValues:
    """Tests for values property."""

    def test_values_returns_data(self):
        """values returns the same as data."""
        data = np.array([1, 2, 3])
        arr = Array(data, {"x": ["a", "b", "c"]})

        assert_array_equal(arr.values, arr.data)

    def test_values_is_same_object(self):
        """values is the same object as data."""
        arr = Array(np.array([[1, 2], [3, 4]]), {"x": ["a", "b"], "y": [0, 1]})

        assert arr.values is arr.data
