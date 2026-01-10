"""Tests for reduction methods: sum, mean, min, max, std, prod."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from nimblend import Array


class TestSum:
    """Tests for sum() - already covered in xarray comparison, basic here."""

    def test_sum_all(self):
        arr = Array(np.array([[1, 2], [3, 4]]), {"x": ["a", "b"], "y": [0, 1]})
        assert arr.sum() == 10

    def test_sum_dim(self):
        arr = Array(np.array([[1, 2], [3, 4]]), {"x": ["a", "b"], "y": [0, 1]})
        result = arr.sum("x")
        assert_array_equal(result.data, [4, 6])


class TestMean:
    """Tests for mean()."""

    def test_mean_all(self):
        arr = Array(np.array([[1.0, 2.0], [3.0, 4.0]]), {"x": ["a", "b"], "y": [0, 1]})
        assert arr.mean() == 2.5

    def test_mean_dim(self):
        arr = Array(np.array([[1.0, 2.0], [3.0, 4.0]]), {"x": ["a", "b"], "y": [0, 1]})
        result = arr.mean("x")
        assert_array_equal(result.data, [2.0, 3.0])
        assert result.dims == ["y"]

    def test_mean_multiple_dims(self):
        arr = Array(np.arange(24.0).reshape(2, 3, 4),
                    {"x": ["a", "b"], "y": [0, 1, 2], "z": [0, 1, 2, 3]})
        result = arr.mean(["x", "z"])
        assert result.dims == ["y"]
        assert result.shape == (3,)


class TestMin:
    """Tests for min()."""

    def test_min_all(self):
        arr = Array(np.array([[5, 2], [3, 8]]), {"x": ["a", "b"], "y": [0, 1]})
        assert arr.min() == 2

    def test_min_dim(self):
        arr = Array(np.array([[5, 2], [3, 8]]), {"x": ["a", "b"], "y": [0, 1]})
        result = arr.min("x")
        assert_array_equal(result.data, [3, 2])


class TestMax:
    """Tests for max()."""

    def test_max_all(self):
        arr = Array(np.array([[5, 2], [3, 8]]), {"x": ["a", "b"], "y": [0, 1]})
        assert arr.max() == 8

    def test_max_dim(self):
        arr = Array(np.array([[5, 2], [3, 8]]), {"x": ["a", "b"], "y": [0, 1]})
        result = arr.max("y")
        assert_array_equal(result.data, [5, 8])


class TestStd:
    """Tests for std()."""

    def test_std_all(self):
        arr = Array(np.array([[1.0, 2.0], [3.0, 4.0]]), {"x": ["a", "b"], "y": [0, 1]})
        assert_allclose(arr.std(), np.std([1, 2, 3, 4]))

    def test_std_dim(self):
        arr = Array(np.array([[1.0, 2.0], [3.0, 4.0]]), {"x": ["a", "b"], "y": [0, 1]})
        result = arr.std("x")
        assert_allclose(result.data, [1.0, 1.0])


class TestProd:
    """Tests for prod()."""

    def test_prod_all(self):
        arr = Array(np.array([[1, 2], [3, 4]]), {"x": ["a", "b"], "y": [0, 1]})
        assert arr.prod() == 24

    def test_prod_dim(self):
        arr = Array(np.array([[1, 2], [3, 4]]), {"x": ["a", "b"], "y": [0, 1]})
        result = arr.prod("x")
        assert_array_equal(result.data, [3, 8])


class TestReductionEdgeCases:
    """Edge cases for all reductions."""

    def test_reduction_preserves_name(self):
        arr = Array(np.arange(6).reshape(2, 3),
                    {"x": ["a", "b"], "y": [0, 1, 2]}, name="test")
        result = arr.sum("x")
        assert result.name == "test"

    def test_reduction_1d_to_scalar(self):
        arr = Array(np.array([1, 2, 3]), {"x": ["a", "b", "c"]})
        assert arr.sum("x") == 6
        assert arr.mean("x") == 2.0
        assert arr.min("x") == 1
        assert arr.max("x") == 3

    def test_reduction_preserves_coords(self):
        arr = Array(np.arange(6).reshape(2, 3),
                    {"x": ["a", "b"], "y": [10, 20, 30]})
        result = arr.sum("x")
        assert_array_equal(result.coords["y"], [10, 20, 30])
