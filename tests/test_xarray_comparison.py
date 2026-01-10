"""
Systematic tests comparing nimblend operations against xarray.

nimblend uses outer-join with zero-fill by default, so we configure xarray
with join='outer' and fill_value=0 to get equivalent behavior.
"""
import numpy as np
import xarray as xr
from numpy.testing import assert_array_equal

from nimblend import Array


def align_xarray_outer(xr1, xr2, fill_value=0):
    """Align two xarray DataArrays with outer join and fill_value."""
    aligned = xr.align(xr1, xr2, join="outer", fill_value=fill_value)
    return aligned[0], aligned[1]


def compare_with_reordering(nb_result, xr_result):
    """
    Compare nimblend and xarray results accounting for coordinate reordering.

    nimblend uses np.union1d which sorts coordinates, while xarray preserves
    original order. This function reindexes xarray result to match nimblend's
    coordinate order before comparing values.
    """
    # Reindex xarray result to match nimblend's coordinate order
    reindex_coords = {dim: nb_result.coords[dim] for dim in nb_result.dims}
    xr_reindexed = xr_result.reindex(**reindex_coords)
    assert_array_equal(nb_result.data, xr_reindexed.values)


class TestBinaryOperationsAligned:
    """Binary operations with fully aligned dimensions and coordinates."""

    def test_add_aligned_2d(self):
        """Addition with identical dimensions and coordinates."""
        data1 = np.array([[1, 2], [3, 4]])
        data2 = np.array([[10, 20], [30, 40]])
        coords = {"x": ["a", "b"], "y": [0, 1]}

        xr1 = xr.DataArray(data1, dims=["x", "y"], coords=coords)
        xr2 = xr.DataArray(data2, dims=["x", "y"], coords=coords)
        nb1 = Array(data1, coords)
        nb2 = Array(data2, coords)

        expected = (xr1 + xr2).values
        result = (nb1 + nb2).data

        assert_array_equal(result, expected)

    def test_sub_aligned_2d(self):
        """Subtraction with identical dimensions and coordinates."""
        data1 = np.array([[10, 20], [30, 40]])
        data2 = np.array([[1, 2], [3, 4]])
        coords = {"x": ["a", "b"], "y": [0, 1]}

        xr1 = xr.DataArray(data1, dims=["x", "y"], coords=coords)
        xr2 = xr.DataArray(data2, dims=["x", "y"], coords=coords)
        nb1 = Array(data1, coords)
        nb2 = Array(data2, coords)

        expected = (xr1 - xr2).values
        result = (nb1 - nb2).data

        assert_array_equal(result, expected)

    def test_mul_aligned_2d(self):
        """Multiplication with identical dimensions and coordinates."""
        data1 = np.array([[1, 2], [3, 4]])
        data2 = np.array([[10, 20], [30, 40]])
        coords = {"x": ["a", "b"], "y": [0, 1]}

        xr1 = xr.DataArray(data1, dims=["x", "y"], coords=coords)
        xr2 = xr.DataArray(data2, dims=["x", "y"], coords=coords)
        nb1 = Array(data1, coords)
        nb2 = Array(data2, coords)

        expected = (xr1 * xr2).values
        result = (nb1 * nb2).data

        assert_array_equal(result, expected)

    def test_div_aligned_2d(self):
        """Division with identical dimensions and coordinates."""
        data1 = np.array([[10.0, 20.0], [30.0, 40.0]])
        data2 = np.array([[2.0, 4.0], [5.0, 8.0]])
        coords = {"x": ["a", "b"], "y": [0, 1]}

        xr1 = xr.DataArray(data1, dims=["x", "y"], coords=coords)
        xr2 = xr.DataArray(data2, dims=["x", "y"], coords=coords)
        nb1 = Array(data1, coords)
        nb2 = Array(data2, coords)

        expected = (xr1 / xr2).values
        result = (nb1 / nb2).data

        assert_array_equal(result, expected)

    def test_aligned_3d(self):
        """Operations with 3D arrays."""
        data1 = np.arange(24).reshape(2, 3, 4)
        data2 = np.arange(24, 48).reshape(2, 3, 4)
        coords = {"x": ["a", "b"], "y": [0, 1, 2], "z": ["p", "q", "r", "s"]}

        xr1 = xr.DataArray(data1, dims=["x", "y", "z"], coords=coords)
        xr2 = xr.DataArray(data2, dims=["x", "y", "z"], coords=coords)
        nb1 = Array(data1, coords)
        nb2 = Array(data2, coords)

        assert_array_equal((nb1 + nb2).data, (xr1 + xr2).values)
        assert_array_equal((nb1 - nb2).data, (xr1 - xr2).values)
        assert_array_equal((nb1 * nb2).data, (xr1 * xr2).values)


class TestBinaryOperationsMisaligned:
    """Binary operations with different dimension orders."""

    def test_different_dim_order_2d(self):
        """Arrays with same dimensions but different order."""
        data1 = np.arange(6).reshape(2, 3)
        data2 = np.arange(6).reshape(3, 2)

        xr1 = xr.DataArray(
            data1,
            dims=["region", "year"],
            coords={"region": ["DE", "FR"], "year": [2020, 2021, 2022]},
        )
        xr2 = xr.DataArray(
            data2,
            dims=["year", "region"],
            coords={"year": [2020, 2021, 2022], "region": ["DE", "FR"]},
        )

        nb1 = Array(data1, {"region": ["DE", "FR"], "year": [2020, 2021, 2022]})
        nb2 = Array(data2, {"year": [2020, 2021, 2022], "region": ["DE", "FR"]})

        xr_result = xr1 + xr2
        nb_result = nb1 + nb2

        assert nb_result.dims == list(xr_result.dims)
        assert_array_equal(nb_result.data, xr_result.values)

    def test_different_dim_order_3d(self):
        """3D arrays with different dimension orders."""
        shape1 = (2, 3, 4)
        shape2 = (4, 2, 3)
        data1 = np.arange(np.prod(shape1)).reshape(shape1)
        data2 = np.arange(np.prod(shape2)).reshape(shape2)

        xr1 = xr.DataArray(
            data1,
            dims=["a", "b", "c"],
            coords={"a": [1, 2], "b": [10, 20, 30], "c": ["x", "y", "z", "w"]},
        )
        xr2 = xr.DataArray(
            data2,
            dims=["c", "a", "b"],
            coords={"c": ["x", "y", "z", "w"], "a": [1, 2], "b": [10, 20, 30]},
        )

        nb1 = Array(data1, {"a": [1, 2], "b": [10, 20, 30], "c": ["x", "y", "z", "w"]})
        nb2 = Array(data2, {"c": ["x", "y", "z", "w"], "a": [1, 2], "b": [10, 20, 30]})

        xr_result = xr1 + xr2
        nb_result = nb1 + nb2

        assert nb_result.dims == list(xr_result.dims)
        compare_with_reordering(nb_result, xr_result)


class TestPartiallyOverlappingCoordinates:
    """Operations with partially overlapping coordinate values."""

    def test_partial_overlap_one_dim(self):
        """Arrays sharing some but not all coordinate values in one dim."""
        data1 = np.array([[1, 2], [3, 4]])
        data2 = np.array([[10, 20], [30, 40]])

        xr1 = xr.DataArray(
            data1, dims=["x", "y"], coords={"x": ["a", "b"], "y": [0, 1]}
        )
        xr2 = xr.DataArray(
            data2, dims=["x", "y"], coords={"x": ["b", "c"], "y": [0, 1]}
        )

        nb1 = Array(data1, {"x": ["a", "b"], "y": [0, 1]})
        nb2 = Array(data2, {"x": ["b", "c"], "y": [0, 1]})

        xr1_aligned, xr2_aligned = align_xarray_outer(xr1, xr2)
        xr_result = xr1_aligned + xr2_aligned
        nb_result = nb1 + nb2

        assert set(nb_result.coords["x"]) == set(xr_result.coords["x"].values)
        assert set(nb_result.coords["y"]) == set(xr_result.coords["y"].values)
        assert_array_equal(nb_result.data, xr_result.values)

    def test_partial_overlap_columns_only(self):
        """Arrays sharing some but not all coordinate values in second dim."""
        # This tests the column-misaligned fast path in _fast_aligned_binop_2d
        data1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        data2 = np.array([[10.0, 20.0], [30.0, 40.0]])

        xr1 = xr.DataArray(
            data1, dims=["x", "y"], coords={"x": ["a", "b"], "y": [0, 1]}
        )
        xr2 = xr.DataArray(
            data2, dims=["x", "y"], coords={"x": ["a", "b"], "y": [1, 2]}
        )

        nb1 = Array(data1, {"x": ["a", "b"], "y": [0, 1]})
        nb2 = Array(data2, {"x": ["a", "b"], "y": [1, 2]})

        xr1_aligned, xr2_aligned = align_xarray_outer(xr1, xr2)
        xr_result = xr1_aligned + xr2_aligned
        nb_result = nb1 + nb2

        assert set(nb_result.coords["x"]) == set(xr_result.coords["x"].values)
        assert set(nb_result.coords["y"]) == set(xr_result.coords["y"].values)
        assert_array_equal(nb_result.data, xr_result.values)

    def test_partial_overlap_both_dims(self):
        """Arrays sharing some coordinates in both dimensions."""
        data1 = np.array([[1, 2], [3, 4]])
        data2 = np.array([[10, 20], [30, 40]])

        xr1 = xr.DataArray(
            data1, dims=["x", "y"], coords={"x": ["a", "b"], "y": [0, 1]}
        )
        xr2 = xr.DataArray(
            data2, dims=["x", "y"], coords={"x": ["b", "c"], "y": [1, 2]}
        )

        nb1 = Array(data1, {"x": ["a", "b"], "y": [0, 1]})
        nb2 = Array(data2, {"x": ["b", "c"], "y": [1, 2]})

        xr1_aligned, xr2_aligned = align_xarray_outer(xr1, xr2)
        xr_result = xr1_aligned + xr2_aligned
        nb_result = nb1 + nb2

        assert set(nb_result.coords["x"]) == set(xr_result.coords["x"].values)
        assert set(nb_result.coords["y"]) == set(xr_result.coords["y"].values)
        assert_array_equal(nb_result.data, xr_result.values)

    def test_disjoint_coordinates(self):
        """Arrays with no overlapping coordinates."""
        data1 = np.array([[1, 2], [3, 4]])
        data2 = np.array([[10, 20], [30, 40]])

        xr1 = xr.DataArray(
            data1, dims=["x", "y"], coords={"x": ["a", "b"], "y": [0, 1]}
        )
        xr2 = xr.DataArray(
            data2, dims=["x", "y"], coords={"x": ["c", "d"], "y": [2, 3]}
        )

        nb1 = Array(data1, {"x": ["a", "b"], "y": [0, 1]})
        nb2 = Array(data2, {"x": ["c", "d"], "y": [2, 3]})

        xr1_aligned, xr2_aligned = align_xarray_outer(xr1, xr2)
        xr_result = xr1_aligned + xr2_aligned
        nb_result = nb1 + nb2

        # With disjoint coords, addition just preserves values (other is zero)
        assert set(nb_result.coords["x"]) == set(xr_result.coords["x"].values)
        assert set(nb_result.coords["y"]) == set(xr_result.coords["y"].values)
        assert_array_equal(nb_result.data, xr_result.values)

    def test_disjoint_multiplication(self):
        """Multiplication with disjoint coordinates yields zeros at non-overlap."""
        data1 = np.array([[1, 2], [3, 4]])
        data2 = np.array([[10, 20], [30, 40]])

        xr1 = xr.DataArray(
            data1, dims=["x", "y"], coords={"x": ["a", "b"], "y": [0, 1]}
        )
        xr2 = xr.DataArray(
            data2, dims=["x", "y"], coords={"x": ["c", "d"], "y": [2, 3]}
        )

        nb1 = Array(data1, {"x": ["a", "b"], "y": [0, 1]})
        nb2 = Array(data2, {"x": ["c", "d"], "y": [2, 3]})

        xr1_aligned, xr2_aligned = align_xarray_outer(xr1, xr2)
        xr_result = xr1_aligned * xr2_aligned
        nb_result = nb1 * nb2

        # With disjoint coords, multiplication gives all zeros
        assert_array_equal(nb_result.data, xr_result.values)
        assert np.all(nb_result.data == 0)

    def test_partial_overlap_different_dim_order(self):
        """Partial overlap combined with different dimension order."""
        data1 = np.arange(6).reshape(2, 3)
        data2 = np.arange(6, 12).reshape(3, 2)

        xr1 = xr.DataArray(
            data1,
            dims=["region", "year"],
            coords={"region": ["DE", "FR"], "year": [2020, 2021, 2022]},
        )
        xr2 = xr.DataArray(
            data2,
            dims=["year", "region"],
            coords={"year": [2021, 2022, 2023], "region": ["FR", "ES"]},
        )

        nb1 = Array(data1, {"region": ["DE", "FR"], "year": [2020, 2021, 2022]})
        nb2 = Array(data2, {"year": [2021, 2022, 2023], "region": ["FR", "ES"]})

        xr1_aligned, xr2_aligned = align_xarray_outer(xr1, xr2)
        xr_result = xr1_aligned + xr2_aligned
        nb_result = nb1 + nb2

        assert set(nb_result.coords["region"]) == set(xr_result.coords["region"].values)
        assert set(nb_result.coords["year"]) == set(xr_result.coords["year"].values)
        compare_with_reordering(nb_result, xr_result)


class TestReductionOperations:
    """Reduction operations (sum) comparison."""

    def test_sum_single_dim(self):
        """Sum over a single dimension."""
        data = np.arange(12).reshape(2, 3, 2)
        coords = {"x": ["a", "b"], "y": [0, 1, 2], "z": ["p", "q"]}

        xr_arr = xr.DataArray(data, dims=["x", "y", "z"], coords=coords)
        nb_arr = Array(data, coords)

        assert_array_equal(nb_arr.sum(dim="x").data, xr_arr.sum(dim="x").values)
        assert_array_equal(nb_arr.sum(dim="y").data, xr_arr.sum(dim="y").values)
        assert_array_equal(nb_arr.sum(dim="z").data, xr_arr.sum(dim="z").values)

    def test_sum_multiple_dims(self):
        """Sum over multiple dimensions."""
        data = np.arange(24).reshape(2, 3, 4)
        coords = {"x": ["a", "b"], "y": [0, 1, 2], "z": ["p", "q", "r", "s"]}

        xr_arr = xr.DataArray(data, dims=["x", "y", "z"], coords=coords)
        nb_arr = Array(data, coords)

        xr_result = xr_arr.sum(dim=["x", "z"])
        nb_result = nb_arr.sum(dim=["x", "z"])

        assert_array_equal(nb_result.data, xr_result.values)
        assert nb_result.dims == list(xr_result.dims)

    def test_sum_all_dims(self):
        """Sum over all dimensions (total sum)."""
        data = np.arange(24).reshape(2, 3, 4)
        coords = {"x": ["a", "b"], "y": [0, 1, 2], "z": ["p", "q", "r", "s"]}

        xr_arr = xr.DataArray(data, dims=["x", "y", "z"], coords=coords)
        nb_arr = Array(data, coords)

        assert nb_arr.sum() == float(xr_arr.sum().values)

    def test_sum_preserves_remaining_coords(self):
        """Sum preserves coordinates of remaining dimensions."""
        data = np.arange(6).reshape(2, 3)
        coords = {"x": ["a", "b"], "y": [10, 20, 30]}

        xr_arr = xr.DataArray(data, dims=["x", "y"], coords=coords)
        nb_arr = Array(data, coords)

        xr_result = xr_arr.sum(dim="x")
        nb_result = nb_arr.sum(dim="x")

        assert_array_equal(nb_result.coords["y"], xr_result.coords["y"].values)
        assert_array_equal(nb_result.data, xr_result.values)


class TestScalarOperations:
    """Operations with scalar values."""

    def test_add_scalar(self):
        """Add scalar to array."""
        data = np.array([[1, 2], [3, 4]])
        coords = {"x": ["a", "b"], "y": [0, 1]}

        xr_arr = xr.DataArray(data, dims=["x", "y"], coords=coords)
        nb_arr = Array(data, coords)

        assert_array_equal((nb_arr + 10).data, (xr_arr + 10).values)
        assert_array_equal((10 + nb_arr).data, (10 + xr_arr).values)

    def test_sub_scalar(self):
        """Subtract scalar from array and vice versa."""
        data = np.array([[10, 20], [30, 40]])
        coords = {"x": ["a", "b"], "y": [0, 1]}

        xr_arr = xr.DataArray(data, dims=["x", "y"], coords=coords)
        nb_arr = Array(data, coords)

        assert_array_equal((nb_arr - 5).data, (xr_arr - 5).values)
        assert_array_equal((100 - nb_arr).data, (100 - xr_arr).values)

    def test_mul_scalar(self):
        """Multiply array by scalar."""
        data = np.array([[1, 2], [3, 4]])
        coords = {"x": ["a", "b"], "y": [0, 1]}

        xr_arr = xr.DataArray(data, dims=["x", "y"], coords=coords)
        nb_arr = Array(data, coords)

        assert_array_equal((nb_arr * 3).data, (xr_arr * 3).values)
        assert_array_equal((3 * nb_arr).data, (3 * xr_arr).values)

    def test_div_scalar(self):
        """Divide array by scalar and vice versa."""
        data = np.array([[10.0, 20.0], [30.0, 40.0]])
        coords = {"x": ["a", "b"], "y": [0, 1]}

        xr_arr = xr.DataArray(data, dims=["x", "y"], coords=coords)
        nb_arr = Array(data, coords)

        assert_array_equal((nb_arr / 2).data, (xr_arr / 2).values)
        assert_array_equal((100 / nb_arr).data, (100 / xr_arr).values)

    def test_power_scalar(self):
        """Raise array to scalar power."""
        data = np.array([[1, 2], [3, 4]])
        coords = {"x": ["a", "b"], "y": [0, 1]}

        xr_arr = xr.DataArray(data, dims=["x", "y"], coords=coords)
        nb_arr = Array(data, coords)

        assert_array_equal((nb_arr**2).data, (xr_arr**2).values)


class TestChainedOperations:
    """Multiple operations chained together."""

    def test_chained_arithmetic(self):
        """Chain multiple arithmetic operations."""
        data1 = np.array([[1, 2], [3, 4]])
        data2 = np.array([[10, 20], [30, 40]])
        coords = {"x": ["a", "b"], "y": [0, 1]}

        xr1 = xr.DataArray(data1, dims=["x", "y"], coords=coords)
        xr2 = xr.DataArray(data2, dims=["x", "y"], coords=coords)
        nb1 = Array(data1, coords)
        nb2 = Array(data2, coords)

        xr_result = (xr1 + xr2) * 2 - 5
        nb_result = (nb1 + nb2) * 2 - 5

        assert_array_equal(nb_result.data, xr_result.values)

    def test_operation_then_reduction(self):
        """Perform operation then reduce."""
        data1 = np.arange(6).reshape(2, 3)
        data2 = np.arange(6, 12).reshape(2, 3)
        coords = {"x": ["a", "b"], "y": [0, 1, 2]}

        xr1 = xr.DataArray(data1, dims=["x", "y"], coords=coords)
        xr2 = xr.DataArray(data2, dims=["x", "y"], coords=coords)
        nb1 = Array(data1, coords)
        nb2 = Array(data2, coords)

        xr_result = (xr1 + xr2).sum(dim="y")
        nb_result = (nb1 + nb2).sum(dim="y")

        assert_array_equal(nb_result.data, xr_result.values)

    def test_operation_with_partial_overlap_then_reduction(self):
        """Operation with partial overlap followed by reduction."""
        data1 = np.arange(6).reshape(2, 3)
        data2 = np.arange(6).reshape(2, 3)

        xr1 = xr.DataArray(
            data1,
            dims=["x", "y"],
            coords={"x": ["a", "b"], "y": [0, 1, 2]},
        )
        xr2 = xr.DataArray(
            data2,
            dims=["x", "y"],
            coords={"x": ["b", "c"], "y": [1, 2, 3]},
        )

        nb1 = Array(data1, {"x": ["a", "b"], "y": [0, 1, 2]})
        nb2 = Array(data2, {"x": ["b", "c"], "y": [1, 2, 3]})

        xr1_aligned, xr2_aligned = align_xarray_outer(xr1, xr2)
        xr_result = (xr1_aligned + xr2_aligned).sum(dim="x")
        nb_result = (nb1 + nb2).sum(dim="x")

        assert set(nb_result.coords["y"]) == set(xr_result.coords["y"].values)
        assert_array_equal(nb_result.data, xr_result.values)


class TestEdgeCases:
    """Edge cases and special scenarios."""

    def test_single_element_arrays(self):
        """Operations on single-element arrays."""
        data1 = np.array([[5]])
        data2 = np.array([[3]])
        coords = {"x": ["a"], "y": [0]}

        xr1 = xr.DataArray(data1, dims=["x", "y"], coords=coords)
        xr2 = xr.DataArray(data2, dims=["x", "y"], coords=coords)
        nb1 = Array(data1, coords)
        nb2 = Array(data2, coords)

        assert_array_equal((nb1 + nb2).data, (xr1 + xr2).values)
        assert_array_equal((nb1 * nb2).data, (xr1 * xr2).values)

    def test_1d_arrays(self):
        """Operations on 1D arrays."""
        data1 = np.array([1, 2, 3])
        data2 = np.array([10, 20, 30])
        coords = {"x": ["a", "b", "c"]}

        xr1 = xr.DataArray(data1, dims=["x"], coords=coords)
        xr2 = xr.DataArray(data2, dims=["x"], coords=coords)
        nb1 = Array(data1, coords)
        nb2 = Array(data2, coords)

        assert_array_equal((nb1 + nb2).data, (xr1 + xr2).values)

    def test_1d_partial_overlap(self):
        """1D arrays with partial coordinate overlap."""
        data1 = np.array([1, 2, 3])
        data2 = np.array([10, 20, 30])

        xr1 = xr.DataArray(data1, dims=["x"], coords={"x": ["a", "b", "c"]})
        xr2 = xr.DataArray(data2, dims=["x"], coords={"x": ["b", "c", "d"]})
        nb1 = Array(data1, {"x": ["a", "b", "c"]})
        nb2 = Array(data2, {"x": ["b", "c", "d"]})

        xr1_aligned, xr2_aligned = align_xarray_outer(xr1, xr2)
        xr_result = xr1_aligned + xr2_aligned
        nb_result = nb1 + nb2

        assert set(nb_result.coords["x"]) == set(xr_result.coords["x"].values)
        assert_array_equal(nb_result.data, xr_result.values)

    def test_negation(self):
        """Unary negation."""
        data = np.array([[1, 2], [3, 4]])
        coords = {"x": ["a", "b"], "y": [0, 1]}

        xr_arr = xr.DataArray(data, dims=["x", "y"], coords=coords)
        nb_arr = Array(data, coords)

        assert_array_equal((-nb_arr).data, (-xr_arr).values)

    def test_float_data(self):
        """Operations with floating point data."""
        data1 = np.array([[1.5, 2.7], [3.1, 4.9]])
        data2 = np.array([[0.5, 0.3], [0.9, 0.1]])
        coords = {"x": ["a", "b"], "y": [0, 1]}

        xr1 = xr.DataArray(data1, dims=["x", "y"], coords=coords)
        xr2 = xr.DataArray(data2, dims=["x", "y"], coords=coords)
        nb1 = Array(data1, coords)
        nb2 = Array(data2, coords)

        np.testing.assert_allclose((nb1 + nb2).data, (xr1 + xr2).values)
        np.testing.assert_allclose((nb1 * nb2).data, (xr1 * xr2).values)

    def test_string_coordinates(self):
        """Arrays with string coordinates."""
        data1 = np.array([[1, 2], [3, 4]])
        data2 = np.array([[10, 20], [30, 40]])
        coords = {"region": ["Germany", "France"], "tech": ["solar", "wind"]}

        xr1 = xr.DataArray(data1, dims=["region", "tech"], coords=coords)
        xr2 = xr.DataArray(data2, dims=["region", "tech"], coords=coords)
        nb1 = Array(data1, coords)
        nb2 = Array(data2, coords)

        # Use compare_with_reordering because np.union1d sorts coords
        compare_with_reordering(nb1 + nb2, xr1 + xr2)

    def test_numeric_string_coords_mixed(self):
        """Arrays with mixed numeric and string coordinates."""
        data = np.arange(6).reshape(2, 3)

        xr_arr = xr.DataArray(
            data,
            dims=["region", "year"],
            coords={"region": ["DE", "FR"], "year": [2020, 2021, 2022]},
        )
        nb_arr = Array(data, {"region": ["DE", "FR"], "year": [2020, 2021, 2022]})

        xr_result = xr_arr.sum(dim="region")
        nb_result = nb_arr.sum(dim="region")

        assert_array_equal(nb_result.data, xr_result.values)
        assert_array_equal(nb_result.coords["year"], xr_result.coords["year"].values)
