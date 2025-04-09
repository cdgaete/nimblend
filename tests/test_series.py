"""
Tests for the Series import/export functionality.
"""
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import nimblend as nd
from nimblend.core import Array
from nimblend.io.series import from_series, to_series

# Skip tests if pandas/polars not available
pandas_available = pytest.importorskip("pandas", reason="pandas not installed")
pd = pandas_available
try:
    import polars as pl
    polars_available = True
except ImportError:
    polars_available = False

class TestPandasSeries:
    """Test pandas Series import/export functionality."""

    def setup_method(self):
        """Set up test data."""
        # Create a simple MultiIndex series
        idx = pd.MultiIndex.from_tuples(
            [('A', 1), ('A', 2), ('B', 1), ('B', 3)],
            names=['dim1', 'dim2']
        )
        self.series = pd.Series([10.5, 20.0, 30.5, 40.0], index=idx, name='values')

        # Create a more complex MultiIndex series with missing values
        idx2 = pd.MultiIndex.from_tuples(
            [('A', 1, 'x'), ('A', 2, 'y'), ('B', 1, 'x'), ('B', 3, 'z')],
            names=['dim1', 'dim2', 'dim3']
        )
        self.series_with_nans = pd.Series([10.5, np.nan, 30.5, 40.0],
                                         index=idx2, name='complex_values')

        # Create a single-index series
        self.single_index_series = pd.Series([1, 2, 3],
                                            index=['a', 'b', 'c'],
                                            name='single')

        # Create a NimbleNd array for testing to_series
        coords = {
            'dim1': np.array(['A', 'B']),
            'dim2': np.array([1, 2, 3]),
        }
        data = np.array([
            [10.5, 20.0, np.nan],
            [30.5, np.nan, 40.0],
        ])
        self.array = Array(data, coords, ['dim1', 'dim2'], 'test_array')

    def test_from_series_basic(self):
        """Test basic conversion from pandas Series to Array."""
        array = from_series(self.series)

        # Check dimensions and coordinates
        assert array.dims == ['dim1', 'dim2']
        assert_array_equal(array.coords['dim1'], np.array(['A', 'B']))
        assert_array_equal(array.coords['dim2'], np.array([1, 2, 3]))

        # Check data values
        expected = np.array([
            [10.5, 20.0, np.nan],
            [30.5, np.nan, 40.0],
        ])
        # Use allclose with equal_nan=True to handle NaN values
        assert_allclose(array.data, expected, equal_nan=True)

        # Check name
        assert array.name == 'values'

    def test_from_series_with_nans(self):
        """Test conversion from pandas Series with NaN values."""
        array = from_series(self.series_with_nans)

        # Check dimensions and coordinates
        assert array.dims == ['dim1', 'dim2', 'dim3']
        assert_array_equal(array.coords['dim1'], np.array(['A', 'B']))
        assert_array_equal(array.coords['dim2'], np.array([1, 2, 3]))
        assert_array_equal(array.coords['dim3'], np.array(['x', 'y', 'z']))

        # Value at ('A', 2, 'y') should be NaN
        idx = (0, 1, 1)  # A, 2, y
        assert np.isnan(array.data[idx])

        # Check name
        assert array.name == 'complex_values'

    def test_from_series_with_custom_dims(self):
        """Test conversion with custom dimension names."""
        array = from_series(self.series, dims=['x', 'y'], value_name='custom')

        # Check dimensions and name
        assert array.dims == ['x', 'y']
        assert array.name == 'custom'

    def test_from_single_index_series(self):
        """Test conversion from a single-index Series."""
        array = from_series(self.single_index_series)

        # Check dimensions
        assert len(array.dims) == 1
        assert_array_equal(array.coords[array.dims[0]], np.array(['a', 'b', 'c']))

        # Check data
        assert_array_equal(array.data, np.array([1, 2, 3]))

    def test_to_series_basic(self):
        """Test basic conversion from Array to pandas Series."""
        series = to_series(self.array, format='pandas')

        # Check index names
        assert series.index.names == ['dim1', 'dim2']

        # Check that we have all 6 possible combinations of coordinates
        assert len(series) == 6

        # Check specific values
        assert series[('A', 1)] == 10.5
        assert series[('B', 3)] == 40.0
        assert np.isnan(series[('A', 3)])

    def test_to_series_dropna(self):
        """Test conversion to Series dropping NaN values."""
        series = to_series(self.array, format='pandas', dropna=True)

        # Check that NaN values are dropped
        assert len(series) == 4  # 6 combinations - 2 NaN values = 4

        # Check that NaN values are not present
        assert ('A', 3) not in series.index
        assert ('B', 2) not in series.index

    def test_to_series_custom_name(self):
        """Test conversion to Series with custom name."""
        series = to_series(self.array, format='pandas', name='custom_name')
        assert series.name == 'custom_name'

    def test_roundtrip(self):
        """Test round-trip conversion (Series -> Array -> Series)."""
        array = from_series(self.series)
        series_again = to_series(array, format='pandas')

        # Check that we have all values in the original series
        for idx, value in self.series.items():
            assert series_again[idx] == value

        # Check that extra NaN values are also included in the roundtrip
        assert len(series_again) >= len(self.series)

    def test_array_to_series_to_array(self):
        """Test round-trip conversion (Array -> Series -> Array)."""
        series = to_series(self.array, format='pandas')
        array_again = from_series(series)

        # Check dimensions and coordinates
        assert array_again.dims == self.array.dims
        for dim in self.array.dims:
            assert_array_equal(array_again.coords[dim], self.array.coords[dim])

        # Check data (allowing NaN equality)
        assert_allclose(array_again.data, self.array.data, equal_nan=True)


@pytest.mark.skipif(not polars_available, reason="polars not installed")
class TestPolarsSeries:
    """Test polars Series import/export functionality."""

    def setup_method(self):
        """Set up test data."""
        # Create a polars DataFrame and Series
        self.df = pl.DataFrame({
            'dim1': ['A', 'A', 'B', 'B'],
            'dim2': [1, 2, 1, 3],
            'values': [10.5, 20.0, 30.5, 40.0]
        })
        self.series = self.df['values']

        # DataFrame with NaN values
        self.df_with_nans = pl.DataFrame({
            'dim1': ['A', 'A', 'B', 'B'],
            'dim2': [1, 2, 1, 3],
            'dim3': ['x', 'y', 'x', 'z'],
            'values': [10.5, None, 30.5, 40.0]
        })
        self.series_with_nans = self.df_with_nans['values']

        # Create a NimbleNd array for testing to_series
        coords = {
            'dim1': np.array(['A', 'B']),
            'dim2': np.array([1, 2, 3]),
        }
        data = np.array([
            [10.5, 20.0, np.nan],
            [30.5, np.nan, 40.0],
        ])
        self.array = Array(data, coords, ['dim1', 'dim2'], 'test_array')

    def test_from_series_basic(self):
        """Test basic conversion from polars Series to Array."""
        # In polars, we need to explicitly specify dimensions because
        # polars Series don't have a notion of index names
        array = from_series(self.series, dims=['dim1', 'dim2'])

        # Check dimensions and coordinates
        assert array.dims == ['dim1', 'dim2']
        assert_array_equal(array.coords['dim1'], np.array(['A', 'B']))
        assert_array_equal(array.coords['dim2'], np.array([1, 2, 3]))

        # Check data values
        expected = np.array([
            [10.5, 20.0, np.nan],
            [30.5, np.nan, 40.0],
        ])
        # Use allclose with equal_nan=True to handle NaN values
        assert_allclose(array.data, expected, equal_nan=True)

    def test_from_series_with_nans(self):
        """Test conversion from polars Series with NaN values."""
        array = from_series(self.series_with_nans, dims=['dim1', 'dim2', 'dim3'])

        # Check dimensions and coordinates
        assert array.dims == ['dim1', 'dim2', 'dim3']
        assert_array_equal(array.coords['dim1'], np.array(['A', 'B']))
        assert_array_equal(array.coords['dim2'], np.array([1, 2, 3]))
        assert_array_equal(array.coords['dim3'], np.array(['x', 'y', 'z']))

        # Value at ('A', 2, 'y') should be NaN
        idx = (0, 1, 1)  # A, 2, y
        assert np.isnan(array.data[idx])

    def test_to_series_basic(self):
        """Test basic conversion from Array to polars Series."""
        series = to_series(self.array, format='polars')

        # Polars returns a Series not a DataFrame
        # But we can get the parent DataFrame
        df = series._df

        # Check column names
        assert set(df.columns) == {'dim1', 'dim2', 'test_array'}

        # Check that we have all 6 possible combinations
        assert len(df) == 6

        # Check specific values using filter
        a1_row = df.filter((pl.col('dim1') == 'A') & (pl.col('dim2') == 1))
        assert a1_row['test_array'][0] == 10.5

        b3_row = df.filter((pl.col('dim1') == 'B') & (pl.col('dim2') == 3))
        assert b3_row['test_array'][0] == 40.0

    def test_to_series_dropna(self):
        """Test conversion to Series dropping NaN values."""
        series = to_series(self.array, format='polars', dropna=True)
        df = series._df

        # Check that NaN values are dropped
        assert len(df) == 4  # 6 combinations - 2 NaN values = 4

        # Check that a3 is not in the result (it's NaN)
        a3_row = df.filter((pl.col('dim1') == 'A') & (pl.col('dim2') == 3))
        assert len(a3_row) == 0

    def test_roundtrip(self):
        """Test round-trip conversion (Series -> Array -> Series)."""
        # Need to specify dims for polars Series
        array = from_series(self.series, dims=['dim1', 'dim2'])
        series_again = to_series(array, format='polars')
        df_again = series_again._df

        # Check that we have right number of rows
        # (includes NaN combinations that weren't in original)
        assert len(df_again) >= len(self.df)

        # Check values in original dataframe match in result
        for row in self.df.iter_rows(named=True):
            matching = df_again.filter(
                (pl.col('dim1') == row['dim1']) &
                (pl.col('dim2') == row['dim2'])
            )
            assert matching[series_again.name][0] == row['values']


class TestEdgeCases:
    """Test edge cases for Series import/export."""

    def setup_method(self):
        """Set up test data."""
        # Only run if pandas is available
        if not 'pd' in globals():
            pytest.skip("pandas not installed")

    def test_empty_series(self):
        """Test handling of empty series."""
        # Create an empty Series with MultiIndex
        idx = pd.MultiIndex.from_tuples([], names=['dim1', 'dim2'])
        empty_series = pd.Series([], index=idx, name='empty')

        # Converting empty series should still create correct dimensions
        array = from_series(empty_series)
        assert array.dims == ['dim1', 'dim2']
        assert array.data.size == 0

    def test_duplicate_coords(self):
        """Test handling of duplicate coordinate values."""
        # Create a Series with duplicate index values
        idx = pd.MultiIndex.from_tuples(
            [('A', 1), ('A', 1), ('B', 2)],
            names=['dim1', 'dim2']
        )
        duplicate_series = pd.Series([10, 20, 30], index=idx, name='duplicates')

        # This should raise an error because coordinates must be unique
        with pytest.raises(ValueError):
            from_series(duplicate_series)

    def test_missing_dims(self):
        """Test error when dimensions not provided for polars Series."""
        if not polars_available:
            pytest.skip("polars not installed")

        # Create a polars DataFrame and Series
        df = pl.DataFrame({
            'dim1': ['A', 'B'],
            'values': [1.0, 2.0]
        })
        series = df['values']

        # Not providing dims should raise an error for polars Series
        with pytest.raises(ValueError):
            from_series(series)

    def test_complex_types(self):
        """Test handling of complex data types."""
        # Create a Series with string values and datetime indices
        idx = pd.MultiIndex.from_product([
            ['A', 'B'],
            pd.date_range('2023-01-01', periods=3)
        ], names=['category', 'date'])

        complex_series = pd.Series(
            ['value1', 'value2', 'value3', 'value4', 'value5', 'value6'],
            index=idx,
            name='strings'
        )

        # Test conversion
        array = from_series(complex_series)
        assert array.dims == ['category', 'date']
        assert array.data.dtype == complex_series.dtype

        # Test round trip
        series_again = to_series(array)
        assert series_again.equals(complex_series)
