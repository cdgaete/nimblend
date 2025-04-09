"""
Pandas and Polars integration for NimbleNd arrays.
"""

from typing import Any, List, Optional

import numpy as np

from ..core import Array

# Check for pandas availability
try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# Check for polars availability
try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


def from_series(
    series: Any, dims: Optional[List[str]] = None, value_name: str = "value"
) -> Array:
    """
    Create a NimbleNd Array from a pandas Series or polars Series with a MultiIndex.

    The Series is assumed to be in long format where the index contains dimension values
    and the Series values represent the data at those coordinates.

    Parameters
    ----------
    series : Union[pd.Series, pl.Series]
        The input Series with a MultiIndex that represents dimension coordinates
    dims : Optional[List[str]], optional
        Names of dimensions to use. If not provided, will be extracted from the MultiIndex names
        for pandas or automatically generated for polars
    value_name : str, default: "value"
        Name to assign to the resulting array

    Returns
    -------
    Array
        A NimbleNd Array constructed from the Series data

    Notes
    -----
    For a pandas Series, the MultiIndex names will be used as dimension names if dims is not provided.
    For a polars Series, you should provide dimension names explicitly as polars doesn't have named indices.
    """
    if not (HAS_PANDAS or HAS_POLARS):
        raise ImportError("Either pandas or polars is required for this functionality")

    # Handle pandas Series
    if HAS_PANDAS and isinstance(series, pd.Series):
        return _from_pandas_series(series, dims, value_name)

    # Handle polars Series
    elif HAS_POLARS and isinstance(series, pl.Series):
        return _from_polars_series(series, dims, value_name)

    else:
        raise TypeError(
            f"Expected a pandas or polars Series, got {type(series).__name__}"
        )


def _from_pandas_series(
    series: "pd.Series", dims: Optional[List[str]], value_name: str
) -> Array:
    """Convert a pandas Series to a NimbleNd Array."""
    # Check for MultiIndex
    if not isinstance(series.index, pd.MultiIndex):
        # Handle single-index case by converting to a MultiIndex
        series = series.copy()
        series.index = pd.MultiIndex.from_arrays(
            [series.index], names=[series.index.name or "dim_0"]
        )

    # Extract dimension names from the MultiIndex if not provided
    if dims is None:
        dims = list(series.index.names)
        # Replace None names with default names
        dims = [f"dim_{i}" if name is None else name for i, name in enumerate(dims)]

    # Get unique values for each dimension
    unique_coords = {}
    for i, dim_name in enumerate(dims):
        # Get unique level values in original order if possible
        if hasattr(series.index, "get_level_values"):
            level_values = series.index.get_level_values(i)
            unique_values = pd.unique(level_values)
        else:
            # Fallback for older pandas versions
            unique_values = pd.unique([idx[i] for idx in series.index])

        unique_coords[dim_name] = np.array(unique_values)

    # Determine the appropriate dtype for the array
    # If we have NaN values or the dtype is object, use float64
    # Otherwise, preserve the original dtype
    has_na = series.isna().any()
    if has_na or np.issubdtype(series.dtype, np.inexact) or series.dtype == object:
        # Use float64 for any series with NA values or object/float dtypes
        dtype = np.float64
    else:
        # Use the original dtype for integer, boolean, etc.
        dtype = series.dtype

    # Create shape of the array
    shape = tuple(len(vals) for vals in unique_coords.values())

    # Create empty array (not filled with NaN to avoid warnings)
    data = np.empty(shape, dtype=dtype)

    # If using float dtype, fill with NaN
    if np.issubdtype(dtype, np.inexact):
        data.fill(np.nan)

    # Create mapping from coordinate values to positions
    position_maps = {
        dim: {val: idx for idx, val in enumerate(unique_coords[dim])} for dim in dims
    }

    # Fill the array with values from the Series
    for idx, value in series.items():
        # Skip NaN values for integer arrays
        if pd.isna(value) and not np.issubdtype(dtype, np.inexact):
            continue

        # Convert index tuple to position tuple
        if not isinstance(idx, tuple):
            idx = (idx,)

        positions = tuple(position_maps[dim][idx[i]] for i, dim in enumerate(dims))
        data[positions] = value

    return Array(data, unique_coords, dims, value_name)


def _from_polars_series(
    series: "pl.Series", dims: Optional[List[str]], value_name: str
) -> Array:
    """Convert a polars Series to a NimbleNd Array."""
    # For polars, we assume the series is one column of a DataFrame
    # with dimension values in other columns. We need to get the parent DataFrame.
    if not hasattr(series, "_df"):
        raise ValueError(
            "The polars Series must be a column of a DataFrame to extract dimensional coordinates"
        )

    df = series._df
    series_name = series.name

    # If dimensions not specified, use all columns except the value column
    if dims is None:
        dims = [col for col in df.columns if col != series_name]

    # Get unique values for each dimension
    unique_coords = {}
    for dim_name in dims:
        if dim_name not in df.columns:
            raise ValueError(
                f"Dimension '{dim_name}' not found in the DataFrame columns"
            )

        # Get unique values for this dimension
        unique_values = df[dim_name].unique().to_numpy()
        unique_coords[dim_name] = unique_values

    # Determine if we have NA values and choose appropriate dtype
    if df[series_name].null_count() > 0 or df[series_name].dtype == pl.Float64:
        dtype = np.float64
    else:
        # Convert polars dtype to numpy dtype
        dtype_str = str(df[series_name].dtype).lower()
        if "int" in dtype_str:
            dtype = np.int64
        elif "bool" in dtype_str:
            dtype = np.bool_
        else:
            dtype = np.float64

    # Create shape of the array
    shape = tuple(len(vals) for vals in unique_coords.values())

    # Create empty array
    data = np.empty(shape, dtype=dtype)

    # If using float dtype, fill with NaN
    if np.issubdtype(dtype, np.inexact):
        data.fill(np.nan)

    # Create mapping from coordinate values to positions
    position_maps = {
        dim: {val: idx for idx, val in enumerate(unique_coords[dim])} for dim in dims
    }

    # Convert to pandas DataFrame for easier processing
    # (polars doesn't have a simple way to iterate through rows with named columns)
    if HAS_PANDAS:
        pdf = df.to_pandas()
        for _, row in pdf.iterrows():
            # Skip NaN values for integer arrays
            if pd.isna(row[series_name]) and not np.issubdtype(dtype, np.inexact):
                continue

            # Get positions for each dimension
            positions = tuple(position_maps[dim][row[dim]] for dim in dims)
            data[positions] = row[series_name]
    else:
        # Fallback for when pandas is not available
        for row in df.iter_rows(named=True):
            # Skip NaN values for integer arrays
            if row[series_name] is None and not np.issubdtype(dtype, np.inexact):
                continue

            # Get positions for each dimension
            positions = tuple(position_maps[dim][row[dim]] for dim in dims)
            data[positions] = row[series_name]

    return Array(data, unique_coords, dims, value_name)


def to_series(
    array: Array,
    format: str = "pandas",
    dropna: bool = False,
    name: Optional[str] = None,
) -> Any:
    """
    Convert a NimbleNd Array to a pandas or polars Series with a MultiIndex.

    The output Series will be in long format with a MultiIndex representing the
    coordinate values for each dimension.

    Parameters
    ----------
    array : Array
        The input NimbleNd Array to convert
    format : str, default: 'pandas'
        Output format, either 'pandas' or 'polars'
    dropna : bool, default: False
        Whether to exclude NaN values from the result
    name : Optional[str], optional
        Name for the resulting Series. If None, uses the array's name

    Returns
    -------
    Union[pd.Series, pl.Series]
        A Series representation of the Array

    Raises
    ------
    ImportError
        If the requested output format library is not available
    """
    if format.lower() == "pandas":
        if not HAS_PANDAS:
            raise ImportError(
                "pandas is required for this functionality. Please install it with pip install pandas"
            )
        return _to_pandas_series(array, dropna, name)

    elif format.lower() == "polars":
        if not HAS_POLARS:
            raise ImportError(
                "polars is required for this functionality. Please install it with pip install polars"
            )
        return _to_polars_series(array, dropna, name)

    else:
        raise ValueError(f"Unknown format: {format}. Expected 'pandas' or 'polars'")


def _to_pandas_series(array: Array, dropna: bool, name: Optional[str]) -> "pd.Series":
    """Convert a NimbleNd Array to a pandas Series."""
    if array.is_lazy:
        # Compute lazy arrays first
        array = array.compute()

    # Generate all combinations of coordinate indices
    indices = np.meshgrid(
        *[np.arange(len(array.coords[dim])) for dim in array.dims], indexing="ij"
    )
    indices = [idx.flatten() for idx in indices]

    # Convert indices to actual coordinate values
    coord_values = []
    for dim_idx, dim in enumerate(array.dims):
        dim_coords = array.coords[dim]
        coord_values.append(dim_coords[indices[dim_idx]])

    # Create MultiIndex
    multi_idx = pd.MultiIndex.from_arrays(coord_values, names=array.dims)

    # Get flattened data values
    values = array.data.flatten()

    # Create Series
    series_name = name if name is not None else array.name
    series = pd.Series(values, index=multi_idx, name=series_name)

    # Drop NaN values if requested
    if dropna:
        series = series.dropna()

    return series


def _to_polars_series(array: Array, dropna: bool, name: Optional[str]) -> "pl.Series":
    """Convert a NimbleNd Array to a polars Series via a DataFrame."""
    if array.is_lazy:
        # Compute lazy arrays first
        array = array.compute()

    # Generate all combinations of coordinate indices
    indices = np.meshgrid(
        *[np.arange(len(array.coords[dim])) for dim in array.dims], indexing="ij"
    )
    indices = [idx.flatten() for idx in indices]

    # Create a dictionary for the DataFrame
    data_dict = {}

    # Add coordinate columns
    for dim_idx, dim in enumerate(array.dims):
        dim_coords = array.coords[dim]
        data_dict[dim] = dim_coords[indices[dim_idx]]

    # Add value column
    series_name = name if name is not None else (array.name or "value")
    data_dict[series_name] = array.data.flatten()

    # Create DataFrame
    df = pl.DataFrame(data_dict)

    # Drop NaN values if requested
    if dropna:
        df = df.drop_nulls(series_name)

    # Return the series column
    return df[series_name]
