# NimbleNd: Efficient Labeled N-Dimensional Arrays with Flexible Alignment

NimbleNd (`nimblend`) is a Python library for working with labeled N-dimensional arrays. It provides a flexible and intuitive interface for data manipulation with automatic coordinate alignment, lazy computation, and various storage backends.

## Features

- **Labeled Dimensions**: Access your data using meaningful dimension names and coordinate values
- **Automatic Coordinate Alignment**: Operations between arrays automatically align their coordinates
- **Lazy Computation**: Optional integration with Dask for out-of-core processing of large datasets
- **Multiple Storage Backends**:
  - Zarr: For efficient chunked, compressed, N-dimensional arrays
  - IceChunk: For versioned, distributed storage with transactional semantics
- **Data Format Interoperability**:
  - Convert to/from pandas and polars Series/DataFrames
  - Preserve coordinate information during conversion

## Installation

Basic installation with minimal dependencies:

```bash
pip install nimblend
```

With optional features:

```bash
# For lazy computation with Dask
pip install nimblend[dask]

# For Zarr storage
pip install nimblend[io]

# For IceChunk support
pip install nimblend[icechunk]

# For pandas integration
pip install nimblend[pandas]

# For polars integration
pip install nimblend[polars]

# Install all optional dependencies
pip install nimblend[all]
```

## Quick Start

### Creating and Manipulating Arrays

```python
import numpy as np
from nimblend import Array

# Create an array with labeled dimensions
data = np.array([[1, 2, 3], [4, 5, 6]])
coords = {"x": ["a", "b"], "y": [10, 20, 30]}
arr = Array(data, coords)

# Access data by coordinate values
subset = arr[{"x": "a", "y": [10, 20]}]
print(subset.data)  # array([[1, 2]])

# Mathematical operations
arr2 = arr * 2
print(arr2.data)  # array([[2, 4, 6], [8, 10, 12]])
```

### Automatic Coordinate Alignment

```python
# Create two arrays with different coordinates
data1 = np.array([[1, 2], [3, 4]])
coords1 = {"x": ["a", "b"], "y": [0, 1]}
arr1 = Array(data1, coords1)

data2 = np.array([[10, 20, 30], [40, 50, 60]])
coords2 = {"x": ["b", "c"], "y": [0, 1, 2]}
arr2 = Array(data2, coords2)

# Operation automatically aligns coordinates
result = arr1 + arr2  # Handles different dimension sizes
print(result.dims)    # ['x', 'y']
print(result.coords)  # {'x': array(['a', 'b', 'c']), 'y': array([0, 1, 2])}
```

### Lazy Computation with Dask

```python
import numpy as np
from nimblend import Array

# Create a large array with lazy computation
shape = (1000, 1000)
coords = {"x": np.arange(shape[0]), "y": np.arange(shape[1])}

# Create with lazy computation (requires dask[array])
array = Array(np.ones(shape), coords, chunks="auto")

# Operations remain lazy until compute() is called
result = (array * 2 + 10).sum(dim="x")
print(result.is_lazy)  # True

# Compute the result when needed
computed = result.compute()
print(computed.is_lazy)  # False
```

## Storage Options

### Zarr Storage

```python
from nimblend import Array, to_zarr, from_zarr

# Create an array
data = np.random.rand(100, 200)
coords = {"time": np.arange(100), "space": np.arange(200)}
array = Array(data, coords)

# Save to Zarr format
to_zarr(array, "my_array.zarr")

# Load from Zarr (optionally as a lazy array)
loaded = from_zarr("my_array.zarr", chunks="auto")
```

### IceChunk Storage (Versioned)

```python
import icechunk
from nimblend import Array, to_icechunk, from_icechunk

# Create icechunk repository and session
storage = icechunk.local_filesystem_storage("repo")
repo = icechunk.Repository.open_or_create(storage)
session = repo.writable_session("main")

# Store array in icechunk
array = Array(np.random.rand(10, 20), {"x": range(10), "y": range(20)})
to_icechunk(array, session, group="my_data")
session.commit("Initial data")

# Modify and store new version
modified = array * 10
session2 = repo.writable_session("main")
to_icechunk(modified, session2, group="my_data", mode="a")
session2.commit("Modified data")

# Load latest version
latest = repo.readonly_session("main")
loaded = from_icechunk(latest, group="my_data")
```

## Integration with Pandas and Polars

```python
import pandas as pd
from nimblend import from_series, to_series

# Create a pandas Series with MultiIndex
idx = pd.MultiIndex.from_product(
    [["A", "B"], [1, 2, 3]],
    names=["dim_x", "dim_y"]
)
series = pd.Series(np.random.rand(6), index=idx, name="values")

# Convert to NimbleNd Array
array = from_series(series)
print(array.dims)  # ['dim_x', 'dim_y']

# Perform operations
result = array * 100

# Convert back to pandas Series
new_series = to_series(result, format="pandas")

# Also works with polars (with nimblend[polars] installed)
pl_series = to_series(result, format="polars")
```

## What Makes NimbleNd Different?

- **Simplicity**: Clean, intuitive API focused on ease of use
- **Flexibility**: Works with various array types, including eager (NumPy) and lazy (Dask)
- **Coordinate Alignment**: Automatic alignment of coordinates during operations
- **Storage Options**: Multiple storage backends with consistent APIs
- **Interoperability**: Smooth conversion between popular data formats

## Dependencies

- Required: `numpy>=2.2.4`
- Optional:
  - Dask: `dask[array]>=2025.1.0`
  - Zarr storage: `zarr>=3.0.4`
  - IceChunk: `icechunk>=0.2.12`
  - Pandas integration: `pandas>=2.0.0`
  - Polars integration: `polars>=1.26.0`

## License

NimbleNd is released under the MIT License.
