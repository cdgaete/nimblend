# NimbleNd: Efficient Labeled N-Dimensional Arrays

NimbleNd is a lightweight library for working with labeled N-dimensional arrays. It provides an efficient way to handle multi-dimensional data with named axes, allowing for operations between arrays with different dimension sizes and coordinates.

## Key Features

- **Memory Efficient**: Direct operations on NumPy arrays without excessive memory overhead
- **Lazy Computation**: Seamless integration with Dask for delayed computation of large arrays
- **Flexible Alignment**: Properly aligns arrays with different coordinate systems during operations
- **NumPy Compatible**: Built on top of NumPy arrays for high performance
- **Intuitive API**: Simple interface for creating and manipulating labeled arrays
- **Multiple I/O Options**: Support for Zarr, IceChunk, pandas, and polars data formats

## Installation

```bash
pip install nimblend         # Basic installation
pip install nimblend[io]     # With Zarr I/O support
pip install nimblend[dask]   # With Dask support
pip install nimblend[pandas] # With pandas support
pip install nimblend[polars] # With polars support
pip install nimblend[all]    # All optional dependencies
```

## Quick Start

### Creating Arrays

```python
import nimblend as nd
import numpy as np

# Create an array with named dimensions
data = np.array([[1, 2, 3], [4, 5, 6]])
coords = {'time': ['2023-01-01', '2023-01-02'], 'station': ['A', 'B', 'C']}
arr = nd.Array(data, coords=coords, name='temperature')

# Access data using coordinate values
arr[{'time': '2023-01-01', 'station': 'B'}]  # returns 2

# Create a lazy array with Dask
lazy_arr = nd.Array(data, coords=coords, chunks="auto")
print(lazy_arr.is_lazy)  # True

# Compute a lazy array
computed = lazy_arr.compute()
print(computed.is_lazy)  # False
```

### Array Operations

Arrays automatically align dimensions during operations, correctly handling different coordinate systems:

```python
# Two arrays with different coordinate systems
data1 = np.array([[1, 2], [3, 4]])
coords1 = {'lat': [30, 40], 'lon': [10, 20]}
arr1 = nd.Array(data1, coords1)

data2 = np.array([[10, 20, 30], [40, 50, 60]])
coords2 = {'lat': [40, 50], 'lon': [10, 20, 30]}
arr2 = nd.Array(data2, coords2)

# Arithmetic operations automatically align coordinates
result = arr1 + arr2
print(result.coords)
# {'lat': array([30, 40, 50]), 'lon': array([10, 20, 30])}

# Result has correct values with alignment
print(result.data)
# [[  1,   2, nan],
#  [ 13,  24, nan],
#  [nan, nan, nan]]
```

### Combining Arrays

```python
# Concatenate arrays along a dimension
time1 = nd.Array(np.array([1, 2]), {'time': ['01-01', '01-02'], 'station': ['A']})
time2 = nd.Array(np.array([3, 4]), {'time': ['01-03', '01-04'], 'station': ['A']})
combined = nd.concat([time1, time2], dim='time')

# Stack arrays along a new dimension
station1 = nd.Array(np.array([1, 2]), {'time': ['01-01', '01-02']}, name='station_A')
station2 = nd.Array(np.array([3, 4]), {'time': ['01-01', '01-02']}, name='station_B')
multi_station = nd.stack([station1, station2], dim='station',
                        coords={'station': ['A', 'B']})
```

### Array Reduction

```python
# Sum along a dimension
data = np.array([[1, 2, 3], [4, 5, 6]])
coords = {'time': ['2023-01-01', '2023-01-02'], 'station': ['A', 'B', 'C']}
arr = nd.Array(data, coords)

# Sum over the time dimension
station_totals = arr.sum(dim='time')
print(station_totals.data)  # [5, 7, 9]

# Sum over the station dimension
time_totals = arr.sum(dim='station')
print(time_totals.data)  # [6, 15]
```

### Integration with Pandas

Convert between NimbleNd Arrays and pandas Series with MultiIndex:

```python
import pandas as pd
import nimblend as nd

# Create a pandas Series with MultiIndex
idx = pd.MultiIndex.from_tuples(
    [('A', 1), ('A', 2), ('B', 1), ('B', 3)],
    names=['region', 'station']
)
series = pd.Series([10.5, 20.0, 30.5, 40.0], index=idx, name='temperature')

# Convert to NimbleNd Array
arr = nd.from_series(series)
print(arr.dims)  # ['region', 'station']
print(arr.name)  # 'temperature'

# The result is a dense n-dimensional array
print(arr.data)
# [[10.5, 20.0,  nan],
#  [30.5,  nan, 40.0]]

# Convert back to Series format
series_again = nd.to_series(arr)
print(series_again)
# region  station
# A       1         10.5
#         2         20.0
#         3          NaN
# B       1         30.5
#         2          NaN
#         3         40.0
# Name: temperature, dtype: float64
```

### Integration with Polars

Convert between NimbleNd Arrays and polars Series:

```python
import polars as pl
import nimblend as nd

# Create a polars DataFrame
df = pl.DataFrame({
    'region': ['A', 'A', 'B', 'B'],
    'station': [1, 2, 1, 3],
    'temperature': [10.5, 20.0, 30.5, 40.0]
})

# Convert to NimbleNd Array
arr = nd.from_series(df['temperature'], dims=['region', 'station'])
print(arr.dims)  # ['region', 'station']

# Convert back to polars format
pl_series = nd.to_series(arr, format='polars')
```

### Persistence with Zarr

Save and load arrays with Zarr:

```python
import nimblend as nd
import numpy as np

# Create an array
data = np.random.rand(100, 100)
coords = {'x': np.arange(100), 'y': np.arange(100)}
arr = nd.Array(data, coords, name='random_data')

# Save to Zarr format
nd.to_zarr(arr, 'data.zarr')

# Load from Zarr
loaded = nd.from_zarr('data.zarr')

# Load lazily with specific chunks
lazy_loaded = nd.from_zarr('data.zarr', chunks={'x': 20, 'y': 20})
```

### IceChunk Support

NimbleNd integrates with IceChunk for scalable array storage:

```python
import nimblend as nd
import numpy as np
from icechunk import Session

# Create an array
data = np.random.rand(1000, 1000)
coords = {'x': np.arange(1000), 'y': np.arange(1000)}
arr = nd.Array(data, coords, name='large_data')

# Save to IceChunk
with Session() as session:
    nd.to_icechunk(arr, session, group='my_array')

# Use with Dask for efficient distributed computing
lazy_arr = arr.to_dask()
with Session() as session:
    nd.to_icechunk(lazy_arr, session, group='large_data')
```

## Advanced Usage

### Custom Indexing

```python
# Get a specific slice of data
subset = arr[{'time': ['2023-01-01', '2023-01-02'], 'station': 'A'}]

# Standard NumPy-style indexing also works
time_slice = arr[0, :]  # First time index, all stations
```

### Working with Lazy Arrays

```python
# Create a lazy array
lazy_arr = nd.Array(np.random.rand(1000, 1000),
                   {'x': np.arange(1000), 'y': np.arange(1000)},
                   chunks={'x': 100, 'y': 100})

# Perform operations without computing
squared = lazy_arr ** 2
result = squared.sum(dim='x')

# Compute when needed
final = result.compute()

# Or persist in memory while keeping lazy properties
persisted = result.persist()
```

### Type Conversion

```python
# Convert a NumPy array to a Dask array
data = np.random.rand(1000, 1000)
arr = nd.Array(data, {'x': np.arange(1000), 'y': np.arange(1000)})
lazy_arr = arr.to_dask()

# Convert a Dask array to a NumPy array
eager_arr = lazy_arr.to_numpy()
```

## Compared to Other Libraries

Unlike other labeled array libraries, NimbleNd focuses on:

- **Memory Efficiency**: Minimizes memory overhead when handling labeled dimensions
- **Direct Operations**: Works directly with NumPy/Dask arrays whenever possible
- **Alignment-First**: Prioritizes correct alignment during operations between arrays with different coordinate systems
- **Simplicity**: Provides a focused API for essential operations without unnecessary complexity

## Performance Tips

- Use lazy computation with `chunks="auto"` for large arrays
- Persist intermediate results that will be reused with `persist()`
- For operations between arrays with very different coordinates, consider pre-aligning
- When working with IceChunk, choose chunk sizes that match your access patterns

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Example 1: Climate Data Analysis

```python
import nimblend as nd
import numpy as np

# Create temperature data for multiple stations over time
times = [1,2,3,4,5,6,7,8]
stations = ['A', 'B', 'C', 'D']
temps = np.random.normal(15, 5, size=(len(times), len(stations)))

# Create a NimbleNd array
temp_array = nd.Array(
    temps,
    coords={'time': times, 'station': stations},
    dims=['time', 'station'],
    name='temperature'
)

# Apply a 3-day rolling mean along the time dimension
# First convert to pandas for the operation
temp_series = nd.to_series(temp_array)
rolled = temp_series.groupby('station').rolling(3).mean()
rolled = rolled.droplevel(0)

# Convert back to NimbleNd
smoothed_temps = nd.from_series(rolled)

# Extract data for a specific station
station_a = temp_array[{'station': 'A'}]
```

## Example 2: Satellite Image Processing

```python
import nimblend as nd
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# Create sample satellite data with multiple bands
lat = np.linspace(30, 35, 500)
lon = np.linspace(-100, -95, 500)
bands = ['red', 'green', 'blue', 'nir']

# Create some sample data - in practice this would be loaded from files
data = np.zeros((len(lat), len(lon), len(bands)))

# Fill with simulated data
for i, band in enumerate(bands):
    base = np.random.normal(0, 1, size=(len(lat), len(lon)))
    # Apply some smoothing to simulate realistic satellite imagery
    data[:, :, i] = ndimage.gaussian_filter(base, sigma=5)
    # Add some "features"
    if band == 'nir':
        # Higher NIR values in some regions (vegetation)
        data[100:300, 200:400, i] += 2

# Create NimbleNd array
satellite_data = nd.Array(
    data,
    coords={'lat': lat, 'lon': lon, 'band': bands},
    name='satellite_imagery'
)

# Calculate NDVI (Normalized Difference Vegetation Index)
red = satellite_data[{'band': 'red'}]
nir = satellite_data[{'band': 'nir'}]
ndvi = (nir - red) / (nir + red)
ndvi.name = 'ndvi'

# Apply cloud masking (simulated)
cloud_mask = np.random.random(size=(len(lat), len(lon))) > 0.9
masked_ndvi = ndvi.copy()
masked_ndvi.data[cloud_mask] = np.nan

# Visualize
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(red.data, cmap='Reds')
plt.title('Red Band')
plt.colorbar()

plt.subplot(2, 2, 2)
plt.imshow(nir.data, cmap='Greens')
plt.title('NIR Band')
plt.colorbar()

plt.subplot(2, 2, 3)
plt.imshow(ndvi.data, cmap='RdYlGn')
plt.title('NDVI')
plt.colorbar()

plt.subplot(2, 2, 4)
plt.imshow(masked_ndvi.data, cmap='RdYlGn')
plt.title('Cloud-Masked NDVI')
plt.colorbar()

plt.tight_layout()
plt.show()

# Save the NDVI result
nd.to_zarr(ndvi, 'ndvi_result.zarr')
```

## Example 3: Financial Time Series Analysis

```python
import nimblend as nd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create sample financial data
dates = pd.date_range('2020-01-01', periods=252)  # One trading year
tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META']

# Generate random price data with correlations
rng = np.random.RandomState(42)
returns = rng.normal(0, 1, size=(len(dates), len(tickers)))
# Add correlation
correlation = np.array([
    [1.0, 0.7, 0.5, 0.3, 0.2],
    [0.7, 1.0, 0.6, 0.4, 0.3],
    [0.5, 0.6, 1.0, 0.5, 0.4],
    [0.3, 0.4, 0.5, 1.0, 0.7],
    [0.2, 0.3, 0.4, 0.7, 1.0]
])
chol = np.linalg.cholesky(correlation)
correlated_returns = returns @ chol.T

# Convert to prices starting at 100
prices = 100 * np.exp(np.cumsum(correlated_returns * 0.01, axis=0))

# Create NimbleNd array
price_array = nd.Array(
    prices,
    coords={'date': dates, 'ticker': tickers},
    name='stock_prices'
)

# Calculate 20-day moving averages
# First convert to pandas
price_series = nd.to_series(price_array)
ma20 = price_series.groupby('ticker').rolling(20).mean()
ma20 = ma20.dropna()

# Convert back to NimbleNd
ma20_array = nd.from_series(ma20)

# Calculate daily returns
daily_returns = price_array.copy()
daily_returns.data[1:] = price_array.data[1:] / price_array.data[:-1] - 1
daily_returns.data[0] = 0
daily_returns.name = 'daily_returns'

# Calculate volatility (20-day rolling standard deviation)
vol_series = price_series.groupby('ticker').pct_change().rolling(20).std()
vol_array = nd.from_series(vol_series.dropna())

# Visualize results
plt.figure(figsize=(15, 10))

# Plot prices
plt.subplot(2, 2, 1)
for ticker in tickers:
    plt.plot(dates, price_array[{'ticker': ticker}].data, label=ticker)
plt.title('Stock Prices')
plt.legend()

# Plot returns heatmap
plt.subplot(2, 2, 2)
plt.imshow(daily_returns.data.T, aspect='auto', cmap='coolwarm')
plt.colorbar(label='Daily Return')
plt.yticks(range(len(tickers)), tickers)
plt.title('Daily Returns Heatmap')

# Plot volatility
plt.subplot(2, 2, 3)
vol_dates = dates[20:]  # Adjust for rolling window
for i, ticker in enumerate(tickers):
    plt.plot(vol_dates, vol_array[{'ticker': ticker}].data, label=ticker)
plt.title('20-Day Rolling Volatility')
plt.legend()

# Plot correlation matrix
plt.subplot(2, 2, 4)
return_corr = np.corrcoef(daily_returns.data.T)
plt.imshow(return_corr, cmap='viridis')
plt.colorbar(label='Correlation')
plt.xticks(range(len(tickers)), tickers)
plt.yticks(range(len(tickers)), tickers)
plt.title('Return Correlation Matrix')

plt.tight_layout()
plt.show()

# Save the data for future analysis
nd.to_zarr(price_array, 'stock_data.zarr')
```
