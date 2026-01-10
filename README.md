# Nimblend

A lightweight labeled N-dimensional array library for Python. Think xarray, but with:

- **Outer-join alignment**: Operations preserve all coordinates, filling missing values with zero
- **Pure NumPy**: No heavy dependencies—just NumPy
- **Simple API**: Familiar xarray-like interface

## Installation

```bash
pip install nimblend
```

## Quick Start

```python
import numpy as np
from nimblend import Array

# Create labeled arrays
data1 = np.array([[1, 2], [3, 4]])
arr1 = Array(data1, {'region': ['DE', 'FR'], 'year': [2020, 2030]})

data2 = np.array([[10, 20], [30, 40]])
arr2 = Array(data2, {'region': ['FR', 'ES'], 'year': [2020, 2030]})

# Arithmetic automatically aligns coordinates
result = arr1 + arr2
print(result.coords['region'])  # ['DE', 'FR', 'ES']
# DE: [1, 2] + [0, 0] = [1, 2]
# FR: [3, 4] + [10, 20] = [13, 24]
# ES: [0, 0] + [30, 40] = [30, 40]
```

## Why Nimblend?

In energy modeling and similar domains, missing coordinate combinations typically mean "zero contribution," not "missing data." Nimblend's outer-join with zero-fill matches this mental model:

| Feature | xarray | Nimblend |
|---------|--------|----------|
| Alignment | Inner join (intersection) | Outer join (union) |
| Missing values | NaN | 0 |
| Dependencies | pandas, etc. | NumPy only |

## API Reference

### Creating Arrays

```python
# From data and coordinates
arr = Array(data, {'x': ['a', 'b'], 'y': [0, 1, 2]})

# With explicit dimension order
arr = Array(data, coords, dims=['y', 'x'])

# With a name
arr = Array(data, coords, name='temperature')
```

### Properties

- `arr.data` - Underlying NumPy array
- `arr.coords` - Dict of dimension → coordinate array
- `arr.dims` - List of dimension names
- `arr.shape` - Tuple of dimension sizes
- `arr.name` - Optional array name
- `arr.T` - Transposed array (reversed dimensions)

### Selection

```python
# By label (like xarray .sel)
arr.sel({'x': 'a'})              # Single value → reduces dimension
arr.sel({'x': ['a', 'b']})       # Multiple values → keeps dimension
arr.sel({'x': 'a', 'y': 0})      # Multiple dimensions → returns scalar

# By integer index (like xarray .isel)
arr.isel({'x': 0})               # First element
arr.isel({'x': -1})              # Last element (negative indexing)
arr.isel({'x': [0, 2]})          # Multiple indices
```

### Arithmetic Operations

All operations automatically align coordinates using outer join:

```python
result = arr1 + arr2    # Addition
result = arr1 - arr2    # Subtraction
result = arr1 * arr2    # Multiplication
result = arr1 / arr2    # Division
result = arr1 ** 2      # Power (scalar only)
result = -arr1          # Negation

# Scalar operations
result = arr * 2.5
result = arr + 100
```

### Comparison Operations

Comparisons return boolean Arrays:

```python
mask = arr > 0          # Greater than
mask = arr >= threshold # Greater or equal
mask = arr < limit      # Less than
mask = arr <= value     # Less or equal
mask = arr == target    # Equal
mask = arr != exclude   # Not equal

# Compare with another array (with alignment)
mask = arr1 > arr2
```

### Reductions

```python
arr.sum()                    # Sum all → scalar
arr.sum('x')                 # Sum over x → Array
arr.sum(['x', 'y'])          # Sum over multiple → Array

arr.mean(dim='x')            # Mean
arr.min(dim='x')             # Minimum
arr.max(dim='x')             # Maximum
arr.std(dim='x')             # Standard deviation
arr.prod(dim='x')            # Product
```

### Shape Manipulation

```python
# Transpose dimensions
arr.transpose('y', 'x')      # Explicit order
arr.T                        # Reverse order

# Add/remove dimensions
arr.expand_dims('z', coord='new')   # Add size-1 dimension
arr.squeeze()                        # Remove all size-1 dimensions
arr.squeeze('z')                     # Remove specific dimension
```

### Utilities

```python
arr.rename({'x': 'rows', 'y': 'cols'})  # Rename dimensions
arr.copy()                               # Deep copy
arr.astype(float)                        # Convert dtype
```

## Dimension Order Independence

Arrays with the same dimensions in different orders align automatically:

```python
arr1 = Array(data, {'region': r, 'tech': t, 'year': y})  # shape: (3, 4, 5)
arr2 = Array(data, {'tech': t, 'region': r, 'year': y})  # shape: (4, 3, 5)

result = arr1 + arr2  # Aligns by name, not position
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks (runs ruff + tests on every commit)
pre-commit install

# Run tests
pytest

# Run linter
ruff check src/ tests/
```

## License

MIT
