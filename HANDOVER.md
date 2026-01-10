# Nimblend Development Handover

## Project Overview

Nimblend is a lightweight labeled N-dimensional array library with:

- Outer-join alignment on binary operations
- Zero-fill for missing values
- Pure NumPy backend (no pandas, dask, or other dependencies)

## API Reference

### Properties
- `data`: Underlying NumPy array
- `coords`: Dict of dimension name → coordinate array
- `dims`: List of dimension names
- `shape`: Tuple of dimension sizes
- `name`: Optional array name
- `T`: Transpose (reverse dimensions)

### Selection
- `sel(indexers)`: Select by coordinate labels
- `isel(indexers)`: Select by integer indices

### Reductions
- `sum(dim=None)`: Sum over dimension(s)
- `mean(dim=None)`: Mean over dimension(s)
- `min(dim=None)`: Minimum over dimension(s)
- `max(dim=None)`: Maximum over dimension(s)
- `std(dim=None)`: Standard deviation
- `prod(dim=None)`: Product

### Shape Manipulation
- `transpose(*dims)`: Reorder dimensions
- `squeeze(dim=None)`: Remove size-1 dimensions
- `expand_dims(dim, coord)`: Add new dimension

### Utilities
- `rename(name_map)`: Rename dimensions
- `copy()`: Deep copy
- `astype(dtype)`: Convert data type

### Binary Operations
- `+`, `-`, `*`, `/`, `**`: With automatic alignment
- Scalar operations apply element-wise

## Key Behaviors

```python
# Automatic dimension alignment
arr1 = Array(data1, {'region': ['DE', 'FR'], 'year': [2020, 2030]})
arr2 = Array(data2, {'year': [2020, 2030], 'region': ['DE', 'FR']})
result = arr1 + arr2  # Aligns by name, not position

# Outer join with zero-fill
arr1 = Array(data1, {'x': ['a', 'b']})
arr2 = Array(data2, {'x': ['b', 'c']})
result = arr1 + arr2  # coords: ['a', 'b', 'c'], missing filled with 0

# Coordinate order preserved
# Result uses first array's order, then appends new from second
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run linter
ruff check src/ tests/

# Run tests
pytest tests/
```

## Test Coverage

- 95 tests across 7 test files
- `test_xarray_comparison.py`: Verify correctness vs xarray
- `test_sel.py`: Label-based selection
- `test_isel.py`: Index-based selection
- `test_reductions.py`: sum, mean, min, max, std, prod
- `test_transpose.py`: Dimension reordering
- `test_shape.py`: squeeze, expand_dims
- `test_utils.py`: rename, copy, astype

## Notes

- Coordinate order preserved from first array in operations
- Missing values fill with 0, not NaN
- All reductions return scalar when no dims remain
