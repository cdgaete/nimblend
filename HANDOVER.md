# Nimblend Development Handover

## Project Overview

Nimblend is a labeled N-dimensional array library for energy modeling with:

- Outer-join alignment on binary operations
- Zero-fill for missing values
- Pure NumPy backend

## Implementation

Location: `src/nimblend/core.py`

### Array Class

#### Initialization

- `__init__(data, coords, dims=None, name=None)`: Create array with data and coordinate mapping
- `_validate()`: Verify dimensions match data shape

#### Alignment

- `_align_with(other)`: Expand two arrays to common shape using outer join
- `_fill_expanded(expanded, all_dims, union_coords)`: Place values at correct positions with dimension reordering

#### Operations

- Binary: `+`, `-`, `*`, `/`, `**` with arrays or scalars
- Reduction: `sum(dim=None)` over specified dimensions

### Verified Behavior

```python
# Different dimension orders
arr1 = Array(data1, {'region': r, 'tech': t, 'year': y})
arr2 = Array(data2, {'tech': t, 'region': r, 'year': y})
result = arr1 + arr2  # Aligns dimensions automatically

# Partially overlapping coordinates
regions1 = ['DE', 'FR', 'ES']
regions2 = ['FR', 'IT', 'PL']
# Result contains union: ['DE', 'FR', 'ES', 'IT', 'PL'] (order preserved from first array)

# Disjoint coordinates
years1 = [2010, 2020, 2030]
years2 = [2040, 2050, 2060]
# Addition preserves all values
# Multiplication yields zeros where coordinates don't overlap
```

## Next Steps

### Compare with xarray

```python
import xarray as xr
from nimblend import Array

xr_arr = xr.DataArray(data, dims=['region', 'tech'], coords={'region': regions, 'tech': techs})
nb_arr = Array(data, {'region': regions, 'tech': techs})

# Compare operation results (xarray uses NaN, nimblend uses 0)
```

### Features to Implement

- `sel()`: Select by coordinate values
- `isel()`: Select by integer index
- `mean()`, `min()`, `max()`: Additional reductions
- `transpose()`: Reorder dimensions
- `rename()`: Rename dimensions
- `expand_dims()`: Add new dimension
- `squeeze()`: Remove size-1 dimensions

### Edge Cases to Test

- Empty arrays
- Single-element arrays
- Single dimension arrays
- String vs numeric coordinates

## Development Environment

```bash
cd /home/carlos/projects/nimblend
python -c "from nimblend import Array"
```

Run linter before code changes:

```bash
ruff check src/nimblend/
```

## Notes

- Coordinate order is preserved: first array's coords come first, then new coords from second array
- Missing values fill with 0, not NaN
