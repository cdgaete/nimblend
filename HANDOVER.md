# Nimblend Development Handover

## Project Overview

Nimblend is a lightweight labeled N-dimensional array library with:

- Outer-join alignment on binary operations
- Zero-fill for missing values
- Pure NumPy backend (no pandas, dask, or other dependencies)

## Current API (26 methods/properties)

### Properties
- `data`, `coords`, `dims`, `shape`, `name`, `T`

### Selection
- `sel(indexers)` - Select by coordinate labels
- `isel(indexers)` - Select by integer indices

### Arithmetic (with alignment)
- `+`, `-`, `*`, `/`, `**`, `-` (negation)

### Comparison (returns boolean Array)
- `==`, `!=`, `<`, `<=`, `>`, `>=`

### Reductions
- `sum`, `mean`, `min`, `max`, `std`, `prod`

### Shape Manipulation
- `transpose(*dims)`, `squeeze(dim)`, `expand_dims(dim, coord)`

### Utilities
- `rename(name_map)`, `copy()`, `astype(dtype)`

## Test Coverage

- 105 tests across 8 test files
- All tests passing

## Features to Implement

### High Priority

1. **`__getitem__` with slicing**
   ```python
   arr['x']           # Select dimension by name
   arr[0:2]           # Slice first dimension
   arr[{'x': 'a'}]    # Dict-based selection (alias for sel)
   ```

2. **`where(cond, other)`** - Conditional replacement
   ```python
   arr.where(arr > 0, 0)  # Replace negatives with 0
   ```

3. **`clip(min, max)`** - Bound values
   ```python
   arr.clip(0, 100)  # Clamp to range
   ```

### Medium Priority

4. **`equals(other)`** - Full array comparison
   ```python
   arr1.equals(arr2)  # True if identical
   ```

5. **`broadcast_like(other)`** - Expand to match shape
   ```python
   arr.broadcast_like(larger_arr)
   ```

6. **`values` property** - Alias for `.data`
   ```python
   arr.values  # Same as arr.data
   ```

### Lower Priority (nimblend uses 0, not NaN)

7. **`fillna(value)`** - Replace NaN with value
8. **`dropna(dim)`** - Remove coords with NaN

## Development Commands

```bash
cd /home/carlos/projects/nimblend
source .venv/bin/activate

# Run tests
pytest tests/

# Run specific test file
pytest tests/test_sel.py -v

# Run linter
ruff check src/ tests/

# Fix linting issues
ruff check --fix src/ tests/
```

## Code Style

- Use `ruff` for linting (line length 88)
- Type hints on public methods
- Docstrings for all public methods
- Error messages should include available options/context

## Architecture

```
src/nimblend/
├── __init__.py    # Exports Array
└── core.py        # Array class implementation

tests/
├── test_xarray_comparison.py  # Correctness vs xarray
├── test_sel.py                # Label selection
├── test_isel.py               # Index selection
├── test_reductions.py         # sum, mean, etc.
├── test_transpose.py          # Dimension reordering
├── test_shape.py              # squeeze, expand_dims
├── test_comparison.py         # ==, <, >, etc.
└── test_utils.py              # rename, copy, astype
```

## Key Design Decisions

1. **Outer join alignment**: Result contains union of coordinates
2. **Zero fill**: Missing values become 0, not NaN
3. **Coordinate order preserved**: First array's order, then new from second
4. **Scalars on full reduction**: `arr.sum()` returns Python scalar, not 0-d Array
5. **Pure NumPy**: No optional backends or lazy evaluation
