# Nimblend Development Handover

## Project Overview

Nimblend is a lightweight labeled N-dimensional array library with:

- Outer-join alignment on binary operations
- Zero-fill for missing values
- Pure NumPy backend (no pandas, dask, or other dependencies)

## Current API (34 methods/properties)

### Properties
- `data`, `coords`, `dims`, `shape`, `name`, `values`, `T`

### Selection
- `sel(indexers)` - Select by coordinate labels
- `isel(indexers)` - Select by integer indices
- `__getitem__` - Dict, int/slice, or dim name indexing

### Arithmetic (with alignment)
- `+`, `-`, `*`, `/`, `**`, `-` (negation)

### Comparison (returns boolean Array)
- `==`, `!=`, `<`, `<=`, `>`, `>=`
- `equals(other)` - Full array comparison

### Reductions
- `sum`, `mean`, `min`, `max`, `std`, `prod`

### Shape Manipulation
- `transpose(*dims)`, `squeeze(dim)`, `expand_dims(dim, coord)`
- `broadcast_like(other)` - Expand to match shape

### Conditional
- `where(cond, other)` - Replace where condition is False
- `clip(min, max)` - Bound values to range

### NaN Handling
- `fillna(value)` - Replace NaN with value
- `dropna(dim)` - Remove coords with NaN

### Utilities
- `rename(name_map)`, `copy()`, `astype(dtype)`

## Test Coverage

- 164 tests across 14 test files
- All tests passing

## Documentation

- mkdocs-material setup in `/docs`
- Run locally: `mkdocs serve -a 0.0.0.0:8000`
- Deploy: `mkdocs gh-deploy`

## Next Steps: Performance

### Benchmarks vs xarray
- Create benchmark suite comparing nimblend vs xarray
- Key operations: creation, arithmetic, alignment, reductions, selection
- Measure both time and memory
- Generate comparison charts

### Profile Hot Paths
- Identify performance bottlenecks
- Focus on `_align_with()` and `_fill_expanded()` methods
- Consider optimizations for common cases

## Development Commands

```bash
cd /home/carlos/projects/nimblend
source .venv/bin/activate

# Run tests
pytest tests/

# Run linter
ruff check src/ tests/

# Build docs
mkdocs build

# Serve docs locally
mkdocs serve -a 0.0.0.0:8000
```

## Architecture

```
src/nimblend/
├── __init__.py    # Exports Array
└── core.py        # Array class (~870 lines)

tests/              # 14 test files, 164 tests
docs/               # mkdocs-material documentation
```

## Key Design Decisions

1. **Outer join alignment**: Result contains union of coordinates
2. **Zero fill**: Missing values become 0, not NaN
3. **Coordinate order preserved**: First array's order, then new from second
4. **Scalars on full reduction**: `arr.sum()` returns Python scalar, not 0-d Array
5. **Pure NumPy**: No optional backends or lazy evaluation
