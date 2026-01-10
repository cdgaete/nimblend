# Nimblend Development Handover

## Project Overview

Nimblend is a lightweight labeled N-dimensional array library with:

- Outer-join alignment on binary operations
- Zero-fill for missing values
- Pure NumPy backend + optional Rust acceleration

## Performance vs xarray (median 5.6x faster)

| Operation | nimblend | xarray | Speedup |
|-----------|----------|--------|---------|
| Creation 100x100 | 0.02ms | 0.60ms | **29x** |
| Aligned add 100x100 | 0.03ms | 0.56ms | **19x** |
| Misaligned add 1000x1000 | 2.34ms | 8.71ms | **3.7x** |
| Reductions | 0.28ms | 1.74ms | **6.3x** |
| sel single 1000x1000 | 0.08ms | 0.11ms | **1.5x** |

**Nimblend faster in all 27 benchmarks** (up from 25/27 previously)

## Recent Changes (This Session)

### 1. Optimized `sel()` for Large Arrays
- Batch dtype checking: `_normalize_labels_batch()` checks dtype once, not per label
- `searchsorted` for single label lookups on sorted numeric/datetime coords
- Linear search fallback avoids building full dict for single lookups
- Result: `sel single (1000x1000)` went from 0.7x → 1.5x vs xarray

### 2. Column-Misaligned 2D Fast Path
- Extended `_fast_aligned_binop_2d` to handle column-misaligned arrays
- Uses transpose → row-aligned op → transpose back pattern
- Column-misaligned 1000x1000: **4.8x faster** than xarray outer join

### 3. Datetime64 Coordinate Support (Previous)
- `sel()` accepts: `datetime64`, string, `date`, `datetime`
- Outer join alignment works with datetime coords
- Uses int64 views for fast hashing (6-7x faster than native datetime64)

### 4. Rust `aligned_binop_2d` (Previous)
- Fast path for 2D arrays with one misaligned dimension
- Fuses fill + operation in single pass
- **2.6x faster, 65% less memory** vs previous approach

## Rust Acceleration (`rust/src/lib.rs`)

```bash
cd rust && maturin develop --release
```

Functions:
- `fill_expanded_2d_f64` - Fill 2D array at indices
- `fill_expanded_nd_from_indices` - Fill ND array at indices  
- `aligned_binop_2d` - Fused fill+op for 2D misaligned (row or col)
- `scatter_add_2d_rows` - Scatter-add rows (unused currently)

## Current API (34 methods/properties)

### Core
- `data`, `coords`, `dims`, `shape`, `name`, `values`, `T`

### Selection
- `sel(indexers)` - By labels (supports datetime strings)
- `isel(indexers)` - By integer indices

### Arithmetic (with alignment)
- `+`, `-`, `*`, `/`, `**`, unary `-`

### Reductions  
- `sum`, `mean`, `min`, `max`, `std`, `prod`

### Shape
- `transpose`, `squeeze`, `expand_dims`, `broadcast_like`

### Other
- `where`, `clip`, `fillna`, `dropna`, `rename`, `copy`, `astype`
- Comparison ops: `==`, `!=`, `<`, `<=`, `>`, `>=`

## Test Coverage

- **172 tests** across 15 test files
- All passing


## Potential Future Optimizations

1. `aligned_binop_nd` - Generalize to N dimensions in Rust
2. `coord_union_sorted` - Fast union for sorted coords in Rust
3. Parallel Rust with rayon for arrays >1M elements
4. `sel` batch operations in Rust

## Development Commands

```bash
cd /home/carlos/projects/nimblend
source .venv/bin/activate

pytest tests/ --tb=short      # Run tests
ruff check src/ tests/        # Lint
python benchmarks/bench_comparison.py  # Full benchmark

# Rebuild Rust
cd rust && maturin develop --release
```

## Architecture

```
src/nimblend/
├── __init__.py     # Exports Array
├── core.py         # Array class (~1200 lines)
└── _accel.py       # Rust bindings with NumPy fallback

rust/src/lib.rs     # Rust acceleration functions
benchmarks/         # Performance comparison scripts
tests/              # 15 test files, 172 tests
```

## Key Design Decisions

1. **Outer join alignment**: Result contains union of coordinates
2. **Zero fill**: Missing values become 0, not NaN  
3. **Coordinate order preserved**: First array's order, then new from second
4. **Scalars on full reduction**: `arr.sum()` returns Python scalar
5. **Pure NumPy + optional Rust**: No pandas/dask dependencies
