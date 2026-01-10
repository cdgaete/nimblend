# Nimblend Development Handover

## Project Overview

Nimblend is a lightweight labeled N-dimensional array library with:

- Outer-join alignment on binary operations
- Zero-fill for missing values
- Pure NumPy backend + optional Rust acceleration

## Performance vs xarray (median 6.1x faster)

| Operation | nimblend | xarray | Speedup |
|-----------|----------|--------|---------|
| Creation 100x100 | 0.02ms | 0.59ms | **29x** |
| Aligned add 100x100 | 0.03ms | 0.56ms | **19x** |
| Misaligned add 1000x1000 | 2.34ms | 8.71ms | **3.7x** |
| Reductions | 0.28ms | 1.74ms | **6.3x** |

## Recent Changes (This Session)

### 1. Datetime64 Coordinate Support
- `sel()` accepts: `datetime64`, string, `date`, `datetime`
- Outer join alignment works with datetime coords
- Uses int64 views for fast hashing (6-7x faster than native datetime64)
- Uses `searchsorted` for large sorted numeric/datetime arrays

### 2. Rust `aligned_binop_2d` Optimization
- New fast path for 2D arrays with one misaligned dimension
- Fuses fill + operation in single pass
- **2.6x faster, 65% less memory** vs previous approach
- Supports add/sub/mul/div

## Rust Acceleration (`rust/src/lib.rs`)

```bash
cd rust && maturin develop --release
```

Functions:
- `fill_expanded_2d_f64` - Fill 2D array at indices
- `fill_expanded_nd_from_indices` - Fill ND array at indices  
- `aligned_binop_2d` - **NEW** - Fused fill+op for 2D misaligned
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

- **171 tests** across 15 test files (including new `test_datetime.py`)
- All passing

## Known Limitations

1. `sel()` slower than xarray for large arrays (1000x1000) - needs searchsorted
2. Column-misaligned 2D not optimized (only row-misaligned uses fast path)
3. No parallel Rust yet (could use rayon for >1M elements)

## Potential Future Rust Optimizations

1. `aligned_binop_nd` - Generalize to N dimensions
2. `coord_union_sorted` - Fast union for sorted coords
3. Parallel versions with rayon for large arrays
4. `sel` optimization with searchsorted in Rust

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
├── core.py         # Array class (~1100 lines)
└── _accel.py       # Rust bindings with NumPy fallback

rust/src/lib.rs     # Rust acceleration functions
benchmarks/         # Performance comparison scripts
tests/              # 15 test files, 171 tests
```

## Key Design Decisions

1. **Outer join alignment**: Result contains union of coordinates
2. **Zero fill**: Missing values become 0, not NaN  
3. **Coordinate order preserved**: First array's order, then new from second
4. **Scalars on full reduction**: `arr.sum()` returns Python scalar
5. **Pure NumPy + optional Rust**: No pandas/dask dependencies
