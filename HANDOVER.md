# Nimblend Development Handover

## Project Overview

Nimblend is a lightweight labeled N-dimensional array library with:

- Outer-join alignment on binary operations
- Zero-fill for missing values
- Pure NumPy backend + optional Rust acceleration

## Performance vs xarray (median 5.8x faster)

| Operation | nimblend | xarray | Speedup |
|-----------|----------|--------|---------|
| Creation 100x100 | 0.02ms | 0.60ms | **28x** |
| Aligned add 100x100 | 0.03ms | 0.57ms | **21x** |
| Misaligned add 1000x1000 | 2.32ms | 8.62ms | **3.7x** |
| Misaligned add 3000x3000 | 27ms | 87ms | **3.2x** |
| Reductions 1000x1000 | 0.27ms | 1.75ms | **6.4x** |
| sel single 1000x1000 | 0.07ms | 0.11ms | **1.5x** |

**Nimblend faster in all 27 benchmarks** (identical results with outer join, fill=0)

## Recent Changes (Latest Session)

### 1. Fixed Parallel Rust Compilation ✓
- Added `PyUntypedArrayMethods` import for `strides()` method
- Made `SendPtr` wrapper `Copy + Clone` with a `ptr()` method
- `_accel.py` now auto-selects parallel for arrays >1M elements
- Result: Parallel Rust now compiles and works correctly

### 2. New Methods Added
- `shift(shifts, fill_value=0)`: Shift array values along dimension(s)
- `roll(shifts)`: Roll array values with wrap-around
- `coarsen(windows, func='mean', boundary='trim')`: Downsample by aggregating over windows

### Previous Session Methods (still present)
- `diff(dim, n=1)`: n-th discrete differences along dimension
- `cumsum(dim)`: cumulative sum along dimension  
- `cumprod(dim)`: cumulative product along dimension
- `argmax(dim)`: indices of maximum values
- `argmin(dim)`: indices of minimum values
- `concat(arrays, dim)`: concatenate arrays along dimension (module function)

## Rust Acceleration (`rust/src/lib.rs`)

Working functions:
- `fill_expanded_2d_f64` - Fill 2D array at indices
- `fill_expanded_nd_from_indices` - Fill ND array at indices  
- `aligned_binop_2d` - Fused fill+op for 2D misaligned (row or col)
- `aligned_binop_2d_parallel` - Parallel version for large arrays (>1M elements)
- `elementwise_binop_parallel` - Parallel element-wise ops
- `scatter_add_2d_rows` - Scatter-add rows

## Current API

### Module Functions
- `concat(arrays, dim)` - Concatenate arrays along dimension

### Array Methods/Properties (35 total)
**Core**: `data`, `coords`, `dims`, `shape`, `name`, `values`, `T`
**Selection**: `sel`, `isel`
**Arithmetic**: `+`, `-`, `*`, `/`, `**`, unary `-`
**Reductions**: `sum`, `mean`, `min`, `max`, `std`, `prod`
**Cumulative**: `diff`, `cumsum`, `cumprod`, `argmax`, `argmin`
**Window**: `shift`, `roll`, `coarsen`
**Shape**: `transpose`, `squeeze`, `expand_dims`, `broadcast_like`
**Other**: `where`, `clip`, `fillna`, `dropna`, `rename`, `copy`, `astype`, `equals`
**Comparison**: `==`, `!=`, `<`, `<=`, `>`, `>=`

## Test Coverage

- **220 tests** across 19 test files
- All passing

## Potential Future Work

1. `aligned_binop_nd` - Generalize to N dimensions in Rust
2. `coord_union_sorted` - Fast union for sorted coords in Rust
3. `groupby` / `rolling` window operations
4. `interp` - Interpolation along dimensions

## Development Commands

```bash
cd /home/carlos/projects/nimblend
source .venv/bin/activate

pytest tests/ --tb=short      # Run tests
ruff check src/ tests/        # Lint
python benchmarks/bench_comparison.py  # Full benchmark

# Rebuild Rust (after sourcing cargo env)
source $HOME/.cargo/env
cd rust && maturin develop --release
```

## Architecture

```
src/nimblend/
├── __init__.py     # Exports Array, concat
├── core.py         # Array class (~1600 lines)
└── _accel.py       # Rust bindings with NumPy fallback

rust/src/lib.rs     # Rust acceleration functions
benchmarks/         # Performance comparison scripts
tests/              # 19 test files, 220 tests
```

## Key Design Decisions

1. **Outer join alignment**: Result contains union of coordinates
2. **Zero fill**: Missing values become 0, not NaN  
3. **Coordinate order preserved**: First array's order, then new from second
4. **Scalars on full reduction**: `arr.sum()` returns Python scalar
5. **Pure NumPy + optional Rust**: No pandas/dask dependencies
6. **Parallel threshold**: Arrays >1M elements use parallel Rust automatically
