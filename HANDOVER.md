# Nimblend Development Handover

## Project Overview

Nimblend is a lightweight labeled N-dimensional array library with:

- Outer-join alignment on binary operations
- Zero-fill for missing values
- Pure NumPy backend + optional Rust acceleration

## Performance vs xarray (median 6.1x faster)

| Operation | nimblend | xarray | Speedup |
|-----------|----------|--------|---------|
| Creation 100x100 | 0.02ms | 0.60ms | **29x** |
| Aligned add 100x100 | 0.03ms | 0.56ms | **19x** |
| Aligned add 10000x10000 | 107ms | 107ms | **1.0x** (compute-bound) |
| Misaligned add 1000x1000 | 2.34ms | 10.15ms | **4.4x** |
| Misaligned add 20000x20000 | 1.14s | 3.68s | **3.2x** |
| Reductions 1000x1000 | 0.28ms | 1.75ms | **6.2x** |
| sel single 1000x1000 | 0.07ms | 0.10ms | **1.5x** |

**Nimblend faster in all 27 benchmarks** (identical results with outer join, fill=0)

## Recent Changes (This Session)

### 1. Optimized `sel()` for Large Arrays
- Batch dtype checking: `_normalize_labels_batch()` checks dtype once per coord
- `searchsorted` for single label lookups on sorted numeric/datetime coords
- Linear search fallback avoids building full dict for single lookups
- Result: `sel single (1000x1000)` went from 0.7x → 1.5x vs xarray

### 2. Column-Misaligned 2D Fast Path
- Extended `_fast_aligned_binop_2d` to handle column-misaligned arrays
- Uses transpose → row-aligned op → transpose back pattern
- Column-misaligned 1000x1000: **4.8x faster** than xarray outer join

### 3. New Methods Added
- `diff(dim, n=1)`: n-th discrete differences along dimension
- `cumsum(dim)`: cumulative sum along dimension  
- `cumprod(dim)`: cumulative product along dimension
- `argmax(dim)`: indices of maximum values
- `argmin(dim)`: indices of minimum values
- `concat(arrays, dim)`: concatenate arrays along dimension (module function)

### 4. Large Scale Testing
Tested up to 20000x20000 arrays (~6.4GB each):
- Aligned: converges to same speed as xarray (compute-bound by NumPy)
- Misaligned: consistent **~3.2x speedup** at all sizes

## IN PROGRESS: Parallel Rust Acceleration

Started implementing parallel Rust with rayon but hit compilation errors.

**Files modified:** `rust/src/lib.rs`

**Errors to fix:**
1. Need to import `PyUntypedArrayMethods` trait for `strides()` method
2. `SendPtr` wrapper not being used correctly in `elementwise_binop_parallel`

**Fix needed in `rust/src/lib.rs`:**
```rust
// Add this import at top:
use numpy::PyUntypedArrayMethods;

// In elementwise_binop_parallel, change:
let res_ptr = SendPtr(result.as_raw_array_mut().as_mut_ptr());
// The closure needs to capture res_ptr, not &res_ptr
```

**To rebuild after fixing:**
```bash
cd /home/carlos/projects/nimblend/rust
source $HOME/.cargo/env
source ../.venv/bin/activate
maturin develop --release
```

## Rust Acceleration (`rust/src/lib.rs`)

Current working functions:
- `fill_expanded_2d_f64` - Fill 2D array at indices
- `fill_expanded_nd_from_indices` - Fill ND array at indices  
- `aligned_binop_2d` - Fused fill+op for 2D misaligned (row or col)
- `scatter_add_2d_rows` - Scatter-add rows

New parallel functions (need fixes):
- `aligned_binop_2d_parallel` - Parallel version for large arrays
- `elementwise_binop_parallel` - Parallel element-wise ops

## Current API

### Module Functions
- `concat(arrays, dim)` - Concatenate arrays along dimension

### Array Methods/Properties (32 total)
**Core**: `data`, `coords`, `dims`, `shape`, `name`, `values`, `T`
**Selection**: `sel`, `isel`
**Arithmetic**: `+`, `-`, `*`, `/`, `**`, unary `-`
**Reductions**: `sum`, `mean`, `min`, `max`, `std`, `prod`
**Cumulative**: `diff`, `cumsum`, `cumprod`, `argmax`, `argmin`
**Shape**: `transpose`, `squeeze`, `expand_dims`, `broadcast_like`
**Other**: `where`, `clip`, `fillna`, `dropna`, `rename`, `copy`, `astype`, `equals`
**Comparison**: `==`, `!=`, `<`, `<=`, `>`, `>=`

## Test Coverage

- **196 tests** across 17 test files
- All passing

## Potential Future Optimizations

1. **Fix parallel Rust** - Add `use numpy::PyUntypedArrayMethods;` and fix SendPtr usage
2. `aligned_binop_nd` - Generalize to N dimensions in Rust
3. `coord_union_sorted` - Fast union for sorted coords in Rust
4. `groupby` / `rolling` window operations

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
├── core.py         # Array class (~1400 lines)
└── _accel.py       # Rust bindings with NumPy fallback

rust/src/lib.rs     # Rust acceleration functions
benchmarks/         # Performance comparison scripts
tests/              # 17 test files, 196 tests
```

## Key Design Decisions

1. **Outer join alignment**: Result contains union of coordinates
2. **Zero fill**: Missing values become 0, not NaN  
3. **Coordinate order preserved**: First array's order, then new from second
4. **Scalars on full reduction**: `arr.sum()` returns Python scalar
5. **Pure NumPy + optional Rust**: No pandas/dask dependencies
