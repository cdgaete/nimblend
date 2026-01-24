# Nimblend Development Handover

## Project Overview

Nimblend is a lightweight labeled N-dimensional array library with:

- Outer-join alignment on binary operations
- Zero-fill for missing values
- Pure NumPy backend + optional Rust acceleration

Location: `/home/carlos/projects/nimblend`

## Installation

```bash
pip install nimblend              # Base (numpy only)
pip install nimblend[test]        # + pytest
pip install nimblend[dev]         # + pytest, ruff, pre-commit
pip install nimblend[benchmark]   # + xarray (for comparison benchmarks)
```

## Performance vs xarray (median 5.8x faster)

| Operation | nimblend | xarray | Speedup |
|-----------|----------|--------|---------|
| Creation 100x100 | 0.02ms | 0.60ms | **28x** |
| Aligned add 100x100 | 0.03ms | 0.57ms | **21x** |
| Misaligned add 1000x1000 | 2.32ms | 8.62ms | **3.7x** |
| Misaligned add 3000x3000 | 27ms | 87ms | **3.2x** |
| Reductions 1000x1000 | 0.27ms | 1.75ms | **6.4x** |

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

## Rust Acceleration

Optional Rust extension for large arrays (>1M elements):
- `fill_expanded_2d_f64` - Fill 2D array at indices
- `fill_expanded_nd_from_indices` - Fill ND array at indices
- `aligned_binop_2d` / `aligned_binop_2d_parallel` - Fused fill+op
- `elementwise_binop_parallel` - Parallel element-wise ops

Build: `cd rust && source ~/.cargo/env && maturin develop --release`

## Development Commands

```bash
cd /home/carlos/projects/nimblend
PYTHONPATH=src pytest tests/ --tb=short
python3 -m ruff check src/ tests/
```

## Architecture

```
src/nimblend/
├── __init__.py     # Exports Array, concat
├── core.py         # Array class (~1600 lines)
└── _accel.py       # Rust bindings with NumPy fallback
rust/src/lib.rs     # Rust acceleration functions
tests/              # 19 test files, 220 tests
```

## Key Design Decisions

1. **Outer join alignment**: Result contains union of coordinates
2. **Zero fill**: Missing values become 0, not NaN
3. **Coordinate order preserved**: First array's order, then new from second
4. **Scalars on full reduction**: `arr.sum()` returns Python scalar
5. **Minimal dependencies**: Only numpy required
6. **Parallel threshold**: Arrays >1M elements use parallel Rust automatically
7. **NotImplemented for unknown types**: Binary ops return `NotImplemented` for unknown types, allowing Python to try the other operand's `__rmul__` etc.
