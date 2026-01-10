# Nimblend Rust Acceleration

Optional Rust extension for accelerating hot paths in nimblend.

## Requirements

- Rust toolchain (rustup recommended)
- maturin (`pip install maturin`)

## Building

```bash
# Development build (for testing)
cd rust/
maturin develop --release

# Build wheel for distribution
maturin build --release
```

## What it accelerates

The Rust extension provides faster implementations for:

1. **`fill_expanded_2d_f64`**: The core bottleneck in misaligned array operations.
   Uses parallel iteration with rayon for large arrays.

2. **`map_coords_to_indices`**: Coordinate lookup for alignment.

3. **`fill_expanded_nd_f64`**: Generic N-dimensional fill using flat indices.

## Fallback

If the Rust extension is not installed, nimblend automatically falls back to
pure NumPy implementations. The API is identical.

## Benchmarks

Without Rust (pure NumPy):
- Misaligned 1000x1000 add: ~17ms

With Rust acceleration (expected):
- Misaligned 1000x1000 add: ~3-5ms (3-5x speedup)

The main speedup comes from avoiding Python overhead in the inner loops
and using SIMD operations where possible.
