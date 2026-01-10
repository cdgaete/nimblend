"""
Optional Rust acceleration for nimblend.

This module provides a fallback mechanism: if the Rust extension is available,
use it for hot paths. Otherwise, fall back to pure NumPy implementations.

To build the Rust extension:
    cd rust/
    pip install maturin
    maturin develop --release

Functions provided:
    - map_coords_to_indices: Build index mapping between coordinate arrays
    - fill_expanded_2d: Fill 2D expanded array at specified indices
    - fill_expanded_nd: Fill ND expanded array at specified indices
"""

import numpy as np

# Try to import Rust extension
try:
    from nimblend_rust import (
        aligned_binop_2d as _rust_aligned_binop_2d,
    )
    from nimblend_rust import (
        fill_expanded_2d_f64 as _rust_fill_2d,
    )
    from nimblend_rust import (
        fill_expanded_nd_from_indices as _rust_fill_nd_indices,
    )
    from nimblend_rust import (
        map_coords_to_indices as _rust_map_coords,
    )
    HAS_RUST = True
except ImportError:
    HAS_RUST = False


# Re-export for use in core.py
def aligned_binop_2d(
    result: np.ndarray,
    source1: np.ndarray,
    row_indices1: np.ndarray,
    source2: np.ndarray,
    row_indices2: np.ndarray,
    op: str,
) -> None:
    """
    Compute aligned binary operation on two 2D arrays with row index mappings.

    result = zeros(result_shape)
    result[row_idx1, :] = source1
    result[row_idx2, :] op= source2
    """
    if HAS_RUST:
        _rust_aligned_binop_2d(
            result,
            source1,
            row_indices1.astype(np.int64),
            source2,
            row_indices2.astype(np.int64),
            op,
        )
    else:
        # Pure NumPy fallback
        result[row_indices1, :] = source1
        if op == "add":
            result[row_indices2, :] += source2
        elif op == "sub":
            result[row_indices2, :] -= source2
        elif op == "mul":
            result[row_indices2, :] *= source2
        elif op == "div":
            result[row_indices2, :] /= source2


def map_coords_to_indices(
    source_coords: np.ndarray,
    target_coords: np.ndarray,
) -> np.ndarray:
    """Map source coordinates to their indices in target coordinates."""
    if HAS_RUST:
        return _rust_map_coords(
            source_coords.tolist(),
            target_coords.tolist(),
        )
    else:
        target_to_idx = {v: i for i, v in enumerate(target_coords)}
        return np.array([target_to_idx.get(v, -1) for v in source_coords])


def fill_expanded_2d(
    expanded: np.ndarray,
    source: np.ndarray,
    row_indices: np.ndarray,
    col_indices: np.ndarray,
) -> None:
    """Fill expanded 2D array with source data at specified indices."""
    if HAS_RUST and expanded.dtype == np.float64:
        _rust_fill_2d(
            expanded,
            source.astype(np.float64),
            row_indices.astype(np.int64),
            col_indices.astype(np.int64),
        )
    else:
        idx_arrays = np.meshgrid(row_indices, col_indices, indexing="ij")
        expanded[tuple(idx_arrays)] = source


def fill_expanded_nd(
    expanded: np.ndarray,
    source: np.ndarray,
    index_arrays: list,
) -> None:
    """Fill expanded ND array with source data at specified indices."""
    if HAS_RUST and expanded.dtype == np.float64:
        # Compute strides for the expanded array
        strides = []
        stride = 1
        for s in reversed(expanded.shape):
            strides.append(stride)
            stride *= s
        strides = list(reversed(strides))

        # Convert index arrays to int64
        idx_arrays_i64 = [idx.astype(np.int64) for idx in index_arrays]

        _rust_fill_nd_indices(
            expanded.ravel(),
            source.ravel().astype(np.float64),
            idx_arrays_i64,
            strides,
        )
    else:
        idx_arrays = np.meshgrid(*index_arrays, indexing="ij")
        expanded[tuple(idx_arrays)] = source
