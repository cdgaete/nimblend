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
"""

import numpy as np

# Try to import Rust extension
try:
    from nimblend_rust import (
        fill_expanded_2d_f64 as _rust_fill_2d,
    )
    from nimblend_rust import (
        fill_expanded_nd_f64 as _rust_fill_nd,
    )
    from nimblend_rust import (
        map_coords_to_indices as _rust_map_coords,
    )
    HAS_RUST = True
except ImportError:
    HAS_RUST = False


def map_coords_to_indices(
    source_coords: np.ndarray,
    target_coords: np.ndarray,
) -> np.ndarray:
    """
    Map source coordinates to their indices in target coordinates.

    Returns array of indices where each source coord appears in target.
    """
    if HAS_RUST:
        return _rust_map_coords(
            source_coords.tolist(),
            target_coords.tolist(),
        )
    else:
        # Pure NumPy fallback
        target_to_idx = {v: i for i, v in enumerate(target_coords)}
        return np.array([target_to_idx.get(v, -1) for v in source_coords])


def fill_expanded_2d(
    expanded: np.ndarray,
    source: np.ndarray,
    row_indices: np.ndarray,
    col_indices: np.ndarray,
) -> None:
    """
    Fill expanded 2D array with source data at specified indices.

    Modifies expanded in place.
    """
    if HAS_RUST and expanded.dtype == np.float64:
        _rust_fill_2d(
            expanded,
            source.astype(np.float64),
            row_indices.astype(np.int64),
            col_indices.astype(np.int64),
        )
    else:
        # Pure NumPy fallback using meshgrid
        idx_arrays = np.meshgrid(row_indices, col_indices, indexing="ij")
        expanded[tuple(idx_arrays)] = source


def fill_expanded_nd(
    expanded: np.ndarray,
    source: np.ndarray,
    index_arrays: list,
) -> None:
    """
    Fill expanded ND array with source data.

    Uses flat indexing for Rust acceleration.
    """
    if HAS_RUST and expanded.dtype == np.float64 and expanded.ndim <= 4:
        # Compute flat indices
        mesh = np.meshgrid(*index_arrays, indexing="ij")
        flat_idx = np.ravel_multi_index(
            [m.ravel() for m in mesh],
            expanded.shape,
        )
        _rust_fill_nd(
            expanded.ravel(),
            source.ravel().astype(np.float64),
            flat_idx.astype(np.int64),
        )
    else:
        # Pure NumPy fallback
        idx_arrays = np.meshgrid(*index_arrays, indexing="ij")
        expanded[tuple(idx_arrays)] = source
