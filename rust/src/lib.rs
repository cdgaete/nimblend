//! Rust accelerated functions for nimblend
//!
//! This module provides high-performance implementations of the hot paths
//! identified in profiling: coordinate alignment and index mapping.

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

/// Build a coordinate-to-index mapping from a numpy array of string coordinates.
/// Returns indices for each source coord in target coords.
#[pyfunction]
fn map_coords_to_indices(
    py: Python<'_>,
    source_coords: Vec<String>,
    target_coords: Vec<String>,
) -> PyResult<Bound<'_, PyArray1<i64>>> {
    // Build target lookup
    let target_map: HashMap<&str, i64> = target_coords
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i as i64))
        .collect();

    // Map source to target indices
    let indices: Vec<i64> = source_coords
        .iter()
        .map(|s| *target_map.get(s.as_str()).unwrap_or(&-1))
        .collect();

    Ok(PyArray1::from_vec(py, indices))
}

/// Fill expanded 2D array with source data at specified row/col indices.
/// This is the core hot path - replaces meshgrid + advanced indexing.
///
/// Uses parallel iteration with rayon for large arrays.
#[pyfunction]
fn fill_expanded_2d_f64(
    _py: Python<'_>,
    expanded: &Bound<'_, PyArray2<f64>>,
    source: PyReadonlyArray2<'_, f64>,
    row_indices: PyReadonlyArray1<'_, i64>,
    col_indices: PyReadonlyArray1<'_, i64>,
) -> PyResult<()> {
    let src = source.as_array();
    let row_idx = row_indices.as_slice()?;
    let col_idx = col_indices.as_slice()?;

    // SAFETY: exclusive access via GIL
    unsafe {
        let mut exp = expanded.as_array_mut();
        let (nrows, ncols) = (src.nrows(), src.ncols());

        // Parallel fill for large arrays
        if nrows * ncols > 10000 {
            // Collect updates first (can't mutate in parallel directly)
            let updates: Vec<(usize, usize, f64)> = (0..nrows)
                .into_par_iter()
                .flat_map(|i| {
                    let ri = row_idx[i] as usize;
                    (0..ncols)
                        .map(move |j| (ri, col_idx[j] as usize, src[[i, j]]))
                        .collect::<Vec<_>>()
                })
                .collect();

            for (r, c, v) in updates {
                exp[[r, c]] = v;
            }
        } else {
            // Sequential for small arrays
            for i in 0..nrows {
                let ri = row_idx[i] as usize;
                for j in 0..ncols {
                    let ci = col_idx[j] as usize;
                    exp[[ri, ci]] = src[[i, j]];
                }
            }
        }
    }
    Ok(())
}

/// Fill expanded ND array - generic version using flat indices.
/// Slower than specialized 2D but works for any dimensionality.
#[pyfunction]
fn fill_expanded_nd_f64(
    _py: Python<'_>,
    expanded: &Bound<'_, PyArray1<f64>>,  // Flattened view
    source: PyReadonlyArray1<'_, f64>,     // Flattened source
    flat_indices: PyReadonlyArray1<'_, i64>,
) -> PyResult<()> {
    let src = source.as_slice()?;
    let indices = flat_indices.as_slice()?;

    unsafe {
        let mut exp = expanded.as_array_mut();
        for (i, &idx) in indices.iter().enumerate() {
            exp[idx as usize] = src[i];
        }
    }
    Ok(())
}

/// Python module definition
#[pymodule]
fn nimblend_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(map_coords_to_indices, m)?)?;
    m.add_function(wrap_pyfunction!(fill_expanded_2d_f64, m)?)?;
    m.add_function(wrap_pyfunction!(fill_expanded_nd_f64, m)?)?;
    Ok(())
}
