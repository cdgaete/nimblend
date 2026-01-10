//! Rust accelerated functions for nimblend
//!
//! This module provides high-performance implementations of the hot paths
//! identified in profiling: coordinate alignment and index mapping.

use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods};
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use std::collections::HashMap;

/// Build a coordinate-to-index mapping from string coordinates.
/// Returns indices for each source coord in target coords.
#[pyfunction]
fn map_coords_to_indices<'py>(
    py: Python<'py>,
    source_coords: Vec<String>,
    target_coords: Vec<String>,
) -> Bound<'py, PyArray1<i64>> {
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

    Array1::from_vec(indices).into_pyarray_bound(py)
}

/// Fill expanded 2D array with source data at specified row/col indices.
/// This is the core hot path - replaces meshgrid + advanced indexing.
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

        // Simple sequential fill - avoids allocation overhead
        for i in 0..nrows {
            let ri = row_idx[i] as usize;
            for j in 0..ncols {
                let ci = col_idx[j] as usize;
                exp[[ri, ci]] = src[[i, j]];
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
    expanded: &Bound<'_, PyArray1<f64>>,
    source: PyReadonlyArray1<'_, f64>,
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
