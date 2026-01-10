//! Rust accelerated functions for nimblend
//!
//! This module provides high-performance implementations of the hot paths
//! identified in profiling: coordinate alignment and index mapping.

use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

/// Wrapper to make raw pointer Send+Sync for parallel access.
/// SAFETY: Caller must ensure no data races (disjoint index access).
#[derive(Clone, Copy)]
struct SendPtr(*mut f64);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

impl SendPtr {
    #[inline(always)]
    fn ptr(self) -> *mut f64 {
        self.0
    }
}

/// Build a coordinate-to-index mapping from string coordinates.
#[pyfunction]
fn map_coords_to_indices<'py>(
    py: Python<'py>,
    source_coords: Vec<String>,
    target_coords: Vec<String>,
) -> Bound<'py, PyArray1<i64>> {
    let target_map: HashMap<&str, i64> = target_coords
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i as i64))
        .collect();

    let indices: Vec<i64> = source_coords
        .iter()
        .map(|s| *target_map.get(s.as_str()).unwrap_or(&-1))
        .collect();

    Array1::from_vec(indices).into_pyarray_bound(py)
}

/// Fill expanded 2D array with source data at specified row/col indices.
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

    unsafe {
        let mut exp = expanded.as_array_mut();
        let (nrows, ncols) = (src.nrows(), src.ncols());
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

/// Fill expanded ND array using per-dimension index arrays.
#[pyfunction]
fn fill_expanded_nd_from_indices(
    _py: Python<'_>,
    expanded: &Bound<'_, PyArray1<f64>>,
    source: PyReadonlyArray1<'_, f64>,
    index_arrays: Vec<PyReadonlyArray1<'_, i64>>,
    expanded_strides: Vec<i64>,
) -> PyResult<()> {
    let src = source.as_slice()?;
    let n_elements = src.len();
    let idx_slices: Vec<&[i64]> = index_arrays
        .iter()
        .map(|arr| arr.as_slice())
        .collect::<Result<Vec<_>, _>>()?;
    let ndim = idx_slices.len();
    let strides: Vec<i64> = expanded_strides;
    let shape: Vec<usize> = idx_slices.iter().map(|s| s.len()).collect();

    unsafe {
        let mut exp = expanded.as_array_mut();
        let mut coords = vec![0usize; ndim];
        for src_idx in 0..n_elements {
            let mut flat_idx: i64 = 0;
            for d in 0..ndim {
                flat_idx += idx_slices[d][coords[d]] * strides[d];
            }
            exp[flat_idx as usize] = src[src_idx];
            for d in (0..ndim).rev() {
                coords[d] += 1;
                if coords[d] < shape[d] { break; }
                coords[d] = 0;
            }
        }
    }
    Ok(())
}

/// Fill expanded ND array using flat indices.
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

/// Scatter-add rows.
#[pyfunction]
fn scatter_add_2d_rows(
    _py: Python<'_>,
    result: &Bound<'_, PyArray2<f64>>,
    source: PyReadonlyArray2<'_, f64>,
    row_indices: PyReadonlyArray1<'_, i64>,
) -> PyResult<()> {
    let src = source.as_array();
    let row_idx = row_indices.as_slice()?;
    unsafe {
        let mut res = result.as_array_mut();
        let (nrows, ncols) = (src.nrows(), src.ncols());
        for i in 0..nrows {
            let ri = row_idx[i] as usize;
            for j in 0..ncols {
                res[[ri, j]] += src[[i, j]];
            }
        }
    }
    Ok(())
}

/// Sequential aligned binary operation (for smaller arrays)
#[pyfunction]
fn aligned_binop_2d(
    _py: Python<'_>,
    result: &Bound<'_, PyArray2<f64>>,
    source1: PyReadonlyArray2<'_, f64>,
    row_indices1: PyReadonlyArray1<'_, i64>,
    source2: PyReadonlyArray2<'_, f64>,
    row_indices2: PyReadonlyArray1<'_, i64>,
    op: &str,
) -> PyResult<()> {
    let src1 = source1.as_array();
    let src2 = source2.as_array();
    let row_idx1 = row_indices1.as_slice()?;
    let row_idx2 = row_indices2.as_slice()?;

    unsafe {
        let mut res = result.as_array_mut();
        let ncols = src1.ncols();

        for (i, &ri) in row_idx1.iter().enumerate() {
            let ri = ri as usize;
            for j in 0..ncols {
                res[[ri, j]] = src1[[i, j]];
            }
        }

        match op {
            "add" => {
                for (i, &ri) in row_idx2.iter().enumerate() {
                    let ri = ri as usize;
                    for j in 0..ncols { res[[ri, j]] += src2[[i, j]]; }
                }
            }
            "sub" => {
                for (i, &ri) in row_idx2.iter().enumerate() {
                    let ri = ri as usize;
                    for j in 0..ncols { res[[ri, j]] -= src2[[i, j]]; }
                }
            }
            "mul" => {
                for (i, &ri) in row_idx2.iter().enumerate() {
                    let ri = ri as usize;
                    for j in 0..ncols { res[[ri, j]] *= src2[[i, j]]; }
                }
            }
            "div" => {
                for (i, &ri) in row_idx2.iter().enumerate() {
                    let ri = ri as usize;
                    for j in 0..ncols { res[[ri, j]] /= src2[[i, j]]; }
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// Parallel aligned binary operation for large 2D arrays.
/// Uses rayon to parallelize across rows.
#[pyfunction]
fn aligned_binop_2d_parallel(
    py: Python<'_>,
    result: &Bound<'_, PyArray2<f64>>,
    source1: PyReadonlyArray2<'_, f64>,
    row_indices1: PyReadonlyArray1<'_, i64>,
    source2: PyReadonlyArray2<'_, f64>,
    row_indices2: PyReadonlyArray1<'_, i64>,
    op: &str,
) -> PyResult<()> {
    let src1 = source1.as_array();
    let src2 = source2.as_array();
    let row_idx1 = row_indices1.as_slice()?;
    let row_idx2 = row_indices2.as_slice()?;
    let ncols = src1.ncols();
    let op_code: u8 = match op {
        "add" => 0, "sub" => 1, "mul" => 2, "div" => 3, _ => 0,
    };

    let res_ptr = SendPtr(result.as_raw_array_mut().as_mut_ptr());
    let res_stride = result.strides()[0] as usize / 8;

    py.allow_threads(|| {
        // Phase 1: Fill from source1
        row_idx1.par_iter().enumerate().for_each(|(i, &ri)| {
            let ri = ri as usize;
            unsafe {
                let row_ptr = res_ptr.ptr().add(ri * res_stride);
                for j in 0..ncols {
                    *row_ptr.add(j) = src1[[i, j]];
                }
            }
        });

        // Phase 2: Apply operation with source2
        row_idx2.par_iter().enumerate().for_each(|(i, &ri)| {
            let ri = ri as usize;
            unsafe {
                let row_ptr = res_ptr.ptr().add(ri * res_stride);
                match op_code {
                    0 => { for j in 0..ncols { *row_ptr.add(j) += src2[[i, j]]; } }
                    1 => { for j in 0..ncols { *row_ptr.add(j) -= src2[[i, j]]; } }
                    2 => { for j in 0..ncols { *row_ptr.add(j) *= src2[[i, j]]; } }
                    3 => { for j in 0..ncols { *row_ptr.add(j) /= src2[[i, j]]; } }
                    _ => {}
                }
            }
        });
    });
    Ok(())
}

/// Parallel element-wise binary operation on flat arrays.
#[pyfunction]
fn elementwise_binop_parallel(
    py: Python<'_>,
    result: &Bound<'_, PyArray1<f64>>,
    source1: PyReadonlyArray1<'_, f64>,
    source2: PyReadonlyArray1<'_, f64>,
    op: &str,
) -> PyResult<()> {
    let src1 = source1.as_slice()?;
    let src2 = source2.as_slice()?;
    let op_code: u8 = match op {
        "add" => 0, "sub" => 1, "mul" => 2, "div" => 3, _ => 0,
    };

    let res_ptr = SendPtr(result.as_raw_array_mut().as_mut_ptr());
    let n = src1.len();

    py.allow_threads(|| {
        (0..n).into_par_iter().for_each(|i| {
            unsafe {
                let val = match op_code {
                    0 => src1[i] + src2[i],
                    1 => src1[i] - src2[i],
                    2 => src1[i] * src2[i],
                    3 => src1[i] / src2[i],
                    _ => src1[i] + src2[i],
                };
                *res_ptr.ptr().add(i) = val;
            }
        });
    });
    Ok(())
}

/// Python module definition
#[pymodule]
fn nimblend_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(map_coords_to_indices, m)?)?;
    m.add_function(wrap_pyfunction!(fill_expanded_2d_f64, m)?)?;
    m.add_function(wrap_pyfunction!(fill_expanded_nd_f64, m)?)?;
    m.add_function(wrap_pyfunction!(fill_expanded_nd_from_indices, m)?)?;
    m.add_function(wrap_pyfunction!(scatter_add_2d_rows, m)?)?;
    m.add_function(wrap_pyfunction!(aligned_binop_2d, m)?)?;
    m.add_function(wrap_pyfunction!(aligned_binop_2d_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(elementwise_binop_parallel, m)?)?;
    Ok(())
}
