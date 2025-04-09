"""
Operations between arrays for NimbleNd.
"""
import numpy as np
from typing import List, Dict, Optional, Union, Tuple, Any
from .core import Array, HAS_DASK

# Import dask conditionally
if HAS_DASK:
    import dask.array as da

def concat(arrays: List[Array], dim: str, coords: Optional[np.ndarray] = None) -> Array:
    """
    Concatenate arrays along a dimension.

    Parameters
    ----------
    arrays : List[Array]
        Arrays to concatenate
    dim : str
        Dimension along which to concatenate
    coords : Optional[np.ndarray], optional
        Coordinates to use for the concatenated dimension.
        If not provided, coordinates will be inferred.

    Returns
    -------
    Array
        Concatenated array
    """
    if not arrays:
        raise ValueError("No arrays provided for concatenation")

    # Check if all arrays have the same dimensions except the concat dimension
    base_dims = set(arrays[0].dims)
    for arr in arrays[1:]:
        if set(arr.dims) != base_dims and set(arr.dims) != base_dims | {dim}:
            raise ValueError(f"Arrays have inconsistent dimensions: {arr.dims} vs {arrays[0].dims}")

    # Check if inputs are lazy arrays
    is_lazy = any(arr.is_lazy for arr in arrays)

    # Check if the concat dimension exists in all arrays
    dim_exists = [dim in arr.dims for arr in arrays]

    if all(dim_exists):
        # Dimension exists in all arrays - standard concatenation
        # Get the dimensions from the first array
        result_dims = arrays[0].dims.copy()

        # Get the coordinates for non-concatenated dimensions
        # (ensuring they match across all arrays)
        result_coords = {}
        for d in result_dims:
            if d != dim:
                coords_set = set(tuple(arr.coords[d]) for arr in arrays)
                if len(coords_set) > 1:
                    raise ValueError(f"Arrays have different coordinates for dimension '{d}'")
                result_coords[d] = arrays[0].coords[d].copy()

        # Concatenate the coordinates for the concat dimension
        if coords is None:
            concat_coords = []
            for arr in arrays:
                concat_coords.append(arr.coords[dim])
            result_coords[dim] = np.concatenate(concat_coords)
        else:
            result_coords[dim] = coords
            if len(coords) != sum(arr.coords[dim].size for arr in arrays):
                raise ValueError("Provided coordinates do not match the concatenated array size")

        # Concatenate the data
        concat_axis = result_dims.index(dim)
        data_arrays = [arr.data for arr in arrays]

        # Determine if we should use dask.array.concatenate
        if is_lazy and HAS_DASK:
            result_data = da.concatenate(data_arrays, axis=concat_axis)
        else:
            result_data = np.concatenate(data_arrays, axis=concat_axis)

        return Array(result_data, result_coords, result_dims)

    elif not any(dim_exists):
        # Dimension doesn't exist in any array - create a new dimension
        # Get the dimensions from the first array
        result_dims = arrays[0].dims.copy()
        result_dims.append(dim)

        # Get the coordinates for existing dimensions
        # (ensuring they match across all arrays)
        result_coords = {}
        for d in arrays[0].dims:
            coords_set = set(tuple(arr.coords[d]) for arr in arrays)
            if len(coords_set) > 1:
                raise ValueError(f"Arrays have different coordinates for dimension '{d}'")
            result_coords[d] = arrays[0].coords[d].copy()

        # Create coordinates for the new dimension
        if coords is None:
            result_coords[dim] = np.arange(len(arrays))
        else:
            result_coords[dim] = coords
            if len(coords) != len(arrays):
                raise ValueError("Provided coordinates do not match the number of arrays")

        # Stack the data
        data_arrays = [arr.data for arr in arrays]

        # Determine if we should use dask.array.stack
        if is_lazy and HAS_DASK:
            result_data = da.stack(data_arrays, axis=-1)
        else:
            result_data = np.stack(data_arrays, axis=-1)

        return Array(result_data, result_coords, result_dims)

    else:
        # Mixed case - some arrays have the dimension, some don't
        raise ValueError("Inconsistent presence of concatenation dimension across arrays")


def stack(arrays: List[Array], dim: str, coords: Optional[np.ndarray] = None) -> Array:
    """
    Stack arrays along a new dimension.

    Parameters
    ----------
    arrays : List[Array]
        Arrays to stack
    dim : str
        Name of the new dimension
    coords : Optional[np.ndarray], optional
        Coordinates to use for the new dimension.
        If not provided, will use range(len(arrays)).

    Returns
    -------
    Array
        Stacked array
    """
    if not arrays:
        raise ValueError("No arrays provided for stacking")

    # Check if inputs are lazy arrays
    is_lazy = any(arr.is_lazy for arr in arrays)

    # Check if all arrays have the same dimensions
    base_dims = tuple(arrays[0].dims)
    for arr in arrays[1:]:
        if tuple(arr.dims) != base_dims:
            raise ValueError(f"Arrays have inconsistent dimensions: {arr.dims} vs {base_dims}")

    # Check if the new dimension already exists
    if dim in base_dims:
        raise ValueError(f"Dimension '{dim}' already exists in the arrays")

    # Get the coordinates for existing dimensions
    # (ensuring they match across all arrays)
    result_coords = {}
    for d in base_dims:
        coords_set = set(tuple(arr.coords[d]) for arr in arrays)
        if len(coords_set) > 1:
            raise ValueError(f"Arrays have different coordinates for dimension '{d}'")
        result_coords[d] = arrays[0].coords[d].copy()

    # Create coordinates for the new dimension
    if coords is None:
        result_coords[dim] = np.arange(len(arrays))
    else:
        result_coords[dim] = coords
        if len(coords) != len(arrays):
            raise ValueError("Provided coordinates do not match the number of arrays")

    # Stack the data
    data_arrays = [arr.data for arr in arrays]

    # Determine if we should use dask.array.stack
    if is_lazy and HAS_DASK:
        result_data = da.stack(data_arrays, axis=0)
    else:
        result_data = np.stack(data_arrays, axis=0)

    # Create the result dimensions by adding the new dimension at the beginning
    result_dims = [dim] + list(base_dims)

    return Array(result_data, result_coords, result_dims)
