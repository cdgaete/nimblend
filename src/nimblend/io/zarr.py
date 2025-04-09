"""
Zarr storage support for NimbleNd arrays.
"""
from typing import Dict, Optional, Union, Tuple
from ..core import Array, HAS_DASK

def to_zarr(
    array: Array,
    path: str,
    mode: str = "w",
    compression: Optional[str] = "blosc",
    dimension_separator: str = "/",
) -> None:
    """
    Save a NimbleNd array to Zarr format.

    Parameters
    ----------
    array : Array
        The array to save
    path : str
        Path to save the Zarr array
    mode : str, optional
        Mode to open the Zarr store ('w' for new, 'a' for append, default: 'w')
    compression : Optional[str], optional
        Compression method (default: "blosc")
    dimension_separator : str, optional
        Separator for dimension coordinates (default: "/")
    """
    try:
        import zarr
        # Check Zarr version to handle API differences
        import pkg_resources
        zarr_version = pkg_resources.get_distribution("zarr").version
        is_zarr_v3 = zarr_version.startswith("3.")
    except ImportError:
        raise ImportError("Zarr is required for this functionality. Please install it with pip install zarr")

    # Create a Zarr group
    root = zarr.open_group(path, mode=mode)

    # Store the array data with appropriate chunking
    if array.is_lazy:
        # Get chunks from Dask array if available
        chunks = array.data.chunks

        # Store data - dask.array.to_zarr is more efficient for lazy arrays
        import dask.array as da
        da.to_zarr(
            array.data,
            url=path,
            component='data',
            overwrite=True,
            compute=True,
            return_stored=False,
            storage_options=None,
            chunks=chunks
        )
    else:
        # Store data directly - handle Zarr v2 vs v3 differences
        if is_zarr_v3:
            # For Zarr v3
            # Always provide chunks parameter - use "auto" if none specified
            # This is critical for Zarr v3 which can't handle None for chunks
            if compression == "blosc":
                compressors = "auto"
            else:
                compressors = None  # No compression
            
            # Create the array with automatic chunking if none specified
            data_array = root.create_array(
                'data',
                shape=array.data.shape,
                chunks="auto",  # Always provide a valid chunks value
                dtype=array.data.dtype,
                compressors=compressors
            )
            # Write the data
            data_array[:] = array.data
        else:
            # For Zarr v2
            root.create_dataset(
                'data',
                data=array.data,
                compression=compression
            )

    # Store dimensions
    root.attrs['dims'] = array.dims

    # Store coordinates for each dimension
    coords_group = root.create_group('coords')
    for dim, coord_values in array.coords.items():
        if is_zarr_v3:
            # For Zarr v3
            if compression == "blosc":
                compressors = "auto"
            else:
                compressors = None
            
            # Create the array for coordinates - always specify chunks explicitly
            coord_array = coords_group.create_array(
                dim,
                shape=coord_values.shape,
                dtype=coord_values.dtype,
                chunks="auto",  # Always use auto-chunking to avoid None
                compressors=compressors
            )
            # Write the coordinate data
            coord_array[:] = coord_values
        else:
            # For Zarr v2
            coords_group.create_dataset(dim, data=coord_values, compression=compression)

    # Store array name if it exists
    if array.name is not None:
        root.attrs['name'] = array.name


def from_zarr(path: str, chunks: Optional[Union[str, Dict[str, int], Tuple[int, ...]]] = None) -> Array:
    """
    Load a NimbleNd array from Zarr format.

    Parameters
    ----------
    path : str
        Path to the Zarr array
    chunks : Optional[Union[str, Dict[str, int], Tuple[int, ...]]], optional
        Chunks to use when loading the array. If provided and dask is available,
        will return a lazy array with these chunks. If None, returns an eager NumPy array.

    Returns
    -------
    Array
        The loaded array
    """
    try:
        import zarr
    except ImportError:
        raise ImportError("Zarr is required for this functionality. Please install it with pip install zarr")

    # Open the Zarr group
    root = zarr.open_group(path, mode='r')

    # Load dimensions
    dims = root.attrs['dims']

    # Load coordinates
    coords = {}
    coords_group = root['coords']
    for dim in dims:
        coords[dim] = coords_group[dim][:]

    # Load name if it exists
    name = root.attrs.get('name', None)

    # Determine if we should load as Dask array
    if chunks is not None and HAS_DASK:
        import dask.array as da
        data = da.from_zarr(path, component='data', chunks=chunks)
    else:
        # Load as NumPy array
        data = root['data'][:]

    return Array(data, coords, dims, name)
