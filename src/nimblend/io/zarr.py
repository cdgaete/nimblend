"""
Zarr storage support for NimbleNd arrays.
"""

import json
import os
from typing import Dict, Optional, Tuple, Union

import numpy as np

from ..core import HAS_DASK, Array


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
        # Check Zarr version to handle API differences
        import pkg_resources
        import zarr

        zarr_version = pkg_resources.get_distribution("zarr").version
        is_zarr_v3 = zarr_version.startswith("3.")
    except ImportError:
        raise ImportError(
            "Zarr is required for this functionality. Please install it with pip install zarr"
        )

    # Use file system storage directly to avoid string type warnings
    # Create directory if it doesn't exist
    os.makedirs(path, exist_ok=True)

    # Store metadata in a separate JSON file instead of Zarr attributes
    # This avoids Zarr dealing with string types in attributes
    metadata = {
        "dims": array.dims,
        "name": array.name,
        "data_shape": array.data.shape if hasattr(array.data, "shape") else None,
        "coords": {},
    }

    # Create a Zarr group for the data and coordinates
    root = zarr.open_group(path, mode=mode)

    # Process coordinates first
    coords_group = root.create_group("coords")
    for dim, coord_values in array.coords.items():
        # Save metadata about this coordinate
        coord_metadata = {
            "shape": coord_values.shape,
            "dtype": str(coord_values.dtype),
            "is_string": coord_values.dtype.kind == "U",
        }
        metadata["coords"][dim] = coord_metadata

        # If string coordinate, convert to bytes for storage
        if is_zarr_v3 and coord_values.dtype.kind == "U":
            # Convert to bytes as NumPy object array
            bytes_coords = np.array(
                [str(x).encode("utf-8") for x in coord_values.flatten()], dtype=object
            )

            # Save the bytes to a npy file instead of zarr to avoid warnings
            np_path = os.path.join(path, f"coords_{dim}.npy")
            np.save(np_path, bytes_coords)
        else:
            # Create the array for coordinates
            if is_zarr_v3:
                # Always use auto-chunking for Zarr v3
                if compression == "blosc":
                    compressors = "auto"
                else:
                    compressors = None

                coord_array = coords_group.create_array(
                    dim,
                    shape=coord_values.shape,
                    dtype=coord_values.dtype,
                    chunks="auto",
                    compressors=compressors,
                )
            else:
                # For Zarr v2
                coord_array = coords_group.create_dataset(
                    dim, data=coord_values, compression=compression
                )

            # Write the coordinate data
            coord_array[:] = coord_values

    # Handle the main data array
    is_string_data = hasattr(array.data, "dtype") and array.data.dtype.kind == "U"
    metadata["is_string_data"] = is_string_data

    if is_string_data and is_zarr_v3:
        # For string data in Zarr v3, save as NumPy file to avoid warnings
        metadata["data_dtype"] = str(array.data.dtype)

        if array.is_lazy and HAS_DASK:
            # For dask arrays, compute first
            computed_data = array.data.compute()
            # Convert to bytes array
            bytes_data = np.array(
                [str(x).encode("utf-8") for x in computed_data.flatten()], dtype=object
            )
        else:
            # For numpy arrays, convert directly
            bytes_data = np.array(
                [str(x).encode("utf-8") for x in array.data.flatten()], dtype=object
            )

        # Save to NumPy file
        np_path = os.path.join(path, "data.npy")
        np.save(np_path, bytes_data)
    else:
        # For non-string data or Zarr v2, use zarr directly
        if array.is_lazy:
            # Get chunks from Dask array if available
            chunks = array.data.chunks

            # Store data - dask.array.to_zarr is more efficient for lazy arrays
            import dask.array as da

            da.to_zarr(
                array.data,
                url=path,
                component="data",
                overwrite=True,
                compute=True,
                return_stored=False,
                storage_options=None,
                chunks=chunks,
            )
        else:
            # Store data directly - handle Zarr v2 vs v3 differences
            if is_zarr_v3:
                # For Zarr v3
                if compression == "blosc":
                    compressors = "auto"
                else:
                    compressors = None  # No compression

                data_array = root.create_array(
                    "data",
                    shape=array.data.shape,
                    chunks="auto",  # Always provide a valid chunks value
                    dtype=array.data.dtype,
                    compressors=compressors,
                )
                # Write the data
                data_array[:] = array.data
            else:
                # For Zarr v2
                root.create_array("data", data=array.data, compression=compression)

    # Write metadata to JSON file
    with open(os.path.join(path, "nimblend_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def from_zarr(
    path: str, chunks: Optional[Union[str, Dict[str, int], Tuple[int, ...]]] = None
) -> Array:
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
        raise ImportError(
            "Zarr is required for this functionality. Please install it with pip install zarr"
        )

    # Load metadata from JSON file
    metadata_path = os.path.join(path, "nimblend_metadata.json")

    if os.path.exists(metadata_path):
        # New format - load metadata from JSON
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        dims = metadata["dims"]
        name = metadata["name"]
        is_string_data = metadata.get("is_string_data", False)

        # Load coordinates
        coords = {}
        for dim in dims:
            dim_metadata = metadata["coords"].get(dim, {})
            is_string_coord = dim_metadata.get("is_string", False)

            if is_string_coord:
                # Load from NumPy file
                np_path = os.path.join(path, f"coords_{dim}.npy")
                if os.path.exists(np_path):
                    bytes_data = np.load(np_path, allow_pickle=True)

                    # Convert bytes back to strings
                    orig_dtype = dim_metadata["dtype"]
                    orig_shape = tuple(dim_metadata["shape"])

                    # Convert and reshape
                    string_data = np.array(
                        [b.decode("utf-8", errors="replace") for b in bytes_data],
                        dtype=orig_dtype,
                    )
                    if len(orig_shape) > 1:
                        string_data = string_data.reshape(orig_shape)

                    coords[dim] = string_data
                else:
                    # Fallback to zarr stored data
                    coords_group = zarr.open_group(path, mode="r")["coords"]
                    coords[dim] = coords_group[dim][:]
            else:
                # Load from Zarr
                coords_group = zarr.open_group(path, mode="r")["coords"]
                coords[dim] = coords_group[dim][:]

        # Load the main data array
        if is_string_data:
            # Load string data from NumPy file
            np_path = os.path.join(path, "data.npy")
            if os.path.exists(np_path):
                bytes_data = np.load(np_path, allow_pickle=True)

                # Convert bytes back to strings
                orig_dtype = metadata["data_dtype"]
                orig_shape = tuple(metadata["data_shape"])

                # Convert and reshape
                string_data = np.array(
                    [b.decode("utf-8", errors="replace") for b in bytes_data],
                    dtype=orig_dtype,
                )
                data = string_data.reshape(orig_shape)

                # Convert to dask if needed
                if chunks is not None and HAS_DASK:
                    import dask.array as da

                    data = da.from_array(data, chunks=chunks)
            else:
                # Fallback to zarr data
                if chunks is not None and HAS_DASK:
                    import dask.array as da

                    data = da.from_zarr(path, component="data", chunks=chunks)
                else:
                    data = zarr.open_group(path, mode="r")["data"][:]
        else:
            # Load regular data from Zarr
            if chunks is not None and HAS_DASK:
                import dask.array as da

                data = da.from_zarr(path, component="data", chunks=chunks)
            else:
                data = zarr.open_group(path, mode="r")["data"][:]
    else:
        # Old format or standard zarr - fall back to previous approach
        root = zarr.open_group(path, mode="r")

        # Load dimensions from attributes
        dims = root.attrs["dims"]

        # Load coordinates
        coords = {}
        coords_group = root["coords"]
        for dim in dims:
            coords[dim] = coords_group[dim][:]

        # Load name if it exists
        name = root.attrs.get("name", None)

        # Load the data array
        if chunks is not None and HAS_DASK:
            import dask.array as da

            data = da.from_zarr(path, component="data", chunks=chunks)
        else:
            data = root["data"][:]

    return Array(data, coords, dims, name)
