"""
IceChunk integration for NimbleNd arrays.
"""

from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Hashable,
    List,
    Literal,
    Mapping,
    MutableMapping,
    Optional,
    Tuple,
    Union,
)

import numpy as np
from packaging.version import Version

from ..core import HAS_DASK, Array

# Define types for clarity
Region = Mapping[str, slice | Literal["auto"]] | Literal["auto"] | None
ZarrWriteModes = Literal["w", "w-", "a", "a-", "r+", "r"]

# Import dask conditionally
if HAS_DASK:
    import dask
    import dask.array as da

# Check for IceChunk
try:
    from icechunk import IcechunkStore
    from icechunk.distributed import extract_session, merge_sessions
    from icechunk.vendor.xarray import _choose_default_mode

    HAS_ICECHUNK = True
except ImportError:
    HAS_ICECHUNK = False


@dataclass
class _NimbleNdArrayWriter:
    """
    Write NimbleNd Arrays to a group in an Icechunk store.

    This class handles the writing of NimbleNd arrays to IceChunk stores,
    supporting both eager (in-memory) and lazy (dask) arrays.
    """

    array: Array = field(repr=False)
    store: Any = field(kw_only=True)  # IcechunkStore type

    safe_chunks: bool = field(kw_only=True, default=True)
    _initialized: bool = field(default=False, repr=False)

    # Fields that will be set later
    zarr_store: Any = field(init=False, repr=False, default=None)
    eager_sources: List[np.ndarray] = field(
        init=False, repr=False, default_factory=list
    )
    eager_targets: List[Any] = field(init=False, repr=False, default_factory=list)
    lazy_sources: List[Any] = field(init=False, repr=False, default_factory=list)
    lazy_targets: List[Any] = field(init=False, repr=False, default_factory=list)

    def __post_init__(self) -> None:
        if not HAS_ICECHUNK:
            raise ImportError(
                "IceChunk is required for this functionality. Please install it with pip install icechunk"
            )

        if not isinstance(self.store, IcechunkStore):
            raise ValueError(
                f"Please pass in an icechunk.Session. Received {type(self.store)!r} instead."
            )

    def _open_group(
        self,
        *,
        group: str | None = None,
        mode: ZarrWriteModes | None = None,
        append_dim: Hashable | None = None,
        region: Region = None,
    ) -> None:
        """
        Open a Zarr group in the IceChunk store.

        Parameters
        ----------
        group : str | None, optional
            Group path in the store
        mode : ZarrWriteModes | None, optional
            Write mode ('w', 'a', etc.)
        append_dim : Hashable | None, optional
            Dimension along which to append
        region : Region, optional
            Region in the existing array to write to
        """
        import zarr

        # Use the icechunk mode selection logic
        concrete_mode = _choose_default_mode(
            mode=mode, append_dim=append_dim, region=region
        )

        # Open the group in the store with Zarr v3 format
        root = zarr.open_group(
            store=self.store,
            path=group or "/",
            mode=concrete_mode,
            zarr_format=3,
        )
        self.zarr_store = root

    def _prepare_string_data(self, data, is_lazy=False):
        """
        Prepare string data for storage by converting to bytes.
        Returns the converted data and whether conversion occurred.
        """
        if hasattr(data, "dtype") and data.dtype.kind == "U":
            # Store the original dtype info
            data_dtype = str(data.dtype)

            # Handle different array shapes for string conversion
            if np.isscalar(data) or data.ndim == 0:
                # For scalar data
                if is_lazy and HAS_DASK:
                    # Convert to uint8 array - dask doesn't handle this well,
                    # so compute first and create a new dask array
                    string_bytes = str(data).encode("utf-8")
                    uint8_array = np.frombuffer(string_bytes, dtype=np.uint8)
                    converted_data = da.from_array(uint8_array, chunks="auto")
                else:
                    # For NumPy, directly convert the string to bytes to uint8
                    string_bytes = str(data).encode("utf-8")
                    converted_data = np.frombuffer(string_bytes, dtype=np.uint8)
            else:
                # For array data
                if is_lazy and HAS_DASK:
                    # For dask arrays, compute first
                    data_computed = data.compute()
                    # Convert each string to bytes and then to uint8
                    flat_data = data_computed.flatten()
                    byte_arrays = []
                    for item in flat_data:
                        item_bytes = str(item).encode("utf-8")
                        byte_arrays.append(np.frombuffer(item_bytes, dtype=np.uint8))

                    # Pad to the same length
                    max_len = max(len(arr) for arr in byte_arrays) if byte_arrays else 0
                    padded_arrays = []
                    for arr in byte_arrays:
                        padded = np.zeros(max_len, dtype=np.uint8)
                        padded[: len(arr)] = arr
                        padded_arrays.append(padded)

                    # Stack arrays along a new axis
                    stacked = np.stack(padded_arrays)
                    # Reshape back to original shape plus byte dimension
                    result_shape = data_computed.shape + (max_len,)
                    converted_data = stacked.reshape(result_shape)
                    # Convert back to dask
                    converted_data = da.from_array(converted_data, chunks="auto")
                else:
                    # For NumPy arrays
                    # Convert each string to bytes and then to uint8
                    flat_data = data.flatten()
                    byte_arrays = []
                    for item in flat_data:
                        item_bytes = str(item).encode("utf-8")
                        byte_arrays.append(np.frombuffer(item_bytes, dtype=np.uint8))

                    # Pad to the same length
                    max_len = max(len(arr) for arr in byte_arrays) if byte_arrays else 0
                    padded_arrays = []
                    for arr in byte_arrays:
                        padded = np.zeros(max_len, dtype=np.uint8)
                        padded[: len(arr)] = arr
                        padded_arrays.append(padded)

                    # Stack arrays along a new axis
                    stacked = np.stack(padded_arrays)
                    # Reshape back to original shape plus byte dimension
                    result_shape = data.shape + (max_len,)
                    converted_data = stacked.reshape(result_shape)

            return converted_data, data_dtype

        return data, None

    def write_metadata(self, encoding: Mapping[Any, Any] | None = None) -> None:
        """
        Write metadata and prepare for array writing.

        Parameters
        ----------
        encoding : Mapping[Any, Any] | None, optional
            Encoding options for arrays
        """
        # Create a group for coordinates
        coords_group = self.zarr_store.create_group("coords")

        # Store dimensions in attributes
        self.zarr_store.attrs["dims"] = self.array.dims

        # Store array name if it exists
        if self.array.name is not None:
            self.zarr_store.attrs["name"] = self.array.name

        # Store coordinates for each dimension
        for dim, coord_values in self.array.coords.items():
            # Check if coordinates are strings and need conversion
            coord_values_to_store, coord_dtype = self._prepare_string_data(coord_values)

            # Store original dtype if string conversion was applied
            if coord_dtype is not None:
                coords_group.attrs[f"{dim}_dtype"] = coord_dtype

            # Create the array for coordinates - always use auto-chunking for Zarr v3
            coord_array = coords_group.create_array(
                dim,
                shape=coord_values_to_store.shape,
                dtype=coord_values_to_store.dtype,
                chunks="auto",  # Use auto-chunking for Zarr v3
                compressors="auto",  # Use default compression
            )
            # Write the coordinate data
            coord_array[:] = coord_values_to_store

        # Prepare for data writing
        if encoding is None:
            encoding = {}

        # Extract encoding options
        chunks = encoding.get("chunks", None)
        compression = encoding.get("compression", "blosc")

        # Use Dask chunks if available and not specified in encoding
        if chunks is None and self.array.is_lazy:
            chunks = self.array.data.chunks

        # For Zarr v3, ensure we have a valid chunks value
        if chunks is None:
            chunks = "auto"

        # Check if data is string type and needs conversion
        data_to_store, data_dtype = self._prepare_string_data(
            self.array.data, is_lazy=self.array.is_lazy
        )

        # Store original dtype if string conversion was applied
        if data_dtype is not None:
            self.zarr_store.attrs["data_dtype"] = data_dtype

        # Create the data array with Zarr v3 API
        compressors = "auto" if compression == "blosc" else None

        data_array = self.zarr_store.create_array(
            "data",
            shape=data_to_store.shape,
            chunks=chunks,
            dtype=data_to_store.dtype,
            compressors=compressors,
        )

        # Determine if the source is a lazy array
        if self.array.is_lazy:
            self.lazy_sources.append(data_to_store)
            self.lazy_targets.append(data_array)
        else:
            self.eager_sources.append(data_to_store)
            self.eager_targets.append(data_array)

        self._initialized = True

    def write_eager(self) -> None:
        """
        Write in-memory arrays to the store.
        """
        if not self._initialized:
            raise ValueError("Please call `write_metadata` first.")

        for source, target in zip(self.eager_sources, self.eager_targets):
            target[:] = source

        # Clear the lists after writing
        self.eager_sources = []
        self.eager_targets = []

    def write_lazy(
        self,
        chunkmanager_store_kwargs: MutableMapping[Any, Any] | None = None,
        split_every: int | None = None,
    ) -> None:
        """
        Write lazy (dask) arrays to the store.

        Parameters
        ----------
        chunkmanager_store_kwargs : MutableMapping[Any, Any] | None, optional
            Additional keyword arguments for dask array store operations
        split_every : int | None, optional
            Number of tasks to merge at every tree reduction level
        """
        if not self._initialized:
            raise ValueError("Please call `write_metadata` first.")

        if not self.lazy_sources or not HAS_DASK:
            return

        # Check Dask version for compatibility
        if Version(dask.__version__) < Version("2024.11.0"):
            raise ValueError(
                f"Writing to icechunk requires dask>=2024.11.0 but you have {dask.__version__}. Please upgrade."
            )

        # Set up store kwargs for dask
        store_kwargs = chunkmanager_store_kwargs or {}
        store_kwargs["load_stored"] = False
        store_kwargs["return_stored"] = True

        # Store each lazy array and collect the resulting dask arrays
        stored_arrays = []
        for source, target in zip(self.lazy_sources, self.lazy_targets):
            stored = da.store(source, target, compute=False, **store_kwargs)
            stored_arrays.append(stored)

        # Tree-reduce all changesets if we have them
        if stored_arrays:
            merged_sessions = [
                da.reduction(
                    arr,
                    name="ice-changeset",
                    chunk=extract_session,
                    aggregate=merge_sessions,
                    split_every=split_every,
                    concatenate=False,
                    dtype=object,
                )
                for arr in stored_arrays
            ]
            computed = da.compute(*merged_sessions)
            merged_session = merge_sessions(*computed)
            self.store.session.merge(merged_session)


def to_icechunk(
    obj: Array,
    session: Any,  # Session type
    *,
    group: str | None = None,
    mode: ZarrWriteModes | None = None,
    safe_chunks: bool = True,
    encoding: Mapping[Any, Any] | None = None,
    append_dim: Hashable | None = None,
    region: Region = None,
    chunkmanager_store_kwargs: MutableMapping[Any, Any] | None = None,
    split_every: int | None = None,
) -> None:
    """
    Write a NimbleNd Array to an IceChunk store.

    Parameters
    ----------
    obj : Array
        NimbleNd Array to write
    session : Session
        Writable IceChunk Session
    group : str | None, optional
        Group path in the store
    mode : ZarrWriteModes | None, optional
        Write mode ('w', 'w-', 'a', 'a-', 'r+', 'r')
    safe_chunks : bool, default: True
        If True, only allow writes with safe chunking patterns
    encoding : Mapping[Any, Any] | None, optional
        Encoding options for arrays (compression, chunks, etc.)
    append_dim : Hashable | None, optional
        Dimension along which to append
    region : Region, optional
        Region in the existing array to write to
    chunkmanager_store_kwargs : MutableMapping[Any, Any] | None, optional
        Additional keyword arguments for dask array store operations
    split_every : int | None, optional
        Number of tasks to merge at every tree reduction level

    Returns
    -------
    None
    """
    if not HAS_ICECHUNK:
        raise ImportError(
            "IceChunk is required for this functionality. Please install it with pip install icechunk"
        )

    with session.allow_pickling():
        store = session.store
        writer = _NimbleNdArrayWriter(obj, store=store, safe_chunks=safe_chunks)

        # Open the group
        writer._open_group(group=group, mode=mode, append_dim=append_dim, region=region)

        # Write metadata and prepare for array writing
        writer.write_metadata(encoding)

        # Write in-memory arrays
        writer.write_eager()

        # Write lazy (dask) arrays if present
        writer.write_lazy(
            chunkmanager_store_kwargs=chunkmanager_store_kwargs, split_every=split_every
        )


def from_icechunk(
    session: Any,  # Session type
    *,
    group: str | None = None,
    chunks: Optional[Union[str, Dict[str, int], Tuple[int, ...]]] = None,
) -> Array:
    """
    Read a NimbleNd Array from an IceChunk store.

    Parameters
    ----------
    session : Session
        IceChunk Session
    group : str | None, optional
        Group path in the store
    chunks : Optional[Union[str, Dict[str, int], Tuple[int, ...]]], optional
        Chunks to use when loading the array. If provided and dask is available,
        will return a lazy array with these chunks. If None, returns an eager NumPy array.

    Returns
    -------
    Array
        The loaded array
    """
    if not HAS_ICECHUNK:
        raise ImportError(
            "IceChunk is required for this functionality. Please install it with pip install icechunk"
        )

    import zarr

    # Open the zarr store from the session
    store = session.store
    root = zarr.open_group(
        store=store,
        path=group or "/",
        mode="r",
    )

    # Load dimensions and name from attributes
    dims = root.attrs["dims"]
    name = root.attrs.get("name", None)

    # Load data (potentially as a lazy Dask array)
    if chunks is not None and HAS_DASK:
        import dask.array as da

        component_path = f"{group}/data" if group else "data"
        data = da.from_zarr(store, component=component_path, chunks=chunks)
    else:
        data = root["data"][:]

    # Check if data was stored as strings
    if "data_dtype" in root.attrs:
        original_dtype = root.attrs["data_dtype"]
        if original_dtype.startswith("<U") or original_dtype.startswith("|U"):
            # Convert uint8 data back to strings
            if data.ndim > 1 and data.shape[-1] > 0:
                # Multi-dimensional string array
                flat_shape = data.shape[:-1]
                byte_dim = data.shape[-1]

                # Reshape to 2D for processing
                reshaped = data.reshape(-1, byte_dim)

                # Convert each byte array back to string
                strings = []
                for i in range(reshaped.shape[0]):
                    # Find non-zero bytes
                    mask = reshaped[i] != 0
                    if np.any(mask):
                        bytes_array = reshaped[i][mask].tobytes()
                        strings.append(bytes_array.decode("utf-8", errors="replace"))
                    else:
                        strings.append("")

                # Reshape back to original dimensions
                string_array = np.array(strings, dtype=original_dtype).reshape(
                    flat_shape
                )
                data = string_array
            else:
                # Scalar or 1D string
                if np.any(data != 0):
                    string_val = data.tobytes().decode("utf-8", errors="replace")
                    data = np.array(string_val, dtype=original_dtype)
                else:
                    data = np.array("", dtype=original_dtype)

    # Load coordinates
    coords = {}
    coords_group = root["coords"]

    for dim in dims:
        # Check if this dimension had string coordinates
        dim_dtype = coords_group.attrs.get(f"{dim}_dtype", None)

        # Load coordinates directly from Zarr - there's no path attribute to check
        # for separate .npy files in IcechunkStore
        coord_values = coords_group[dim][:]

        # Check if these were string coordinates
        if dim_dtype is not None and (
            dim_dtype.startswith("<U") or dim_dtype.startswith("|U")
        ):
            # Convert uint8 data back to strings
            if coord_values.ndim > 1 and coord_values.shape[-1] > 0:
                # Multi-dimensional string array
                flat_shape = coord_values.shape[:-1]
                byte_dim = coord_values.shape[-1]

                # Reshape to 2D for processing
                reshaped = coord_values.reshape(-1, byte_dim)

                # Convert each byte array back to string
                strings = []
                for i in range(reshaped.shape[0]):
                    mask = reshaped[i] != 0
                    if np.any(mask):
                        bytes_array = reshaped[i][mask].tobytes()
                        strings.append(bytes_array.decode("utf-8", errors="replace"))
                    else:
                        strings.append("")

                # Reshape back to original dimensions
                coord_values = np.array(strings, dtype=dim_dtype).reshape(flat_shape)

        coords[dim] = coord_values

    return Array(data, coords, dims, name)
