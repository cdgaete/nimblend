"""
IceChunk integration for NimbleNd arrays.
"""

from dataclasses import dataclass, field
from typing import Any, Hashable, List, Literal, Mapping, MutableMapping

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
            # Create the array for coordinates - always use auto-chunking for Zarr v3
            coord_array = coords_group.create_array(
                dim,
                shape=coord_values.shape,
                dtype=coord_values.dtype,
                chunks="auto",  # Use auto-chunking for Zarr v3
                compressors="auto",  # Use default compression
            )
            # Write the coordinate data
            coord_array[:] = coord_values

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

        # Create the data array with Zarr v3 API
        compressors = "auto" if compression == "blosc" else None

        data_array = self.zarr_store.create_array(
            "data",
            shape=self.array.data.shape,
            chunks=chunks,
            dtype=self.array.data.dtype,
            compressors=compressors,
        )

        # Determine if the source is a lazy array
        if self.array.is_lazy:
            self.lazy_sources.append(self.array.data)
            self.lazy_targets.append(data_array)
        else:
            self.eager_sources.append(self.array.data)
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
